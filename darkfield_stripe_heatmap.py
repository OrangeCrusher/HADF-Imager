#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dark-field stripe heatmap generator v4: soft-gradient stripe map.

This version keeps the v3 idea of detecting the whole broad dark stripe by its
left and right boundaries, but the final heatmap is NOT a hard rectangular fill.
It creates a smooth, original-image-like gradient inside the stripe.

Typical usage:
    python darkfield_stripe_heatmap_v4.py 21.png --out-dir fig3_v4
    python darkfield_stripe_heatmap_v4.py 1_5.png --out-dir fig3_v4

Recommended for a soft gradient result:
    python darkfield_stripe_heatmap_v4.py 21.png --out-dir fig3_v4 --gradient-mode combined --soft-edge 80

Manual boundary mode:
    python darkfield_stripe_heatmap_v4.py 21.png --out-dir fig3_v4 --x1 470 --x2 1590
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imwrite_unicode(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        ext = ".png"
        path = path.with_suffix(ext)
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def to_gray_float(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    gray = gray.astype(np.float32)
    lo, hi = np.percentile(gray, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(gray.min()), float(gray.max())
    return np.clip((gray - lo) / (hi - lo + 1e-8), 0, 1)


def robust_minmax(x: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_p, hi_p])
    if hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    return cv2.GaussianBlur(x.reshape(1, -1).astype(np.float32), (0, 0), sigma).ravel()


def nonmax_suppression_1d(indices: np.ndarray, min_distance: int, max_count: int) -> list[int]:
    chosen: list[int] = []
    for idx in indices:
        idx_i = int(idx)
        if all(abs(idx_i - c) >= min_distance for c in chosen):
            chosen.append(idx_i)
        if len(chosen) >= max_count:
            break
    return chosen


def select_edge_candidates(
    derivative: np.ndarray,
    border: int,
    min_distance: int,
    max_candidates: int,
) -> tuple[list[int], list[int]]:
    w = len(derivative)
    valid = np.arange(border, w - border)

    neg_strength = -derivative[valid]
    neg_order = valid[np.argsort(neg_strength)[::-1]]

    pos_strength = derivative[valid]
    pos_order = valid[np.argsort(pos_strength)[::-1]]

    left_candidates = nonmax_suppression_1d(
        neg_order,
        min_distance=min_distance,
        max_count=max_candidates,
    )
    right_candidates = nonmax_suppression_1d(
        pos_order,
        min_distance=min_distance,
        max_count=max_candidates,
    )

    return left_candidates, right_candidates


def detect_stripe_by_edge_pair(
    gray: np.ndarray,
    profile_sigma: float,
    ignore_border_frac: float,
    min_width: int,
    max_width_frac: float,
    edge_min_distance: int,
    max_candidates: int,
    center_prior: float,
    min_contrast: float,
    manual_x1: int | None,
    manual_x2: int | None,
) -> tuple[int, int, dict]:
    h, w = gray.shape
    gray_smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)

    profile = np.median(gray_smooth, axis=0)
    profile_sm = smooth_1d(profile, sigma=profile_sigma)
    derivative = np.gradient(profile_sm)

    if manual_x1 is not None and manual_x2 is not None:
        x1 = int(np.clip(min(manual_x1, manual_x2), 0, w - 1))
        x2 = int(np.clip(max(manual_x1, manual_x2), 0, w - 1))
        return x1, x2, {
            "method": "manual",
            "profile": profile,
            "profile_smooth": profile_sm,
            "derivative": derivative,
            "left_candidates": [],
            "right_candidates": [],
            "best_score": None,
        }

    border = int(round(w * ignore_border_frac))
    max_width = int(round(w * max_width_frac))

    left_candidates, right_candidates = select_edge_candidates(
        derivative=derivative,
        border=border,
        min_distance=edge_min_distance,
        max_candidates=max_candidates,
    )

    p_lo, p_hi = np.percentile(profile_sm, [1, 99])
    intensity_range = max(float(p_hi - p_lo), 1e-6)

    center_x = (w - 1) / 2.0
    center_sigma = max(center_prior * w, 1.0)

    best = None

    for left in left_candidates:
        for right in right_candidates:
            if right <= left:
                continue

            width = right - left + 1
            if width < min_width or width > max_width:
                continue

            segment = profile_sm[left:right + 1]
            inside_min = float(segment.min())
            shoulder_level = 0.5 * (float(profile_sm[left]) + float(profile_sm[right]))
            valley_contrast = (shoulder_level - inside_min) / intensity_range

            if valley_contrast < min_contrast:
                continue

            left_strength = max(float(-derivative[left]), 0.0)
            right_strength = max(float(derivative[right]), 0.0)

            pair_center = 0.5 * (left + right)
            spatial_weight = np.exp(-0.5 * ((pair_center - center_x) / center_sigma) ** 2)

            score = left_strength * right_strength * valley_contrast * spatial_weight

            candidate = {
                "left": int(left),
                "right": int(right),
                "width": int(width),
                "score": float(score),
                "valley_contrast": float(valley_contrast),
                "left_strength": float(left_strength),
                "right_strength": float(right_strength),
                "center_weight": float(spatial_weight),
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "No valid stripe edge pair found. Try lowering --min-contrast, "
            "increasing --max-width-frac, or using --x1 and --x2."
        )

    return int(best["left"]), int(best["right"]), {
        "method": "edge_pair",
        "profile": profile,
        "profile_smooth": profile_sm,
        "derivative": derivative,
        "left_candidates": left_candidates,
        "right_candidates": right_candidates,
        "best_score": best,
    }


def smooth_window_1d(w: int, x1: int, x2: int, soft_edge: float) -> np.ndarray:
    """
    Create a 1D soft stripe window.
    It is close to 1 inside the stripe and gradually decays near boundaries.
    """
    x = np.arange(w, dtype=np.float32)

    if soft_edge <= 0:
        win = np.zeros(w, dtype=np.float32)
        win[x1:x2 + 1] = 1.0
        return win

    left = 1.0 / (1.0 + np.exp(-(x - x1) / soft_edge))
    right = 1.0 / (1.0 + np.exp((x - x2) / soft_edge))
    win = left * right
    win = robust_minmax(win, 0.5, 99.5)
    return win.astype(np.float32)


def make_gradient_outputs(
    gray: np.ndarray,
    x1: int,
    x2: int,
    aux: dict,
    gradient_mode: str,
    soft_edge: float,
    outside_value: float,
    cmap_name: str,
    alpha: float,
    overlay_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape
    gray_smooth = cv2.GaussianBlur(gray, (0, 0), 3.0)

    hard_mask = np.zeros((h, w), dtype=np.uint8)
    hard_mask[:, x1:x2 + 1] = 1

    win_x = smooth_window_1d(w, x1, x2, soft_edge)

    profile_sm = aux["profile_smooth"]
    profile_darkness = 1.0 - robust_minmax(profile_sm, 1.0, 99.0)
    profile_darkness = smooth_1d(profile_darkness, sigma=max(soft_edge * 0.20, 1.0))
    profile_darkness = robust_minmax(profile_darkness, 1.0, 99.0)

    # 2D darkness preserves original texture and subtle gradient.
    img_darkness = 1.0 - robust_minmax(gray_smooth, 1.0, 99.0)
    img_darkness = cv2.GaussianBlur(img_darkness.astype(np.float32), (0, 0), 2.0)

    profile_map = np.tile(profile_darkness[None, :], (h, 1))

    if gradient_mode == "profile":
        score = profile_map
    elif gradient_mode == "image":
        score = img_darkness
    elif gradient_mode == "combined":
        score = 0.65 * profile_map + 0.35 * img_darkness
    else:
        raise ValueError(f"Unknown gradient mode: {gradient_mode}")

    # Constrain high confidence to the stripe region, but with soft boundaries.
    heatmap_float = outside_value + (1.0 - outside_value) * score * win_x[None, :]
    heatmap_float = robust_minmax(heatmap_float, 0.5, 99.5)

    # Keep non-stripe area visibly low.
    heatmap_float = np.maximum(heatmap_float, outside_value)
    heatmap_float[:, win_x < 0.03] = outside_value

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(heatmap_float, 0, 1))
    heatmap_rgb = np.uint8(rgba[:, :, :3] * 255)
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    base_u8 = np.uint8(np.clip(gray * 255, 0, 255))
    base_bgr = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base_bgr, 1 - alpha, heatmap_bgr, alpha, 0)

    if overlay_mode == "soft":
        # Use soft confidence as alpha mask; produces natural gradient overlay.
        alpha_map = np.clip(win_x[None, :] * heatmap_float, 0, 1)
        alpha_map = alpha_map[:, :, None]
        overlay = (base_bgr.astype(np.float32) * (1 - alpha * alpha_map)
                   + heatmap_bgr.astype(np.float32) * (alpha * alpha_map))
        overlay = np.uint8(np.clip(overlay, 0, 255))
    else:
        overlay = base_bgr.copy()
        m3 = np.repeat(hard_mask[:, :, None].astype(bool), 3, axis=2)
        overlay[m3] = blended[m3]

    return heatmap_bgr, hard_mask, overlay, heatmap_float


def save_float_heatmap_with_colorbar(path: Path, heatmap_float: np.ndarray, cmap_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=260)
    im = ax.imshow(heatmap_float, cmap=cmap_name, vmin=0, vmax=1)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Stripe confidence")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_diagnostic(
    path: Path,
    gray: np.ndarray,
    heatmap_float: np.ndarray,
    mask: np.ndarray,
    overlay_bgr: np.ndarray,
    cmap_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=220)

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input")

    axes[1].imshow(heatmap_float, cmap=cmap_name, vmin=0, vmax=1)
    axes[1].set_title("Soft-gradient heatmap")

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Hard mask")

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Soft overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_profile_plot(
    path: Path,
    profile: np.ndarray,
    profile_sm: np.ndarray,
    derivative: np.ndarray,
    x1: int,
    x2: int,
    left_candidates: list[int],
    right_candidates: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(profile))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=220, sharex=True)

    axes[0].plot(x, profile, label="Column median profile", linewidth=1.0)
    axes[0].plot(x, profile_sm, label="Smoothed profile", linewidth=1.5)
    axes[0].axvspan(x1, x2, alpha=0.18, label="Detected stripe region")
    axes[0].axvline(x1, linestyle="--", linewidth=1.2)
    axes[0].axvline(x2, linestyle="--", linewidth=1.2)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Detected stripe boundaries")
    axes[0].legend(loc="best")

    axes[1].plot(x, derivative, label="d(profile)/dx", linewidth=1.0)
    if left_candidates:
        axes[1].scatter(left_candidates, derivative[left_candidates], marker="v", label="Left-edge candidates")
    if right_candidates:
        axes[1].scatter(right_candidates, derivative[right_candidates], marker="^", label="Right-edge candidates")
    axes[1].axvline(x1, linestyle="--", linewidth=1.2)
    axes[1].axvline(x2, linestyle="--", linewidth=1.2)
    axes[1].axhline(0, linewidth=0.8)
    axes[1].set_xlabel("Column index")
    axes[1].set_ylabel("Gradient")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a soft-gradient heatmap for a broad vertical dark stripe.")

    parser.add_argument("input", type=str, help="Input image path.")
    parser.add_argument("--out-dir", type=str, default="fig3_heatmap_v4", help="Output directory.")

    parser.add_argument("--profile-sigma", type=float, default=35.0, help="Smoothing sigma for column profile.")
    parser.add_argument("--ignore-border-frac", type=float, default=0.06, help="Ignore left/right border fraction.")
    parser.add_argument("--min-width", type=int, default=120, help="Minimum stripe width in pixels.")
    parser.add_argument("--max-width-frac", type=float, default=0.75, help="Maximum stripe width as fraction of image width.")
    parser.add_argument("--edge-min-distance", type=int, default=80, help="Minimum distance between candidate edges.")
    parser.add_argument("--max-candidates", type=int, default=20, help="Number of left/right edge candidates to test.")
    parser.add_argument("--center-prior", type=float, default=0.45, help="Larger value weakens center preference.")
    parser.add_argument("--min-contrast", type=float, default=0.05, help="Minimum valley contrast.")

    parser.add_argument("--x1", type=int, default=None, help="Manual left boundary. Use together with --x2.")
    parser.add_argument("--x2", type=int, default=None, help="Manual right boundary. Use together with --x1.")

    parser.add_argument("--gradient-mode", choices=["profile", "image", "combined"], default="combined",
                        help="profile: smooth column gradient; image: original texture; combined: both.")
    parser.add_argument("--soft-edge", type=float, default=80.0, help="Soft boundary width in pixels.")
    parser.add_argument("--outside-value", type=float, default=0.02, help="Heatmap value outside stripe.")
    parser.add_argument("--overlay-mode", choices=["soft", "hard"], default="soft", help="Soft or hard overlay boundary.")
    parser.add_argument("--alpha", type=float, default=0.60, help="Overlay strength.")
    parser.add_argument("--cmap", type=str, default="turbo", help="Matplotlib colormap.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = imread_unicode(in_path)
    gray = to_gray_float(img)

    x1, x2, aux = detect_stripe_by_edge_pair(
        gray=gray,
        profile_sigma=args.profile_sigma,
        ignore_border_frac=args.ignore_border_frac,
        min_width=args.min_width,
        max_width_frac=args.max_width_frac,
        edge_min_distance=args.edge_min_distance,
        max_candidates=args.max_candidates,
        center_prior=args.center_prior,
        min_contrast=args.min_contrast,
        manual_x1=args.x1,
        manual_x2=args.x2,
    )

    heatmap_bgr, mask, overlay_bgr, heatmap_float = make_gradient_outputs(
        gray=gray,
        x1=x1,
        x2=x2,
        aux=aux,
        gradient_mode=args.gradient_mode,
        soft_edge=args.soft_edge,
        outside_value=args.outside_value,
        cmap_name=args.cmap,
        alpha=args.alpha,
        overlay_mode=args.overlay_mode,
    )

    stem = in_path.stem
    heatmap_path = out_dir / f"{stem}_stripe_heatmap.png"
    heatmap_colorbar_path = out_dir / f"{stem}_stripe_heatmap_colorbar.png"
    mask_path = out_dir / f"{stem}_stripe_mask.png"
    overlay_path = out_dir / f"{stem}_stripe_overlay.png"
    diagnostic_path = out_dir / f"{stem}_diagnostic.png"
    profile_path = out_dir / f"{stem}_profile_plot.png"
    confidence_path = out_dir / f"{stem}_stripe_confidence_16bit.png"
    metrics_path = out_dir / f"{stem}_metrics.json"

    imwrite_unicode(heatmap_path, heatmap_bgr)
    imwrite_unicode(mask_path, np.uint8(mask * 255))
    imwrite_unicode(overlay_path, overlay_bgr)
    imwrite_unicode(confidence_path, np.uint16(np.clip(heatmap_float, 0, 1) * 65535))
    save_float_heatmap_with_colorbar(heatmap_colorbar_path, heatmap_float, args.cmap)
    save_diagnostic(diagnostic_path, gray, heatmap_float, mask, overlay_bgr, args.cmap)
    save_profile_plot(
        profile_path,
        aux["profile"],
        aux["profile_smooth"],
        aux["derivative"],
        x1,
        x2,
        aux["left_candidates"],
        aux["right_candidates"],
    )

    metrics = {
        "input": str(in_path),
        "method": aux["method"],
        "detected_x_range": [int(x1), int(x2)],
        "stripe_width_pixels": int(x2 - x1 + 1),
        "stripe_area_fraction_hard_mask": float(mask.mean()),
        "best_score": aux["best_score"],
        "parameters": {
            "profile_sigma": args.profile_sigma,
            "ignore_border_frac": args.ignore_border_frac,
            "min_width": args.min_width,
            "max_width_frac": args.max_width_frac,
            "edge_min_distance": args.edge_min_distance,
            "max_candidates": args.max_candidates,
            "center_prior": args.center_prior,
            "min_contrast": args.min_contrast,
            "x1": args.x1,
            "x2": args.x2,
            "gradient_mode": args.gradient_mode,
            "soft_edge": args.soft_edge,
            "outside_value": args.outside_value,
            "overlay_mode": args.overlay_mode,
            "alpha": args.alpha,
            "cmap": args.cmap,
        },
        "outputs": {
            "heatmap": str(heatmap_path),
            "heatmap_with_colorbar": str(heatmap_colorbar_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "diagnostic": str(diagnostic_path),
            "profile_plot": str(profile_path),
            "confidence_16bit": str(confidence_path),
        },
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Detected stripe x range: [{x1}, {x2}]")
    print(f"Stripe width: {x2 - x1 + 1} px")
    print(f"Hard mask area fraction: {mask.mean():.2%}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Mask: {mask_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Diagnostic: {diagnostic_path}")
    print(f"Profile plot: {profile_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
