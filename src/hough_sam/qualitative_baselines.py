#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from hough_sam import (
    FRAMES_DIR,
    SEG_RESULTS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    MEDSAM_BASELINE_DIR,
    build_sam2_video_predictor,
    sam2_point_prompt_single_frame,
    sam2_box_prompt_single_frame,
    show_mask,
    show_points,
    show_box,
)

DEFAULT_CASES = [
    "case_2000_50fps",
    "case_2025_50fps",
    "case_2030_50fps",
]

DEFAULT_FRAMES = ["0448", "0157", "0215"]
DEFAULT_POINT_PROMPTS = [
    np.array([[256, 225], [256, 235], [281, 230], [256, 200]], dtype=np.float32),
    np.array([[285, 265], [285, 275], [300, 270], [285, 250]], dtype=np.float32),
    np.array([[325, 255], [325, 265], [340, 260], [325, 240]], dtype=np.float32),
]
DEFAULT_BOX_PROMPTS = [
    np.array([240, 200, 300, 250], dtype=np.float32),
    np.array([250, 225, 325, 275], dtype=np.float32),
    np.array([300, 225, 375, 275], dtype=np.float32),
]


def _load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _safe_mask(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=np.float32)
    if mask.max() > 0:
        return mask / float(mask.max())
    return mask


def build_predictor(model_cfg: Path, checkpoint_path: Path, device: str = "cpu"):
    if not model_cfg.exists():
        raise FileNotFoundError(f"SAM model config not found: {model_cfg}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    return build_sam2_video_predictor(str(model_cfg), str(checkpoint_path), device=device)


def plot_baseline_cases(
    predictor,
    cases: list[str] = DEFAULT_CASES,
    frames: list[str] = DEFAULT_FRAMES,
    output_path: Path | None = None,
):
    n = len(cases)
    fig, axes = plt.subplots(n, 4, figsize=(4 * 5, n * 5), squeeze=False)

    for r, (case, frame_name) in enumerate(zip(cases, frames)):
        video_dir = FRAMES_DIR / case
        if not video_dir.exists():
            for c in range(4):
                axes[r, c].axis("off")
            axes[r, 0].set_title(f"{case} (missing dir)")
            continue

        ann_frame_idx = int(frame_name) - 1
        img_path = video_dir / f"{frame_name}.jpeg"
        if not img_path.exists():
            for c in range(4):
                axes[r, c].axis("off")
            axes[r, 0].set_title(f"{case} / {frame_name} (frame not found)")
            continue

        img_rgb = _load_rgb_image(img_path)
        labels = np.array([1, 1, 1, 0], dtype=np.int32)
        points = DEFAULT_POINT_PROMPTS[r]
        box_xyxy = DEFAULT_BOX_PROMPTS[r]

        sam2_mask_points = sam2_point_prompt_single_frame(
            predictor=predictor,
            video_dir=video_dir,
            ann_frame_idx=ann_frame_idx,
            points=points,
            labels=labels,
            obj_id=1,
        )
        sam2_mask_box = sam2_box_prompt_single_frame(
            predictor=predictor,
            video_dir=video_dir,
            ann_frame_idx=ann_frame_idx,
            box_xyxy=box_xyxy,
            obj_id=1,
        )

        our_res_path = SEG_RESULTS_DIR / case / "intersection" / f"{frame_name}.jpeg"
        our_res = None
        if our_res_path.exists():
            our_res = cv2.imread(str(our_res_path), cv2.IMREAD_GRAYSCALE)

        medsam_mask = None
        medsam_path = MEDSAM_BASELINE_DIR / f"{frame_name}_mask.png"
        if medsam_path.exists():
            medsam_mask = cv2.imread(str(medsam_path), cv2.IMREAD_GRAYSCALE)

        row_axes = axes[r]
        row_axes[0].imshow(img_rgb)
        if our_res is not None:
            show_mask(_safe_mask(our_res), row_axes[0], obj_id=1)
        row_axes[0].set_title(f"{case[:-6]}\nHoughSAM(our method)")
        row_axes[0].axis("off")

        row_axes[1].imshow(img_rgb)
        show_points(points, labels, row_axes[1])
        show_mask(_safe_mask(sam2_mask_points), row_axes[1], obj_id=1)
        row_axes[1].set_title("SAM2: point prompt")
        row_axes[1].axis("off")

        row_axes[2].imshow(img_rgb)
        show_box(box_xyxy, row_axes[2])
        show_mask(_safe_mask(sam2_mask_box), row_axes[2], obj_id=1)
        row_axes[2].set_title("SAM2: box prompt")
        row_axes[2].axis("off")

        row_axes[3].imshow(img_rgb)
        if medsam_mask is not None:
            show_mask(_safe_mask(medsam_mask), row_axes[3], obj_id=1)
        row_axes[3].set_title("MedSAM: box prompt")
        row_axes[3].axis("off")

    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        print(f"Saved qualitative baseline figure to {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qualitative baseline visualization for Hough-SAM.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS_DIR / "sam2.1_hiera_small.pt")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs" / "sam2.1" / "sam2.1_hiera_s.yaml")
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "qualitative_baseline.png")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = build_predictor(args.model_config, args.checkpoint, device=args.device)
    plot_baseline_cases(predictor, output_path=args.output)


if __name__ == "__main__":
    main()
