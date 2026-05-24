"""hough_sam_segmentation_pipeline.py

an executable script. This script performs:
  - ring detection (Hough + structured edge)
  - automatic prompting for SAM2 (sampling ring points)
  - segmentation propagation through video
  - postprocessing & incision geometry extraction
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Try importing SAM predictor builder at module import time; if unavailable,
try:
    from sam2.build_sam import build_sam2_video_predictor
except Exception:
    build_sam2_video_predictor = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hough_sam.constants import (
    FRAMES_DIR,
    SEG_RESULTS_DIR,
    FIGURES_DIR,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    SEG_VIDEOS_DIR,
)

from hough_sam.segmentation.visualization import show_mask, show_points, show_box
from hough_sam.segmentation.hough_ring import (
    find_optic_disk_circles_structured,
    sample_ring_points,
    RingDetector,
)
from hough_sam.segmentation.sam_utils import (
    segment_with_sam2,
    propagate_in_video,
)
from hough_sam.segmentation.postprocessing import (
    IncisionDetector,
    IncisionParams,
    mask_center,
    polygon_area,
    contour_centroid,
    dist,
    trapezoid_side,
    triangle_side,
)


def ensure_dir(p: Path) -> None:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)


def detect_ring_for_case(case_dir: Path, start_frame_idx: int = 0, N: int = 2) -> tuple | None:
    """Run RingDetector on a case folder and return inner, outer circles.

    Returns (inner, outer) where each is (x, y, r) or (None, None).
    """
    det = RingDetector(N=N)
    # list frames in folder and pick frames starting at start_frame_idx
    frame_names = [p.name for p in sorted(case_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not frame_names:
        print(f"no frames in {case_dir}")
        return None, None
    for j in range(det.N):
        idx = start_frame_idx + j
        if idx >= len(frame_names):
            break
        f = case_dir / frame_names[idx]
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"Failed to read {f}, skipping")
            continue
        det.add_frame(frame)
    inner, outer = det.detect_on_average()
    return inner, outer


def save_circle_visual(case_dir: Path, inner, outer, out_path: Path, ann_frame_idx: int = 0):
    ensure_dir(out_path.parent)
    frame_names = [p.name for p in sorted(case_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not frame_names:
        return
    f = case_dir / frame_names[ann_frame_idx]
    img = cv2.imread(str(f))
    if img is None:
        return
    if inner is not None:
        cv2.circle(img, (int(inner[0]), int(inner[1])), int(inner[2]), (255, 0, 0), 2)
    if outer is not None:
        cv2.circle(img, (int(outer[0]), int(outer[1])), int(outer[2]), (0, 91, 255), 3)
    cv2.imwrite(str(out_path), img)


def segment_case_with_sam(predictor, inference_state, case_name: str, ann_frame_idx: int = 0, num_frames: int = 500):
    """High-level wrapper that performs the same steps as the notebook:
    - automatic prompting via ring sampling
    - add prompts to predictor state
    - propagate through video
    - write segmentation masks to SEG_RESULTS_DIR/{case_name}/{subdirs}
    """
    video_dir = FRAMES_DIR / case_name
    # run ring detection and build prompts inside segment_with_sam2 helper
    segment_with_sam2(predictor, inference_state, str(video_dir), ann_frame_idx=ann_frame_idx)
    video_segments = propagate_in_video(predictor, inference_state, ann_frame_idx, num_frames=num_frames)

    # prepare output directories
    out_base = SEG_RESULTS_DIR / case_name
    pupil_dir = out_base / "pupil"
    intersection_dir = out_base / "intersection"
    optic_dir = out_base / "optic_disk"
    knife_dir = out_base / "knife"
    for d in [pupil_dir, intersection_dir, optic_dir, knife_dir]:
        ensure_dir(d)

    # iterate frames and save masks like the notebook
    for out_frame_idx in sorted(video_segments.keys()):
        frame_maps = video_segments[out_frame_idx]
        # notebook assumed obj_id 1 => optic_disk, 2 => pupil
        od_mask_logits = frame_maps.get(1)
        p_mask_logits = frame_maps.get(2)
        if od_mask_logits is None or p_mask_logits is None:
            continue
        od_mask = (od_mask_logits[0] > 0.0).astype(np.uint8) * 255
        p_mask = (p_mask_logits[0] > 0.0).astype(np.uint8) * 255

        # find contours and build convex hulled optic disk mask like notebook
        od_contours, _ = cv2.findContours(od_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        p_contours, _ = cv2.findContours(p_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not od_contours or not p_contours:
            continue
        od_contour = max(od_contours, key=len)
        p_contour = max(p_contours, key=len)
        epsilon = 0.005 * cv2.arcLength(od_contour, True)
        approx = cv2.approxPolyDP(od_contour, epsilon, True)
        od_mask_color = cv2.cvtColor(np.zeros_like(od_mask), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(od_mask_color, [approx], -1, (255, 255, 255), cv2.FILLED)
        od_mask_convex = cv2.cvtColor(od_mask_color, cv2.COLOR_BGR2GRAY)
        od_mask_convex = od_mask_convex.astype(np.uint8)

        k_mask = cv2.subtract(od_mask_convex, od_mask)
        p_mask_convex = np.zeros_like(p_mask)
        try:
            cv2.fillConvexPoly(p_mask_convex, cv2.convexHull(p_contour), 255)
        except Exception:
            p_mask_convex = p_mask

        i_mask = np.logical_and(np.logical_not(p_mask_convex.astype(bool)), k_mask.astype(bool)).astype(np.uint8) * 255

        fname = f"{str(out_frame_idx+1).zfill(4)}.jpeg"
        cv2.imwrite(str(pupil_dir / fname), p_mask)
        cv2.imwrite(str(intersection_dir / fname), i_mask)
        cv2.imwrite(str(optic_dir / fname), od_mask_convex)
        cv2.imwrite(str(knife_dir / fname), k_mask)


def postprocess_all_cases(output_tracking_dir_base: Path | str = None):
    """Run the postprocessing loop from the notebook: for each case in SEG_RESULTS_DIR,
    run IncisionDetector on intersection masks, compute geometry, save per-frame visualization
    and pickle results for each case.
    """
    output_tracking_dir_base = Path(output_tracking_dir_base) if output_tracking_dir_base is not None else SEG_VIDEOS_DIR
    case_names = [p.name for p in sorted(SEG_RESULTS_DIR.iterdir()) if p.is_dir()]
    COLUMNS = [
        "case_name",
        "frame_index",
        "trapezoid_coord",
        "trapezoid_area",
        "trapezoid_s1",
        "trapezoid_s2",
        "triangle_coord",
        "triangle_area",
        "triangle_s1",
        "triangle_s2",
        "incision_side_ratio",
    ]

    det = IncisionDetector(IncisionParams(min_area=20, parallel_tol_deg=12, perpendicular_tol_deg=18))

    for case_name in case_names:
        try:
            video_dir = FRAMES_DIR / case_name
            res_dir = SEG_RESULTS_DIR / case_name / "intersection"
            p_dir = SEG_RESULTS_DIR / case_name / "pupil"
            o_dir = SEG_RESULTS_DIR / case_name / "optic_disk"
            video_tracking_dir = output_tracking_dir_base / case_name
            ensure_dir(video_tracking_dir)

            frame_names = [p.name for p in sorted(res_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            result_df = pd.DataFrame(columns=COLUMNS)
            for f in frame_names:
                frame_idx = int(Path(f).stem)
                mask_i = cv2.imread(str(res_dir / f), cv2.IMREAD_GRAYSCALE)
                if mask_i is None:
                    continue
                res_quad, res_tri = det.process_frame(mask_i)
                row = {c: None for c in COLUMNS}
                row["case_name"] = case_name
                row["frame_index"] = frame_idx

                mask_p = cv2.imread(str(p_dir / f), cv2.IMREAD_GRAYSCALE)
                if mask_p is None:
                    cy_p = cx_p = None
                else:
                    cy_p, cx_p = mask_center(mask_p)

                accepted_xor = (res_quad["accepted"] ^ res_tri["accepted"]) if (res_quad and res_tri) else False
                if accepted_xor:
                    if res_quad["accepted"]:
                        res = res_quad
                        h1, h2 = trapezoid_side(res["poly"], [cx_p, cy_p])
                        if dist(h1[0], h2[0]) < dist(h1[0], h2[1]):
                            s1 = np.array([h1[0], h2[0]])
                            s2 = np.array([h1[1], h2[1]])
                        else:
                            s1 = np.array([h1[0], h2[1]])
                            s2 = np.array([h1[1], h2[0]])
                    else:
                        res = res_tri
                        s1, s2 = triangle_side(res["poly"], [cx_p, cy_p])

                    mask_o = cv2.imread(str(o_dir / f), cv2.IMREAD_GRAYSCALE)
                    area = polygon_area(res["poly"])
                    cx_i, cy_i = contour_centroid(res["poly"])
                    incision_ratio = dist(*s1) / dist(*s2)
                    h, w = mask_i.shape
                    # conditions mirrored from notebook
                    valid = (area > 120) and (mask_o is not None) and (mask_o[int(cx_i) % mask_o.shape[0]][int(cy_i) % mask_o.shape[1]] == 255) and (cx_i > w / 2) and (cy_i > h / 2)
                    if valid:
                        if res_quad["accepted"]:
                            row["trapezoid_coord"] = res["poly"].tolist()
                            row["trapezoid_area"] = area
                            row["trapezoid_s1"] = s1.tolist()
                            row["trapezoid_s2"] = s2.tolist()
                        else:
                            row["triangle_coord"] = res["poly"].tolist()
                            row["triangle_area"] = area
                            row["triangle_s1"] = s1.tolist()
                            row["triangle_s2"] = s2.tolist()
                        row["incision_side_ratio"] = incision_ratio

                # visualization per frame
                plt.figure(figsize=(6, 4))
                try:
                    img = plt.imread(str(video_dir / f"{str(frame_idx).zfill(4)}.jpeg"))
                    plt.imshow(img)
                except Exception:
                    pass
                ax = plt.gca()
                try:
                    show_mask(mask_i / 255, ax, obj_id=1)
                except Exception:
                    pass
                try:
                    if mask_p is not None:
                        show_points(np.array([[cx_p, cy_p]]), ax, color="red")
                except Exception:
                    pass

                if row.get("trapezoid_coord") is not None:
                    s1 = np.array(row["trapezoid_s1"]) if row["trapezoid_s1"] is not None else None
                    s2 = np.array(row["trapezoid_s2"]) if row["trapezoid_s2"] is not None else None
                    if s1 is not None:
                        show_points(s1, plt.gca(), marker_size=50)
                    if s2 is not None:
                        show_points(s2, plt.gca(), marker_size=50, color="blue")
                    plt.text(cy_i + 20, cx_i + 20, f"Area: {area}", fontsize=10, color="blue")
                    plt.text(cy_i + 20, cx_i + 40, f"Ratio: {incision_ratio:.4g}", fontsize=10, color="blue")

                if row.get("triangle_coord") is not None:
                    s1 = np.array(row["triangle_s1"]) if row["triangle_s1"] is not None else None
                    s2 = np.array(row["triangle_s2"]) if row["triangle_s2"] is not None else None
                    if s1 is not None:
                        show_points(s1, plt.gca(), marker_size=50)
                    if s2 is not None:
                        show_points(s2, plt.gca(), marker_size=50, color="blue")
                    plt.text(cy_i + 20, cx_i + 20, f"Area: {area}", fontsize=10, color="blue")
                    plt.text(cy_i + 20, cx_i + 40, f"Ratio: {incision_ratio:.4g}", fontsize=10, color="blue")

                plt.axis("off")
                out_fig = video_tracking_dir / f"frame_{str(frame_idx).zfill(4)}.png"
                plt.savefig(str(out_fig), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

            # dump per-case pickle
            pickle.dump(result_df, open(str(video_tracking_dir / f"{case_name}.pkl"), "wb"))
        except Exception as e:
            print(f"Error processing {case_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hough-SAM segmentation pipeline (scripted from notebook).")
    parser.add_argument("--case", type=str, default=None, help="Specific case folder name under frames/ to process (default: all).")
    parser.add_argument("--ann-frame-idx", type=int, default=0, help="Annotation frame index to use for prompting.")
    parser.add_argument("--num-frames", type=int, default=500, help="Number of frames to propagate.")
    parser.add_argument("--predictor-cfg", type=str, default=None, help="Optional SAM model cfg path.")
    parser.add_argument("--predictor-checkpoint", type=str, default=None, help="Optional SAM checkpoint path.")
    parser.add_argument("--run-postproc", action="store_true", help="Run postprocessing step after segmentation.")
    args = parser.parse_args()

    # create figures dir
    ensure_dir(FIGURES_DIR)

    # Try to import and build SAM predictor if available
    predictor = None
    if args.predictor_checkpoint is None:
        default_ckpt = CHECKPOINTS_DIR / "sam2.1_hiera_small.pt"
        ckpt_path = default_ckpt if default_ckpt.exists() else None
    else:
        ckpt_path = Path(args.predictor_checkpoint)

    if build_sam2_video_predictor is not None and ckpt_path is not None:
        model_cfg = args.predictor_cfg or "configs/sam2.1/sam2.1_hiera_s.yaml"
        try:
            predictor = build_sam2_video_predictor(model_cfg, str(ckpt_path), device="cpu")
        except Exception as e:
            predictor = None
            print(f"Failed to build SAM predictor: {e}")
    else:
        if build_sam2_video_predictor is None:
            print("SAM predictor builder not available (sam2 not installed).")
        else:
            print("SAM checkpoint not provided or not found; SAM-based segmentation will fail if attempted.")

    # Choose cases
    frame_cases = [p.name for p in sorted(FRAMES_DIR.iterdir()) if p.is_dir()]
    if args.case:
        if args.case not in frame_cases:
            print(f"Case {args.case} not found under {FRAMES_DIR}")
            return
        frame_cases = [args.case]

    for case_name in frame_cases:
        print(f"Processing {case_name}...")
        case_dir = FRAMES_DIR / case_name
        inner, outer = detect_ring_for_case(case_dir, start_frame_idx=args.ann_frame_idx, N=2)
        # save visualization of circle detection
        out_img = FIGURES_DIR / f"{case_name}_circle.png"
        save_circle_visual(case_dir, inner, outer, out_img, ann_frame_idx=args.ann_frame_idx)

        if predictor is None:
            print("Skipping SAM segmentation because predictor is not available.")
            continue

        # init predictor state and segment
        inference_state = predictor.init_state(video_path=str(case_dir))
        segment_case_with_sam(predictor, inference_state, case_name, ann_frame_idx=args.ann_frame_idx, num_frames=args.num_frames)

    if args.run_postproc:
        postprocess_all_cases()


if __name__ == "__main__":
    main()
