import os
import numpy as np
import cv2
from .hough_ring import sample_ring_points

def sam2_point_prompt_single_frame(predictor, video_dir: str, ann_frame_idx: int, points: np.ndarray, labels: np.ndarray, obj_id: int = 1):
    inference_state = predictor.init_state(video_path=video_dir)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
    )
    mask_bool = None
    for i, oid in enumerate(out_obj_ids):
        if int(oid) == int(obj_id):
            mask_bool = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze()
            break
    if mask_bool is None:
        mask_bool = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
    return mask_bool

def sam2_box_prompt_single_frame(predictor, video_dir: str, ann_frame_idx: int, box_xyxy: np.ndarray, obj_id: int = 1):
    inference_state = predictor.init_state(video_path=video_dir)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=obj_id,
        box=box_xyxy.astype(np.float32),
    )
    mask_bool = None
    for i, oid in enumerate(out_obj_ids):
        if int(oid) == int(obj_id):
            mask_bool = (out_mask_logits[i] > 0.0).detach().cpu().numpy().squeeze()
            break
    if mask_bool is None:
        mask_bool = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze()
    return mask_bool

def make_default_point_prompt(H: int, W: int):
    pts = np.array([
        [0.50 * W, 0.50 * H],
        [0.55 * W, 0.50 * H],
        [0.50 * W, 0.55 * H],
        [0.10 * W, 0.10 * H],
    ], dtype=np.float32)
    lbl = np.array([1, 1, 1, 0], dtype=np.int32)
    return pts, lbl

def make_default_box_prompt(H: int, W: int):
    return np.array([0.35 * W, 0.35 * H, 0.65 * W, 0.65 * H], dtype=np.float32)

def propagate_in_video(sam_predictor, inference_state, ann_frame_idx, num_frames = 500):
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        if out_frame_idx >= ann_frame_idx + num_frames:
            break
    return video_segments

def segment_with_sam2(sam_predictor, inference_state, video_dir, ann_frame_idx = 0, ring_detector=None):
    prompts = {}
    det = ring_detector
    if det is None:
        raise ValueError("ring_detector required")
    for j in range(det.N):
        frame = cv2.imread(os.path.join(video_dir, f"{ann_frame_idx+j+1:04d}.jpeg"))
        if frame is None:
            continue
        det.add_frame(frame)
    inner, outer = det.detect_on_average()
    if inner is None:
        raise RuntimeError("no circles found – try a different ann_frame_idx")
    center = (inner[0], inner[1])
    od_points, p_points = sample_ring_points(inner, outer, center, num_points=3)
    ann_obj_id = 1
    points = np.array(od_points, dtype=np.float32)
    labels = np.array([1, 1, 1], np.int32)
    prompts[ann_obj_id] = points, labels
    sam_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
    ann_obj_id = 2
    points = np.array(p_points, dtype=np.float32)
    labels = np.array([1, 1, 1], np.int32)
    prompts[ann_obj_id] = points, labels
    sam_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)
    return prompts
