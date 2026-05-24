from typing import Tuple
import numpy as np
import pandas as pd

def _impute_cell(a: pd.Series, b: pd.Series, f: int) -> pd.Series:
  result = a.copy()
  result["frame_index"] = f
  if a["trapezoid_coord"] is not None and b["trapezoid_coord"] is not None:
    result["trapezoid_coord"] = [[(c1[0]+c2[0])/2, (c1[1]+c2[1])/2] for c1, c2 in zip(a["trapezoid_coord"], b["trapezoid_coord"])]
  result["trapezoid_area"] = (a["trapezoid_area"] + b["trapezoid_area"]) / 2
  if a["trapezoid_s1"] is not None and b["trapezoid_s1"] is not None:
    result["trapezoid_s1"] = [[(c1[0]+c2[0])/2, (c1[1]+c2[1])/2] for c1, c2 in zip(a["trapezoid_s1"], b["trapezoid_s1"])]
  if a["trapezoid_s2"] is not None and b["trapezoid_s2"] is not None:
    result["trapezoid_s2"] = [[(c1[0]+c2[0])/2, (c1[1]+c2[1])/2] for c1, c2 in zip(a["trapezoid_s2"], b["trapezoid_s2"])]
  result["side_ratio"] = (a.get("side_ratio", 0) + b.get("side_ratio", 0)) / 2
  return result

def find_longest_continuous_detection(df: pd.DataFrame, max_tolerable_jump: int = 2) -> Tuple[int, int]:
    if len(df) == 0:
        raise ValueError("df is empty.")
    out = df.copy()
    out.index = out.index.astype(int)
    det_mask = out["trapezoid_coord"].isnull() == False
    det_frames = out.index[det_mask.values].to_numpy(dtype=int)
    if len(det_frames) == 0:
        raise ValueError("No detected frames found; cannot define a continuous detection segment.")
    segments = []
    seg_start = int(det_frames[0])
    prev = int(det_frames[0])
    for f in det_frames[1:]:
        f = int(f)
        missing_count = f - prev - 1
        if missing_count <= max_tolerable_jump:
            prev = f
        else:
            segments.append((seg_start, prev))
            seg_start = f
            prev = f
    segments.append((seg_start, prev))
    best_start, best_stop = max(segments, key=lambda s: (s[1] - s[0] + 1, -s[0]))
    return int(best_start), int(best_stop)

def impute_gaps_within_segment(df: pd.DataFrame, start_frame: int, stop_frame: int, max_tolerable_jump: int = 2) -> pd.DataFrame:
    if start_frame > stop_frame:
        raise ValueError("start_frame must be <= stop_frame")
    out = df.copy()
    out.index = out.index.astype(int)
    seg_mask = (out.index >= int(start_frame)) & (out.index <= int(stop_frame))
    det_mask = df["trapezoid_coord"].isnull() == False
    det_in_seg = det_mask.values & seg_mask
    det_frames_in_seg = out.index[det_in_seg].to_numpy(dtype=int)
    if len(det_frames_in_seg) < 2:
        return out
    for left, right in zip(det_frames_in_seg[:-1], det_frames_in_seg[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_tolerable_jump:
            continue
        for missing_frame in range(left + 1, right):
            row_left = out.iloc[left]
            row_right = out.iloc[right]
            out.iloc[missing_frame] = _impute_cell(row_left, row_right, missing_frame)
    return out

def smooth(df: pd.DataFrame, max_tolerable_jump: int = 2):
    s, e = find_longest_continuous_detection(df, max_tolerable_jump=max_tolerable_jump)
    out = impute_gaps_within_segment(df, s, e, max_tolerable_jump=max_tolerable_jump)
    return s, e, out

def calculate_case_level_geometric_feature(features_by_frame, start, stop) -> dict:
  ratios = features_by_frame["side_ratio"]
  min_ratio = ratios.min()
  max_ratio = ratios.max()
  deviations = (ratios - 1).abs()
  mean_deviation = deviations.mean()
  max_deviation = deviations.max()
  return {"min_ratio": min_ratio, "max_ratio": max_ratio, "mean_deviation": float(mean_deviation), "max_deviation": float(max_deviation)}

def calculate_slopes(values: list) -> list:
    n = len(values)
    if n < 2:
        return []
    return [values[i-1] - values[i+1] for i in range(1, n-1, 2)]

def calculate_case_level_area_feature(features_by_frame, start, stop) -> dict:
  areas = features_by_frame["trapezoid_area"]
  min_area = areas.min()
  max_area = areas.max()
  max_index = areas.idxmax()
  slopes = calculate_slopes(areas.to_list())
  slopes_early = slopes[0:max_index-start] if len(slopes)>0 else []
  slopes_late = slopes[max_index-start:] if len(slopes)>0 else []
  from numpy import mean
  return {"min_area": min_area, "max_area": max_area, "area_growth_slope_early": float(mean(slopes_early)) if len(slopes_early) > 0 else 0, "area_growth_slope_late": float(mean(slopes_late)) if len(slopes_late) > 0 else 0}

def calculate_case_level_detection_stability(start, stop) -> dict:
  return {"longest_detection_run": int(stop-start+1)}

def calculate_case_level_features(case_df) -> dict:
  start, stop, case_df = smooth(case_df, max_tolerable_jump = 5)
  if start < stop:
    input = case_df.iloc[start:stop+1]
    geometric_features = calculate_case_level_geometric_feature(input, start, stop)
    area_features = calculate_case_level_area_feature(input, start, stop)
    detection_stability = calculate_case_level_detection_stability(start, stop)
    return {**geometric_features, **area_features, **detection_stability}
  else:
    return {}
