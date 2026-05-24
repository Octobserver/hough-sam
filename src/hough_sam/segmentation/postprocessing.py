from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
import math
import numpy as np
import cv2

@dataclass
class IncisionParams:
    open_kernel: int = 3
    close_kernel: int = 7
    blur_sigma: float = 0.8
    min_area: int = 150
    min_area_triangle: int = 150
    keep_top_k: int = 5
    approx_eps_frac: float = 0.03
    parallel_tol_deg: float = 12.0
    perpendicular_tol_deg: float = 18.0
    trapezoid_min_aspect: float = 0.25
    require_trapezoid: bool = False

def _angle_deg(v: np.ndarray) -> float:
    ang = math.degrees(math.atan2(float(v[1]), float(v[0])))
    return abs(ang) % 180.0

def _ang_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def _edge_angles_and_lengths(poly: np.ndarray) -> Tuple[List[float], List[float], List[np.ndarray]]:
    n = poly.shape[0]
    vecs, angs, lens = [], [], []
    for i in range(n):
        j = (i + 1) % n
        v = poly[j] - poly[i]
        vecs.append(v)
        angs.append(_angle_deg(v))
        lens.append(float(np.linalg.norm(v)))
    return angs, lens, vecs

class IncisionDetector:
    def __init__(self, params: Optional[IncisionParams] = None) -> None:
        self.P = params or IncisionParams()

    def process_frame(self, mask: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        m = self._preprocess_mask(mask)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: self.P.keep_top_k]
        best_quad_score = None
        best_quad = None
        quad_debug = {"tested": 0, "rejected_reason": []}
        best_tri_score = None
        best_tri = None
        tri_debug = {"tested": 0, "rejected_reason": []}
        for c in cnts:
            quad_debug["tested"] += 1
            quad = self._contour_to_quad(c)
            if quad is None:
                quad_debug["rejected_reason"].append("not-quad")
            else:
                trap_ok, _ = self._is_trapezoid(quad)
                if self.P.require_trapezoid and not trap_ok:
                    quad_debug["rejected_reason"].append("not-trapezoid")
                else:
                    score = self._quad_score(quad)
                    if best_quad_score is None or score > best_quad_score:
                        best_quad_score, best_quad = score, quad.copy()
            tri_debug["tested"] += 1
            tri = self._contour_to_triangle(c)
            if tri is None:
                tri_debug["rejected_reason"].append("not-triangle")
            else:
                score = self._triangle_score(tri)
                if best_tri_score is None or score > best_tri_score:
                    best_tri_score, best_tri = score, tri.copy()
        best_quad_result = {"accepted": best_quad is not None, "poly": best_quad, "score": best_quad_score, "debug": quad_debug, "mask_clean": m}
        best_tri_result = {"accepted": best_tri is not None, "poly": best_tri, "score": best_tri_score, "debug": tri_debug, "mask_clean": m}
        return best_quad_result, best_tri_result

    def _quad_score(self, quad: np.ndarray) -> float:
        area = cv2.contourArea(quad.astype(np.int32))
        return 0.001 * float(area)

    def _triangle_score(self, tri: np.ndarray) -> float:
        area = cv2.contourArea(tri.astype(np.int32))
        return 0.001 * float(area)

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        P = self.P
        m = mask.astype(np.uint8)
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        if P.blur_sigma and P.blur_sigma > 0:
            m = cv2.GaussianBlur(m, (0, 0), P.blur_sigma, P.blur_sigma)
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        if P.open_kernel and P.open_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P.open_kernel, P.open_kernel))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if P.close_kernel and P.close_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P.close_kernel, P.close_kernel))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        comps.sort(key=lambda x: x[1], reverse=True)
        kept = 0
        for i, area in comps:
            if area >= P.min_area and kept < P.keep_top_k:
                keep[labels == i] = 255
                kept += 1
        return keep

    def _order_clockwise(self, pts: np.ndarray) -> np.ndarray:
        c = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
        return pts[order]

    def _contour_to_poly(self, cnt: np.ndarray, n_vertices: int) -> Optional[np.ndarray]:
        P = self.P
        peri = cv2.arcLength(cnt, True)
        eps = P.approx_eps_frac * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) > max(8, n_vertices + 2):
            hull = cv2.convexHull(cnt)
            peri = cv2.arcLength(hull, True)
            eps = P.approx_eps_frac * peri
            approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) != n_vertices:
            return None
        pts = approx.reshape(-1, 2).astype(np.float32)
        return self._order_clockwise(pts)

    def _contour_to_quad(self, cnt: np.ndarray) -> Optional[np.ndarray]:
        return self._contour_to_poly(cnt, n_vertices=4)

    def _contour_to_triangle(self, cnt: np.ndarray) -> Optional[np.ndarray]:
        return self._contour_to_poly(cnt, n_vertices=3)

    def _is_trapezoid(self, quad: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int]]]:
        P = self.P
        angs, lens, _ = _edge_angles_and_lengths(quad)
        for a, b in [(0, 2), (1, 3)]:
            if _ang_diff_deg(angs[a], angs[b]) <= P.parallel_tol_deg:
                denom = max(lens[a], lens[b]) if max(lens[a], lens[b]) > 0 else 1.0
                short_over_long = min(lens[a], lens[b]) / denom
                if short_over_long >= P.trapezoid_min_aspect:
                    return True, (a, b)
        return False, None

def mask_center(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Mask is empty")
    cy = ys.mean()
    cx = xs.mean()
    return cy, cx

def polygon_area(approx: np.ndarray) -> float:
    contour = approx.reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour))
    return area

def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    M = cv2.moments(contour)
    if M["m00"] == 0:
        raise ValueError("Contour area is zero, cannot compute centroid")
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy

def dist(c1, c2):
    import numpy as _np
    return _np.linalg.norm(_np.array(c1) - _np.array(c2))

def trapezoid_side(coords, center):
    def _dist(c1, c2):
        import numpy as _np
        return _np.linalg.norm(_np.array(c1) - _np.array(c2))
    sorted_l = sorted(coords, key=lambda x: _dist(x, center))
    return _np.array([sorted_l[0], sorted_l[1]]), _np.array([sorted_l[2], sorted_l[3]])

def triangle_side(coords, center):
    def _dist(c1, c2):
        import numpy as _np
        return _np.linalg.norm(_np.array(c1) - _np.array(c2))
    sorted_l = sorted(coords, key=lambda x: _dist(x, center))
    return _np.array([sorted_l[0], sorted_l[1]]), _np.array([sorted_l[0], sorted_l[2]])
