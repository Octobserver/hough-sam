import cv2
import numpy as np
from collections import deque

def find_optic_disk_circles_structured(image, edge_model_path: str = None):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2,a,b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB).astype(np.float32)
    denoised = cv2.fastNlMeansDenoisingColored((enhanced).astype(np.uint8), None, 10,10,7,21)
    denoised = denoised.astype(np.float32) / 255.0

    if edge_model_path is not None and hasattr(cv2.ximgproc, 'createStructuredEdgeDetection'):
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection(str(edge_model_path))
        edges = edge_detector.detectEdges(denoised) * 255.0
    else:
        gray = cv2.cvtColor((denoised*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed = cv2.convertScaleAbs(closed)

    raw = cv2.HoughCircles(
        closed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=0.1,
        param1=100,
        param2=40,
        minRadius=60,
        maxRadius=100,
    )
    if raw is None:
        return None, None
    circles = np.round(raw[0]).astype(int)

    h, w = image.shape[:2]
    img_c = np.array([w//2, h//2])

    def dist2_center(c):
        return (c[0]-img_c[0])**2 + (c[1]-img_c[1])**2

    candidates = sorted(circles, key=dist2_center)[:64]

    valid_pairs = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            c1, c2 = candidates[i], candidates[j]
            center_dist = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
            radius_diff = abs(c1[2] - c2[2])
            inner_r = min(c1[2], c2[2])
            if center_dist < inner_r*0.1 and radius_diff < inner_r *0.6 and radius_diff > inner_r *0.2:
                valid_pairs.append((center_dist-radius_diff, c1, c2))

    if valid_pairs:
        _, c1, c2 = min(valid_pairs, key=lambda x: x[0])
        inner, outer = sorted((c1, c2), key=lambda c: c[2])
    else:
        sorted_by_r = sorted(candidates, key=lambda c: c[2])
        inner, outer = sorted_by_r[0], sorted_by_r[-1]

    return inner, outer

def sample_ring_points(inner, outer, center, num_points=3):
    cx, cy = center
    r_out = (inner[2] + outer[2]) / 2.0
    r_in = 2 * inner[2] - outer[2]
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    outer_pts, inner_pts = [], []
    for a in angles:
        x_o = int(cx + r_out * np.cos(a))
        y_o = int(cy + r_out * np.sin(a))
        x_i = int(cx + r_in * np.cos(a))
        y_i = int(cy + r_in * np.sin(a))
        outer_pts.append([x_o,y_o])
        inner_pts.append([x_i,y_i])
    return outer_pts, inner_pts

class RingDetector:
    def __init__(self, N=5):
        self.N = N
        self.buf = deque(maxlen=N)

    def add_frame(self, frame):
        self.buf.append(frame)

    def get_average(self):
        arr = np.stack(self.buf, axis=0).astype(np.float32)
        avg = np.mean(arr, axis=0)
        return avg.astype(np.uint8)

    def detect_on_average(self):
        if len(self.buf) < self.N:
            return None, None
        avg_frame = self.get_average()
        inner, outer = find_optic_disk_circles_structured(avg_frame)
        return inner, outer
