"""Measuring the empty space in centimetres.

A single photo has no scale. If the user tapes an A4 sheet (21.0 x 29.7 cm) to
the wall, we detect its four corners and compute a homography: a mapping from
image pixels to centimetres on the wall plane. That also corrects perspective
when the photo is taken at an angle. Only valid for points on the same wall.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

A4_SHORT_CM, A4_LONG_CM = 21.0, 29.7
A4_RATIO = A4_LONG_CM / A4_SHORT_CM  # ~1.414


@dataclass
class Measurement:
    width_cm: float
    height_cm: float
    source: str  # "a4" | "manual"


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _intersect(l1, l2) -> np.ndarray | None:
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    s_, _ = np.linalg.solve(A, p2 - p1)
    return p1 + s_ * d1


def refine_corners(gray: np.ndarray, quad: np.ndarray, samples: int = 40, reach: float = 4.0) -> np.ndarray:
    """Sub-pixel corners. Along each side of the rough quad, find where the brightness
    crosses halfway between paper and wall (searching along the side's normal), fit a
    line through those points and intersect neighbouring lines. The rough quad is only
    pixel-accurate, and on a small sheet a 1 px corner error tilts the whole wall plane."""
    g = gray.astype(np.float32)
    q = quad.astype(np.float64)
    centre = q.mean(0)
    offs = np.linspace(-reach, reach, int(reach * 8) + 1)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        if n @ (centre - a) < 0:
            n = -n  # normal points into the paper
        pts = []
        for t in np.linspace(0.15, 0.85, samples):
            base = a + t * (b - a)
            xy = (base[None, :] + offs[:, None] * n[None, :]).astype(np.float32)
            prof = cv2.remap(g, xy[:, 0].reshape(1, -1), xy[:, 1].reshape(1, -1), cv2.INTER_LINEAR).ravel()
            lo, hi = prof[:4].mean(), prof[-4:].mean()
            if hi - lo < 15:  # no clear paper/wall step here
                continue
            mid = (lo + hi) / 2
            k = np.flatnonzero((prof[:-1] < mid) & (prof[1:] >= mid))
            if len(k) != 1:
                continue
            k = k[0]
            f = (mid - prof[k]) / (prof[k + 1] - prof[k])
            pts.append(base + (offs[k] + f * (offs[1] - offs[0])) * n)
        if len(pts) < samples // 3:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):  # corner i is where side i-1 meets side i
        c = _intersect(lines[i - 1], lines[i])
        if c is None:
            return quad
        out.append(c)
    out = np.array(out, np.float32)
    # sanity: refinement should only nudge the corners
    return out if np.abs(out - quad).max() < reach + 2 else quad


def find_a4(image_rgb: np.ndarray, ratio_tol: float = 0.18) -> np.ndarray | None:
    """Find a bright, low-saturation quadrilateral with A4 proportions.
    Returns ordered corners (4, 2) or None."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_area = gray.shape[0] * gray.shape[1]

    best, best_score = None, 0.0
    # Paper is usually the brightest flat thing; try a few thresholds for robustness.
    for thr in (None, 200, 180, 160):
        if thr is None:
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.001 * img_area or area > 0.5 * img_area:
                continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            q = order_corners(approx)
            sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
            w, h = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
            ratio = max(w, h) / max(1e-6, min(w, h))
            if abs(ratio - A4_RATIO) / A4_RATIO > ratio_tol:
                continue
            m = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(m, q.astype(np.int32), 1)
            sat = hsv[..., 1][m.astype(bool)].mean()
            if sat > 60:  # paper is not saturated
                continue
            fill = area / cv2.contourArea(q)  # rectangular-ness
            score = area * fill * (1 - abs(ratio - A4_RATIO) / A4_RATIO)
            if score > best_score:
                best, best_score = q, score
    return refine_corners(gray, best) if best is not None else None


def a4_mask(corners: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, corners.astype(np.int32), 1)
    return m.astype(bool)


def homography_from_a4(corners: np.ndarray) -> np.ndarray:
    """Pixel -> cm homography on the wall plane, choosing portrait/landscape
    from which pair of sides is longer in the image."""
    q = order_corners(corners)
    top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    side = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    w_cm, h_cm = (A4_LONG_CM, A4_SHORT_CM) if top > side else (A4_SHORT_CM, A4_LONG_CM)
    dst = np.array([[0, 0], [w_cm, 0], [w_cm, h_cm], [0, h_cm]], np.float32)
    return cv2.getPerspectiveTransform(q, dst)


def measure_box(H: np.ndarray, box: tuple[int, int, int, int]) -> Measurement:
    x, y, w, h = box
    pts = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], np.float32)
    cm = cv2.perspectiveTransform(pts, H).reshape(4, 2)
    width = (np.linalg.norm(cm[1] - cm[0]) + np.linalg.norm(cm[2] - cm[3])) / 2
    height = (np.linalg.norm(cm[3] - cm[0]) + np.linalg.norm(cm[2] - cm[1])) / 2
    return Measurement(float(width), float(height), "a4")
