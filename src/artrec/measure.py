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


def refine_corners(gray: np.ndarray, quad: np.ndarray, samples: int = 40, reach: float = 7.0) -> np.ndarray:
    """Sub-pixel corners. Along each side of the rough quad, find the paper's edge as the
    steepest rise in brightness into the paper (searching along the side's normal; this
    also works when the sheet casts a thin shadow), fit a line through those points and
    intersect neighbouring lines. The rough quad is only pixel-accurate, and on a small
    sheet a 1 px corner error tilts the whole wall plane."""
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
            step = offs[1] - offs[0]
            per_px = int(round(1 / step))
            rise = np.convolve(np.diff(prof), np.ones(3) / 3, mode="same")  # towards the paper
            k = int(np.argmax(rise))
            if rise[k] * per_px < 2:  # under ~2 grey levels per pixel: no edge here
                continue
            # The edge is where brightness passes halfway between the dark side (wall, or the
            # sheet's thin shadow) and the paper, right around the steepest rise
            w = 2 * per_px
            lo = prof[max(0, k - w):k + 1].min()
            hi = prof[k + 1:k + 1 + w].max() if k + 1 < len(prof) else prof[k]
            mid = (lo + hi) / 2
            js = [j for j in range(max(0, k - w), min(len(prof) - 1, k + w))
                  if prof[j] < mid <= prof[j + 1]]
            if not js:
                continue
            j = min(js, key=lambda j: abs(j - k))
            f = (mid - prof[j]) / (prof[j + 1] - prof[j])
            pts.append(base + (offs[j] + f * step) * n)
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


def _candidate_masks(gray: np.ndarray):
    """Binary images in which the sheet may show up as a blob or an outline."""
    k5 = np.ones((5, 5), np.uint8)
    # 1) globally bright: paper is often the brightest flat thing in the photo
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k5)
    for thr in (200, 180, 160):
        yield cv2.morphologyEx(cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)[1], cv2.MORPH_CLOSE, k5)
    # 2) locally bright: a little brighter than the wall around it (white or light walls,
    #    dim rooms, uneven light)
    g = gray.astype(np.float32)
    for frac in (0.06, 0.15):
        local = cv2.GaussianBlur(g, (0, 0), max(5.0, frac * min(gray.shape)))
        for delta in (3, 6, 12):
            bw = ((g - local) > delta).astype(np.uint8) * 255
            yield cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, k5)
    # 3) outlines: works even when paper and wall are about equally bright
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    for lo, hi in ((6, 18), (15, 45)):
        yield cv2.dilate(cv2.Canny(smooth, lo, hi), np.ones((3, 3), np.uint8))


def find_a4(image_rgb: np.ndarray, ratio_tol: float = 0.18) -> np.ndarray | None:
    """Find a bright, low-saturation quadrilateral with A4 proportions.
    Returns ordered corners (4, 2) or None."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    img_area = gray.shape[0] * gray.shape[1]

    best, best_score = None, 0.0
    for bw in _candidate_masks(gray):
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.0005 * img_area or area > 0.5 * img_area:
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
            inside = m.astype(bool)
            if hsv[..., 1][inside].mean() > 60:  # paper is not saturated
                continue
            # Paper is plain: rules out windows, framed pictures, switches with print on them
            core = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool)
            if core.sum() > 20 and gray[core].std() > 14:
                continue
            # ...and a little brighter than the wall right around it
            ring = cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(bool) & ~cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
            if gray[core].mean() < gray[ring].mean() + 2:
                continue
            fill = cv2.contourArea(c) / max(1.0, cv2.contourArea(q))  # rectangular-ness
            score = area * min(fill, 1 / fill) * (1 - abs(ratio - A4_RATIO) / A4_RATIO)
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
