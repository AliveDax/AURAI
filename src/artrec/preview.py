"""Where an artwork would appear in the room photo, for the "see it on your wall" preview.

Returns the four image-pixel corners (top-left, top-right, bottom-right, bottom-left)
of the artwork centred in the empty space:
  - true size, in perspective: space measured from an A4 sheet and the artwork's size
    is known (cm on the wall plane -> pixels through the A4 homography)
  - true size, flat: space size entered by hand (cm -> px scale from the space box)
  - approximate: otherwise, filling FILL of the space with the artwork's aspect ratio
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .measure import Measurement, homography_from_a4
from .room import Box

FILL = 0.7  # middle of the 60-75% "ideal fill" band in FitConfig


@dataclass
class Placement:
    quad: list[list[float]]  # TL, TR, BR, BL in image pixels
    true_size: bool
    size_cm: tuple[float, float] | None = None  # (w, h) as shown on the wall, if the space was measured


def _rect(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                     [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], np.float64)


def place_artwork(space: Box, art_w_cm: float | None, art_h_cm: float | None, art_aspect: float,
                  measurement: Measurement | None, a4_corners: np.ndarray | None) -> Placement:
    x, y, w, h = space
    cx, cy = x + w / 2, y + h / 2
    sized = art_w_cm is not None and art_h_cm is not None and math.isfinite(art_w_cm) and math.isfinite(art_h_cm)

    if measurement and sized and measurement.source == "a4" and a4_corners is not None:
        H = homography_from_a4(a4_corners)  # image px -> wall cm
        box = np.float32([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
        centre_cm = cv2.perspectiveTransform(box, H).reshape(4, 2).mean(0)
        quad_cm = _rect(centre_cm[0], centre_cm[1], art_w_cm, art_h_cm).astype(np.float32).reshape(-1, 1, 2)
        quad = cv2.perspectiveTransform(quad_cm, np.linalg.inv(H)).reshape(4, 2)
        return Placement(np.round(quad, 1).tolist(), True, (art_w_cm, art_h_cm))

    if measurement and sized:  # entered by hand: the space box is the measured area
        sx, sy = w / measurement.width_cm, h / measurement.height_cm
        return Placement(np.round(_rect(cx, cy, art_w_cm * sx, art_h_cm * sy), 1).tolist(), True,
                         (art_w_cm, art_h_cm))

    # Size unknown (or space not measured): fill FILL of the space with the artwork's shape
    bw, bh = w * FILL, h * FILL
    if bw / bh > art_aspect:
        bw = bh * art_aspect
    else:
        bh = bw / art_aspect
    size_cm = None
    if measurement:  # the size it's shown at = a suggested print size, to the nearest 5 cm
        size_cm = (5 * round(bw / w * measurement.width_cm / 5), 5 * round(bh / h * measurement.height_cm / 5))
    return Placement(np.round(_rect(cx, cy, bw, bh), 1).tolist(), False, size_cm)
