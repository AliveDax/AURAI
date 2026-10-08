import cv2
import numpy as np

from artrec.measure import Measurement, homography_from_a4
from artrec.preview import FILL, place_artwork


def size_of(quad):
    q = np.array(quad)
    return np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])


def test_manual_measurement_gives_true_scale():
    # 200 x 100 px space measured as 100 x 50 cm -> 2 px/cm
    p = place_artwork((100, 50, 200, 100), 40.0, 30.0, 4 / 3, Measurement(100, 50, "manual"), None)
    assert p.true_size
    w, h = size_of(p.quad)
    assert np.isclose(w, 80) and np.isclose(h, 60)
    assert np.allclose(np.mean(p.quad, 0), [200, 100])  # centred in the space


def test_unknown_size_fills_space_with_artwork_aspect():
    p = place_artwork((0, 0, 400, 200), None, None, 0.5, Measurement(100, 50, "manual"), None)
    assert not p.true_size
    w, h = size_of(p.quad)
    assert np.isclose(h, 200 * FILL) and np.isclose(w / h, 0.5)
    assert p.size_cm == (20, 35)  # shown size in cm (space 100 x 50 cm), rounded to 5 cm


def test_a4_placement_round_trips_through_the_homography():
    # A4 sheet seen at 3 px/cm with mild perspective
    a4 = np.array([[500, 100], [563, 104], [563, 193], [500, 190]], np.float32)
    H = homography_from_a4(a4)
    space = (100, 80, 300, 200)
    p = place_artwork(space, 60.0, 40.0, 1.5, Measurement(0, 0, "a4"), a4)
    assert p.true_size
    cm = cv2.perspectiveTransform(np.float32(p.quad).reshape(-1, 1, 2), H).reshape(4, 2)
    assert np.isclose(np.linalg.norm(cm[1] - cm[0]), 60, rtol=1e-3)
    assert np.isclose(np.linalg.norm(cm[3] - cm[0]), 40, rtol=1e-3)
