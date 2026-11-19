import numpy as np
from PIL import Image

from artrec.result_sheet import result_sheet


def test_sheet_has_room_and_artworks_in_order_on_white(cfg):
    s = cfg.save_sheet
    room = Image.new("RGB", (800, 500), (10, 120, 10))
    arts = [Image.new("RGB", (300, 400), (20 * i, 0, 200)) for i in range(s["n_artworks"])]
    page = np.asarray(result_sheet([room], arts, s))
    assert page.shape[1] == s["width_px"]
    assert (page[0, 0] == 255).all() and (page[-1, -1] == 255).all()        # white page, no border art
    pad = s["padding_px"]
    cell = (s["width_px"] - pad * (s["columns"] + 1)) // s["columns"]
    top = pad + min(s["room_max_height_px"], int(500 * (s["width_px"] - 2 * pad) / 800)) + pad
    # first artwork in the first cell, second in the second (rank order, left to right)
    for n in (0, 1):
        x = pad + n * (cell + pad) + cell // 2
        assert tuple(page[top + cell // 2, x]) == (20 * n, 0, 200)


def test_two_room_photos_and_few_artworks(cfg):
    rooms = [Image.new("RGB", (600, 400), "red"), Image.new("RGB", (400, 600), "blue")]
    page = result_sheet(rooms, [Image.new("RGB", (100, 100), "black")] * 3, cfg.save_sheet)
    assert page.width == cfg.save_sheet["width_px"] and page.height > 0
