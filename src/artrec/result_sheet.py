"""One image to keep a result: a white page with the room photo(s) on top and the recommended
artworks below, in rank order (left to right, then the next row). Nothing else is drawn on it."""
from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageOps


def result_sheet(rooms: Sequence[Image.Image], artworks: Sequence[Image.Image], s: dict) -> Image.Image:
    """rooms: the room photo(s); artworks: the recommendations, best first; s: config save_sheet."""
    W, pad, cols = s["width_px"], s["padding_px"], s["columns"]

    # Room photo(s): side by side in one row, as large as the page allows
    slot_w = (W - pad * (len(rooms) + 1)) // max(len(rooms), 1)
    room_imgs = [ImageOps.contain(r.convert("RGB"), (slot_w, s["room_max_height_px"])) for r in rooms]
    room_h = max((r.height for r in room_imgs), default=0)

    # Artworks: a grid of square cells, each artwork fitted inside its cell
    cell = (W - pad * (cols + 1)) // cols
    rows = math.ceil(len(artworks) / cols)
    page = Image.new("RGB", (W, pad + room_h + pad + rows * (cell + pad)), "white")

    x = (W - sum(r.width for r in room_imgs) - pad * (len(room_imgs) - 1)) // 2  # centred
    for r in room_imgs:
        page.paste(r, (x, pad + (room_h - r.height) // 2))
        x += r.width + pad

    top = pad + room_h + pad
    for n, art in enumerate(artworks):
        a = ImageOps.contain(art.convert("RGB"), (cell, cell))
        cx = pad + (n % cols) * (cell + pad)
        cy = top + (n // cols) * (cell + pad)
        page.paste(a, (cx + (cell - a.width) // 2, cy + (cell - a.height) // 2))
    return page
