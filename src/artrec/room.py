"""Room photo analysis.

From one photo we need:
  - where the wall is                         (pretrained segmentation, no training)
  - where on the wall there is clear space     (our own geometry: largest empty rectangle)
  - the wall colour near that space            (for colour harmony)
  - the colours and style of the surroundings  (furniture/decor, for harmony + style)

The user can override the detected space by drawing/choosing a box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .color import Palette, extract_palette
from .config import PHOTO_MAX_SIDE, SEGMENTATION_MODEL, ColorConfig

Box = tuple[int, int, int, int]  # x, y, w, h in pixels


def load_photo(src, max_side: int = PHOTO_MAX_SIDE) -> np.ndarray:
    """Open a room photo (path or file-like) as uint8 RGB, upright and downscaled.
    Phones store photos unrotated plus an EXIF orientation flag; without
    exif_transpose a portrait photo would be analysed lying on its side."""
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Wall masks
# ---------------------------------------------------------------------------


class WallSegmenter:
    """Pretrained SegFormer (ADE20K), used as-is. ADE20K includes a 'wall' class."""

    def __init__(self, model_name: str = SEGMENTATION_MODEL, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.wall_id = labels.get("wall", 0)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=image_rgb.shape[:2], mode="bilinear", align_corners=False
        )
        return (logits.argmax(1)[0] == self.wall_id).cpu().numpy()


def heuristic_wall_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Fallback without a segmentation model: the wall is usually the largest
    smooth, low-texture region. Works for plain walls; fails on wallpaper."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = max(5, (min(gray.shape) // 40) | 1)
    mean = cv2.blur(gray, (k, k))
    sq = cv2.blur(gray * gray, (k, k))
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0))
    smooth = (local_std < 8).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, connectivity=4)
    if n <= 1:
        return np.ones(gray.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def clear_wall_mask(image_rgb: np.ndarray, wall_mask: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Wall pixels with nothing on them: remove anything with edges (switches,
    frames, shelves) and a safety margin around it."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    pad = max(3, min(gray.shape) // 60)
    edges = cv2.dilate(edges, np.ones((pad, pad), np.uint8))
    clear = wall_mask.astype(bool) & (edges == 0)
    if exclude is not None:
        clear &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((pad, pad), np.uint8)).astype(bool)
    clear = cv2.morphologyEx(clear.astype(np.uint8), cv2.MORPH_OPEN, np.ones((pad, pad), np.uint8))
    return clear.astype(bool)


# ---------------------------------------------------------------------------
# Largest empty rectangle
# ---------------------------------------------------------------------------


def largest_rectangle(mask: np.ndarray, max_side: int = 240) -> Box | None:
    """Largest axis-aligned rectangle fully inside `mask` (classic histogram/stack
    algorithm, O(H*W)). Runs on a downscaled mask for speed."""
    h, w = mask.shape
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(mask.astype(np.uint8), (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    sh, sw = small.shape
    heights = np.zeros(sw, dtype=int)
    best, best_box = 0, None
    for row in range(sh):
        heights = np.where(small[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(sw + 1):
            cur = heights[col] if col < sw else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                height = heights[top]
                left = stack[-1] + 1 if stack else 0
                area = height * (col - left)
                if area > best:
                    best = area
                    best_box = (left, row - height + 1, col - left, height)
            stack.append(col)
    if best_box is None or best == 0:
        return None
    x, y, bw, bh = best_box
    return (int(x / scale), int(y / scale), int(bw / scale), int(bh / scale))


def shrink_box(box: Box, frac: float = 0.05) -> Box:
    x, y, w, h = box
    dx, dy = int(w * frac), int(h * frac)
    return (x + dx, y + dy, max(1, w - 2 * dx), max(1, h - 2 * dy))


def expand_box(box: Box, frac: float, shape: tuple[int, int]) -> Box:
    x, y, w, h = box
    H, W = shape
    dx, dy = int(w * frac), int(h * frac)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(W, x + w + dx), min(H, y + h + dy)
    return (x0, y0, x1 - x0, y1 - y0)


def box_mask(box: Box, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(shape, bool)
    x, y, w, h = box
    m[y:y + h, x:x + w] = True
    return m


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


@dataclass
class RoomAnalysis:
    wall_mask: np.ndarray
    space_box: Box | None
    space_source: str               # "detected" | "user" | "none"
    wall_palette: Palette
    decor_palette: Palette | None
    surroundings_image: np.ndarray  # photo with the empty wall greyed out, for CLIP

    @property
    def space_aspect(self) -> float | None:
        if self.space_box is None:
            return None
        _, _, w, h = self.space_box
        return w / h


def analyze_room(
    image_rgb: np.ndarray,
    color_cfg: ColorConfig,
    segmenter=None,
    user_box: Box | None = None,
    exclude_mask: np.ndarray | None = None,
) -> RoomAnalysis:
    """`segmenter`: a WallSegmenter, or None to use the heuristic fallback.
    `exclude_mask`: pixels to ignore (e.g. the A4 reference sheet)."""
    shape = image_rgb.shape[:2]
    wall = segmenter(image_rgb) if segmenter is not None else heuristic_wall_mask(image_rgb)
    if exclude_mask is not None:
        wall = wall & ~exclude_mask

    if user_box is not None:
        space, source = user_box, "user"
    else:
        found = largest_rectangle(clear_wall_mask(image_rgb, wall, exclude_mask))
        space, source = (shrink_box(found), "detected") if found else (None, "none")

    # Wall colour: wall pixels in and around the chosen space (local colour matters,
    # lighting varies across a wall). Fall back to all wall pixels, then the box.
    region = wall & box_mask(expand_box(space, 0.5, shape), shape) if space else wall
    if region.sum() < 500:
        region = wall if wall.sum() >= 500 else (box_mask(space, shape) if space else np.ones(shape, bool))
    wall_palette = extract_palette(image_rgb, color_cfg.k_wall, mask=region, max_pixels=color_cfg.max_pixels)

    decor_region = ~wall
    if exclude_mask is not None:
        decor_region &= ~exclude_mask
    decor_palette = (
        extract_palette(image_rgb, color_cfg.k_room, mask=decor_region, max_pixels=color_cfg.max_pixels)
        if decor_region.sum() >= 500 else None
    )

    surroundings = image_rgb.copy()
    surroundings[wall] = (128, 128, 128)

    return RoomAnalysis(wall, space, source, wall_palette, decor_palette, surroundings)
