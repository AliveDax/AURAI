"""HTTP API for the phone app. The models run here; the phone only sends a photo.

    uvicorn artrec.api:app --host 0.0.0.0 --port 8000

Serves the mobile web app (web/) at /, plus:
    POST /api/recommend         room photo + preferences -> ranked artworks with explanations
    GET  /api/art/{art_id}.jpg  artwork thumbnail
    GET  /api/health

Set ARTREC_NO_SEGMENTER=1 to use the heuristic wall detector instead of SegFormer.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import threading
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import CACHE_DIR, ROOT
from .recommender import EXPLANATION_LABELS, Query, Recommender
from .room import load_photo

WEB_DIR = ROOT / "web"
THUMB_DIR = CACHE_DIR / "thumbs"
THUMB_SIDE = 640
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ORIENTATIONS = {"portrait", "landscape", "square"}
SIZE_MODES = {"a4", "manual", "skip"}


def _num(v) -> float | None:
    return float(v) if v is not None and math.isfinite(float(v)) else None


def _box(b) -> list[int] | None:
    return [int(v) for v in b] if b is not None else None


def _parse_box(text: str | None, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """'x,y,w,h' as fractions of the photo (0-1), so the phone needn't know the analysed size."""
    if not text:
        return None
    try:
        fx, fy, fw, fh = (float(v) for v in text.split(","))
    except ValueError:
        raise HTTPException(422, "box must be 'x,y,w,h' fractions") from None
    if not (0 <= fx < 1 and 0 <= fy < 1 and 0 < fw <= 1 - fx + 1e-6 and 0 < fh <= 1 - fy + 1e-6):
        raise HTTPException(422, "box is outside the photo")
    H, W = shape
    return int(fx * W), int(fy * H), max(1, int(fw * W)), max(1, int(fh * H))


def create_app(recommender: Recommender | None = None, load=None) -> FastAPI:
    """`recommender`: a ready Recommender (tests), or None to build the real one
    lazily via `load()` on the first request."""
    api = FastAPI(title="Art for your wall")
    state = {"rec": recommender}
    lock = threading.Lock()  # one analysis at a time: the models aren't thread-safe

    def get_rec() -> Recommender:
        if state["rec"] is None:
            with lock:
                if state["rec"] is None:
                    state["rec"] = (load or _load_default)()
        return state["rec"]

    @api.get("/api/health")
    def health():
        return {"ok": True, "loaded": state["rec"] is not None}

    @api.post("/api/recommend")
    def recommend(
        photo: UploadFile = File(...),
        mood: str = Form(""),
        description: str = Form(""),
        colour: str = Form(""),
        orientation: str = Form(""),
        size_mode: str = Form("a4"),
        width_cm: float | None = Form(None),
        height_cm: float | None = Form(None),
        box: str = Form(""),
        top_n: int = Form(5),
    ):
        data = photo.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Photo is too large (max 25 MB)")
        try:
            img = load_photo(io.BytesIO(data))
        except (UnidentifiedImageError, OSError):
            raise HTTPException(415, "Couldn't read that photo. Try a JPEG or PNG.") from None
        if size_mode not in SIZE_MODES:
            raise HTTPException(422, f"size_mode must be one of {sorted(SIZE_MODES)}")
        manual = None
        if size_mode == "manual":
            if not (width_cm and height_cm and 5 <= width_cm <= 2000 and 5 <= height_cm <= 2000):
                raise HTTPException(422, "Enter the space's width and height in cm (5-2000)")
            manual = (width_cm, height_cm)

        q = Query(
            room_image=img, mood=mood.strip() or None, description=description.strip() or None,
            preferred_color=colour.strip() or None, user_box=_parse_box(box, img.shape[:2]),
            orientation=orientation if orientation in ORIENTATIONS else None,
            manual_size_cm=manual, use_a4=size_mode == "a4",
        )
        rec = get_rec()
        with lock:
            res = rec.recommend(q, top_n=max(1, min(top_n, 20)))

        m = res.measurement
        return {
            "photo": {"width": img.shape[1], "height": img.shape[0]},
            "space": {"box": _box(res.room.space_box), "source": res.room.space_source},
            "a4_corners": res.a4_corners.round(1).tolist() if res.a4_corners is not None else None,
            "a4_requested": size_mode == "a4",
            "measurement": {"width_cm": round(m.width_cm, 1), "height_cm": round(m.height_cm, 1),
                            "source": m.source} if m else None,
            "room_style": [{"style": k, "p": v} for k, v in list(res.room_style.items())[:3]],
            "wall_colours": res.room.wall_palette.to_hex(),
            "decor_colours": res.room.decor_palette.to_hex() if res.room.decor_palette else [],
            "n_candidates": res.n_candidates,
            "n_size_unknown": res.n_size_unknown,
            "weights": res.weights_used,
            "message": res.message,
            "recommendations": [
                {
                    "rank": r.rank, "art_id": r.art_id,
                    "title": r.row.get("title") or "Untitled", "artist": r.row.get("artist") or "Unknown artist",
                    "style": r.row.get("style") or "", "emotion": r.row.get("emotion") or "",
                    "width_cm": _num(r.row.get("width_cm")), "height_cm": _num(r.row.get("height_cm")),
                    "size_checked": r.details.get("size_checked", False),
                    "palette": r.details["palette_hex"],
                    "image_url": f"/api/art/{quote(r.art_id)}.jpg",  # ids contain spaces
                    "score": round(r.score, 3),
                    "reasons": [{"key": k, "label": EXPLANATION_LABELS.get(k, k), "value": round(v, 3)}
                                for k, v in sorted(r.contributions.items(), key=lambda kv: -kv[1])],
                }
                for r in res.recommendations
            ],
        }

    @api.get("/api/art/{art_id}.jpg")
    def art_image(art_id: str):
        rec = get_rec()
        meta = rec.catalog.meta
        hit = meta.index[meta["art_id"].astype(str) == art_id]
        if len(hit) == 0:
            raise HTTPException(404, "Unknown artwork")
        src = Path(str(meta.at[hit[0], "image_path"]))
        src = src if src.is_absolute() else ROOT / src
        thumb = THUMB_DIR / f"{hashlib.sha1(art_id.encode()).hexdigest()[:16]}.jpg"
        if not thumb.exists():
            if not src.exists():
                raise HTTPException(404, "Image file missing")
            THUMB_DIR.mkdir(parents=True, exist_ok=True)
            im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            im.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
            im.save(thumb, "JPEG", quality=85)
        return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})

    if WEB_DIR.exists():
        api.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return api


def _load_default() -> Recommender:
    from .catalog import Catalog
    from .clip_model import ClipEncoder
    seg = None
    if not os.environ.get("ARTREC_NO_SEGMENTER"):
        from .room import WallSegmenter
        seg = WallSegmenter()
    return Recommender(Catalog.load(), ClipEncoder(), seg)


app = create_app()
