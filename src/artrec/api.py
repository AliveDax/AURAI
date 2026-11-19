"""HTTP API for the phone app (web/). The models run here; the phone only sends photos and text.

    uvicorn artrec.api:app --host 0.0.0.0 --port 8000

The ranking is exactly recommend.py's (analyze -> score -> rank): this file only turns a request
into a UserInput and the Recommendations into JSON, like app/streamlit_app.py does for the
Streamlit page. "See it on my wall" is wall_preview.py and "Save" is result_sheet.py, unchanged.

    POST /api/recommend           room photo(s) + mood text + colour -> ranked artworks, explained
    POST /api/preview             one artwork at real size on a room photo, where the user tapped
    GET  /api/sheet/{session}.jpg room photo(s) + the top artworks as one image
    GET  /api/art/{artwork_id}.jpg artwork thumbnail
    GET  /api/health

Each /api/recommend keeps its photos for a while under a session id (newest few only), so the
wall preview and the Save image can reuse them without uploading again.
"""
from __future__ import annotations

import hashlib
import io
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .clip_embed import load_image
from .color_extract import palette_at
from .config import Config, load_config
from .recommend import Resources, UserInput, analyze, load_resources, rank, score, to_recommendations
from .result_sheet import result_sheet
from .wall_preview import CM_PER_INCH, load_artwork_sizes, open_room, place_artwork, px_per_inch

MAX_PHOTOS = 4
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SESSIONS = 8   # photo sets kept for the wall preview and Save
THUMB_SIDE = 640


class Sessions:
    """The photos of the newest MAX_SESSIONS requests, plus what was derived from them."""

    def __init__(self, size: int = MAX_SESSIONS):
        self.size, self.items, self.lock = size, OrderedDict(), threading.Lock()

    def add(self, photos: list[bytes], ranked_ids: list[int]) -> str:
        sid = uuid.uuid4().hex
        with self.lock:
            self.items[sid] = {"photos": photos, "ids": ranked_ids, "rooms": {}, "sheet": None}
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return sid

    def get(self, sid: str) -> dict:
        with self.lock:
            s = self.items.get(sid)
        if s is None:
            raise HTTPException(410, "These results have expired. Run the search again.")
        return s


def create_app(resources: Resources | None = None, cfg: Config | None = None, thumb_dir=None) -> FastAPI:
    """resources: ready Resources (tests), or None to load the real catalog, embeddings and CLIP
    on the first request. thumb_dir: where artwork thumbnails are cached."""
    cfg = cfg or load_config()
    thumb_dir = Path(thumb_dir) if thumb_dir else cfg.root / "data" / "cache" / "thumbs_5k"
    api = FastAPI(title="Art for your wall")
    state = {"res": resources}
    lock = threading.Lock()  # one analysis at a time: CLIP isn't thread-safe
    sessions = Sessions()
    sizes = load_artwork_sizes(cfg)  # optional known real sizes (data/artwork_sizes.csv)
    w = cfg.wall_preview

    def get_res() -> Resources:
        if state["res"] is None:
            with lock:
                if state["res"] is None:
                    state["res"] = load_resources(cfg)
        return state["res"]

    def art_row(artwork_id: int):
        cat = get_res().catalog
        if not 0 <= artwork_id < len(cat):
            raise HTTPException(404, "Unknown artwork")
        return cat.iloc[artwork_id]

    def art_path(row):
        return cfg.root / row["image_path"]

    @api.get("/api/health")
    def health():
        return {"ok": True, "loaded": state["res"] is not None}

    @api.post("/api/recommend")
    def recommend(photos: list[UploadFile] = File(...), mood: str = Form(""), colour: str = Form("")):
        if not photos:
            raise HTTPException(422, "Add at least one room photo.")
        if len(photos) > MAX_PHOTOS:
            raise HTTPException(422, f"Up to {MAX_PHOTOS} photos of the same room.")
        data = []
        for p in photos:
            b = p.file.read(MAX_UPLOAD_BYTES + 1)
            if len(b) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "A photo is too large (max 25 MB).")
            try:
                with Image.open(io.BytesIO(b)) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                raise HTTPException(415, "Couldn't read a photo. Try a JPEG or PNG.") from None
            data.append(b)

        res = get_res()
        user = UserInput([io.BytesIO(b) for b in data], mood.strip() or None, colour.strip() or None)
        with lock:
            try:
                q = analyze(user, cfg, res)
            except ValueError as err:  # e.g. a malformed colour
                raise HTTPException(422, str(err)) from None
            s = score(q, cfg, res)
            idx, n_color, subject = rank(s, cfg.scoring["top_n"], cfg)
            recs = to_recommendations(s, idx, res, q.parsed, n_color, subject)
        sid = sessions.add(data, [r.artwork_id for r in recs])

        p = q.parsed
        subject_info = None
        if s.subject_z is not None:
            clear = int((s.subject_z >= cfg.subject["min_z"]).sum())
            subject_info = {"text": p.description_text, "clear_matches": clear,
                            "closest_only": clear < cfg.subject["min_pool"]}

        def rec_json(r):
            row = res.catalog.iloc[r.artwork_id]
            try:
                with Image.open(art_path(row)) as im:  # header only: the artwork's proportions
                    aspect = im.width / im.height
            except OSError:
                aspect = 1.0
            known = sizes.get(r.image_path)
            return {
                "rank": r.rank, "artwork_id": r.artwork_id,
                "title": r.title or "Untitled", "artist": r.artist or "Unknown artist", "style": r.style,
                "emotion_label": r.dominant_emotion,  # EmoArt's label: shown, never scored
                "final_score": round(r.final_score, 4),
                "components": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r.components.items()},
                "subject_match": r.subject_match,
                "color_match": None if r.color_match is None else round(r.color_match, 4),
                "picked_for_color": r.picked_for_color,
                "exact_colour": r.color_match is not None and r.color_match >= cfg.color["color_match_min_share"],
                "palette": palette_at(res.palettes, r.artwork_id).to_hex(),
                "aspect": round(aspect, 4),
                "size_cm": list(known) if known else None,
                "image_url": f"/api/art/{r.artwork_id}.jpg",
            }

        return {
            "session": sid,
            "photos": len(data),
            "parsed": {"emotion": p.emotion_text, "emotion_labels": p.emotion_labels,
                       "description": p.description_text, "colours": p.colors, "colour_words": p.color_words},
            "room_colours": q.room_palette.to_hex(),
            "colour_picks": n_color,
            "subject": subject_info,
            "weights": {k: round(v, 4) for k, v in s.weights.items()},
            "n_catalog": len(res.catalog),
            "preview_defaults": {"distance_ft": w["camera_distance_ft"],
                                 "artwork_width_cm": w["default_artwork_width_cm"],
                                 "max_artwork_cm": w["max_artwork_cm"]},
            "recommendations": [rec_json(r) for r in recs],
        }

    @api.post("/api/preview")
    def preview(session: str = Form(...), artwork_id: int = Form(...), photo: int = Form(0),
                x: float = Form(...), y: float = Form(...), width_cm: float | None = Form(None),
                height_cm: float | None = Form(None), distance_ft: float = Form(w["camera_distance_ft"])):
        """The artwork centred on (x, y) (fractions of the photo) at its real size, with a shadow.
        Same steps as the Streamlit page's "See it on your wall" dialog."""
        s = sessions.get(session)
        if not 0 <= photo < len(s["photos"]):
            raise HTTPException(422, "Unknown photo")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise HTTPException(422, "Tap inside the photo")
        if not 2 <= distance_ft <= 30:
            raise HTTPException(422, "The distance must be between 2 and 30 ft")
        row = art_row(artwork_id)
        known = sizes.get(row["image_path"])
        if known:
            width_cm, height_cm = known
        top = w["max_artwork_cm"]
        if not (width_cm and height_cm and 1 <= width_cm <= top and 1 <= height_cm <= top):
            raise HTTPException(422, f"Enter the artwork's width and height (1-{top} cm)")

        room = s["rooms"].get(photo)
        if room is None:  # decode the photo once per session, not on every tap
            room = s["rooms"][photo] = open_room(io.BytesIO(s["photos"][photo]), w)
        size_in = (width_cm / CM_PER_INCH, height_cm / CM_PER_INCH)
        ppi = px_per_inch(room.image.size, room.focal_35mm, distance_ft * 12)
        if size_in[0] * ppi > 2 * room.image.width or size_in[1] * ppi > 2 * room.image.height:
            raise HTTPException(422, "At this scale the artwork is far bigger than the whole photo. "
                                     "Check its size and the distance.")
        with Image.open(art_path(row)) as art:
            shown, light = place_artwork(room.image, art.convert("RGB"),
                                         (x * room.image.width, y * room.image.height), size_in, ppi, w)
        buf = io.BytesIO()
        shown.save(buf, "JPEG", quality=88)
        info = {"X-Light": light.describe(), "X-Lens": room.focal_source, "X-Scale": f"{ppi:.1f}",
                "X-Size-Cm": f"{width_cm:g}x{height_cm:g}", "X-Known-Size": "1" if known else "0"}
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={k: quote(v) for k, v in info.items()} | {"Cache-Control": "no-store"})

    @api.get("/api/sheet/{session}.jpg")
    def sheet(session: str):
        s = sessions.get(session)
        if s["sheet"] is None:
            ss = cfg.save_sheet
            rooms = [load_image(io.BytesIO(b), ss["width_px"]) for b in s["photos"]]
            arts = []
            for i in s["ids"][: ss["n_artworks"]]:
                with Image.open(art_path(art_row(i))) as im:
                    arts.append(im.convert("RGB"))
            buf = io.BytesIO()
            result_sheet(rooms, arts, ss).save(buf, "JPEG", quality=92)
            s["sheet"] = buf.getvalue()
        name = f"art_recommendations_{datetime.now():%Y%m%d_%H%M}.jpg"
        return Response(s["sheet"], media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @api.get("/api/art/{artwork_id}.jpg")
    def art_image(artwork_id: int):
        row = art_row(artwork_id)
        src = art_path(row)
        key = hashlib.sha1(f"{row['image_path']}|{THUMB_SIDE}".encode()).hexdigest()[:16]
        thumb = thumb_dir / f"{key}.jpg"
        if not thumb.exists():
            if not src.exists():
                raise HTTPException(404, "Image file missing")
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
            tmp = thumb.with_suffix(".tmp")
            im.save(tmp, "JPEG", quality=85)
            tmp.replace(thumb)
        return FileResponse(thumb, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})

    web = cfg.root / "web"
    if web.exists():
        api.mount("/", StaticFiles(directory=web, html=True), name="web")
    return api


app = create_app()
