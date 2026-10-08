"""Demo UI. Deliberately simple: the project is graded on the recommendation
system, not the interface. Run with:  streamlit run app/streamlit_app.py"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from artrec.catalog import Catalog
from artrec.config import CATALOG_PARQUET, ROOMS_DIR, ROOT, SETTINGS
from artrec.recommender import Query, Recommender
from artrec.room import load_photo

LABELS = {
    "content": "Matches your description",
    "surroundings": "Suits your room's style",
    "harmony": "Colours work with your wall and decor",
    "preferred_color": "Contains your colour",
    "emotion": "Matches the mood",
    "fit": "Fits the space",
    "comfort_penalty": "Less comforting mood",
}

st.set_page_config(page_title="Art for your wall", layout="wide")


@st.cache_resource(show_spinner="Loading models and catalog…")
def load_system(use_segmenter: bool):
    from artrec.clip_model import ClipEncoder
    seg = None
    if use_segmenter:
        from artrec.room import WallSegmenter
        seg = WallSegmenter()
    return Recommender(Catalog.load(), ClipEncoder(), seg, SETTINGS)


def overlay(img: np.ndarray, box, corners) -> np.ndarray:
    out = img.copy()
    t = max(2, min(img.shape[:2]) // 200)
    if box:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (40, 200, 90), t)
    if corners is not None:
        cv2.polylines(out, [corners.astype(np.int32)], True, (230, 60, 60), t)
    return out


st.title("Find art for your wall")
if not CATALOG_PARQUET.exists():
    st.error("The artwork catalog hasn't been built yet. Run `python scripts/build_catalog.py` first.")
    st.stop()

with st.sidebar:
    st.header("Your wall")
    upload = st.file_uploader("Photo of the wall", type=["jpg", "jpeg", "png", "webp"])
    samples = sorted(p.name for p in ROOMS_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    sample = st.selectbox("Or use one of the test rooms", ["None"] + samples) if samples else "None"
    mood = st.text_input("How should the room feel?", placeholder="calm, cozy, energetic…")
    description = st.text_input("What kind of art? (optional)", placeholder="a landscape, something abstract…")
    use_colour = st.checkbox("I'd like a particular colour in the artwork")
    colour = st.color_picker("Colour", "#2a9d8f") if use_colour else None

    st.header("Where it goes")
    placement = st.radio("Placement", ["Find the empty space for me", "I'll choose the spot"])
    orientation = st.selectbox("Shape", ["Any", "portrait", "landscape", "square"])

    st.header("Size")
    size_mode = st.radio("How should we measure the space?",
                         ["Use the A4 sheet in my photo", "I'll enter the measurements", "Skip the size check"])
    manual = None
    if size_mode == "I'll enter the measurements":
        c1, c2 = st.columns(2)
        manual = (c1.number_input("Width (cm)", 10.0, 1000.0, 120.0), c2.number_input("Height (cm)", 10.0, 1000.0, 80.0))

    use_seg = st.checkbox("Use the wall-segmentation model", value=True,
                          help="Turn off to use a simpler built-in wall detector (no extra download).")

if upload is None and sample == "None":
    st.info("Upload a photo of the wall to get started. For measurements, tape an A4 sheet to the wall first.")
    st.stop()

img = load_photo(upload if upload is not None else ROOMS_DIR / sample)
H, W = img.shape[:2]
user_box = None
if placement == "I'll choose the spot":
    st.subheader("Mark where the artwork should go")
    c1, c2, c3, c4 = st.columns(4)
    left = c1.slider("Left edge %", 0, 95, 30)
    top = c2.slider("Top edge %", 0, 95, 20)
    width = c3.slider("Width %", 5, 100 - left, min(40, 100 - left))
    height = c4.slider("Height %", 5, 100 - top, min(35, 100 - top))
    user_box = (int(W * left / 100), int(H * top / 100), int(W * width / 100), int(H * height / 100))
    st.image(overlay(img, user_box, None))

if not st.button("Find artwork", type="primary"):
    st.stop()

rec = load_system(use_seg)
with st.spinner("Looking at your room…"):
    res = rec.recommend(Query(
        room_image=img, mood=mood or None, description=description or None, preferred_color=colour,
        user_box=user_box, orientation=None if orientation == "Any" else orientation,
        manual_size_cm=manual, use_a4=size_mode == "Use the A4 sheet in my photo",
    ))

left, right = st.columns([3, 2])
with left:
    st.image(overlay(img, res.room.space_box, res.a4_corners),
             caption="Green: where the art goes. Red: the A4 reference sheet.")
with right:
    st.subheader("What we saw")
    if res.measurement:
        st.write(f"Space: **{res.measurement.width_cm:.0f} × {res.measurement.height_cm:.0f} cm** "
                 f"({'from the A4 sheet' if res.measurement.source == 'a4' else 'as entered'})")
    elif size_mode == "Use the A4 sheet in my photo":
        st.warning("No A4 sheet found in the photo, so sizes weren't checked. Enter measurements instead.")
    if res.room.space_source == "none":
        st.warning("Couldn't find a clear area on the wall. Choose the spot yourself.")
    st.write("Room style: " + ", ".join(f"{k} ({v:.0%})" for k, v in list(res.room_style.items())[:3]))
    swatch = lambda hexes: " ".join(  # noqa: E731
        f"<span style='display:inline-block;width:28px;height:28px;background:{h};border:1px solid #999'></span>"
        for h in hexes)
    st.markdown("Wall colour " + swatch(res.room.wall_palette.to_hex()), unsafe_allow_html=True)
    if res.room.decor_palette:
        st.markdown("Decor colours " + swatch(res.room.decor_palette.to_hex()), unsafe_allow_html=True)
    if res.measurement and res.n_size_unknown:
        st.caption(f"{res.n_candidates - res.n_size_unknown} artworks are known to fit the space; "
                   f"{res.n_size_unknown} more have no recorded size, so only their shape was checked.")
    else:
        st.caption(f"{res.n_candidates} artworks fit the space.")
    st.caption("Weights used: " + ", ".join(f"{k} {v:.0%}" for k, v in res.weights_used.items()))

if not res.recommendations:
    st.error(res.message)
    st.stop()

st.subheader("Recommendations")
for r in res.recommendations:
    c1, c2 = st.columns([1, 2])
    p = Path(r.row["image_path"])
    c1.image(str(p if p.is_absolute() else ROOT / p))
    with c2:
        st.markdown(f"**{r.rank}. {r.row['title'] or 'Untitled'}**, {r.row['artist'] or 'unknown artist'}")
        size = (f"{r.row['width_cm']:.0f} × {r.row['height_cm']:.0f} cm (w × h)"
                if np.isfinite(r.row["width_cm"])
                else "size unknown" + (", not checked against your wall" if res.measurement else ""))
        st.write(f"{r.row['style']}, {size}, labelled *{r.row['emotion']}*")
        st.markdown(swatch(r.details["palette_hex"]), unsafe_allow_html=True)
        for k, v in sorted(r.contributions.items(), key=lambda kv: -kv[1]):
            st.write(f"{'▲' if v >= 0 else '▼'} {LABELS.get(k, k)}: {v:+.2f}")
    st.divider()
