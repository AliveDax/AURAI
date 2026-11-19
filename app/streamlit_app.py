"""The whole UI: one basic Streamlit page that calls recommend() and shows the result.

    .\\venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py      (or double-click start_app.bat)
"""
import base64
import io
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from artrec.clip_embed import load_image
from artrec.config import load_config
from artrec.recommend import UserInput, load_resources, recommend
from artrec.result_sheet import result_sheet
from artrec.wall_preview import CM_PER_INCH, load_artwork_sizes, open_room, place_artwork, px_per_inch

st.set_page_config(page_title="Art recommender")

# An image that reports where it was clicked, as fractions of its width and height
_wall_click = st.components.v2.component(
    "wall_click",
    css="img { width: 100%; display: block; cursor: crosshair; border-radius: 0.5rem; }",
    js="""
export default function(component) {
    const { data, setStateValue, parentElement } = component;
    let img = parentElement.querySelector("img");
    if (!img) {
        img = document.createElement("img");
        parentElement.appendChild(img);
    }
    img.src = data.src;
    img.onclick = (e) => {
        const r = img.getBoundingClientRect();
        setStateValue("point", { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height });
    };
}
""")


@st.cache_resource(show_spinner="Loading CLIP and the artwork catalog...")
def get_system():
    """Load the config, catalog, cached embeddings/palettes and CLIP once per server process."""
    cfg = load_config()
    return cfg, load_resources(cfg)


@st.cache_data(max_entries=4, show_spinner=False)
def get_room(data: bytes, w: dict):
    """A room photo upright and resized, with its EXIF focal length (decoded once, not on every click)."""
    return open_room(io.BytesIO(data), w)


@st.dialog("See it on your wall", width="large")
def wall_preview(r):
    """Hang recommendation r on the user's room photo at its real size, where they click."""
    cfg, _ = get_system()
    w = cfg.wall_preview
    rooms = st.session_state.get("rooms")
    if not rooms:
        st.error("Upload your room photo and click Recommend again.")
        return
    keep = {"persist_state": "page"}  # inputs keep their values when the dialog is closed and reopened
    st.write(f"**{r.title or 'Untitled'}** by {r.artist or 'unknown'}")
    i = (st.radio("Room photo", range(len(rooms)), format_func=lambda k: f"Photo {k + 1}", horizontal=True,
                  key="wp_room", **keep) if len(rooms) > 1 else 0)
    room = get_room(rooms[i], w)
    art = Image.open(cfg.root / r.image_path)

    # Real size: from the optional sizes file, else typed in
    known = load_artwork_sizes(cfg).get(r.image_path)
    if known:
        size_in = (known[0] / CM_PER_INCH, known[1] / CM_PER_INCH)
        st.caption(f"Real size: {known[0]:g} x {known[1]:g} cm")
    else:
        st.caption("This artwork's real size is not known. Enter it:")
        lock = st.checkbox("Keep the artwork's proportions", value=True, key="wp_lock", **keep)
        c1, c2, c3 = st.columns(3)
        unit = c1.radio("Unit", ["cm", "in"], horizontal=True, key="wp_unit", **keep)
        per_in = CM_PER_INCH if unit == "cm" else 1.0
        top = round(w["max_artwork_cm"] / CM_PER_INCH * per_in, 1)
        width = c2.number_input(f"Width ({unit})", min_value=1.0, max_value=top, step=1.0,
                                value=round(w["default_artwork_width_cm"] / CM_PER_INCH * per_in, 1),
                                key=f"wp_w_{r.artwork_id}_{unit}", **keep)
        tall = round(width * art.height / art.width, 1)
        if lock:
            height = tall
            c3.number_input(f"Height ({unit})", value=tall, disabled=True)  # display only
        else:
            height = c3.number_input(f"Height ({unit})", min_value=1.0, max_value=top, step=1.0,
                                     value=min(max(tall, 1.0), top), key=f"wp_h_{r.artwork_id}_{unit}", **keep)
        size_in = (width / per_in, height / per_in)

    dist_ft = st.number_input("How far from the wall was the photo taken? (ft)", min_value=2.0, max_value=30.0,
                              value=float(w["camera_distance_ft"]), step=0.5, key="wp_dist", **keep)
    ppi = px_per_inch(room.image.size, room.focal_35mm, dist_ft * 12)
    st.caption(f"Scale: {ppi:.1f} px per inch of wall. Lens: {room.focal_source}. "
               "This assumes the photo was taken straight at the wall and not cropped.")

    # The click: from this run's component state, else the spot used last time for this photo
    key = f"wp_click_{i}"
    saved = st.session_state.setdefault("wp_points", {})
    point = (st.session_state.get(key) or {}).get("point") or saved.get(i)
    shown, light = room.image, None
    if point:
        saved[i] = point
        if size_in[0] * ppi > 2 * room.image.width or size_in[1] * ppi > 2 * room.image.height:
            st.warning("At this scale the artwork is far bigger than the whole photo. Check its size and the distance.")
        else:
            shown, light = place_artwork(room.image, art, (point["x"] * room.image.width, point["y"] * room.image.height),
                                         size_in, ppi, w)
    st.write("Click anywhere on the wall to move it." if light else
             "Click on the wall where the centre of the artwork should go.")
    buf = io.BytesIO()
    shown.save(buf, "JPEG", quality=90)
    _wall_click(key=key, data={"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()},
                on_point_change=lambda: None)
    if light:
        st.caption(f"Shadow: {light.describe()}.")
        st.download_button("Save this preview", buf.getvalue(), file_name=f"wall_preview_{r.artwork_id}.jpg",
                           mime="image/jpeg", on_click="ignore")


st.title("Art recommender")
st.write("Upload photos of your room. Optionally describe the mood or subject you want, "
         "and pick a colour.")

photos = st.file_uploader("Room photo(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
mood = st.text_input("Mood or description (optional)", placeholder="e.g. calming forest, joyful, melancholic sea")
use_color = st.checkbox("I want a specific colour")
color = st.color_picker("Colour", "#2a6f97") if use_color else None

if st.button("Recommend", type="primary"):
    if not photos:
        st.error("Please upload at least one room photo.")
        st.stop()
    try:
        cfg, res = get_system()
    except (FileNotFoundError, RuntimeError, ValueError) as err:
        st.error(f"The system could not be loaded: {err}")
        st.stop()
    with st.spinner("Analysing your room..."):
        recs = recommend(UserInput(photos, mood or None, color), cfg=cfg, resources=res)
        # The image behind the Save button: room photo(s) + the top recommendations, nothing else
        s = cfg.save_sheet
        sheet = result_sheet([load_image(p, s["width_px"]) for p in photos],
                             [Image.open(cfg.root / r.image_path) for r in recs[: s["n_artworks"]]], s)
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=92)
    # Kept in the session, so the results stay on screen when the Save button is clicked
    st.session_state["result"] = (recs, buf.getvalue(), datetime.now().strftime("%Y%m%d_%H%M"))
    # The photos these results are for, for "See it on my wall" (the uploader may change later)
    st.session_state["rooms"] = [p.getvalue() for p in photos]
    st.session_state["wp_points"] = {}

if "result" in st.session_state:
    cfg, _ = get_system()
    recs, sheet_bytes, stamp = st.session_state["result"]
    st.download_button(f"Save room + top {min(len(recs), cfg.save_sheet['n_artworks'])} as one image",
                       sheet_bytes, file_name=f"art_recommendations_{stamp}.jpg", mime="image/jpeg",
                       on_click="ignore")

    p = recs[0].parsed_text
    st.caption(f"Text understood as: emotion = {p.emotion_text or '-'}, description = {p.description_text or '-'}, "
               f"colours = {', '.join(p.colors) or '-'}")

    for r in recs:
        st.subheader(f"{r.rank}. {r.title or 'Untitled'}")
        left, right = st.columns([1, 1])
        left.image(str(cfg.root / r.image_path))
        right.write(f"**Artist:** {r.artist or 'unknown'}  \n**Style:** {r.style}  \n"
                    f"**EmoArt label:** {r.dominant_emotion}  \n**Final score:** {r.final_score:.3f}")
        if r.subject_match is not None:
            right.write(f"**Shows '{p.description_text}':** {'yes' if r.subject_match else 'no'}")
        if r.color_match is not None:
            exact = r.color_match >= cfg.color["color_match_min_share"]
            right.write(f"**Your colour:** {r.color_match:.0%} of the artwork"
                        + (f" (picked for your colour{'' if exact else ', closest shade'})" if r.picked_for_color else ""))
        table = pd.DataFrame(r.components).T[["raw", "norm", "weight", "share"]]
        table["share"] = (table["share"] * 100).round(0).astype(int).astype(str) + "%"
        right.dataframe(table)
        if right.button("See it on my wall", key=f"wall_{r.rank}"):
            wall_preview(r)
