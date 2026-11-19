"""Emotion-only page: type an emotion, see the paintings CLIP matches to it. Nothing else is scored
(no room photo, style, colour or description), so the human check of CLIP's emotion matching
sees CLIP's emotion score alone. The full system stays in streamlit_app.py.

    .\\venv\\Scripts\\python.exe -m streamlit run app/emotion_app.py      (or double-click start_emotion_app.bat)
"""
import io
import re
import zipfile
from collections import Counter

import pandas as pd
import streamlit as st
from PIL import Image

from artrec.config import load_config
from artrec.recommend import load_resources, recommend_emotion

st.set_page_config(page_title="Emotion search")


@st.cache_resource(show_spinner="Loading CLIP and the artwork catalog...")
def get_system():
    """Load the config, catalog, cached embeddings and CLIP once per server process."""
    cfg = load_config()
    return cfg, load_resources(cfg)


def results_table(recs) -> pd.DataFrame:
    return pd.DataFrame([{"rank": r.rank, "file": f"{r.rank:02d}_{r.artwork_id}.jpg", "artwork_id": r.artwork_id,
                          "title": r.title, "artist": r.artist, "style": r.style,
                          "emoart_label": r.dominant_emotion,
                          "emotion_cosine": round(r.components["emotion"]["raw"], 4),
                          "image_path": r.image_path} for r in recs])


def results_zip(recs, cfg) -> bytes:
    """Each result as its own image file (01_<artwork_id>.jpg, ...) plus results.csv, so the images
    can be uploaded one by one into a Google Form. Files are re-saved as JPEG so every name ends in .jpg."""
    table = results_table(recs)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for r, name in zip(recs, table["file"]):
            img = io.BytesIO()
            Image.open(cfg.root / r.image_path).convert("RGB").save(img, "JPEG", quality=92)
            z.writestr(name, img.getvalue())
        z.writestr("results.csv", table.to_csv(index=False))
    return buf.getvalue()


st.title("Emotion search")
st.write("Type an emotion. Paintings are ranked by CLIP's emotion match only: "
         "no room photo, style, colour or description.")

emotion = st.text_input("Emotion", placeholder="e.g. calm, sad, excited")

if st.button("Search", type="primary"):
    if not emotion.strip():
        st.error("Please type an emotion.")
        st.stop()
    try:
        cfg, res = get_system()
    except (FileNotFoundError, RuntimeError, ValueError) as err:
        st.error(f"The system could not be loaded: {err}")
        st.stop()
    with st.spinner("Searching..."):
        recs = recommend_emotion(emotion, cfg=cfg, resources=res)
        zipped = results_zip(recs, cfg)
    # Kept in the session, so the results stay on screen when the Save button is clicked
    st.session_state["emotion_result"] = (recs, zipped)

if "emotion_result" in st.session_state:
    cfg, _ = get_system()
    recs, zipped = st.session_state["emotion_result"]
    phrase = recs[0].parsed_text.emotion_text
    slug = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_") or "emotion"
    st.download_button(f"Save these {len(recs)} images (zip)", zipped, file_name=f"emotion_{slug}.zip",
                       mime="application/zip", on_click="ignore")

    counts = Counter(r.dominant_emotion for r in recs)
    st.caption(f"Emotion used: '{phrase}'.  EmoArt labels of these {len(recs)}: "
               + ", ".join(f"{label} {n}" for label, n in counts.most_common()))

    n = cfg.emotion_only["columns"]
    for start in range(0, len(recs), n):
        for col, r in zip(st.columns(n), recs[start:start + n]):
            col.image(str(cfg.root / r.image_path))
            col.caption(f"**{r.rank}.** {r.title or 'Untitled'}  \n{r.style}  \n"
                        f"EmoArt: {r.dominant_emotion} · cosine {r.components['emotion']['raw']:.3f}")
