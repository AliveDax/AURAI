# Mood & colour aware art recommendation

Upload photos of a room, optionally type a mood or subject ("calming blue forest") and pick a
colour. The system ranks the 5,532 real artworks of EmoArt-5k and explains every pick
(e.g. "colour 52%, emotion 31%, style 17%").

No model is trained and CLIP is not fine-tuned. A frozen pretrained CLIP (`open_clip`,
ViT-L-14-336, OpenAI weights) only turns images and text into embeddings. Everything else is our
own code: text parsing, colour harmony rules, score normalization, fusion and validation.

| Component | How | Default weight |
|---|---|---|
| style | cosine(room photo embedding, artwork embedding), image to image | 0.20 |
| color | colour-harmony rules between the room palette and the artwork palette | 0.25 |
| emotion | cosine(emotion words from your text, artwork) | 0.20 |
| description | cosine(subject words from your text, artwork) | 0.35 |

The first design used 0.30 / 0.35 / 0.20 / 0.15. The description weight was raised
because most users give a description and CLIP matches subjects well.

Components without input are dropped and the weights renormalized. All weights and thresholds are
placeholders in `config.yaml` until the human survey is analysed.

**If you give a description** ("sea", "calm forest"), content has priority. Paintings that clearly
show it come first, ordered by the normal score; then everything else. So "sea" gives sea paintings,
not abstracts, unless you ask for "abstract".

**If you ask for a colour** (picker, or a colour word like "teal"), ranking happens in two steps
(`color_reference_mode: two_step`):
1. The first 5 picks must contain that colour or a shade of it (navy and sky blue count as blue).
   They are ordered half by how much of the colour they contain and half by the normal score, so
   your description and mood still decide between them.
2. Everything after that is the normal ranking.

**Colour + description** ("red sea"): the description has priority.
- Only paintings that clearly show the subject can be colour picks. A painting "clearly shows" the
  subject when CLIP's match to the description, minus its match to a generic "a painting", is well
  above the catalog average.
- If no sea painting is red, the colour slots go to sea paintings in the closest shade (orange,
  maroon), never to red paintings of something else.
- The rest of the list puts the other sea paintings first.

Without a colour, only the normal ranking is used, with harmony between each artwork and the room's colours.

## Setup (Windows, PowerShell)

```powershell
python -m venv venv                       # only if venv\ is missing
.\venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip install -e .
.\venv\Scripts\python.exe -m pytest -q    # fast, no GPU needed
```

Data: EmoArt-5k images + `annotation.json` in `dataset/` (extract `Images.tar.gz` with
`python extract_emoart.py`). Put 10-15 varied room photos in `data/rooms/`.

## Pipeline

```powershell
.\venv\Scripts\python.exe scripts/00_check_env.py         # environment, GPU, packages, data
.\venv\Scripts\python.exe scripts/01_inspect_dataset.py   # real JSON keys and emotion labels
.\venv\Scripts\python.exe scripts/02_build_catalog.py     # -> data/catalog.csv
.\venv\Scripts\python.exe scripts/03_benchmark_clip.py    # speed + VRAM
.\venv\Scripts\python.exe scripts/04_embed_artworks.py    # -> embeddings/ (once, ~6 min, resumable)
.\venv\Scripts\python.exe scripts/05_extract_palettes.py  # -> embeddings/ (once, ~10 min)
.\venv\Scripts\python.exe scripts/06_validate_emotion.py  # E1-E3 -> results/
.\venv\Scripts\python.exe scripts/07_make_ablation_outputs.py
.\venv\Scripts\python.exe scripts/08_make_survey.py
.\venv\Scripts\python.exe scripts/09_analyze_survey.py results/survey/responses.csv
.\venv\Scripts\python.exe scripts/10_fit_weights.py results/survey/responses.csv   # optional
```

## Use it

Easiest: double-click **`start_app.bat`** in the project folder. The web page opens in your browser
by itself; close the black window to stop it.

The page shows the top 15 recommendations. The **Save** button downloads one image: your room
photo on top and the top 10 artworks below, in order, on a white page.

**See it on my wall** (button under each recommendation) hangs that artwork on your room photo at
its real size. Click on the wall where its centre should go; click again to move it.
- Scale: the photo is assumed to be taken straight at the wall from 6 ft (editable). The 35 mm-equivalent
  focal length from the photo's EXIF turns that into pixels per inch. Without EXIF (screenshots, photos
  sent through WhatsApp) a typical phone camera (26 mm) is assumed.
- Size: EmoArt has no real sizes, so you type width and height (cm or inches; height follows the
  artwork's proportions unless you untick that). Known sizes can be added in `data/artwork_sizes.csv`
  (`image_path,width_cm,height_cm`), and those artworks are then placed without asking.
- Shadow: the brightness slope of the wall around the spot (furniture left out) gives the light
  direction. The shadow falls away from the light, straight down under even light.
- This is display only (`src/artrec/wall_preview.py`, `wall_preview:` in `config.yaml`). It does not
  change the recommendations or their scores.

Or from PowerShell:

```powershell
.\venv\Scripts\python.exe main.py --room data/rooms/synthetic_living_room.png --mood "calming forest" --top 5
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py   # basic web page in the browser
```

### Phone app

The same recommender as a mobile web app (`web/`), served by a small FastAPI server
(`src/artrec/api.py`). The server only turns requests into `recommend()` calls, so the ranking is
identical to the Streamlit page and the CLI. On the phone: take or choose room photos, tap a mood,
type a subject, pick a colour, then get the ranked picks with each score's breakdown, "See it on
my wall" (tap the wall; `wall_preview.py`) and Save (`result_sheet.py`).

```bash
.venv/bin/uvicorn artrec.api:app --host 0.0.0.0 --port 8000
```

Open `http://<this computer's IP>:8000` on a phone on the same Wi-Fi (it can be added to the home
screen). On a Mac, CLIP runs on the Apple-silicon GPU (MPS) when there is no CUDA GPU; setup is the
same with `.venv/bin/python` in place of `.\venv\Scripts\python.exe`.

### Emotion only (for the human check of CLIP's emotion matching)

Double-click **`start_emotion_app.bat`**, type an emotion ("calm", "sad") and click Search. The
page ranks all paintings by CLIP's emotion match alone: no room photo, style, colour or
description. If the text also has colour or subject words, only the emotion words are used. The
top 20 (`emotion_only.top_n` in `config.yaml`) are shown with their EmoArt label and cosine. The
Save button downloads them as separate images plus `results.csv`, ready for a Google Form.

```powershell
.\venv\Scripts\python.exe main.py --emotion-only --mood calm --top 20
```

## Layout

```
config.yaml          every tunable number
main.py              CLI
app/streamlit_app.py the whole UI (one basic page)
app/emotion_app.py   emotion-only search, for the human emotion check
web/                 the phone app (HTML/CSS/JS, installable), served by src/artrec/api.py
src/artrec/          config, data_loader, clip_embed, color_extract, color_harmony,
                     text_parse, scorer, recommend, wall_preview (artwork on the room photo),
                     api (HTTP server for the phone app)
data/                catalog.csv, rooms/, color_words.yaml, emotion_lexicon.yaml, emotion_paraphrases.yaml
dataset/             EmoArt-5k (not in git, never modified)
embeddings/          cached CLIP embeddings + palettes (not in git)
results/             all metrics, tables and figures
```

## Limitations

- EmoArt's emotion labels come from GPT-4o (with verification), not human raters, and are skewed
  (52% "Calm", only 2 "Glad"). Always compare against the baselines that 06 prints.
- The style score measures the overall visual "vibe" of the room photo against each artwork. It
  is not object detection or art-style classification.
- Artist and title are recovered from file names (the annotations have neither), so some words stay glued.
- Artwork size is deliberately not part of the ranking. It is only used by "See it on my wall".
- "See it on my wall" assumes a straight-on, uncropped photo; a tilted photo or a wrong distance
  makes the artwork look too big or too small.
