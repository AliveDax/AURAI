# Art for your wall

Take a photo of the wall where you want to hang art. The system finds the empty
space, reads the colours and style of the room around it, and recommends real
artworks that suit the space, the room's decor and the mood you want, so the
room feels more comfortable.

The core is a matching and scoring system built on a pretrained CLIP model
(ViT-L/14@336, OpenAI weights, **no fine-tuning**) plus our own explainable
logic for colour, geometry, sizing and ranking.

## How it works

```
room photo ─┬─ wall segmentation (pretrained SegFormer, ADE20K "wall") ─ largest empty rectangle
            │      └─ A4 sheet on the wall → homography → space size in cm ─ size filter
            ├─ wall colours near the space ──┐
            ├─ decor colours (non-wall) ─────┴─ colour harmony (our colour theory, LAB/LCh)
            └─ surroundings (wall greyed out) ─ CLIP ─ room style ─ art prompts ─ style score
description ─ CLIP text → artwork  .............................. content score
colour      ─ CIEDE2000 distance to artwork palette ............. preferred-colour score
mood        ─ CLIP text → artwork  .............................. emotion score
            + comfort prior: negative-emotion artworks pushed down unless the user asks for that mood
```

Each score is z-scored across the candidates for that query, then combined with
weights from `src/artrec/config.py`. The starting weights follow the plan
(style 40%, colour 35%, emotion 25%) plus a size/shape fit term. When an
optional input (description, colour) is missing, its weight is shared out
among the rest. Every recommendation shows how much each score contributed.

| Score | Source | Ours or pretrained |
|---|---|---|
| Content (description) | CLIP text–image similarity | pretrained, used as-is |
| Surroundings style | CLIP zero-shot room style → art-style prompts | pretrained + our prompt design |
| Colour harmony | k-means in LAB, hue templates, neutral-wall rule | **ours** |
| Preferred colour | CIEDE2000 distance weighted by palette share | **ours** |
| Emotion | CLIP text–image similarity | pretrained, used as-is |
| Comfort prior | EmoArt emotion labels (valence) | **ours** |
| Fit | fill ratio (ideal 60–75%) + aspect-ratio match | **ours** |
| Size filter | homography from A4 sheet, or manual entry | **ours** |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip install -e .
pytest                                                  # logic tests, no GPU or downloads needed
```

## Workflow

```bash
# 1. Data (EmoArt is CC BY-NC 4.0; see data/emoart/README.md)
python scripts/download_emoart.py              # downloads and prints the schema
python scripts/build_catalog.py --metadata-only

# 2. Recover physical sizes (EmoArt has none)
python scripts/enrich_dimensions.py --catalog data/cache/metadata_all.parquet

# 3. Build the catalog: palettes + CLIP embeddings (GPU, run once)
python scripts/build_catalog.py --per-style 100 --require-dimensions

# 4. Demo
streamlit run app/streamlit_app.py
```

If `build_catalog.py` can't auto-detect a column, it prints its mapping; fix it
with e.g. `--map emotion=annotation.emotion.domain_emotion`.

## Evaluation

| Level | Script | What it measures | Baseline |
|---|---|---|---|
| Component | `evaluate_emotion.py` | Emotion retrieval (precision/recall/lift@k) and classification (top-1/3, macro recall, valence balanced accuracy) against EmoArt labels | random / always-"calm" / chance |
| Component | `evaluate_color_and_space.py` | Whether top-k results contain the requested colour | rate in a random artwork |
| Component | `evaluate_color_and_space.py` | Empty-space detection (IoU vs hand-drawn box), measurement error vs tape measure | – |
| System | `make_survey.py` → `analyze_survey.py` | Blind human ratings of system picks vs CLIP-only ablation vs random vs mismatched | Mann–Whitney U tests |

Two things to keep in mind when reporting:

- **EmoArt is skewed.** About 56% of labels are "calm", so a model that always says
  "calm" gets 56% accuracy. Always report against the baselines the script prints,
  and use macro recall and lift, not raw accuracy alone.
- **The labels come from GPT-4o** (with human verification). The emotion test
  measures agreement with those labels; the survey is the only purely human judgment.

For the space tests, photograph a few rooms with an A4 sheet taped to the wall,
draw the correct spot by hand and tape-measure it, then fill in
`data/rooms/annotations.csv` (see `annotations_template.csv`).

## Project layout

```
src/artrec/
  config.py        every tunable number (weights, thresholds) in one place
  color.py         palettes, harmony rules, preferred colour, colour parsing
  room.py          wall segmentation, empty-space detection, room regions
  measure.py       A4 detection and homography → centimetres
  scoring.py       normalisation, weighting, size filter, fit scores
  prompts.py       all CLIP prompts and the mood → EmoArt emotion mapping
  clip_model.py    open_clip wrapper
  catalog.py       precomputed catalog (metadata, embeddings, palettes)
  recommender.py   end-to-end pipeline with per-score explanations
scripts/           data preparation, evaluation, survey
app/               Streamlit demo
tests/             unit + end-to-end tests (fake encoder, no downloads)
```

## Known limitations

- CLIP is moderately good at art emotion and weak at fine art-movement distinctions;
  this is documented in prior work. Colour harmony and fit are our own logic,
  partly to compensate.
- Measurements need an A4 sheet on the same wall, or manual entry. Without either,
  the size check is skipped and only the shape of the space is used.
- Physical sizes are known for only about 10% of EmoArt (12,945 of 132,664 artworks,
  matched on Wikidata by artist + title; best for Baroque/Rococo/Romanticism, worst for
  modern and Asian styles). By default, artworks of unknown size are still recommended
  when the space is measured, ranked by shape only and marked "not checked" in the app;
  set `FitConfig.keep_unknown_sizes = False` to recommend only pieces proven to fit.
- The heuristic wall detector (used when the segmentation model is off) only works
  on plain walls.
- Recommendations come from museum collections, so most pieces aren't for sale.
  Prints and purchasable art are future work.
