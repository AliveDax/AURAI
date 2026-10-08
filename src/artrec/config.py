"""Central configuration.

Every tunable number in the system lives here so it can be reported and
justified in the writeup. Values marked TUNE are starting points, not results.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EMOART_DIR = DATA_DIR / "emoart"
ROOMS_DIR = DATA_DIR / "rooms"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = ROOT / "reports"

# Precomputed catalog artifacts (built by scripts/build_catalog.py)
CATALOG_PARQUET = CACHE_DIR / "catalog.parquet"
EMBEDDINGS_NPY = CACHE_DIR / "image_embeddings.npy"
PALETTES_NPZ = CACHE_DIR / "palettes.npz"

# --- Room photos -------------------------------------------------------------
# Phone photos are 12-48 MP; analysis doesn't need more than this on the long side.
PHOTO_MAX_SIDE = 1600

# --- CLIP -------------------------------------------------------------------
CLIP_MODEL = "ViT-L-14-336"
CLIP_PRETRAINED = "openai"

# --- Wall segmentation (pretrained on ADE20K, which has a "wall" class) -------
SEGMENTATION_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# --- EmoArt label space (from the EmoArt paper, Zhang et al. 2025) -----------
EMOART_EMOTIONS = [
    "aroused", "excited", "happy",          # positive, high arousal
    "alarmed", "annoyed", "frustrated",     # negative, high arousal
    "sad", "bored", "tired",                # negative, low arousal
    "content", "calm", "glad",              # positive, low arousal
]
NEGATIVE_EMOTIONS = {"alarmed", "annoyed", "frustrated", "sad", "bored", "tired"}


@dataclass
class Weights:
    """Score weights. Grouped to mirror the project plan:
    style 40% (content + surroundings), colour 35% (harmony + preferred),
    emotion 25%. `fit` is an extra term for how well the size/shape fits.
    Missing optional inputs have their weight redistributed (see scoring.py).
    TUNE all of these.
    """
    content: float = 0.25        # user description  -> artwork (CLIP text-image)
    surroundings: float = 0.15   # room decor style  -> artwork (CLIP, via text)
    harmony: float = 0.25        # room palette      -> artwork palette (colour theory)
    preferred_color: float = 0.10  # user's colour   -> artwork palette
    emotion: float = 0.25        # mood text         -> artwork (CLIP text-image)
    fit: float = 0.10            # size / shape fit of artwork to the empty space

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ColorConfig:
    k_palette: int = 7            # k-means clusters per artwork (enough to keep accents)
    k_room: int = 6               # clusters for room decor
    k_wall: int = 3               # clusters for the wall region
    max_pixels: int = 20_000      # subsample pixels before k-means for speed
    neutral_chroma: float = 12.0  # LCh chroma below this counts as neutral (white/grey/beige)
    wall_share: float = 0.6       # weight of wall vs decor in the harmony score
    neutral_wall_base: float = 0.7  # neutral walls go with almost anything
    preferred_sigma: float = 15.0   # CIEDE2000 scale for "this is the user's colour"
    presence_delta_e: float = 20.0  # eval: colour counts as present below this distance
    presence_min_share: float = 0.05


@dataclass
class ComfortConfig:
    """The app's goal is a more comfortable room, so by default we favour
    positive-valence art and penalise negative emotions."""
    default_mood: str = "calm, peaceful and comforting"
    negative_penalty: float = 0.75  # subtracted (in z-score units) for negative labels


@dataclass
class FitConfig:
    ideal_fill_low: float = 0.60   # interior-design rule of thumb: art fills
    ideal_fill_high: float = 0.75  # roughly 2/3 to 3/4 of the available width
    min_margin_cm: float = 5.0     # artwork must leave at least this much free space
    aspect_tolerance: float = 0.30  # log-ratio scale for aspect-ratio matching
    # Most EmoArt artworks have no known physical size. True: keep them when the
    # space is measured (ranked by shape only, flagged as unchecked in the app).
    # False: only recommend artworks proven to fit.
    keep_unknown_sizes: bool = True


@dataclass
class Settings:
    weights: Weights = field(default_factory=Weights)
    color: ColorConfig = field(default_factory=ColorConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    top_n: int = 5


SETTINGS = Settings()
