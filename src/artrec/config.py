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
    """Score weights. The plan started at style 40% (content + surroundings), colour 35%
    (harmony + preferred) and emotion 25%. In use, the mood the user picked was outvoted
    by the room's style: the same pictures came up for 'cozy' and 'melancholic'. Mood is
    now the largest term and the zero-shot room style the smallest:
    style 28%, colour 30%, emotion 35%, fit 7%.
    Missing optional inputs have their weight redistributed (preferred colour -> harmony,
    see recommender.py; the rest proportionally, see scoring.py). TUNE all of these.
    """
    content: float = 0.22        # user description  -> artwork (CLIP text-image, also filters)
    surroundings: float = 0.06   # room decor style  -> artwork (CLIP, via text)
    harmony: float = 0.21        # room palette      -> artwork palette (colour theory)
    preferred_color: float = 0.09  # user's colour   -> artwork palette
    emotion: float = 0.35        # mood -> CLIP phrases + EmoArt labels + colour psychology
    fit: float = 0.07            # size / shape fit of artwork to the empty space

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ColorConfig:
    k_palette: int = 7            # k-means clusters per artwork (enough to keep accents)
    k_room: int = 8               # clusters for room decor (enough to keep small accents)
    k_wall: int = 3               # clusters for the wall region
    max_pixels: int = 20_000      # subsample pixels before k-means for speed
    neutral_chroma: float = 12.0  # LCh chroma below this counts as neutral (white/grey/beige)
    neutral_wall_base: float = 0.7  # palette_harmony: a neutral colour goes with almost anything
    # Room harmony = echo of the room's accent colours + hue harmony with them + contrast with the wall
    accent_min_chroma: float = 20.0  # LCh chroma from which a room colour counts as an accent
    accent_share_power: float = 0.15  # accent weight = share**power * chroma: mostly how vivid a
                                      # colour is, so cushions count more than a large dull floor
    echo_sigma: float = 14.0        # CIEDE2000 scale for "the artwork contains this room colour"
    echo_full_share: float = 0.25   # this much of the artwork in an accent colour = full echo
    harmony_weights: tuple[float, float, float] = (0.45, 0.35, 0.20)  # echo, hue, contrast
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
class MoodConfig:
    """How the mood score is built. CLIP alone is weak at emotion, so it is blended with
    EmoArt's (human-verified) emotion labels and with colour psychology."""
    clip: float = 0.4     # CLIP: artwork vs descriptive phrases for the mood
    label: float = 0.3    # EmoArt label is one of the mood's emotions (or same valence/arousal)
    colour: float = 0.3   # warmth, lightness and colourfulness that suit the mood


@dataclass
class SubjectConfig:
    """When the user names a subject ('a dog'), only artworks that clearly show it are
    candidates; the other scores then rank within them."""
    z_alone: float = 2.5         # CLIP match this far above average counts on its own
    z_with_keyword: float = 0.5  # ...or this far, if the EmoArt description names the subject
    min_pool: int = 20           # if fewer match, use this many best matches anyway


@dataclass
class DiversityConfig:
    """Re-ranking so the top picks aren't five near-identical works (maximal marginal
    relevance on CLIP image embeddings, plus a cap per artist)."""
    penalty: float = 2.0         # z-score units subtracted per unit of similarity above `floor`
    floor: float = 0.6           # CLIP image-image cosine below which two works count as unrelated
    max_per_artist: int = 1
    diversify_first: int = 20    # only the first picks are re-ranked; the rest stay in score order


@dataclass
class Settings:
    weights: Weights = field(default_factory=Weights)
    color: ColorConfig = field(default_factory=ColorConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    mood: MoodConfig = field(default_factory=MoodConfig)
    subject: SubjectConfig = field(default_factory=SubjectConfig)
    top_n: int = 5


SETTINGS = Settings()
