"""All text prompts in one place. Prompt wording is part of the method, so it
should be reported in the writeup and can be tuned like any other parameter."""
from __future__ import annotations

from .config import EMOART_EMOTIONS, NEGATIVE_EMOTIONS

# Room style -> (prompt describing the room, prompts describing art that suits it).
# The surroundings score goes room photo -> room style (zero-shot) -> art prompts,
# instead of comparing a room photo directly to paintings (which mostly finds
# paintings *of* rooms).
ROOM_STYLES: dict[str, tuple[str, list[str]]] = {
    "minimalist": (
        "a photo of a minimalist interior with clean lines and few objects",
        ["a minimalist painting with simple shapes and lots of empty space",
         "a calm abstract painting with a restrained palette"],
    ),
    "scandinavian": (
        "a photo of a scandinavian interior with light wood and soft white tones",
        ["a soft, light-toned painting with natural subjects",
         "a quiet landscape painting in pale colours"],
    ),
    "modern": (
        "a photo of a modern contemporary living room",
        ["a bold modern abstract painting", "a contemporary artwork with strong colour blocks"],
    ),
    "industrial": (
        "a photo of an industrial interior with exposed brick, metal and concrete",
        ["a gritty expressive painting with dark tones", "an urban cityscape painting"],
    ),
    "rustic": (
        "a photo of a rustic interior with wooden beams and warm natural materials",
        ["a warm landscape painting of the countryside", "a traditional still life painting"],
    ),
    "traditional": (
        "a photo of a classic traditional interior with ornate furniture",
        ["a classical oil painting in a traditional style", "a realist portrait or landscape painting"],
    ),
    "bohemian": (
        "a photo of a bohemian interior with plants, patterns and colourful textiles",
        ["a vibrant, colourful, patterned painting", "a lively painting with rich warm colours"],
    ),
    "eclectic colourful": (
        "a photo of a colourful eclectic room with bright mixed decor",
        ["a bright playful painting full of colour", "a pop art style artwork"],
    ),
    "asian-inspired": (
        "a photo of a calm japanese or asian-inspired interior",
        ["a traditional east asian ink painting", "a serene ukiyo-e style landscape"],
    ),
}

# Prompt templates. Several phrasings are averaged ("prompt ensembling"), a
# standard trick that makes CLIP text embeddings more stable.
CONTENT_TEMPLATES = ["a painting of {}", "an artwork showing {}", "{}"]
MOOD_TEMPLATES = ["a painting that feels {}", "an artwork with a {} mood", "a {} painting"]
EMOTION_EVAL_TEMPLATES = ["a painting that makes you feel {}", "an artwork that evokes a feeling of being {}"]

# Free-text mood words -> EmoArt emotion labels. Used (1) to decide whether the
# user explicitly wants a negative mood (then the comfort penalty is switched off)
# and (2) to evaluate free mood words against EmoArt labels.
MOOD_TO_EMOTIONS: dict[str, list[str]] = {
    "calm": ["calm"], "calming": ["calm"], "peaceful": ["calm", "content"],
    "relaxing": ["calm", "content"], "serene": ["calm"], "tranquil": ["calm"],
    "soothing": ["calm"], "cozy": ["content", "calm"], "cosy": ["content", "calm"],
    "comforting": ["content", "calm"], "warm": ["content", "glad"],
    "content": ["content"], "happy": ["happy", "glad"], "cheerful": ["happy", "glad"],
    "joyful": ["happy", "excited"], "uplifting": ["glad", "happy"],
    "energetic": ["excited", "aroused"], "lively": ["excited", "happy"],
    "exciting": ["excited"], "vibrant": ["excited", "happy"], "bold": ["aroused", "excited"],
    "dramatic": ["aroused", "alarmed"], "intense": ["aroused", "alarmed"],
    "melancholic": ["sad"], "melancholy": ["sad"], "sad": ["sad"], "moody": ["sad", "tired"],
    "gloomy": ["sad", "tired"], "dark": ["sad", "alarmed"],
}


def mood_emotions(mood: str | None) -> list[str]:
    if not mood:
        return []
    found: list[str] = []
    for word in mood.lower().replace(",", " ").split():
        for e in MOOD_TO_EMOTIONS.get(word, []):
            if e not in found:
                found.append(e)
        if word in EMOART_EMOTIONS and word not in found:
            found.append(word)
    return found


def mood_is_negative(mood: str | None) -> bool:
    emos = mood_emotions(mood)
    return bool(emos) and all(e in NEGATIVE_EMOTIONS for e in emos)
