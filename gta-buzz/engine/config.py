"""Réglages globaux du moteur."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
FONTS_DIR = ROOT / "assets" / "fonts"

# Format vertical Shorts / TikTok / Reels
W, H, FPS = 1080, 1920, 30

# Débit de parole estimé quand il n'y a pas de voix off (mots / seconde)
WORDS_PER_SEC = 2.6
MIN_SCENE_SEC = 2.8

# Clés optionnelles (variables d'environnement)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Voix off : auto = ElevenLabs si clé, sinon Kokoro si installé, sinon sans voix
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto")  # auto | kokoro | elevenlabs | none
KOKORO_LANG = {"fr": "f", "en": "a"}
KOKORO_VOICES = {
    "fr": os.getenv("KOKORO_VOICE_FR", "ff_siwis"),
    "en": os.getenv("KOKORO_VOICE_EN", "am_michael"),
}
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "1.08"))

# Voix par défaut selon la langue (à remplacer par tes voice_id ElevenLabs)
DEFAULT_VOICES = {
    "fr": os.getenv("VOICE_FR", "IHngRooVccHyPqB4uQkG"),  # Corentin (GrowthOS), jeune, dynamique
    "en": os.getenv("VOICE_EN", "nPczCjzI2devNBz1zQrb"),  # Brian, voix premade américaine
}

# Palette « néon Floride » générique (aucun asset Rockstar)
THEMES = {
    "vice": {
        "sky_top": (8, 2, 30),
        "sky_mid": (58, 10, 74),
        "sky_horizon": (255, 46, 136),
        "ground": (10, 3, 28),
        "grid": (255, 60, 170),
        "sun_top": (255, 220, 90),
        "sun_bottom": (255, 46, 136),
        "accent": (0, 240, 255),
        "accent2": (255, 220, 90),
        "text": (255, 255, 255),
    },
    "money": {
        "sky_top": (2, 14, 12),
        "sky_mid": (6, 48, 40),
        "sky_horizon": (40, 220, 140),
        "ground": (2, 12, 10),
        "grid": (60, 230, 150),
        "sun_top": (255, 230, 120),
        "sun_bottom": (40, 220, 140),
        "accent": (120, 255, 190),
        "accent2": (255, 220, 90),
        "text": (255, 255, 255),
    },
}


def font_path(weight: str = "Black") -> str:
    """Trouve une police : assets/fonts d'abord, puis le système."""
    candidates = [
        FONTS_DIR / f"InterDisplay-{weight}.otf",
        Path(f"/usr/share/fonts/opentype/inter/InterDisplay-{weight}.otf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError("Aucune police trouvée : place InterDisplay-*.otf dans assets/fonts/")
