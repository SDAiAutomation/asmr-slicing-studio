"""Voix off ElevenLabs (optionnelle) + minutage mot à mot pour les sous-titres.

Sans clé ELEVENLABS_API_KEY, le moteur estime le minutage et rend une vidéo
sans voix (musique seule) : pratique pour tester le visuel gratuitement.
"""
import base64
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import config


def _estimate_words(text: str, duration: float):
    words = text.split()
    weights = [len(w) + 2 for w in words]
    total = sum(weights) or 1
    t, out = 0.0, []
    for w, wt in zip(words, weights):
        d = duration * wt / total
        out.append((w, t, t + d))
        t += d
    return out


def _words_from_alignment(al: dict):
    chars = al["characters"]
    starts = al["character_start_times_seconds"]
    ends = al["character_end_times_seconds"]
    words, cur, ws, we = [], "", None, None
    for c, s, e in zip(chars, starts, ends):
        if c.isspace():
            if cur:
                words.append((cur, ws, we))
            cur, ws = "", None
            continue
        if ws is None:
            ws = s
        cur += c
        we = e
    if cur:
        words.append((cur, ws, we))
    return words


def _audio_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def _elevenlabs(text: str, voice_id: str, dest: Path):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    body = json.dumps({
        "text": text,
        "model_id": config.ELEVENLABS_MODEL,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.8, "style": 0.35},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"ElevenLabs {e.code} pour la voix {voice_id} : {e.read()[:300]!r}") from e
    dest.write_bytes(base64.b64decode(data["audio_base64"]))
    return _words_from_alignment(data["alignment"])


# ---------------------------------------------------------------- Kokoro (gratuit, local)
_KOKORO = {}


def _kokoro(text: str, lang: str, voice: str, dest: Path, speed: float):
    """Synthèse Kokoro-82M (Apache 2.0) sur CPU. Renvoie le minutage estimé."""
    import wave

    import numpy as np
    from kokoro import KPipeline

    code = config.KOKORO_LANG[lang]
    if code not in _KOKORO:
        _KOKORO[code] = KPipeline(lang_code=code)
    chunks = []
    for res in _KOKORO[code](text, voice=voice, speed=speed):
        audio = res[2] if isinstance(res, tuple) else res.audio
        if audio is None:
            continue
        audio = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        chunks.append(audio)
        chunks.append(np.zeros(int(24000 * 0.12), dtype=audio.dtype))
    wav = np.concatenate(chunks)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    dur = len(wav) / 24000
    return _estimate_words(text, dur), dur


def pick_provider(use_tts: bool) -> str:
    if not use_tts:
        return "none"
    want = config.TTS_PROVIDER
    if want in ("elevenlabs", "auto") and config.ELEVENLABS_API_KEY:
        return "elevenlabs"
    if want in ("kokoro", "auto"):
        try:
            import kokoro  # noqa: F401
            return "kokoro"
        except ImportError:
            if want == "kokoro":
                raise RuntimeError("TTS_PROVIDER=kokoro mais le paquet 'kokoro' n'est pas installé")
    return "none"


def voice_scenes(spec: dict, workdir: Path, use_tts: bool = True):
    """Retourne (liste par scène {audio, duration, words}, fournisseur utilisé)."""
    provider = pick_provider(use_tts)
    voice_id = spec.get("voice_id") or config.DEFAULT_VOICES.get(spec["lang"], "")
    k_voice = spec.get("kokoro_voice") or config.KOKORO_VOICES.get(spec["lang"])
    results = []
    for i, sc in enumerate(spec["scenes"]):
        text = sc["voice"]
        if provider == "elevenlabs":
            mp3 = workdir / f"voice_{i:02d}.mp3"
            words = _elevenlabs(text, voice_id, mp3)
            dur = max(config.MIN_SCENE_SEC, _audio_duration(mp3) + 0.35)
            results.append({"audio": mp3, "duration": dur, "words": words})
        elif provider == "kokoro":
            wav = workdir / f"voice_{i:02d}.wav"
            words, d = _kokoro(text, spec["lang"], k_voice, wav, float(spec.get("speed", config.KOKORO_SPEED)))
            results.append({"audio": wav, "duration": max(config.MIN_SCENE_SEC, d + 0.3), "words": words})
        else:
            n = len(text.split())
            dur = max(config.MIN_SCENE_SEC, n / config.WORDS_PER_SEC + 0.6)
            dur = float(sc.get("duration", dur))
            results.append({"audio": None, "duration": dur, "words": _estimate_words(text, dur - 0.4)})
    return results, provider
