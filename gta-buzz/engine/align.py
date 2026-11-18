"""Calage mot à mot des sous-titres sur la voix réelle (faster-whisper, gratuit, CPU).

Le texte affiché reste celui du script : Whisper ne sert qu'à donner les temps.
Si Whisper n'est pas installé ou échoue, on retombe sur une estimation.
"""
import difflib
import re
import unicodedata

_MODEL = None


def script_tokens(text):
    """Découpe en mots en recollant la ponctuation isolée (« jeudi ? » -> un seul mot)."""
    out = []
    for w in text.split():
        if out and re.fullmatch(r"[?!:;»%$€.,…]+", w):
            out[-1] += " " + w
        else:
            out.append(w)
    return out


def estimate(text, duration):
    words = script_tokens(text)
    weights = [len(w) + 2 for w in words]
    total = sum(weights) or 1
    t, out = 0.0, []
    for w, wt in zip(words, weights):
        d = duration * wt / total
        out.append((w, t, t + d))
        t += d
    return out


def _norm(w):
    w = unicodedata.normalize("NFKD", w.lower())
    return "".join(c for c in w if c.isalnum())


def align(text, wav_path, lang, duration):
    words = script_tokens(text)
    try:
        global _MODEL
        from faster_whisper import WhisperModel
        if _MODEL is None:
            _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = _MODEL.transcribe(str(wav_path), language=lang, word_timestamps=True,
                                    initial_prompt=text, vad_filter=False)
        heard = [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or [])]
    except Exception as e:  # pas de whisper : estimation
        print(f"   (calage estimé : {type(e).__name__})")
        return estimate(text, duration)
    if not heard:
        return estimate(text, duration)

    a = [_norm(w) for w in words]
    b = [_norm(w[0]) for w in heard]
    times = [None] * len(words)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            times[blk.a + k] = (heard[blk.b + k][1], heard[blk.b + k][2])
    # mots non reconnus (ex. « treize » entendu « 13 ») : interpolation entre voisins
    i = 0
    while i < len(words):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and times[j] is None:
            j += 1
        t0 = times[i - 1][1] if i > 0 else 0.0
        t1 = times[j][0] if j < len(words) else duration
        span = words[i:j]
        wts = [len(w) + 2 for w in span]
        tot = sum(wts) or 1
        t = t0
        for k, wt in zip(range(i, j), wts):
            d = (t1 - t0) * wt / tot
            times[k] = (t, t + d)
            t += d
        i = j
    return [(w, s, e) for w, (s, e) in zip(words, times)]
