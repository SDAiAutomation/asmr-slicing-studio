"""Rendu motion design 9:16 : PIL dessine chaque image, ffmpeg encode.

Aucun asset du jeu : fond néon généré, chiffres animés, sous-titres mot à mot.
"""
import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import config
from .spec import fmt_number

W, H, FPS = config.W, config.H, config.FPS
HY = 1250            # ligne d'horizon
CONTENT_Y = (330, 860)


# ---------------------------------------------------------------- utilitaires
@lru_cache(maxsize=64)
def font(weight: str, size: int):
    return ImageFont.truetype(config.font_path(weight), size)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def wrap(text, fnt, max_w):
    text = re.sub(r" ([?!:;%])", "\u00a0\\1", text)
    words, lines, cur = [w for w in text.split(" ") if w], [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if fnt.getlength(test) <= max_w or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_font(text, weight, max_size, max_w, max_lines):
    size = max_size
    while size > 40:
        f = font(weight, size)
        lines = wrap(text, f, max_w)
        if len(lines) <= max_lines and all(f.getlength(l) <= max_w for l in lines):
            return f, lines
        size -= 6
    f = font(weight, 40)
    return f, wrap(text, f, max_w)


def text_block(text, weight, max_size, color, max_w=940, max_lines=3, glow=None, spacing=1.08, stroke=0):
    """Image RGBA d'un bloc de texte centré, avec halo optionnel."""
    f, lines = fit_font(text, weight, max_size, max_w, max_lines)
    asc, desc = f.getmetrics()
    lh = int((asc + desc) * spacing)
    pad = 60
    bw = int(max(f.getlength(l) for l in lines)) + 2 * pad
    bh = lh * len(lines) + 2 * pad
    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, l in enumerate(lines):
        x = (bw - f.getlength(l)) / 2
        d.text((x, pad + i * lh), l, font=f, fill=color + (255,),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    if glow:
        g = Image.new("RGBA", layer.size, glow + (0,))
        g.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(22)))
        base = Image.new("RGBA", layer.size, (0, 0, 0, 0))
        base = Image.alpha_composite(base, g)
        base = Image.alpha_composite(base, g)
        layer = Image.alpha_composite(base, layer)
    return layer


def paste(frame, layer, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.01:
        return
    if abs(scale - 1) > 0.01:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BILINEAR)
    if alpha < 0.99:
        a = layer.getchannel("A").point(lambda v: int(v * alpha))
        layer = layer.copy()
        layer.putalpha(a)
    frame.alpha_composite(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)))


# ---------------------------------------------------------------- décor
def build_background(th):
    arr = np.zeros((H, W, 3), dtype=np.float32)
    y = np.arange(H)[:, None]
    top, mid, hor = (np.array(th[k], dtype=np.float32) for k in ("sky_top", "sky_mid", "sky_horizon"))
    t1 = np.clip(y / (HY * 0.62), 0, 1)
    t2 = np.clip((y - HY * 0.62) / (HY * 0.38), 0, 1)
    sky = np.where(y < HY * 0.62, top + (mid - top) * t1, mid + (hor - mid) * t2)
    arr[:] = sky[:, None, :]
    arr[HY:] = np.array(th["ground"], dtype=np.float32)
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")

    # soleil rayé
    r = 230
    sun = Image.new("RGBA", (2 * r, 2 * r), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sun)
    for i in range(2 * r):
        k = i / (2 * r)
        c = tuple(int(th["sun_top"][j] + (th["sun_bottom"][j] - th["sun_top"][j]) * k) for j in range(3))
        sd.line([(0, i), (2 * r, i)], fill=c + (255,))
    mask = Image.new("L", sun.size, 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([0, 0, 2 * r - 1, 2 * r - 1], fill=255)
    for k in range(6):  # bandes horizontales
        yy = int(r * 0.9) + k * 34
        md.rectangle([0, yy, 2 * r, yy + 5 + k * 3], fill=0)
    sun.putalpha(mask)
    glow = sun.filter(ImageFilter.GaussianBlur(40))
    img.alpha_composite(glow, (W // 2 - r, HY - int(r * 1.45)))
    img.alpha_composite(sun, (W // 2 - r, HY - int(r * 1.45)))
    # masque le soleil sous l'horizon
    ImageDraw.Draw(img).rectangle([0, HY, W, H], fill=th["ground"] + (255,))
    # étoiles
    rng = np.random.default_rng(3)
    d = ImageDraw.Draw(img)
    for _ in range(90):
        x, yy = int(rng.integers(0, W)), int(rng.integers(0, int(HY * 0.55)))
        a = int(rng.integers(60, 200))
        d.point((x, yy), fill=(255, 255, 255, a))
    return img


def draw_grid(frame, th, t):
    d = ImageDraw.Draw(frame)
    col = th["grid"]
    vx = W / 2
    # lignes fuyantes
    for i in range(-14, 15):
        x_bottom = vx + i * 160
        d.line([(vx + i * 8, HY), (x_bottom, H)], fill=col + (150,), width=2)
    # lignes horizontales qui avancent
    phase = (t * 0.6) % 1.0
    for k in range(14):
        z = (k + phase) / 14
        yy = HY + (H - HY) * (z ** 2.2)
        a = int(60 + 160 * z)
        d.line([(0, yy), (W, yy)], fill=col + (a,), width=2 if z < 0.5 else 3)
    d.line([(0, HY), (W, HY)], fill=th["grid"] + (255,), width=4)


# ---------------------------------------------------------------- scènes
class SceneRenderer:
    def __init__(self, sc, th, lang):
        self.sc, self.th, self.lang = sc, th, lang
        t = sc["type"]
        acc, acc2, txt = th["accent"], th["accent2"], th["text"]
        self.layers = {}
        if t in ("hook", "text", "cta"):
            big_color = acc2 if t == "hook" else (acc if t == "cta" else txt)
            self.layers["big"] = text_block(sc.get("big", ""), "Black", 150 if t == "hook" else 120,
                                            big_color, glow=big_color, max_lines=3, max_w=880)
            if sc.get("small"):
                self.layers["small"] = text_block(sc["small"], "Bold", 66, txt, max_lines=3, stroke=3)
        elif t == "stat":
            if sc.get("label"):
                self.layers["small"] = text_block(sc["label"], "Bold", 66, txt, max_lines=3, stroke=3)
            if sc.get("top"):
                self.layers["top"] = text_block(sc["top"], "SemiBold", 52, acc, max_lines=2)
        elif t == "versus":
            for side in ("left", "right"):
                s = sc[side]
                self.layers[side + "_top"] = text_block(s.get("top", ""), "SemiBold", 54, acc, max_w=440, max_lines=2)
                self.layers[side + "_big"] = text_block(s.get("big", ""), "Black", 108, acc2, max_w=380,
                                                        max_lines=2, glow=acc2)
            self.layers["vs"] = text_block("VS", "Black", 90, th["text"], glow=th["grid"])
            if sc.get("small"):
                self.layers["small"] = text_block(sc["small"], "Bold", 60, txt, max_lines=3)

    def draw(self, frame, lt, dur):
        sc, th = self.sc, self.th
        intro = ease_out(lt / 0.45)
        outro = 1 - ease_out((lt - (dur - 0.25)) / 0.25)
        a = min(intro, outro)
        dy = (1 - intro) * 70
        t = sc["type"]
        if t in ("hook", "text", "cta"):
            punch = 1 + 0.18 * (1 - ease_out(lt / 0.3)) if t == "hook" else 1.0
            big = self.layers["big"]
            small = self.layers.get("small")
            total = big.height + (small.height - 40 if small else 0)
            y0 = (CONTENT_Y[0] + CONTENT_Y[1]) / 2 - total / 2
            paste(frame, big, W / 2, y0 + big.height / 2 + dy, a, punch)
            if small:
                a2 = min(ease_out((lt - 0.25) / 0.45), outro)
                paste(frame, small, W / 2, y0 + big.height - 40 + small.height / 2 + (1 - a2) * 60, a2)
            if t == "cta":
                self._arrow(frame, lt, a)
        elif t == "stat":
            k = ease_out(lt / 1.3)
            val = sc["value"] * k
            txt = sc.get("prefix", "") + fmt_number(val, sc.get("decimals", 0), self.lang) + sc.get("suffix", "")
            num = text_block(txt, "Black", sc.get("size", 210), th["accent2"], max_w=1000, max_lines=1, glow=th["accent2"])
            top = self.layers.get("top")
            small = self.layers.get("small")
            cy = (CONTENT_Y[0] + CONTENT_Y[1]) / 2 - 40
            if top:
                paste(frame, top, W / 2, cy - num.height / 2 - top.height / 2 + 70 + dy, a)
            paste(frame, num, W / 2, cy + dy, a)
            if small:
                a2 = min(ease_out((lt - 0.4) / 0.45), outro)
                paste(frame, small, W / 2, cy + num.height / 2 + small.height / 2 - 80 + (1 - a2) * 60, a2)
        elif t == "versus":
            cy = 560
            for i, side in enumerate(("left", "right")):
                ai = min(ease_out((lt - 0.2 * i) / 0.45), outro)
                cx = W * (0.27 if side == "left" else 0.73)
                slide = (1 - ai) * (-120 if side == "left" else 120)
                paste(frame, self.layers[side + "_top"], cx + slide, cy - 110, ai)
                paste(frame, self.layers[side + "_big"], cx + slide, cy + 40, ai)
            av = min(ease_out((lt - 0.4) / 0.3), outro)
            paste(frame, self.layers["vs"], W / 2, cy, av, 1 + 0.3 * (1 - av))
            if self.layers.get("small"):
                a2 = min(ease_out((lt - 0.6) / 0.45), outro)
                paste(frame, self.layers["small"], W / 2, cy + 300, a2)

    def _arrow(self, frame, lt, a):
        d = ImageDraw.Draw(frame)
        bounce = abs(math.sin(lt * 4)) * 30
        cx, cy = W / 2, 1300 + bounce
        col = self.th["accent2"] + (int(255 * a),)
        d.polygon([(cx - 50, cy), (cx + 50, cy), (cx, cy + 60)], fill=col)


# ---------------------------------------------------------------- sous-titres
def chunk_words(words, max_words=4, max_chars=24):
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or len(" ".join(x[0] for x in cur + [w])) > max_chars):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if w[0].endswith((".", "!", "?", ",")) and len(cur) >= 2:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def draw_caption(frame, chunks, lt, th):
    cur = None
    for ch in chunks:
        if ch[0][1] - 0.05 <= lt <= ch[-1][2] + 0.25:
            cur = ch
    if not cur:
        return
    f = font("Black", 76)
    words = [w[0] for w in cur]
    space = f.getlength(" ")
    widths = [f.getlength(w) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    lines = [list(range(len(words)))]
    if total > 960:  # 2 lignes
        half = len(words) // 2 or 1
        lines = [list(range(half)), list(range(half, len(words)))]
    d = ImageDraw.Draw(frame)
    y = 1470
    for line in lines:
        lw = sum(widths[i] for i in line) + space * (len(line) - 1)
        x = (W - lw) / 2
        for i in line:
            active = cur[i][1] <= lt <= cur[i][2] + 0.05
            col = th["accent2"] if active else (255, 255, 255)
            d.text((x, y), words[i], font=f, fill=col + (255,), stroke_width=7, stroke_fill=(0, 0, 0, 255))
            x += widths[i] + space
        y += 96


# ---------------------------------------------------------------- vidéo
def render_video(spec, voices, workdir: Path, music_path: Path, out_path: Path, tts_on: bool):
    th = config.THEMES.get(spec["theme"], config.THEMES["vice"])
    bg = build_background(th)
    durations = [v["duration"] for v in voices]
    starts = np.cumsum([0] + durations[:-1]).tolist()
    total = sum(durations)
    renderers = [SceneRenderer(sc, th, spec["lang"]) for sc in spec["scenes"]]
    chunks = [chunk_words(v["words"]) for v in voices]
    tag = text_block(spec["tag"], "Black", 44, th["accent"], max_w=900, max_lines=1)
    src_font = font("SemiBold", 30)

    silent = workdir / "video_silent.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-pix_fmt", "yuv420p", str(silent)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(total * FPS)
    for fi in range(n):
        t = fi / FPS
        si = max(i for i, s in enumerate(starts) if s <= t + 1e-9)
        lt = t - starts[si]
        frame = bg.copy()
        draw_grid(frame, th, t)
        d = ImageDraw.Draw(frame)
        # barre de progression + badge
        d.rectangle([0, 0, int(W * t / total), 12], fill=th["accent"] + (255,))
        d.rounded_rectangle([W / 2 - tag.width / 2 + 40, 150, W / 2 + tag.width / 2 - 40, 150 + tag.height - 70],
                            radius=40, outline=th["accent"] + (255,), width=4, fill=(0, 0, 0, 140))
        paste(frame, tag, W / 2, 150 + (tag.height - 70) / 2)
        renderers[si].draw(frame, lt, durations[si])
        draw_caption(frame, chunks[si], lt, th)
        if spec.get("sources"):
            s = ("Sources : " if spec["lang"] == "fr" else "Sources: ") + spec["sources"]
            d.text(((W - src_font.getlength(s)) / 2, 1840), s, font=src_font, fill=(255, 255, 255, 170))
        proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg a échoué pendant l'encodage vidéo")

    # ---- audio : voix positionnées + musique
    inputs = ["-i", str(silent), "-i", str(music_path)]
    filters, labels = [], []
    k = 2
    for v, s in zip(voices, starts):
        if v["audio"]:
            inputs += ["-i", str(v["audio"])]
            ms = int(s * 1000)
            filters.append(f"[{k}:a]adelay={ms}|{ms},aresample=44100[v{k}]")
            labels.append(f"[v{k}]")
            k += 1
    music_vol = 0.16 if tts_on else 0.55
    filters.append(f"[1:a]volume={music_vol}[m]")
    if labels:
        filters.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0[vo]")
        filters.append("[vo][m]amix=inputs=2:normalize=0:duration=longest[aout]")
    else:
        filters.append("[m]anull[aout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
           "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{total:.2f}", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    return total
