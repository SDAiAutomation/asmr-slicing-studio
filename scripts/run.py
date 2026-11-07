"""One entry point for the whole pipeline.

  python run.py check                          # locate Blender / FFmpeg / FFprobe
  python run.py preview  config.json           # plan + audio + Blender scene + preview stills
  python run.py render   RUN_DIR [--final]     # render PNG frames (resumable)
  python run.py export   RUN_DIR               # MP4 (H.264 + AAC) + ffprobe/decode checks
  python run.py all      config.json [--final] # everything in one go
  python run.py bench    15,50,100,400         # performance table (plan+audio+build+3 frames)

Each preview creates a fresh RUN_DIR under config.output_path, so earlier
videos are never overwritten.
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_config  # noqa: E402


def find_exe(name, env):
    p = os.environ.get(env)
    if p and os.path.exists(p):
        return p
    p = shutil.which(name)
    if p:
        return p
    cands = []
    if platform.system() == "Windows":
        cands += glob.glob(r"C:\Program Files\Blender Foundation\Blender*\blender.exe")
        cands += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Blender*\blender.exe"))
        cands += glob.glob(r"C:\ffmpeg*\bin\%s.exe" % name)
        cands += glob.glob(os.path.expanduser(r"~\ffmpeg*\bin\%s.exe" % name))
    elif platform.system() == "Darwin":
        cands += ["/Applications/Blender.app/Contents/MacOS/Blender"]
    cands = [c for c in cands if os.path.basename(c).lower().startswith(name)]
    return sorted(cands)[-1] if cands else None


def tools():
    t = {"blender": find_exe("blender", "BLENDER_PATH"),
         "ffmpeg": find_exe("ffmpeg", "FFMPEG_PATH"),
         "ffprobe": find_exe("ffprobe", "FFPROBE_PATH")}
    return t


def version(exe, flag="-version"):
    if not exe:
        return None
    try:
        out = subprocess.run([exe, flag], capture_output=True, text=True, timeout=60).stdout
        return out.splitlines()[0] if out else "?"
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def cmd_check():
    t = tools()
    res = {k: {"path": v, "version": version(v, "--version" if k == "blender" else "-version")}
           for k, v in t.items()}
    print(json.dumps(res, indent=1))
    missing = [k for k, v in t.items() if not v]
    if missing:
        print("MISSING: " + ", ".join(missing) +
              " -> set BLENDER_PATH / FFMPEG_PATH / FFPROBE_PATH to the full .exe path")
        return 1
    return 0


def new_run_dir(cfg):
    base = os.path.abspath(cfg["output_path"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{cfg['object_type']}_{cfg['target_piece_count']}p_{cfg['cutting_mode']}"
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=False)
    return d


def run_py(script, *a):
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, script), *a],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f"{script} failed")
    return json.loads(r.stdout.strip().splitlines()[-1]), time.time() - t


def run_blender(args, log_path, blend=None):
    t = tools()["blender"]
    if not t:
        raise SystemExit("Blender not found: set BLENDER_PATH")
    cmd = [t, "-b"] + ([blend] if blend else []) + ["-P", os.path.join(HERE, "blender_build.py"), "--"] + args
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(r.stdout)
    if r.returncode:
        raise SystemExit(f"Blender failed, see {log_path}")
    tag = {"build": "ASMR_REPORT ", "render": "ASMR_RENDER "}
    for line in r.stdout.splitlines():
        for v in tag.values():
            if line.startswith(v):
                return json.loads(line[len(v):]), time.time() - t0
    return {}, time.time() - t0


def contact_sheet(run_dir):
    ff = tools()["ffmpeg"]
    files = sorted(glob.glob(os.path.join(run_dir, "previews", "strokes", "s_*.png")))
    if not ff or not files:
        return None
    cols = min(6, len(files))
    rows = (len(files) + cols - 1) // cols
    out = os.path.join(run_dir, "previews", "05_progression_sheet.png")
    subprocess.run([ff, "-y", "-v", "error", "-framerate", "1", "-i",
                    os.path.join(run_dir, "previews", "strokes", "s_%02d.png"),
                    "-vf", f"tile={cols}x{rows}:padding=6:color=black", "-frames:v", "1", out])
    return out


def cmd_preview(config_path):
    cfg = load_config(config_path)
    rd = new_run_dir(cfg)
    with open(os.path.join(rd, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    perf = {}
    plan_path = os.path.join(rd, "plan.json")
    plan_stats, perf["plan_s"] = run_py("plan_cuts.py", os.path.join(rd, "config.json"), plan_path)
    audio, perf["audio_s"] = run_py("make_audio.py", plan_path, rd)
    rep, perf["blender_build_and_previews_s"] = run_blender(
        ["--plan", plan_path, "--out", rd, "--stage", "build"], os.path.join(rd, "blender.log"))
    sheet = contact_sheet(rd)
    with open(os.path.join(rd, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    impacts = [{"hit": i, "stroke": h["stroke"], "contact_frame": h["contact_frame"],
                "exit_frame": h["exit_frame"], "normal": [round(x, 3) for x in h["plane_normal"]],
                "count_after": h["count_after"]} for i, h in enumerate(plan["hits"])]
    with open(os.path.join(rd, "impact_frames.json"), "w", encoding="utf-8") as f:
        json.dump(impacts, f, indent=1)
    summary = {"run_dir": rd, "plan": plan_stats, "audio": audio, "blender": rep,
               "perf": {k: round(v, 2) for k, v in perf.items()}, "progression_sheet": sheet,
               "strokes": [{"i": s["index"], "direction": s["direction"],
                            "count_after": s["count_after"]} for s in plan["strokes"]]}
    with open(os.path.join(rd, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return rd


def cmd_render(rd, final):
    a = ["--out", rd, "--stage", "render"] + (["--final"] if final else [])
    frames_dir = os.path.join(rd, "frames")
    # resume: drop empty/corrupt PNGs so Blender re-renders them
    for p in glob.glob(os.path.join(frames_dir, "*.png")):
        if os.path.getsize(p) < 1024:
            os.remove(p)
    if final and glob.glob(os.path.join(frames_dir, "*.png")):
        marker = os.path.join(frames_dir, ".final")
        if not os.path.exists(marker):  # preview-res frames must not mix with final ones
            shutil.move(frames_dir, frames_dir + "_preview_" + time.strftime("%H%M%S"))
    os.makedirs(frames_dir, exist_ok=True)
    if final:
        open(os.path.join(frames_dir, ".final"), "w").close()
    rep, secs = run_blender(a, os.path.join(rd, "blender.log"), os.path.join(rd, "scene.blend"))
    n = len(glob.glob(os.path.join(frames_dir, "*.png")))
    out = {"frames_on_disk": n, "render_s": round(secs, 1), **rep}
    print(json.dumps(out))
    return out


def cmd_export(rd):
    t = tools()
    with open(os.path.join(rd, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    fps, total = plan["fps"], plan["total_frames"]
    frames = sorted(glob.glob(os.path.join(rd, "frames", "f_*.png")))
    if len(frames) < total:
        raise SystemExit(f"only {len(frames)}/{total} frames rendered; run `render` again to resume")
    name = os.path.basename(rd) + ".mp4"
    mp4 = os.path.join(rd, name)
    subprocess.run([t["ffmpeg"], "-y", "-v", "error", "-framerate", str(fps), "-start_number", "1",
                    "-i", os.path.join(rd, "frames", "f_%05d.png"),
                    "-i", os.path.join(rd, "soundtrack.wav"),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-movflags", "+faststart", "-shortest", mp4], check=True)
    probe = json.loads(subprocess.run(
        [t["ffprobe"], "-v", "error", "-show_streams", "-show_format", "-of", "json", mp4],
        capture_output=True, text=True).stdout)
    v = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    au = [s for s in probe["streams"] if s["codec_type"] == "audio"]
    dec = subprocess.run([t["ffmpeg"], "-v", "error", "-i", mp4, "-f", "null", "-"],
                         capture_output=True, text=True)
    checks = {
        "mp4": mp4,
        "size_mb": round(os.path.getsize(mp4) / 1e6, 2),
        "width": v["width"], "height": v["height"], "codec": v["codec_name"],
        "pix_fmt": v["pix_fmt"], "frames": int(v.get("nb_frames", 0)),
        "duration_s": float(probe["format"]["duration"]),
        "audio_stream": bool(au), "audio_codec": au[0]["codec_name"] if au else None,
        "decode_errors": dec.stderr.strip()[:500] or None,
    }
    checks["ok"] = (checks["audio_stream"] and not checks["decode_errors"]
                    and abs(checks["frames"] - total) <= 1 and v["height"] > v["width"])
    with open(os.path.join(rd, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks, indent=1))
    return checks


def cmd_bench(counts):
    rows = []
    for n in counts:
        cfg_path = os.path.join(HERE, f"_bench_{n}.json")
        with open(cfg_path, "w") as f:
            json.dump({"target_piece_count": n, "output_path": "bench_runs"}, f)
        t = time.time()
        rd = cmd_preview(cfg_path)
        rep = json.load(open(os.path.join(rd, "blender_report.json")))
        # render 3 frames at preview size to estimate per-frame cost
        r = cmd_render_sample(rd)
        rows.append({"target": n, "validated": rep["final_pieces_validated"],
                     "failures": len(rep["failures"]), "non_manifold": rep["non_manifold"],
                     "objects": rep["objects"], "pipeline_s": round(time.time() - t, 1),
                     "sec_per_preview_frame": r, "blend_mb": round(
                         os.path.getsize(os.path.join(rd, "scene.blend")) / 1e6, 1)})
        os.remove(cfg_path)
    print(json.dumps(rows, indent=1))


def cmd_render_sample(rd):
    with open(os.path.join(rd, "plan.json"), encoding="utf-8") as f:
        total = json.load(f)["total_frames"]
    mid = total // 2
    _, secs = run_blender(["--out", rd, "--stage", "render", "--frames", f"{mid}-{mid + 2}"],
                          os.path.join(rd, "blender.log"), os.path.join(rd, "scene.blend"))
    return round(secs / 3, 2)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    c = sys.argv[1]
    final = "--final" in sys.argv
    if c == "check":
        return cmd_check()
    if c == "preview":
        cmd_preview(sys.argv[2])
    elif c == "render":
        cmd_render(sys.argv[2], final)
    elif c == "export":
        cmd_export(sys.argv[2])
    elif c == "all":
        rd = cmd_preview(sys.argv[2])
        cmd_render(rd, final)
        cmd_export(rd)
    elif c == "bench":
        cmd_bench([int(x) for x in sys.argv[2].split(",")])
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
