"""Download CC0 photo-scanned assets from Poly Haven (https://polyhaven.com, CC0).

  - a wood texture set for the cutting board (colour, roughness, normal map)
  - a studio HDRI for realistic reflections and soft light

Files land in assets/cache/. Everything is optional: if the network or an asset is
unavailable, Blender falls back to the procedural board and lights.

Usage: python fetch_assets.py [out_dir]
"""
import json
import os
import sys
import urllib.request

WOOD = ["fine_grained_wood", "oak_veneer_01", "wood_table_worn"]
HDRI = ["studio_small_09", "studio_small_08", "brown_photostudio_02", "photo_studio_loft_hall"]
UA = {"User-Agent": "asmr-slicing-studio/1.0"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        return True
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return os.path.getsize(path) > 1024


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "assets", "cache")
    os.makedirs(out, exist_ok=True)
    got = {}
    for name in WOOD:
        try:
            files = get_json(f"https://api.polyhaven.com/files/{name}")
            maps = {"diff": "Diffuse", "rough": "Rough", "nor": "nor_gl"}
            ok = True
            for key, mk in maps.items():
                entry = files.get(mk) or files.get(mk.lower()) or files.get(mk.capitalize())
                url = entry["2k"]["jpg"]["url"]
                ok &= download(url, os.path.join(out, f"wood_{key}.jpg"))
            if ok:
                got["wood"] = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"wood {name}: {e}")
    for name in HDRI:
        try:
            files = get_json(f"https://api.polyhaven.com/files/{name}")
            url = files["hdri"]["1k"]["hdr"]["url"]
            if download(url, os.path.join(out, "studio.hdr")):
                got["hdri"] = name
                break
        except Exception as e:  # noqa: BLE001
            print(f"hdri {name}: {e}")
    with open(os.path.join(out, "assets.json"), "w") as f:
        json.dump(got, f)
    print(json.dumps(got))


if __name__ == "__main__":
    main()
