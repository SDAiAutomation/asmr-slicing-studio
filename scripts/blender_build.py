"""Blender entry point of the ASMR slicing studio (Blender 4.2+ / 5.x, background mode).

Shared scene pieces (board, lights, knife, materials, counters, render settings).
The grid scene itself ("N CUTS -> X CUBES") is built by blender_grid.py.

Usage:
  blender -b -P blender_build.py -- --plan plan.json --out RUN_DIR --stage build
  blender -b RUN_DIR/scene.blend -P blender_build.py -- --out RUN_DIR --stage render [--final]
"""
import json
import math
import os
import sys
import time

import bpy
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"plan": None, "out": None, "stage": "build", "final": False, "frames": None}
    i = 0
    while i < len(a):
        k = a[i]
        if k == "--final":
            out["final"] = True
            i += 1
            continue
        out[k.lstrip("-")] = a[i + 1]
        i += 2
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_input(node, names, value):
    for n in names if isinstance(names, (list, tuple)) else [names]:
        if n in node.inputs:
            try:
                node.inputs[n].default_value = value
                return True
            except Exception:
                pass
    return False


def mix_rgb(nt, fac=0.5):
    m = nt.nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    ins = [s for s in m.inputs if s.enabled]
    f = [s for s in ins if s.type == "VALUE"][0]
    a, b = [s for s in ins if s.type == "RGBA"][:2]
    out = [s for s in m.outputs if s.enabled and s.type == "RGBA"][0]
    f.default_value = fac
    return m, f, a, b, out


def ramp(nt, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    while len(els) > len(stops):
        els.remove(els[-1])
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position = pos
        el.color = (*col, 1.0) if len(col) == 3 else col
    return r


def principled(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    return mat, nt, bsdf


def make_skin(M, shape):
    mat, nt, bsdf = principled("Skin")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])
    # stem end (+Y) greener, tip yellower
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -shape.b
    mr.inputs["From Max"].default_value = shape.b
    L.new(sep.outputs["Y"], mr.inputs["Value"])
    grad = ramp(nt, [(0.0, M["skin_base"]), (0.55, M["skin_base"]), (1.0, M["skin_stem"])])
    L.new(mr.outputs["Result"], grad.inputs["Fac"])
    # blush: large soft noise patches biased to the +X cheek
    noise = nt.nodes.new("ShaderNodeTexNoise")
    set_input(noise, "Scale", 9.0)
    set_input(noise, "Detail", 3.0)
    L.new(tc.outputs["Object"], noise.inputs["Vector"])
    side = nt.nodes.new("ShaderNodeMath")
    side.operation = "MULTIPLY_ADD"
    side.inputs[1].default_value = 7.0
    side.inputs[2].default_value = 0.0
    L.new(sep.outputs["X"], side.inputs[0])
    add = nt.nodes.new("ShaderNodeMath")
    add.operation = "ADD"
    L.new(noise.outputs["Fac"], add.inputs[0])
    L.new(side.outputs[0], add.inputs[1])
    bl = ramp(nt, [(0.55, (0, 0, 0)), (0.95, (1, 1, 1))])
    L.new(add.outputs[0], bl.inputs["Fac"])
    m1, f1, a1, b1, o1 = mix_rgb(nt)
    L.new(bl.outputs["Color"], f1)
    L.new(grad.outputs["Color"], a1)
    b1.default_value = (*M["skin_blush"], 1)
    # lenticels: tiny pale dots
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    set_input(vor, "Scale", 260.0)
    L.new(tc.outputs["Object"], vor.inputs["Vector"])
    dots = ramp(nt, [(0.0, (1, 1, 1)), (0.12, (0, 0, 0))])
    L.new(vor.outputs["Distance"], dots.inputs["Fac"])
    m2, f2, a2, b2, o2 = mix_rgb(nt)
    dm = nt.nodes.new("ShaderNodeMath")
    dm.operation = "MULTIPLY"
    dm.inputs[1].default_value = 0.35
    L.new(dots.outputs["Color"], dm.inputs[0])
    L.new(dm.outputs[0], f2)
    L.new(o1, a2)
    b2.default_value = (0.85, 0.80, 0.35, 1)
    L.new(o2, bsdf.inputs["Base Color"])
    set_input(bsdf, "Roughness", M["skin_roughness"])
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.15)
    set_input(bsdf, ["Specular IOR Level", "Specular"], 0.45)
    # fine skin relief
    bn = nt.nodes.new("ShaderNodeTexNoise")
    set_input(bn, "Scale", 140.0)
    L.new(tc.outputs["Object"], bn.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    set_input(bump, "Strength", 0.06)
    L.new(bn.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_flesh(M):
    mat, nt, bsdf = principled("Flesh")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    base = (*M["flesh"], 1)
    if M["fibrous"]:
        # fibres run along the long axis: wave bands in X/Z, distorted
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X"
        set_input(wave, "Scale", 60.0)
        set_input(wave, "Distortion", 6.0)
        set_input(wave, "Detail", 4.0)
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(90))
        L.new(tc.outputs["Object"], mp.inputs["Vector"])
        L.new(mp.outputs["Vector"], wave.inputs["Vector"])
        r = ramp(nt, [(0.2, M["flesh"]), (0.9, M["flesh_deep"])])
        L.new(wave.outputs["Fac"], r.inputs["Fac"])
        L.new(r.outputs["Color"], bsdf.inputs["Base Color"])
        bump = nt.nodes.new("ShaderNodeBump")
        set_input(bump, "Strength", 0.18)
        L.new(wave.outputs["Fac"], bump.inputs["Height"])
        L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        bsdf.inputs["Base Color"].default_value = base
    set_input(bsdf, "Roughness", M["flesh_roughness"])
    set_input(bsdf, ["Subsurface Weight", "Subsurface"], M["flesh_subsurface"])
    set_input(bsdf, "Subsurface Radius", (0.02, 0.008, 0.002))
    set_input(bsdf, "Subsurface Scale", 0.01)
    set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.35 if M["fibrous"] else 0.0)
    set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.08)
    return mat


def simple_mat(name, color, rough=0.5, metal=0.0, emit=None):
    mat, nt, bsdf = principled(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    set_input(bsdf, "Roughness", rough)
    set_input(bsdf, "Metallic", metal)
    if emit is not None:
        set_input(bsdf, ["Emission Color", "Emission"], (*emit, 1))
        set_input(bsdf, "Emission Strength", 1.0)
    return mat


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def build_root_mesh(shape, n_lat=72, n_lon=112):
    import numpy as np
    bm = bmesh.new()
    rows = []
    for i in range(1, n_lat):
        lat = -math.pi / 2 + math.pi * i / n_lat
        row = []
        for j in range(n_lon):
            lon = 2 * math.pi * j / n_lon
            u, v, w = math.cos(lat) * math.cos(lon), math.sin(lat), math.cos(lat) * math.sin(lon)
            x, y, z = shape.forward(u, v, w)
            row.append(bm.verts.new((float(x), float(y), float(z))))
        rows.append(row)
    s = [float(c) for c in shape.forward(0.0, -1.0, 0.0)]
    n = [float(c) for c in shape.forward(0.0, 1.0, 0.0)]
    south, north = bm.verts.new(s), bm.verts.new(n)
    for r0, r1 in zip(rows, rows[1:]):
        for j in range(n_lon):
            k = (j + 1) % n_lon
            bm.faces.new((r0[j], r0[k], r1[k], r1[j]))
    for j in range(n_lon):
        k = (j + 1) % n_lon
        bm.faces.new((south, rows[0][k], rows[0][j]))
        bm.faces.new((north, rows[-1][j], rows[-1][k]))
    for f in bm.faces:
        f.smooth = True
        f.material_index = 0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def interp(kind):
    bpy.context.preferences.edit.keyframe_new_interpolation_type = kind


def key_visible(obj, frame, visible):
    obj.hide_render = not visible
    obj.hide_viewport = not visible
    obj.keyframe_insert("hide_render", frame=frame)
    obj.keyframe_insert("hide_viewport", frame=frame)


def set_visibility(obj, show_from, hide_from):
    interp("CONSTANT")
    if show_from > 1:
        key_visible(obj, 1, False)
    key_visible(obj, max(1, show_from), True)
    if hide_from is not None:
        key_visible(obj, hide_from, False)


ASSETS = os.path.join(HERE, "..", "assets", "cache")


def _asset(name):
    p = os.path.join(ASSETS, name)
    return p if os.path.exists(p) else None


def board_material():
    """The channel's signature board: clean, light maple with a fine straight grain.
    Deliberately procedural, so it is identical in every video."""
    mat, nt, bsdf = principled("Board")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.2, 2.2, 2.2)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    diff = None  # photo textures looked worn and varied with the download
    if diff:
        def img(path, colorspace):
            t = nt.nodes.new("ShaderNodeTexImage")
            t.image = bpy.data.images.load(path, check_existing=True)
            t.image.colorspace_settings.name = colorspace
            t.projection = "BOX"
            t.projection_blend = 0.2
            L.new(mp.outputs["Vector"], t.inputs["Vector"])
            return t
        L.new(img(diff, "sRGB").outputs["Color"], bsdf.inputs["Base Color"])
        if _asset("wood_rough.jpg"):
            L.new(img(_asset("wood_rough.jpg"), "Non-Color").outputs["Color"], bsdf.inputs["Roughness"])
        if _asset("wood_nor.jpg"):
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nm.inputs["Strength"].default_value = 0.6
            L.new(img(_asset("wood_nor.jpg"), "Non-Color").outputs["Color"], nm.inputs["Color"])
            L.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        wave = nt.nodes.new("ShaderNodeTexWave")
        wave.bands_direction = "X"
        set_input(wave, "Scale", 18.0)
        set_input(wave, "Distortion", 0.8)
        set_input(wave, "Detail", 8.0)
        set_input(wave, "Detail Roughness", 0.7)
        sc = nt.nodes.new("ShaderNodeMapping")
        sc.inputs["Scale"].default_value = (0.25, 4.0, 4.0)  # long grain along the board
        L.new(mp.outputs["Vector"], sc.inputs["Vector"])
        L.new(sc.outputs["Vector"], wave.inputs["Vector"])
        r = ramp(nt, [(0.0, (0.60, 0.40, 0.21)), (0.6, (0.72, 0.51, 0.29)), (1.0, (0.56, 0.37, 0.19))])
        L.new(wave.outputs["Fac"], r.inputs["Fac"])
        L.new(r.outputs["Color"], bsdf.inputs["Base Color"])
        set_input(bsdf, "Roughness", 0.42)
        bump = nt.nodes.new("ShaderNodeBump")
        set_input(bump, "Strength", 0.02)
        L.new(wave.outputs["Fac"], bump.inputs["Height"])
        L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def cutting_board(size, thick=0.022, radius=0.028):
    """Rounded-corner cutting board, bevelled edges, a handle hole at one end."""
    import numpy as np
    w, d = size
    pts = []
    for cx, cy, a0 in ((w / 2 - radius, d / 2 - radius, 0), (-w / 2 + radius, d / 2 - radius, 90),
                       (-w / 2 + radius, -d / 2 + radius, 180), (w / 2 - radius, -d / 2 + radius, 270)):
        for k in range(10):
            a = math.radians(a0 + 90 * k / 9)
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    bm = bmesh.new()
    bot = [bm.verts.new((x, y, -thick)) for x, y in pts]
    top = [bm.verts.new((x, y, 0.0)) for x, y in pts]
    bm.faces.new(list(reversed(bot)))
    bm.faces.new(top)
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("Board")
    bm.to_mesh(me)
    bm.free()
    board = bpy.data.objects.new("Board", me)
    bpy.context.scene.collection.objects.link(board)
    bev = board.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments, bev.limit_method = 0.004, 4, "ANGLE"
    # handle hole: a capsule-shaped boolean near the right end
    bpy.ops.mesh.primitive_cylinder_add(radius=0.011, depth=0.1, location=(w / 2 - 0.035, 0, 0))
    cut = bpy.context.object
    cut.name = "HandleCutter"
    cut.scale = (1.0, 2.2, 1.0)
    cut.hide_render = True
    cut.hide_viewport = True
    boo = board.modifiers.new("Hole", "BOOLEAN")
    boo.object = cut
    boo.operation = "DIFFERENCE"
    board.modifiers.move(board.modifiers.find("Hole"), 0)
    for f in me.polygons:
        f.use_smooth = False
    board.data.materials.append(board_material())
    return board


def board_brand(text, size):
    """The account handle burned into the front-left corner of the board, like a pyrography
    mark: it is part of the scene, so a re-upload cannot crop or blur it away without
    damaging the video. Lies flat on the top face and reads correctly from the camera
    (which looks along +Y from the front edge)."""
    w, d = size
    cu = bpy.data.curves.new("BoardBrand", "FONT")
    cu.body = text
    f = _bold_font()
    if f:
        cu.font = f
    cu.size = max(0.020, d * 0.085)
    cu.align_x, cu.align_y = "LEFT", "BOTTOM"
    cu.extrude = 0.00008
    ob = bpy.data.objects.new("BoardBrand", cu)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = (-w / 2 + 0.03, -d / 2 + 0.018, 0.00012)
    mat = simple_mat("BurnedWood", (0.02, 0.009, 0.004), 0.7)
    cu.materials.append(mat)
    return ob


BOARD_SIZE = (0.40, 0.27)  # same board in every video: it is the channel's signature


def setup_environment(scn, board_size=BOARD_SIZE, brand=""):
    """Realistic tabletop: wooden cutting board on a dark surface, studio HDRI light."""
    board = cutting_board(board_size)
    if brand:
        board_brand(brand, board_size)
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -0.0221))
    floor = bpy.context.object
    floor.name = "Table"
    floor.data.materials.append(simple_mat("Table", (0.016, 0.015, 0.019), 0.7))
    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    out = nt.nodes["World Output"]
    hdr = _asset("studio.hdr")
    if hdr:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdr, check_existing=True)
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.9
        # the camera sees a dark studio, the objects receive the real HDRI light
        dark = nt.nodes.new("ShaderNodeBackground")
        dark.inputs["Color"].default_value = (0.012, 0.012, 0.016, 1)
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs["Background"], mix.inputs[1])
        nt.links.new(dark.outputs["Background"], mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        key_power, rim_power = 7, 9
    else:
        bg.inputs[0].default_value = (0.012, 0.013, 0.016, 1)
        bg.inputs[1].default_value = 0.25
        key_power, rim_power = 16, 12

    def area(name, loc, power, size, color=(1, 0.97, 0.92)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = power, size, color
        ob = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(ob)
        ob.location = loc
        d = (Vector((0, 0, 0.03)) - Vector(loc)).normalized()
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return ob

    area("Key", (-0.45, -0.35, 0.55), key_power, 0.45)
    area("Rim", (0.10, 0.60, 0.45), rim_power, 0.35)
    if not hdr:
        area("Fill", (0.55, -0.45, 0.30), 3, 0.6, (0.85, 0.9, 1.0))
    return board


def build_knife():
    """Chef's knife as ONE mesh (copies keep the handle): curved belly, ground bevel, bolster,
    riveted handle. Edge along y~0, tip at -x, heel at +x, handle beyond the heel."""
    import numpy as np
    from knife_traj import edge_y
    steel = simple_mat("Steel", (0.82, 0.83, 0.85), 0.16, 1.0)
    set_input(steel.node_tree.nodes["Principled BSDF"], "Anisotropic", 0.55)
    handle_m = simple_mat("Handle", (0.045, 0.024, 0.014), 0.38)
    rivet_m = simple_mat("Rivet", (0.75, 0.62, 0.35), 0.25, 1.0)
    bm = bmesh.new()
    xs = np.linspace(-0.115, 0.09, 42)

    def spine_y(x):
        return 0.048 if x > -0.02 else 0.013 + 0.035 * ((x + 0.115) / 0.095) ** 0.75

    rows = []
    for x in xs:
        e, s = float(edge_y(x)), spine_y(x)
        grind = e + (s - e) * 0.42
        t = 0.0011 * min(1.0, (x + 0.125) / 0.05)
        rows.append([bm.verts.new((x, e, 0.0)),
                     bm.verts.new((x, grind, t * 0.75)), bm.verts.new((x, s, t)),
                     bm.verts.new((x, s, -t)), bm.verts.new((x, grind, -t * 0.75))])
    for a, b in zip(rows, rows[1:]):
        for k in range(5):
            bm.faces.new((a[k], b[k], b[(k + 1) % 5], a[(k + 1) % 5]))
    bm.faces.new(rows[0])
    bm.faces.new(list(reversed(rows[-1])))

    def box(centre, size, mi, bevel=0.0, smooth=False):
        res = bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation(centre) @ Matrix.Diagonal((*size, 1.0)))
        vs = res["verts"]
        if bevel:
            es = list({e for v in vs for e in v.link_edges})
            out = bmesh.ops.bevel(bm, geom=vs + es, offset=bevel, segments=3, profile=0.5, affect="EDGES")
            faces = out["faces"] + list({f for v in vs if v.is_valid for f in v.link_faces})
        else:
            faces = list({f for v in vs for f in v.link_faces})
        for f in faces:
            f.material_index, f.smooth = mi, smooth

    box((0.097, 0.024, 0.0), (0.014, 0.048, 0.0065), 0, 0.0012)
    box((0.163, 0.026, 0.0), (0.118, 0.026, 0.017), 1, 0.0055, True)
    for x in (0.128, 0.163, 0.198):
        res = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=0.0032, radius2=0.0032, depth=0.0178,
                                    matrix=Matrix.Translation((x, 0.026, 0.0)))
        for f in {f for v in res["verts"] for f in v.link_faces}:
            f.material_index, f.smooth = 2, True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("Knife")
    bm.to_mesh(me)
    bm.free()
    for m in (steel, handle_m, rivet_m):
        me.materials.append(m)
    blade = bpy.data.objects.new("Knife", me)
    bpy.context.scene.collection.objects.link(blade)
    blade.rotation_mode = "QUATERNION"
    blade.scale = (0.85, 0.85, 0.85)
    return blade


_FONT = []


def _bold_font():
    """A heavy sans font reads far better on a phone than Blender's default."""
    if not _FONT:
        cands = [os.environ.get("ASMR_FONT", ""),
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf", "/Library/Fonts/Arial Bold.ttf"]
        font = None
        for c in cands:
            if c and os.path.exists(c):
                try:
                    font = bpy.data.fonts.load(c, check_existing=True)
                    break
                except Exception:
                    pass
        _FONT.append(font)
    return _FONT[0]


def text_obj(name, body, size, loc, parent, mat, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    f = _bold_font()
    if f is not None:
        cu.font = f
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = loc
    return ob


def _fmt(n):
    return f"{int(n):,}"


def _counter(cam, name, events, size, y, D, mat):
    """events: (frame, value) or (frame, value, total) -> shows "value / total"."""
    per_frame = {}
    for e in sorted(events, key=lambda e: e[0]):
        per_frame[max(1, int(e[0]))] = e[1:]  # last value reached in that frame
    ev = sorted(per_frame.items())
    for i, (f, v) in enumerate(ev):
        nxt = ev[i + 1][0] if i + 1 < len(ev) else None
        body = _fmt(v[0]) if len(v) == 1 or v[1] is None else f"{_fmt(v[0])} / {_fmt(v[1])}"
        ob = text_obj(f"{name}_{i:05d}", body, size, (0, y, -D), cam, mat)
        set_visibility(ob, f, nxt)


def build_overlay(cam, cfg, cuts_events, pieces_events, total):
    """Two counters in every format: CUTS on top, PIECES/CUBES under the object.

    Positions stay inside the TikTok/Reels safe zone: the top ~13 % (search bar, tabs)
    and the bottom ~25 % (username, caption) are covered by the app UI on a phone.
    y is in units of the frame half-height: +1 = top edge, -1 = bottom edge."""
    white = simple_mat("Overlay", (1, 1, 1), 1.0, 0.0, emit=(1, 1, 1))
    ink = simple_mat("OverlayInk", (0.018, 0.010, 0.005), 1.0, 0.0, emit=(0.018, 0.010, 0.005))
    accent = simple_mat("OverlayAccent", (1, 0.75, 0.3), 1.0, 0.0, emit=(1, 0.72, 0.25))
    D = 0.1
    hh = D * math.tan(math.atan(18 / 50))
    fps = bpy.context.scene.render.fps
    if cfg["show_counter"]:
        _counter(cam, "Cuts", cuts_events, hh * 0.15, hh * 0.60, D, white)
        text_obj("CutsWord", cfg["label_cuts"], hh * 0.055, (0, hh * 0.47, -D), cam, accent)
        # white on a dark plate: in the three-quarter view this counter straddles the board edge
        _counter(cam, "Pieces", pieces_events, hh * 0.11, -hh * 0.36, D, white)
        text_obj("PiecesWord", cfg["label_pieces"], hh * 0.05, (0, -hh * 0.48, -D), cam, accent)
        # dark plate: the label sits on the dark table in wide shots but on the light board when
        # the camera pushes in, and must stay readable on both
        pm = bpy.data.meshes.new("LabelPlate")
        pm.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
        pm.materials.append(simple_mat("LabelPlate", (0.02, 0.02, 0.022), 1.0, 0.0, emit=(0.02, 0.02, 0.022)))
        plate = bpy.data.objects.new("LabelPlate", pm)
        scn_plate = bpy.context.scene.collection
        scn_plate.objects.link(plate)
        plate.parent = cam
        plate.location = (0, -hh * 0.41, -D * 1.003)
        plate.scale = (hh * 0.36, hh * 0.12, 1)
    if cfg.get("hook_text"):
        ob = text_obj("Hook", _wrap(cfg["hook_text"], 22), hh * 0.065, (0, hh * 0.31, -D), cam, white)
        set_visibility(ob, 1, int(3.0 * fps))
    if cfg.get("end_text"):
        ob = text_obj("EndText", _wrap(cfg["end_text"], 22), hh * 0.065, (0, hh * 0.31, -D), cam, white)
        set_visibility(ob, max(2, total - int(2.5 * fps)), None)
    if cfg["watermark"]:
        text_obj("Watermark", cfg["watermark"], hh * 0.035, (0, -hh * 0.58, -D), cam, white)


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


def setup_render(scn, cfg, out_dir, final):
    eng = cfg["render_engine"]
    if eng == "cycles":
        scn.render.engine = "CYCLES"
        c = scn.cycles
        c.device = "CPU"
        c.samples = cfg["cycles_samples"] if final else cfg.get("preview_samples", 12)
        c.use_adaptive_sampling = True
        c.adaptive_threshold = 0.04 if final else 0.08
        c.use_denoising = True
        try:
            c.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            pass
        # small studio scene: few bounces are enough, caustics never matter here
        c.max_bounces = 4
        scn.render.use_motion_blur = bool(scn.get("knife_blur", False))  # knife-only blur (grid)
        c.diffuse_bounces, c.glossy_bounces = 2, 2
        c.transmission_bounces, c.transparent_max_bounces = 2, 4
        c.caustics_reflective = c.caustics_refractive = False
        scn.render.use_persistent_data = True  # keep BVH/shaders between frames
    else:
        for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                scn.render.engine = name
                break
            except TypeError:
                continue
        try:
            scn.eevee.taa_render_samples = 48 if final else 16
        except AttributeError:
            pass
    res = cfg["resolution"] if final else cfg["preview_resolution"]
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.use_overwrite = False
    scn.render.use_placeholder = False
    scn.render.filepath = os.path.join(out_dir, "frames", "f_#####")
    try:
        scn.view_settings.view_transform = "AgX"
        scn.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass


def render_still(scn, frame, path, res=None):
    scn.frame_set(frame)
    old = scn.render.filepath
    if res:
        scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.filepath = path
    scn.render.use_overwrite = True
    bpy.ops.render.render(write_still=True)
    scn.render.filepath = old
    scn.render.use_overwrite = False


def main():
    a = args()
    out_dir = os.path.abspath(a["out"])
    os.makedirs(out_dir, exist_ok=True)
    if a["stage"] == "build":
        with open(a["plan"], encoding="utf-8") as f:
            plan = json.load(f)
        import blender_grid
        report = blender_grid.build_grid_scene(plan, out_dir)
        scn = bpy.context.scene
        scn["asmr_plan"] = os.path.abspath(a["plan"])
        # embed textures/HDRI: relative paths break when the run dir moves between CI jobs
        # (a missing HDRI renders the whole world magenta)
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "scene.blend"))
        # previews: opening, a cut in progress, mid progression, final + one still per stroke
        pv = os.path.join(out_dir, "previews")
        os.makedirs(pv, exist_ok=True)
        t1 = time.time()
        small = [r // 2 for r in plan["config"]["preview_resolution"]]
        lv = plan["levels"]
        mid = lv[min(3, len(lv) - 1)]
        shots = {"01_opening": 2,
                 "02_cut_in_progress": (mid["strokes_start"] + mid["strokes_end"]) // 2,
                 "03_mid_progression": lv[len(lv) // 2]["collapse_end"] + 2,
                 "04_final": lv[-1]["collapse_end"] + 4}
        bl = [s for s in plan.get("blade_strokes", []) if s["level"] == lv[-1]["level"]]
        if len(lv) == 1 and len(bl) >= 4:
            # single level: a calm stroke mid-object, the middle of the ramp, the fast blur, the end
            def mid_obj(s):
                return int(round((s["time"]["contact"] + s["time"]["exit"]) / 2))
            shots = {"01_opening": 2,
                     "02_first_stroke": mid_obj(bl[1]),
                     "03_mid_strokes": mid_obj(bl[len(bl) // 2]),
                     "04_fast_strokes": mid_obj(bl[int(len(bl) * 0.9)]),
                     "06_final": lv[-1]["collapse_end"] + 4}
        # stills of huge levels cost ~8 min each (full BVH build): keep the sheet to the readable ones
        sheet = [(l["level"], l["collapse_end"] + 2) for l in lv if l["cubes"] <= 150000]
        print(f"ASMR_LOG scene built in {report.get('build_seconds')}s, rendering previews", flush=True)
        for name, f in shots.items():
            t2 = time.time()
            render_still(scn, f, os.path.join(pv, name + ".png"))
            print(f"ASMR_LOG still {name} (frame {f}) {time.time() - t2:.1f}s", flush=True)
        for i, f in sheet:
            t2 = time.time()
            render_still(scn, f, os.path.join(pv, "strokes", f"s_{i:02d}.png"), small)
            print(f"ASMR_LOG sheet still {i} (frame {f}) {time.time() - t2:.1f}s", flush=True)
        report["preview_seconds"] = round(time.time() - t1, 2)
        report["preview_frames"] = shots
        with open(os.path.join(out_dir, "blender_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("ASMR_REPORT " + json.dumps(report))
    elif a["stage"] == "render":
        scn = bpy.context.scene
        plan_path = os.path.join(out_dir, "plan.json")  # run dirs move between CI jobs
        if not os.path.exists(plan_path):
            plan_path = scn["asmr_plan"]
        with open(plan_path, encoding="utf-8") as f:
            cfg = json.load(f)["config"]
        setup_render(scn, cfg, out_dir, final=a["final"])
        if a.get("frames"):
            s, e = a["frames"].split("-")
            scn.frame_start, scn.frame_end = int(s), int(e)
        t1 = time.time()
        bpy.ops.render.render(animation=True)
        print("ASMR_RENDER " + json.dumps({"seconds": round(time.time() - t1, 1),
                                           "frames": scn.frame_end - scn.frame_start + 1}))


if __name__ == "__main__":
    try:
        main()
    except Exception:            # Blender -P exits 0 on a script error: make CI see it
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
