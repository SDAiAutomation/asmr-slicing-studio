"""Blender side of the ASMR slicing studio (Blender 4.2+ / 5.x, background mode).

Replays plan.json cut by cut on real meshes:
  root mesh -> bisect -> cap -> split loose parts -> validate -> next cut
so every piece on screen is the actual geometric result of the previous state.
The on-screen counter is driven by the number of validated mesh parts, not by
the planner's expectation.

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
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import MATERIALS, OBJECTS  # noqa: E402


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


def wood_mat():
    mat, nt, bsdf = principled("Board")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    wave = nt.nodes.new("ShaderNodeTexWave")
    set_input(wave, "Scale", 9.0)
    set_input(wave, "Distortion", 9.0)
    set_input(wave, "Detail", 6.0)
    L.new(tc.outputs["Object"], wave.inputs["Vector"])
    r = ramp(nt, [(0.0, (0.33, 0.17, 0.07)), (1.0, (0.55, 0.32, 0.14))])
    L.new(wave.outputs["Fac"], r.inputs["Fac"])
    L.new(r.outputs["Color"], bsdf.inputs["Base Color"])
    set_input(bsdf, "Roughness", 0.55)
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


def islands(bm):
    seen, out = set(), []
    for f in bm.faces:
        if f in seen:
            continue
        stack, comp = [f], []
        seen.add(f)
        while stack:
            g = stack.pop()
            comp.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
        out.append(comp)
    return out


def bm_subset(faces):
    nb = bmesh.new()
    vmap = {}
    for f in faces:
        vs = []
        for v in f.verts:
            if v not in vmap:
                vmap[v] = nb.verts.new(v.co)
            vs.append(vmap[v])
        nf = nb.faces.new(vs)
        nf.material_index = f.material_index
        nf.smooth = f.smooth
    return nb


def half(bm_src, co, no, keep_positive, rim):
    b = bm_src.copy()
    geom = b.verts[:] + b.edges[:] + b.faces[:]
    bmesh.ops.bisect_plane(b, geom=geom, dist=1e-7, plane_co=co, plane_no=no,
                           clear_inner=keep_positive, clear_outer=not keep_positive)
    boundary = [e for e in b.edges if e.is_boundary]
    caps = []
    if boundary:
        res = bmesh.ops.holes_fill(b, edges=boundary, sides=0)
        caps = res["faces"]
        for f in caps:
            f.material_index = 1
            f.smooth = False
        if rim > 0 and caps:
            ins = bmesh.ops.inset_region(b, faces=caps, thickness=rim, depth=0.0,
                                         use_even_offset=True, use_boundary=True)
            for f in ins["faces"]:
                f.material_index = 2
                f.smooth = False
    bmesh.ops.recalc_face_normals(b, faces=b.faces[:])
    parts = []
    comps = islands(b)
    if len(comps) == 1:
        parts.append(b)
    else:
        for comp in comps:
            parts.append(bm_subset(comp))
        b.free()
    return parts


def validate(b):
    nonman = sum(1 for e in b.edges if len(e.link_faces) != 2)
    vol = b.calc_volume(signed=True)
    c = Vector()
    for v in b.verts:
        c += v.co
    c /= max(1, len(b.verts))
    return {"non_manifold_edges": nonman, "volume": abs(vol), "volume_signed": vol, "center": c}


# ---------------------------------------------------------------------------
# animation helpers
# ---------------------------------------------------------------------------
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


def key_location(obj, keys):
    interp("BEZIER")
    for f, off in keys:
        obj.location = Vector(off)
        obj.keyframe_insert("location", frame=f)


def blade_quat(n, d):
    n, d = Vector(n).normalized(), Vector(d).normalized()
    Y = -d
    Z = n
    X = Y.cross(Z).normalized()
    if X.x < 0:  # keep the handle on the right of the frame
        Z = -Z
        X = Y.cross(Z).normalized()
    m = Matrix((X, Y, Z)).transposed()
    return m.to_quaternion()


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
def build_scene(plan, out_dir):
    t0 = time.time()
    cfg = plan["config"]
    shape = OBJECTS[cfg["object_type"]]
    M = MATERIALS[cfg["material_profile"]]
    fps = plan["fps"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.fps = fps
    scn.frame_start, scn.frame_end = 1, plan["total_frames"]

    skin, flesh = make_skin(M, shape), make_flesh(M)
    rim = simple_mat("Rim", M["rim"], 0.35)
    mats = [skin, flesh, rim]

    # --- environment: board, backdrop, lights, world
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -0.011))
    board = bpy.context.object
    board.name = "Board"
    board.scale = (0.46, 0.34, 0.02)
    board.data.materials.append(wood_mat())
    bpy.ops.object.modifier_add(type="BEVEL")
    board.modifiers[-1].width = 0.004
    board.modifiers[-1].segments = 3
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, -0.022))
    floor = bpy.context.object
    floor.data.materials.append(simple_mat("Backdrop", (0.018, 0.02, 0.024), 0.8))
    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.012, 0.013, 0.016, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

    def area(name, loc, power, size, color=(1, 0.97, 0.92)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = power, size, color
        ob = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(ob)
        ob.location = loc
        d = (Vector((0, 0, 0.03)) - Vector(loc)).normalized()
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return ob

    area("Key", (-0.45, -0.35, 0.55), 55, 0.45)
    area("Fill", (0.55, -0.45, 0.30), 14, 0.6, (0.85, 0.9, 1.0))
    area("Rim", (0.10, 0.60, 0.45), 40, 0.35)

    # --- knife
    knife = build_knife()

    # --- pieces: replay the plan cut by cut
    pieces = {p["id"]: p for p in plan["pieces"]}
    bm_of = {}
    obj_of = {}
    report = {"hits": [], "failures": [], "non_manifold": 0, "mismatch_hits": 0}
    root_ids = [p["id"] for p in plan["pieces"] if p["parent"] is None]
    root_bm = build_root_mesh(shape)
    root_vol = abs(root_bm.calc_volume(signed=True))
    for rid in root_ids:
        bm_of[rid] = root_bm.copy()
        obj_of[rid] = make_piece_obj(rid, bm_of[rid], mats)
    true_count = {}  # level -> current count
    counter_events = []
    for rid in root_ids:
        lvl = pieces[rid]["level"]
        true_count[lvl] = 1
        counter_events.append((pieces[rid]["born_frame"] if lvl else 1, 1))
    # blender-only extra parts get their own ids
    extra_id = max(pieces) + 1
    born, cut_at, keys_of, hide_at = {}, {}, {}, {}
    for pid, p in pieces.items():
        keys_of[pid] = p["keys"]
        born[pid] = p["born_frame"]
        cut_at[pid] = p["cut_frame"]
        hide_at[pid] = p["hide_frame"]

    rim_w = M["rim_width"]
    for hi, h in enumerate(plan["hits"]):
        pid = h["piece"]
        if pid not in bm_of:
            report["failures"].append({"hit": hi, "reason": "parent piece missing"})
            continue
        src = bm_of[pid]
        co, no = Vector(h["plane_point"]), Vector(h["plane_normal"])
        parts = half(src, co, no, False, rim_w) + half(src, co, no, True, rim_w)
        info = [validate(b) for b in parts]
        good = [(b, i) for b, i in zip(parts, info) if i["volume"] > 1e-10]
        kids = h["children"]
        # match blender parts to planned children by nearest centroid
        unused = list(kids)
        assigned = []
        for b, i in sorted(good, key=lambda t: -t[1]["volume"]):
            if unused:
                best = min(unused, key=lambda k: (Vector(pieces[k]["centroid"]) - i["center"]).length)
                unused.remove(best)
                assigned.append((best, b, i))
            else:  # blender found an extra part the planner did not see
                near = min(kids, key=lambda k: (Vector(pieces[k]["centroid"]) - i["center"]).length)
                nid = extra_id
                extra_id += 1
                keys_of[nid], born[nid], cut_at[nid], hide_at[nid] = keys_of[near], born[near], None, hide_at.get(near)
                assigned.append((nid, b, i))
        if unused or len(good) != len(kids):
            report["mismatch_hits"] += 1
        for k in unused:
            report["failures"].append({"hit": hi, "reason": f"planned piece {k} has no mesh part"})
        for cid, b, i in assigned:
            report["non_manifold"] += int(i["non_manifold_edges"] > 0)
            bm_of[cid] = b
            obj_of[cid] = make_piece_obj(cid, b, mats)
        lvl = h["level"]
        true_count[lvl] += len(assigned) - 1
        counter_events.append((h["exit_frame"], true_count[lvl]))
        report["hits"].append({"hit": hi, "parts": len(assigned), "planned": len(kids),
                               "count": true_count[lvl]})
        src.free()
        del bm_of[pid]

    # visibility + motion
    for pid, ob in obj_of.items():
        set_visibility(ob, born[pid] or 1, cut_at.get(pid) or hide_at.get(pid))
        key_location(ob, keys_of[pid])

    # final validation on what is left
    final_vol = sum(abs(b.calc_volume(signed=True)) for b in bm_of.values())
    last_level = max(true_count)
    report.update({
        "final_pieces_validated": true_count[last_level],
        "target": cfg["target_piece_count"],
        "volume_root": root_vol,
        "volume_final": final_vol,
        "volume_error_pct": (round(100 * abs(final_vol / root_vol - 1), 4)
                             if cfg["cutting_mode"] == "cumulative" else None),
        "objects": len(obj_of),
    })

    animate_knife(knife, plan, pieces, keys_of)
    cam = build_camera(plan, cfg)
    build_overlay(cam, cfg, M, counter_events, plan["total_frames"])
    setup_render(scn, cfg, out_dir, final=False)
    report["build_seconds"] = round(time.time() - t0, 2)
    report["counter_events"] = counter_events
    return report


def make_piece_obj(pid, b, mats):
    me = bpy.data.meshes.new(f"piece_{pid:05d}")
    b.to_mesh(me)
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(f"piece_{pid:05d}", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build_knife():
    steel = simple_mat("Steel", (0.75, 0.76, 0.78), 0.12, 1.0)
    handle_m = simple_mat("Handle", (0.03, 0.025, 0.02), 0.45)
    bm = bmesh.new()
    # blade profile in local XY: edge on y=0 from x=-0.11 (tip) to x=0.09 (heel)
    prof = [(-0.115, 0.012), (-0.10, 0.0), (0.09, 0.0), (0.09, 0.048), (-0.02, 0.046), (-0.09, 0.03)]
    t = 0.0009
    edge_t = 0.00008  # a real edge is not infinitely thin; avoids degenerate faces
    top = [bm.verts.new((x, y, t if y > 0.001 else edge_t)) for x, y in prof]
    bot = [bm.verts.new((x, y, -t if y > 0.001 else -edge_t)) for x, y in prof]
    bm.faces.new(top)
    bm.faces.new(list(reversed(bot)))
    n = len(prof)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((top[i], top[j], bot[j], bot[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new("Blade")
    bm.to_mesh(me)
    me.materials.append(steel)
    blade = bpy.data.objects.new("Knife", me)
    bpy.context.scene.collection.objects.link(blade)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.15, 0.022, 0))
    handle = bpy.context.object
    handle.name = "KnifeHandle"
    handle.scale = (0.12, 0.022, 0.016)
    handle.data.materials.append(handle_m)
    handle.parent = blade
    blade.rotation_mode = "QUATERNION"
    return blade


def animate_knife(knife, plan, pieces, keys_of):
    fps = plan["fps"]
    park = Vector((0.35, -0.10, 0.25))
    interp("BEZIER")
    knife.location = park
    knife.rotation_quaternion = blade_quat((0, 1, 0), (0, 0, -1))
    knife.keyframe_insert("location", frame=1)
    knife.keyframe_insert("rotation_quaternion", frame=1)
    prev_q = knife.rotation_quaternion.copy()
    last = 1
    hits = plan["hits"]

    def key(f, loc, q):
        knife.location = loc
        knife.rotation_quaternion = q
        knife.keyframe_insert("location", frame=f)
        knife.keyframe_insert("rotation_quaternion", frame=f)

    for i, h in enumerate(hits):
        c0, c1 = h["contact_frame"], h["exit_frame"]
        if c0 <= last:  # dense wave: several pieces in this frame, knife follows the first
            continue
        n, d = Vector(h["plane_normal"]), Vector(h["travel"])
        q = blade_quat(n, d)
        if q.dot(prev_q) < 0:
            q.negate()
        c = Vector(h["plane_point"]) + Vector(h["offset"])
        ext = h["extent_travel"]
        shift = -q.to_matrix().col[0] * 0.02
        start = c - d * (ext + 0.004) + shift
        end = c + d * (ext + 0.004) + shift
        if d.z < -0.2:  # coming down: stop on the board surface (z = 0)
            end = c + d * ((c.z - 0.0004) / -d.z) + shift
        gap = c0 - last
        if gap >= 4:
            lead = min(6, gap - 2)
            key(c0 - lead, start - d * 0.03, q)
            key(c0, start, q)
            key(c1, end, q)
            last = c1
        else:  # fast chopping: one key per frame, blade through the piece
            key(c0, end, q)
            last = c0
        prev_q = q
        nxt = hits[i + 1]["contact_frame"] if i + 1 < len(hits) else None
        if (nxt is None or nxt - last > 8) and last == c1:
            key(c1 + 4, end - d * 0.05, q)
            last = c1 + 4
    end_f = plan["total_frames"]
    knife.location = park
    knife.keyframe_insert("location", frame=max(last + 10, end_f - 20))
    knife.keyframe_insert("rotation_quaternion", frame=max(last + 10, end_f - 20))


def build_camera(plan, cfg):
    cd = bpy.data.cameras.new("Camera")
    cd.lens = 50
    cd.clip_start = 0.004
    cd.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("Camera", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    w, h = cfg["resolution"]
    vfov = 2 * math.atan(18 / 50)
    hfov = 2 * math.atan(math.tan(vfov / 2) * w / h)
    elev, azim = math.radians(42), math.radians(-78)
    view = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
    interp("BEZIER")
    shots = [{"frame": 1, "start_frame": 1, "center": [0, 0, 0.04], "radius": 0.075}] + plan["camera"]
    if cfg["camera_style"] == "static":
        shots = [max(shots, key=lambda s: s["radius"])] if len(shots) > 1 else shots
        shots[0] = dict(shots[0], frame=1)
    prev_q = None
    last = 0
    for s in shots:
        r = max(0.06, s["radius"])
        dist = r / math.tan(hfov / 2) * 1.05
        target = Vector(s["center"]) + Vector((0, 0, -0.01))
        loc = target + view * dist
        q = (target - loc).to_track_quat("-Z", "Y")
        if prev_q is not None and q.dot(prev_q) < 0:
            q.negate()
        # camera eases during the stroke's pre-roll, holds during cutting
        f = max(last + 1, s.get("start_frame", s["frame"]) - 2)
        cam.location, cam.rotation_quaternion = loc, q
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        last, prev_q = f, q
    return cam


def text_obj(name, body, size, loc, parent, mat, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    ob.location = loc
    return ob


def build_overlay(cam, cfg, M, counter_events, total):
    white = simple_mat("Overlay", (1, 1, 1), 1.0, 0.0, emit=(1, 1, 1))
    accent = simple_mat("OverlayAccent", (1, 0.75, 0.3), 1.0, 0.0, emit=(1, 0.72, 0.25))
    D = 0.1
    w, h = cfg["resolution"]
    half_h = D * math.tan(math.atan(18 / 50))
    half_w = half_h * w / h
    text_obj("Label", M["label"], half_h * 0.06, (0, half_h * 0.80, -D), cam, accent)
    if cfg["show_counter"]:
        text_obj("PiecesWord", "PIECES", half_h * 0.05, (0, half_h * 0.57, -D), cam, white)
        # one text object per validated count value, switched on/off in time
        per_frame = {}
        for f, c in sorted(counter_events):
            per_frame[f] = c  # last value reached in that frame
        ev = sorted(per_frame.items())
        for i, (f, c) in enumerate(ev):
            nxt = ev[i + 1][0] if i + 1 < len(ev) else None
            ob = text_obj(f"Count_{i:05d}", f"{c:,}".replace(",", " "), half_h * 0.13, (0, half_h * 0.68, -D), cam, white)
            set_visibility(ob, f, nxt)
    fps = bpy.context.scene.render.fps
    if cfg.get("hook_text"):
        ob = text_obj("Hook", cfg["hook_text"], half_h * 0.07, (0, -half_h * 0.62, -D), cam, accent)
        ob.data.body = _wrap(cfg["hook_text"], 22)
        set_visibility(ob, 1, int(3.0 * fps))
    if cfg.get("end_text"):
        ob = text_obj("EndText", _wrap(cfg["end_text"], 22), half_h * 0.07, (0, -half_h * 0.62, -D), cam, accent)
        set_visibility(ob, max(2, total - int(2.5 * fps)), None)
    if cfg["watermark"]:
        text_obj("Watermark", cfg["watermark"], half_h * 0.045, (0, -half_h * 0.86, -D), cam, white)


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
        scn.cycles.samples = cfg["cycles_samples"] if final else 16
        scn.cycles.use_denoising = True
        scn.cycles.device = "CPU"
        scn.cycles.max_bounces = 6
        try:
            scn.cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            pass
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
        report = build_scene(plan, out_dir)
        scn = bpy.context.scene
        scn["asmr_plan"] = os.path.abspath(a["plan"])
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "scene.blend"))
        # previews: opening, a cut in progress, mid progression, final + one still per stroke
        pv = os.path.join(out_dir, "previews")
        os.makedirs(pv, exist_ok=True)
        t1 = time.time()
        hits, strokes = plan["hits"], plan["strokes"]
        mid_hit = hits[min(len(hits) - 1, plan["strokes"][min(2, len(strokes) - 1)]["hits"][0])]
        shots = {"01_opening": 2,
                 "02_cut_in_progress": (mid_hit["contact_frame"] + mid_hit["exit_frame"]) // 2,
                 "03_mid_progression": strokes[len(strokes) // 2]["end_frame"] - 1,
                 "04_final": plan["total_frames"] - 5}
        for name, f in shots.items():
            render_still(scn, f, os.path.join(pv, name + ".png"))
        small = [r // 2 for r in plan["config"]["preview_resolution"]]
        for st in strokes:
            render_still(scn, st["end_frame"] - 1,
                         os.path.join(pv, "strokes", f"s_{st['index']:02d}.png"), small)
        report["preview_seconds"] = round(time.time() - t1, 2)
        report["preview_frames"] = shots
        with open(os.path.join(out_dir, "blender_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print("ASMR_REPORT " + json.dumps({k: v for k, v in report.items()
                                           if k not in ("hits", "counter_events")}))
    elif a["stage"] == "render":
        scn = bpy.context.scene
        with open(scn["asmr_plan"], encoding="utf-8") as f:
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
    main()
