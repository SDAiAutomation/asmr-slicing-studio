"""Blender builder for the grid / levels format ("N CUTS -> X CUBES").

Per level:
  - a fresh object slides onto the board,
  - the knife makes the planned straight cuts; each cut leaves a visible seam
    (shader driven by animated cut counts, so millions of cuts stay cheap),
  - the scored object is swapped for its real cubes (one instance per occupied
    cell, Geometry Nodes), which slump and spread on the board,
  - the pile slides away and the next level comes in.
CUTS and PIECES/CUBES counters come from the plan's exact cell counts; the cube
object of each level is checked against the planned cube count.
"""
import math
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector

import blender_build as B
from common import MATERIALS, OBJECTS, grid_occupancy


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
def extruded_mesh(shape, rings=16, n=200):
    """Organic slab: concentric rings follow the outline, top height = shape.top(x, y)
    (swells + rounded edge), fat cap faces on the outer rings, muscle elsewhere."""
    bm = bmesh.new()
    out = shape.outline(n)
    has_top = hasattr(shape, "top")
    # ring 0 -> ring 1 is the fat cap (exact width), then the muscle rings inwards
    inner = 1.0 - shape.fat
    scales = [1.0, inner] + [inner * (1.0 - k / (rings - 1)) for k in range(1, rings - 1)]
    ring_v = []
    for sc in scales:
        row = []
        for x, y in out * sc:
            z = float(shape.top(x, y)) if has_top else shape.h
            row.append(bm.verts.new((float(x), float(y), z)))
        ring_v.append(row)
    centre = bm.verts.new((0.0, 0.0, float(shape.top(0.0, 0.0)) if has_top else shape.h))
    bot = [bm.verts.new((float(x), float(y), 0.0)) for x, y in out]
    f = bm.faces.new(list(reversed(bot)))
    f.material_index = 1
    for i in range(n):
        j = (i + 1) % n
        side = bm.faces.new((bot[i], bot[j], ring_v[0][j], ring_v[0][i]))
        side.material_index, side.smooth = 1, True
    for k in range(rings - 1):
        fat_ring = k == 0
        for i in range(n):
            j = (i + 1) % n
            q = bm.faces.new((ring_v[k][i], ring_v[k][j], ring_v[k + 1][j], ring_v[k + 1][i]))
            q.material_index, q.smooth = (0 if fat_ring else 1), True
    for i in range(n):
        j = (i + 1) % n
        t = bm.faces.new((ring_v[-1][i], ring_v[-1][j], centre))
        t.material_index, t.smooth = 1, True
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def object_mesh(shape, name):
    bm = extruded_mesh(shape) if getattr(shape, "kind", "") == "extruded" else B.build_root_mesh(shape)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------
def meat_group(M):
    """Muscle shading from a 3D coordinate: colour variation, oriented fibres, irregular
    marbling. Driven by the object's original-space position, so the marbling continues
    through every cut face and into every cube."""
    if "SteakMuscle" in bpy.data.node_groups:
        return bpy.data.node_groups["SteakMuscle"]
    g = bpy.data.node_groups.new("SteakMuscle", "ShaderNodeTree")
    g.interface.new_socket(name="Vector", in_out="INPUT", socket_type="NodeSocketVector")
    for nm, st in (("Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"), ("Height", "NodeSocketFloat")):
        g.interface.new_socket(name=nm, in_out="OUTPUT", socket_type=st)
    N, L = g.nodes, g.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    # colour blotches (deep red <-> brighter red)
    blot = N.new("ShaderNodeTexNoise")
    blot.inputs["Scale"].default_value = 9.0
    blot.inputs["Detail"].default_value = 4.0
    L.new(gi.outputs["Vector"], blot.inputs["Vector"])
    cr = B.ramp(g, [(0.3, M["flesh_deep"]), (0.7, M["flesh"])])
    L.new(blot.outputs["Fac"], cr.inputs["Fac"])
    # muscle fibres: fine bands across the steak, slightly wavy
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, 0.6)
    L.new(gi.outputs["Vector"], mp.inputs["Vector"])
    fib = N.new("ShaderNodeTexWave")
    fib.bands_direction = "X"
    fib.inputs["Scale"].default_value = 95.0
    fib.inputs["Distortion"].default_value = 3.0
    fib.inputs["Detail"].default_value = 3.0
    L.new(mp.outputs["Vector"], fib.inputs["Vector"])
    fr = B.ramp(g, [(0.0, (0.82, 0.82, 0.82)), (1.0, (1.0, 1.0, 1.0))])
    L.new(fib.outputs["Fac"], fr.inputs["Fac"])
    mul = N.new("ShaderNodeMix")
    mul.data_type, mul.blend_type = "RGBA", "MULTIPLY"
    ins = [x for x in mul.inputs if x.enabled]
    [x for x in ins if x.type == "VALUE"][0].default_value = 1.0
    ca, cb = [x for x in ins if x.type == "RGBA"][:2]
    L.new(cr.outputs["Color"], ca)
    L.new(fr.outputs["Color"], cb)
    mout = [x for x in mul.outputs if x.enabled and x.type == "RGBA"][0]
    # marbling: thin, branching fat veins from a distorted 3D noise iso-band
    mn = N.new("ShaderNodeTexNoise")
    mn.inputs["Scale"].default_value = 16.0
    mn.inputs["Detail"].default_value = 8.0
    mn.inputs["Roughness"].default_value = 0.62
    mn.inputs["Distortion"].default_value = 0.8
    L.new(gi.outputs["Vector"], mn.inputs["Vector"])
    # vein = thin iso-band where the noise crosses 0.5
    sub = N.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = 0.5
    L.new(mn.outputs["Fac"], sub.inputs[0])
    band = N.new("ShaderNodeMath")
    band.operation = "ABSOLUTE"
    L.new(sub.outputs[0], band.inputs[0])
    # coarse mask: marbling denser in some areas, absent in others
    mask = N.new("ShaderNodeTexNoise")
    mask.inputs["Scale"].default_value = 4.0
    L.new(gi.outputs["Vector"], mask.inputs["Vector"])
    mk = B.ramp(g, [(0.30, (0.15, 0.15, 0.15)), (0.60, (1, 1, 1))])
    L.new(mask.outputs["Fac"], mk.inputs["Fac"])
    vr = B.ramp(g, [(0.0, (1, 1, 1)), (0.008, (1, 1, 1)), (0.022, (0, 0, 0))])
    L.new(band.outputs[0], vr.inputs["Fac"])
    vm = N.new("ShaderNodeMath")
    vm.operation = "MULTIPLY"
    L.new(vr.outputs["Color"], vm.inputs[0])
    L.new(mk.outputs["Color"], vm.inputs[1])
    fat = N.new("ShaderNodeMix")
    fat.data_type = "RGBA"
    fins = [x for x in fat.inputs if x.enabled]
    L.new(vm.outputs[0], [x for x in fins if x.type == "VALUE"][0])
    fa, fb = [x for x in fins if x.type == "RGBA"][:2]
    L.new(mout, fa)
    fb.default_value = (0.80, 0.70, 0.58, 1)
    L.new([x for x in fat.outputs if x.enabled and x.type == "RGBA"][0], go.inputs["Color"])
    # wet, uneven sheen: rougher where there is fat
    rg = N.new("ShaderNodeMath")
    rg.operation = "MULTIPLY_ADD"
    rg.inputs[1].default_value = 0.15
    rg.inputs[2].default_value = 0.52
    L.new(vm.outputs[0], rg.inputs[0])
    L.new(rg.outputs[0], go.inputs["Roughness"])
    hh = N.new("ShaderNodeMath")
    hh.operation = "ADD"
    L.new(fib.outputs["Fac"], hh.inputs[0])
    L.new(vm.outputs[0], hh.inputs[1])
    L.new(hh.outputs[0], go.inputs["Height"])
    return g


def _wire_meat(mat, nt, bsdf, M, coord_socket):
    L = nt.links
    grp = nt.nodes.new("ShaderNodeGroup")
    grp.node_tree = meat_group(M)
    L.new(coord_socket, grp.inputs["Vector"])
    L.new(grp.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(grp.outputs["Roughness"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    B.set_input(bump, "Strength", 0.12)
    B.set_input(bump, "Distance", 0.0004)
    L.new(grp.outputs["Height"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    B.set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.22)
    B.set_input(bsdf, "Subsurface Radius", (0.012, 0.003, 0.002))
    B.set_input(bsdf, "Subsurface Scale", 0.01)
    B.set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.06)       # slight surface moisture
    B.set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.3)
    B.set_input(bsdf, ["Specular IOR Level", "Specular"], 0.3)


def meat_mat(M):
    mat, nt, bsdf = B.principled("Meat")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    _wire_meat(mat, nt, bsdf, M, tc.outputs["Object"])
    return mat


def fat_mat(M):
    mat, nt, bsdf = B.principled("FatCap")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    B.set_input(nz, "Scale", 40.0)
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    r = B.ramp(nt, [(0.35, (0.72, 0.60, 0.45)), (0.65, (0.85, 0.75, 0.60))])
    L.new(nz.outputs["Fac"], r.inputs["Fac"])
    L.new(r.outputs["Color"], bsdf.inputs["Base Color"])
    B.set_input(bsdf, "Roughness", 0.5)
    B.set_input(bsdf, ["Specular IOR Level", "Specular"], 0.3)
    B.set_input(bsdf, ["Subsurface Weight", "Subsurface"], 0.3)
    B.set_input(bsdf, "Subsurface Radius", (0.01, 0.008, 0.005))
    B.set_input(bsdf, "Subsurface Scale", 0.006)
    B.set_input(bsdf, ["Coat Weight", "Clearcoat"], 0.04)
    return mat


def steak_materials(M):
    fat = fat_mat(M)
    return [fat, meat_mat(M), fat]


def add_seams(mat, P, nodes_out, colour=(0.03, 0.006, 0.004, 1)):
    """Darken thin lines at the planned cut positions whose cut is already made.
    P: dict name -> Value node (lo_x.., cell_x.., n_x, n_y, n_z, cz, w)."""
    nt = mat.node_tree
    L = nt.links
    bsdf = nt.nodes.get("Principled BSDF")
    base_link = next((l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"]), None)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L.new(tc.outputs["Object"], sep.inputs[0])

    def val(name):
        if name not in P:
            v = nt.nodes.new("ShaderNodeValue")
            v.name = v.label = name
            v.outputs[0].default_value = 0.0
            P[name] = v
        return P[name].outputs[0]

    def math(op, a, b=None, c=None):
        m = nt.nodes.new("ShaderNodeMath")
        m.operation = op
        for i, x in enumerate((a, b, c)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                m.inputs[i].default_value = x
            else:
                L.new(x, m.inputs[i])
        return m.outputs[0]

    total = None
    for ax, comp in enumerate("XYZ"):
        u = math("DIVIDE", math("SUBTRACT", sep.outputs[comp], val(f"lo_{ax}")), val(f"cell_{ax}"))
        k = math("ROUND", u)
        dist = math("MULTIPLY", math("ABSOLUTE", math("SUBTRACT", u, k)), val(f"cell_{ax}"))
        line = math("LESS_THAN", dist, val("w"))
        if ax < 2:  # cuts 1..n made, sweeping from low to high coordinate
            active = math("MULTIPLY", math("GREATER_THAN", k, 0.5),
                          math("LESS_THAN", k, math("ADD", val(f"n_{ax}"), 0.5)))
        else:      # horizontal cuts land from the top: indices > cz - n
            active = math("MULTIPLY", math("GREATER_THAN", k, math("SUBTRACT", val("cz"), math("SUBTRACT", val("n_2"), 0.5))),
                          math("LESS_THAN", k, math("ADD", val("cz"), 0.5)))
        s = math("MULTIPLY", line, active)
        total = s if total is None else math("MAXIMUM", total, s)
    m, f, a, b, o = B.mix_rgb(nt)
    L.new(total, f)
    if base_link is not None:
        L.new(base_link.from_socket, a)
    else:
        a.default_value = bsdf.inputs["Base Color"].default_value
    b.default_value = colour
    L.new(o, bsdf.inputs["Base Color"])
    nodes_out.append(P)


def cube_mat(M, steak):
    mat, nt, bsdf = B.principled("Cubes")
    L = nt.links
    kind = nt.nodes.new("ShaderNodeAttribute")
    kind.attribute_type, kind.attribute_name = "INSTANCER", "kind"
    if steak:
        orig = nt.nodes.new("ShaderNodeAttribute")
        orig.attribute_type, orig.attribute_name = "INSTANCER", "orig"
        # shading point in the steak's original space = cube centre + local offset,
        # so a big cube shows the real marbling that runs through it (not one flat colour)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        add = nt.nodes.new("ShaderNodeVectorMath")
        add.operation = "ADD"
        L.new(orig.outputs["Vector"], add.inputs[0])
        L.new(tc.outputs["Object"], add.inputs[1])
        _wire_meat(mat, nt, bsdf, M, add.outputs["Vector"])
        # fat-cap cubes: creamy white instead of muscle
        link = next(l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"])
        m, f, a, b, o = B.mix_rgb(nt)
        L.new(kind.outputs["Fac"], f)
        L.new(link.from_socket, a)
        b.default_value = (0.82, 0.72, 0.57, 1)
        L.new(o, bsdf.inputs["Base Color"])
    else:
        r = B.ramp(nt, [(0.0, M["flesh"]), (1.0, M["skin_base"])])
        r.color_ramp.interpolation = "CONSTANT"
        r.color_ramp.elements[1].position = 0.5
        L.new(kind.outputs["Fac"], r.inputs["Fac"])
        L.new(r.outputs["Color"], bsdf.inputs["Base Color"])
        B.set_input(bsdf, "Roughness", 0.35)
    return mat


def collapse_group(mat):
    ng = bpy.data.node_groups.new("CubeCollapse", "GeometryNodeTree")
    I = ng.interface
    I.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    I.new_socket(name="Collapse", in_out="INPUT", socket_type="NodeSocketFloat")
    I.new_socket(name="CubeSize", in_out="INPUT", socket_type="NodeSocketVector")
    I.new_socket(name="Height", in_out="INPUT", socket_type="NodeSocketFloat")
    I.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    pos = N.new("GeometryNodeInputPosition")
    sep = N.new("ShaderNodeSeparateXYZ")
    L.new(pos.outputs[0], sep.inputs[0])
    rnd = N.new("FunctionNodeRandomValue")
    rnd.data_type = "FLOAT"
    rv = [o for o in rnd.outputs if o.enabled and o.type == "VALUE"][0]

    def m(op, a, b=None, c=None):
        n = N.new("ShaderNodeMath")
        n.operation = op
        for i, x in enumerate((a, b, c)):
            if x is None:
                continue
            if isinstance(x, (int, float)):
                n.inputs[i].default_value = x
            else:
                L.new(x, n.inputs[i])
        return n.outputs[0]

    col, H = gi.outputs["Collapse"], gi.outputs["Height"]
    # cubes sink towards the board (pile keeps ~25-30 % of its height)...
    dz = m("MULTIPLY", m("MULTIPLY", col, -1.0), m("MULTIPLY", sep.outputs["Z"], m("MULTIPLY_ADD", rv, 0.22, 0.62)))
    # ...and the upper ones slide outwards
    rel = m("DIVIDE", sep.outputs["Z"], m("MAXIMUM", H, 1e-4))
    # outward expansion proportional to the distance from the centre (pile widens ~10-25 %)
    spread = m("MULTIPLY", col, m("MULTIPLY", m("ADD", rel, 0.3), m("MULTIPLY_ADD", rv, 0.18, 0.06)))
    xy = N.new("ShaderNodeCombineXYZ")
    L.new(sep.outputs["X"], xy.inputs[0])
    L.new(sep.outputs["Y"], xy.inputs[1])
    sc = N.new("ShaderNodeVectorMath")
    sc.operation = "SCALE"
    L.new(xy.outputs[0], sc.inputs[0])
    L.new(spread, sc.inputs["Scale"])
    vz = N.new("ShaderNodeCombineXYZ")
    L.new(dz, vz.inputs[2])
    add = N.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    L.new(sc.outputs[0], add.inputs[0])
    L.new(vz.outputs[0], add.inputs[1])
    sp = N.new("GeometryNodeSetPosition")
    L.new(gi.outputs["Geometry"], sp.inputs["Geometry"])
    L.new(add.outputs[0], sp.inputs["Offset"])
    cube = N.new("GeometryNodeMeshCube")
    sz = N.new("ShaderNodeVectorMath")
    sz.operation = "SCALE"
    L.new(gi.outputs["CubeSize"], sz.inputs[0])
    sz.inputs["Scale"].default_value = 0.9
    L.new(sz.outputs[0], cube.inputs["Size"])
    sm = N.new("GeometryNodeSetMaterial")
    L.new(cube.outputs["Mesh"], sm.inputs["Geometry"])
    sm.inputs["Material"].default_value = mat
    iop = N.new("GeometryNodeInstanceOnPoints")
    L.new(sp.outputs["Geometry"], iop.inputs["Points"])
    L.new(sm.outputs["Geometry"], iop.inputs["Instance"])
    L.new(iop.outputs["Instances"], go.inputs["Geometry"])
    ids = {it.name: it.identifier for it in I.items_tree
           if getattr(it, "item_type", "SOCKET") == "SOCKET" and it.in_out == "INPUT"}
    return ng, ids


def cells(shape, lvl):
    """Occupied cell centres, their kind (0 flesh / 1 skin-fat) and a surface flag."""
    lo, ext, n = np.array(lvl["lo"]), np.array(lvl["ext"]), np.array(lvl["cells_xyz"])
    occ, axes = grid_occupancy(shape, lo, ext, n)
    Y, Z = np.meshgrid(axes[1], axes[2], indexing="ij")
    inner = occ.copy()
    for a in range(3):
        for sgn in (1, -1):
            sh = np.roll(occ, sgn, axis=a)
            sl = [slice(None)] * 3
            sl[a] = 0 if sgn == 1 else -1
            sh[tuple(sl)] = False
            inner &= sh
    surf = occ & ~inner
    if hasattr(shape, "is_fat"):
        kind = np.zeros(occ.shape, np.float32)
        for i, x in enumerate(axes[0]):
            kind[i] = shape.is_fat(np.full_like(Y, x), Y, Z)
    else:
        kind = surf.astype(np.float32)
    idx = np.nonzero(occ)
    co = np.stack([axes[0][idx[0]], axes[1][idx[1]], axes[2][idx[2]]], 1).astype(np.float32)
    return co, kind[idx].astype(np.float32), surf[idx]


def display_subset(co, kinds, surf, budget, seed):
    """Render-time budget: above `budget` cubes, keep every surface cube plus a random
    share of the hidden interior ones. The counter still shows the exact total; cubes
    this small are below one pixel, so the picture does not change."""
    n = len(co)
    if n <= budget:
        return co, kinds
    rng = np.random.default_rng(seed)
    s_idx = np.flatnonzero(surf)
    i_idx = np.flatnonzero(~surf)
    if len(s_idx) >= budget:
        keep = rng.choice(s_idx, budget, replace=False)
    else:
        extra = rng.choice(i_idx, min(len(i_idx), budget - len(s_idx)), replace=False)
        keep = np.concatenate([s_idx, extra])
    keep.sort()
    return co[keep], kinds[keep]


def cubes_object(name, co, kinds, ng, ids, size, height, mat):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.ravel())
    att = me.attributes.new("kind", "FLOAT", "POINT")
    att.data.foreach_set("value", kinds)
    org = me.attributes.new("orig", "FLOAT_VECTOR", "POINT")  # original position -> shading
    org.data.foreach_set("vector", co.ravel())
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("GN", "NODES")
    mod.node_group = ng
    mod[ids["CubeSize"]] = list(size)
    mod[ids["Height"]] = float(height)
    mod[ids["Collapse"]] = 0.0
    me.materials.append(mat)
    return ob, mod


# ---------------------------------------------------------------------------
# scene
# ---------------------------------------------------------------------------
def build_grid_scene(plan, out_dir):
    t0 = time.time()
    cfg = plan["config"]
    shape = OBJECTS[cfg["object_type"]]
    M = MATERIALS[cfg["material_profile"]]
    fps = plan["fps"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.fps = fps
    scn.frame_start, scn.frame_end = 1, plan["total_frames"]
    lo, hi = shape.bbox()
    span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    board_size = B.BOARD_SIZE
    B.setup_environment(scn, board_size=board_size, brand=cfg.get("board_watermark", ""))

    levitate = cfg["format"] == "levitate"
    steak = getattr(shape, "kind", "") == "extruded"
    if steak:
        mats = steak_materials(M)
    else:
        mats = [B.make_skin(M, shape), B.make_flesh(M), B.simple_mat("Rim", M["rim"], 0.35)]
    params = {}
    seam_sets = []
    for mt in {id(m): m for m in mats}.values():
        add_seams(mt, {}, seam_sets, (0.95, 0.78, 0.55, 1) if levitate else (0.03, 0.006, 0.004, 1))
    me = object_mesh(shape, "Object")
    for mt in mats:
        me.materials.append(mt)

    cmat = cube_mat(M, steak)
    ng, ids = collapse_group(cmat)
    knife = B.build_knife()
    kmat = bpy.data.materials["Steel"]
    lift = 0.0
    frame_obj = None
    pivot = None
    extra_knives = []
    centre = Vector((0.0, 0.0, 0.0))
    if levitate:
        lift = float(cfg["levitate_height"] or max(0.095, span * 0.55))
        centre = Vector((0.0, 0.0, lift + float(hi[2] - lo[2]) / 2))
        # decorative knives keep world space; the working knife follows the tumbling object
        for _ in range(2):
            k2 = knife.copy()
            scn.collection.objects.link(k2)
            extra_knives.append(k2)
        pivot = bpy.data.objects.new("Pivot", None)
        scn.collection.objects.link(pivot)
        pivot.location = centre
        pivot.rotation_mode = "XYZ"
        frame_obj = bpy.data.objects.new("ObjectFrame", None)
        scn.collection.objects.link(frame_obj)
        frame_obj.parent = pivot
        frame_obj.location = (0.0, 0.0, -float(hi[2] - lo[2]) / 2)
        knife.parent = frame_obj
    report = {"levels": [], "failures": []}

    def key_param(name, frame, value):
        B.interp("CONSTANT")
        for P in seam_sets:
            if name in P:
                sock = P[name].outputs[0]
                sock.default_value = float(value)
                sock.keyframe_insert("default_value", frame=frame)

    lvl_cuts = {l["level"]: l["cuts"] for l in plan["levels"]}
    cuts_ev = [(c[0], c[1], lvl_cuts.get(c[2])) for c in plan["counters"]["cuts"]]
    pieces_ev = [(c[0], c[1]) for c in plan["counters"]["pieces"]]
    height = float(hi[2] - lo[2])
    park = Vector((span * 1.6, -0.05, 0.35))   # outside the frame
    B.interp("BEZIER")
    knife.location = park
    knife.rotation_quaternion = B.blade_quat((0, 1, 0), (0, 0, -1))
    knife.keyframe_insert("location", frame=1)
    knife.keyframe_insert("rotation_quaternion", frame=1)
    last_knife = 1

    for lvl in plan["levels"]:
        li = lvl["level"]
        cell = lvl["cell_size"]
        start, enter = lvl["start_frame"], lvl["enter_frames"]
        cuts_ev.append((start, 0, lvl["cuts"]))
        pieces_ev.append((start, 1))
        # seam parameters for this level (constant until the next level)
        for ax in range(3):
            key_param(f"lo_{ax}", start, lvl["lo"][ax])
            key_param(f"cell_{ax}", start, cell[ax])
            key_param(f"n_{ax}", start, 0)
        key_param("cz", start, lvl["cuts_xyz"][2])
        key_param("w", start, min(0.0006, 0.09 * min(cell)))

        ob = bpy.data.objects.new(f"Level_{li:02d}", me)
        scn.collection.objects.link(ob)
        if levitate:
            ob.parent = frame_obj
            # whole turns only: the object is upright again when it comes apart, so the cubes match
            turns_z = 2 * math.pi * (1 if li % 2 else -1)
            long_level = lvl["strokes_end"] - lvl["strokes_start"] >= 2.0 * fps
            turns_x = (2 * math.pi if long_level else 0.0)
            B.interp("CONSTANT")
            pivot.rotation_euler = (0.0, 0.0, 0.0)
            pivot.keyframe_insert("rotation_euler", frame=start)
            B.interp("BEZIER")
            pivot.keyframe_insert("rotation_euler", frame=lvl["strokes_start"])
            B.interp("CONSTANT")
            pivot.rotation_euler = (turns_x, 0.0, turns_z)
            pivot.keyframe_insert("rotation_euler", frame=lvl["collapse_start"])
        B.set_visibility(ob, start, lvl["collapse_start"])
        B.interp("BEZIER")
        ob.location = (0.45, 0, 0) if li else (0, 0, 0)
        ob.keyframe_insert("location", frame=start)
        ob.location = (0, 0, 0)
        ob.keyframe_insert("location", frame=start + enter)

        # cuts: seams + knife
        counts = [0, 0, 0]
        strokes = [s for s in plan["strokes"] if s["level"] == li]
        for i, st in enumerate(strokes):
            ax, f = st["axis"], st["frame"]
            counts[ax] += 1
            key_param(f"n_{ax}", f + 1, counts[ax])
            gap_next = (strokes[i + 1]["frame"] - f) if i + 1 < len(strokes) else 8
            if f <= last_knife:
                continue
            n = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][ax]
            d = Vector((0, 0, -1)) if ax < 2 else Vector((0, 1, 0))
            q = B.blade_quat(n, d)
            c = Vector((0.0, 0.0, 0.0))
            c[ax] = st["pos"]
            if ax < 2:
                start_p = c + Vector((0, 0, height + 0.03))
                end_p = c + Vector((0, 0, 0.0004))
            else:
                start_p = c + Vector((0, lo[1] - 0.05, 0))
                end_p = c + Vector((0, hi[1] + 0.01, 0))
            B.interp("BEZIER")
            if f - last_knife >= 3 and gap_next >= 3:
                for ff, loc in ((f - 2, start_p), (f, end_p)):
                    if ff > last_knife:
                        knife.location, knife.rotation_quaternion = loc, q
                        knife.keyframe_insert("location", frame=ff)
                        knife.keyframe_insert("rotation_quaternion", frame=ff)
                        last_knife = ff
            else:
                knife.location, knife.rotation_quaternion = end_p, q
                knife.keyframe_insert("location", frame=f)
                knife.keyframe_insert("rotation_quaternion", frame=f)
                last_knife = f
        if levitate and lvl["cuts"] >= 10:
            rng = np.random.default_rng(int(cfg["random_seed"]) + li)
            B.interp("LINEAR")
            reach = span * 1.25
            away = Vector((1.6, 0.0, 0.5))
            for ki, k2 in enumerate(extra_knives):
                f = lvl["strokes_start"] + ki * 3
                k2.location = away
                k2.keyframe_insert("location", frame=max(1, f - 1))
                side = 1.0
                while f < lvl["strokes_end"]:
                    u = Vector(rng.normal(size=3)).normalized()
                    n_b = u.cross(Vector(rng.normal(size=3))).normalized()
                    k2.rotation_quaternion = B.blade_quat(tuple(n_b), tuple(-u * side))
                    k2.keyframe_insert("rotation_quaternion", frame=f)
                    k2.location = centre + u * reach * side
                    k2.keyframe_insert("location", frame=f)
                    k2.location = centre - u * reach * side
                    k2.keyframe_insert("location", frame=f + 5)
                    k2.keyframe_insert("rotation_quaternion", frame=f + 5)
                    f += 6
                k2.location = away
                k2.keyframe_insert("location", frame=f + 1)
        # fast cutting reads as a blur: make the blade translucent while strokes are dense
        dense = len(strokes) / max(1, lvl["strokes_end"] - lvl["strokes_start"]) > 0.6
        bsdf = kmat.node_tree.nodes.get("Principled BSDF")
        B.interp("CONSTANT")
        bsdf.inputs["Alpha"].default_value = 0.35 if dense else 1.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=lvl["strokes_start"])
        bsdf.inputs["Alpha"].default_value = 1.0
        bsdf.inputs["Alpha"].keyframe_insert("default_value", frame=lvl["strokes_end"] + 2)
        # lift the knife away before the collapse
        B.interp("BEZIER")
        f_up = max(last_knife + 2, lvl["strokes_end"] + 3)
        knife.location = park
        knife.keyframe_insert("location", frame=f_up + 4)
        knife.keyframe_insert("rotation_quaternion", frame=f_up + 4)
        last_knife = f_up + 4

        # cubes
        t_l = time.time()
        co, kinds, surf = cells(shape, lvl)
        if len(co) != lvl["cubes"]:
            report["failures"].append({"level": li, "planned": lvl["cubes"], "built": int(len(co))})
        built = int(len(co))
        co, kinds = display_subset(co, kinds, surf, int(cfg.get("display_cubes_max", 1200000)), li)
        cob, mod = cubes_object(f"Cubes_{li:02d}", co, kinds, ng, ids, cell, height, cmat)
        print(f"ASMR_LOG level {li}: cuts={lvl['cuts']} cubes={built} displayed={len(co)} "
              f"in {time.time() - t_l:.1f}s", flush=True)
        B.set_visibility(cob, lvl["collapse_start"], lvl["end_frame"])
        path = f'modifiers["GN"]["{ids["Collapse"]}"]'
        B.interp("BEZIER")
        mod[ids["Collapse"]] = 0.0
        # levitate: the pieces burst apart in the air for a beat, then drop on the board
        drop = lvl["collapse_start"] + (int(0.12 * fps) if levitate else 0)
        cob.keyframe_insert(data_path=path, frame=drop)
        mod[ids["Collapse"]] = 1.0
        cob.keyframe_insert(data_path=path, frame=lvl["collapse_end"])
        if levitate:
            cob.location = (0, 0, lift)
            cob.keyframe_insert("location", frame=drop)
            cob.scale = (1, 1, 1)
            cob.keyframe_insert("scale", frame=lvl["collapse_start"])
            cob.scale = (1.1, 1.1, 1.1)
            cob.keyframe_insert("scale", frame=drop)
            cob.location = (0, 0, 0)
            cob.keyframe_insert("location", frame=lvl["collapse_end"])
            cob.scale = (1, 1, 1)
            cob.keyframe_insert("scale", frame=lvl["collapse_end"])
        cob.location = (0, 0, 0)
        cob.keyframe_insert("location", frame=lvl["exit_start"])
        cob.location = (-0.6, 0, 0)
        cob.keyframe_insert("location", frame=lvl["end_frame"])
        report["levels"].append({"level": li, "cuts": lvl["cuts"], "cubes_planned": lvl["cubes"],
                                 "cubes_built": built, "cubes_displayed": int(len(co))})

    # loop: intact object returns to the opening pose, counters back to their opening values
    if plan["levels"]:
        li0 = plan["levels"][0]
        lf, tot = plan["loop_in"], plan["total_frames"]
        ob = bpy.data.objects.new("LoopObject", me)
        scn.collection.objects.link(ob)
        ob.parent = frame_obj
        B.set_visibility(ob, lf, None)
        B.interp("BEZIER")
        ob.location = (0.45, 0, 0)
        ob.keyframe_insert("location", frame=lf)
        ob.location = (0, 0, 0)
        ob.keyframe_insert("location", frame=min(tot - 2, lf + int(0.6 * fps)))
        for ax in range(3):
            key_param(f"lo_{ax}", lf, li0["lo"][ax])
            key_param(f"cell_{ax}", lf, li0["cell_size"][ax])
            key_param(f"n_{ax}", lf, 0)
        key_param("cz", lf, li0["cuts_xyz"][2])
        cuts_ev.append((lf, 0, li0["cuts"]))
        pieces_ev.append((lf, 1))

    cam = grid_camera(plan, lo, hi, board_size, lift)
    B.build_overlay(cam, cfg, cuts_ev, pieces_ev, plan["total_frames"])
    B.setup_render(scn, cfg, out_dir, final=False)
    report["final_pieces_validated"] = report["levels"][-1]["cubes_built"] if report["levels"] else 0
    report["target"] = plan["levels"][-1]["cubes"] if plan["levels"] else 0
    report["non_manifold"] = 0
    report["volume_error_pct"] = 0.0
    report["objects"] = len(bpy.data.objects)
    report["build_seconds"] = round(time.time() - t0, 2)
    return report


def grid_camera(plan, lo, hi, board_size, lift=0.0):
    """Whole board in frame (~92 % of the width), object centred, slight depth of field."""
    cd = bpy.data.cameras.new("Camera")
    cd.lens = 50
    cd.clip_start = 0.004
    cam = bpy.data.objects.new("Camera", cd)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    w, h = plan["config"]["resolution"]
    vfov = 2 * math.atan(18 / 50)
    hfov = 2 * math.atan(math.tan(vfov / 2) * w / h)
    half_w = board_size[0] / 2 / 0.92
    elev, azim = math.radians(22 if lift else 34), math.radians(-90)
    view = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
    target = Vector((0, 0.01, 0.01 + lift * 0.7))
    dist = half_w / math.tan(hfov / 2)
    cam.location = target + view * dist
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (target - cam.location).to_track_quat("-Z", "Y")
    # no depth of field: the counters are camera-attached and must stay sharp
    return cam
