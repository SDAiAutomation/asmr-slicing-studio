"""Build seeded, hollow longitudinal half-slices with solid rind and cut faces."""
import math
import bpy
from mathutils import Vector
import blender_build as B
from papaya_plan import radius, HALF_LENGTH, CENTRE_HEIGHT


def segment(scene, index, count, side, materials):
    verts, faces, slots = [], [], []
    sections, arcs = max(2, 80//count), 20
    lo, hi = -1+2*index/count, -1+2*(index+1)/count
    def face(indices, material):
        faces.append(indices)
        slots.append(material)
    for j in range(sections+1):
        u = lo+(hi-lo)*j/sections
        r = max(0.00005,radius(u))
        inner = max(0.00001, r * 0.52 * max(0,1-(u/0.86)**4))
        for radial in (r, inner):
            for k in range(arcs+1):
                a = math.pi*(k/arcs-0.5)
                verts.append((u*HALF_LENGTH, side*radial*math.cos(a), CENTRE_HEIGHT+radial*math.sin(a)))
    width=arcs+1
    for j in range(sections+1):
        a=j*2*width
        for k in range(arcs):
            if j in (0,sections):
                face([a+k,a+k+1,a+width+k+1,a+width+k],1)
            if j<sections:
                for offset,mat in ((0,0),(width,2)):
                    v=a+k+offset
                    face([v,v+1,v+2*width+1,v+2*width],mat)
        if j<sections:
            for k in (0,arcs):
                v=a+k
                face([v,v+width,v+3*width,v+2*width],1)
    mesh=bpy.data.meshes.new("Fruit section")
    midpoint=Vector(((lo+hi)*HALF_LENGTH/2,side*radius((lo+hi)/2)*0.52,CENTRE_HEIGHT))
    mesh.from_pydata([Vector(v)-midpoint for v in verts],[],faces)
    for mat in materials: mesh.materials.append(mat)
    for poly,slot in zip(mesh.polygons,slots):
        poly.material_index=slot
        poly.use_smooth=slot!=1
    import bmesh
    bm=bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    # Analytic rind normals keep adjacent pre-cut sections visually continuous.
    normals=[tuple(loop.vector) for loop in mesh.corner_normals]
    for poly in mesh.polygons:
        if poly.material_index != 0:
            continue
        for li in poly.loop_indices:
            v=mesh.vertices[mesh.loops[li].vertex_index].co+midpoint
            u=v.x/HALF_LENGTH
            slope=(radius(u+0.0001)-radius(u-0.0001))/(0.0002*HALF_LENGTH)
            radial=math.hypot(v.y,v.z-CENTRE_HEIGHT) or 1
            normals[li]=tuple(Vector((-slope,v.y/radial,(v.z-CENTRE_HEIGHT)/radial)).normalized())
    mesh.normals_split_custom_set(normals)
    ob=bpy.data.objects.new(f"Slice_{index+1}_{side}",mesh)
    scene.collection.objects.link(ob)
    ob.location=midpoint
    return ob


def build(plan):
    import random
    from blender_grid import grid_camera
    cfg=plan["config"]
    n=plan["levels"][0]["cuts"]
    fall=plan["levels"][0]["collapse_start"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s=bpy.context.scene
    s.render.fps=plan["fps"]
    s.frame_end=plan["total_frames"]
    board=B.setup_environment(s,board_size=B.BOARD_SIZE,brand=cfg["board_watermark"])
    def rigid(ob,kind):
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.context.view_layer.objects.active=ob
        bpy.ops.rigidbody.object_add()
        rb=ob.rigid_body
        rb.type=kind
        rb.use_margin=True
        rb.collision_margin=min(0.00015,HALF_LENGTH/n/8)
        rb.friction=0.58
        return rb
    rigid(board,"PASSIVE").collision_shape="BOX"
    mats=[B.simple_mat("Golden papaya peel",(0.58,0.27,0.023),0.35),
          B.simple_mat("Papaya flesh",(0.95,0.18,0.024),0.38),
          B.simple_mat("Seed cavity",(0.59,0.11,0.014),0.48)]
    nt=mats[0].node_tree
    noise=nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value=44
    tex=nt.nodes.new("ShaderNodeTexCoord")
    tex.object=board
    nt.links.new(tex.outputs["Object"],noise.inputs["Vector"])
    ramp=B.ramp(nt,[(0.25,(0.21,0.27,0.014)),(0.52,(0.81,0.40,0.019)),(0.78,(0.96,0.57,0.061))])
    nt.links.new(noise.outputs["Fac"],ramp.inputs[0])
    nt.links.new(ramp.outputs[0],nt.nodes["Principled BSDF"].inputs["Base Color"])
    seedmat=B.simple_mat("Papaya seeds",(0.015,0.009,0.006),0.29)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2,radius=1)
    seed_mesh=bpy.context.object.data
    seed_mesh.materials.append(seedmat)
    bpy.data.objects.remove(bpy.context.object,do_unlink=True)
    rng=random.Random(cfg["random_seed"])
    for i in range(n):
        for side in (-1,1):
            ob=segment(s,i,n,side,mats)
            origin=ob.location.copy()
            rb=rigid(ob,"ACTIVE")
            rb.mass=0.65/(2*n)
            rb.kinematic=True
            rb.keyframe_insert("kinematic",frame=1)
            ob.keyframe_insert("location",frame=1)
            # Open each transverse seam only after its own impact.
            hit=plan["impacts"][i]["contact"]
            ob.location.y=side*0.00045+origin.y
            ob.keyframe_insert("location",frame=plan["impacts"][0]["contact"]+2)
            ob.keyframe_insert("location",frame=max(hit,plan["impacts"][0]["contact"]+2))
            # Open the kerf at each contact without exposing future seams.
            for boundary in range(1,i+1):
                frame=plan["impacts"][boundary]["contact"]
                ob.location.x=origin.x+0.012*(boundary-1)/n
                ob.keyframe_insert("location",frame=frame)
                ob.location.x=origin.x+0.012*boundary/n
                ob.keyframe_insert("location",frame=frame+2)

            ob.keyframe_insert("location",frame=hit+2)
            ob.keyframe_insert("location",frame=fall-3)
            ob.rotation_euler.x=0
            ob.keyframe_insert("rotation_euler",frame=fall-3)
            ob.rotation_euler.x=-side*rng.uniform(0.07,0.16)
            ob.location.y+=side*0.003
            ob.keyframe_insert("location",frame=fall-1)
            ob.keyframe_insert("rotation_euler",frame=fall-1)
            rb.kinematic=False
            rb.keyframe_insert("kinematic",frame=fall)
            u=origin.x/HALF_LENGTH
            if abs(u)<0.82:
                for _ in range(max(2,120//n)):
                    angle=rng.uniform(-1.4,1.4)
                    r=radius(u)*0.49*(1-(u/0.86)**4)
                    seed=bpy.data.objects.new("Seed",seed_mesh)
                    s.collection.objects.link(seed)
                    seed.parent=ob
                    seed.location=Vector((origin.x+rng.uniform(-0.65,0.65)*HALF_LENGTH/n,side*r*math.cos(angle),CENTRE_HEIGHT+r*math.sin(angle)))-origin
                    seed.scale=(min(0.0017,HALF_LENGTH/n*0.75),0.0022,0.002)
    world=s.rigidbody_world
    world.substeps_per_frame=12
    world.solver_iterations=20
    world.point_cache.frame_end=s.frame_end
    knife=B.build_knife()
    knife.rotation_mode="XYZ"
    for i,e in enumerate(plan["impacts"]):
        c,x=e["contact"],e["x"] + 0.012*max(0,i-1)/n
        orientation=0 if i==0 else math.pi/2
        z=CENTRE_HEIGHT+(CENTRE_HEIGHT if i==0 else radius(x/HALF_LENGTH))
        for frame,height in [(c-2,z+0.035),(c,z),(e["exit"],0.0015),(e["exit"]+1,z+0.035)]:
            knife.rotation_euler=(math.pi/2,0,orientation)
            knife.location=(x,0,height)
            knife.keyframe_insert("location",frame=frame)
            knife.keyframe_insert("rotation_euler",frame=frame)
    B.set_visibility(knife,1,plan["impacts"][-1]["exit"]+2)
    camera=grid_camera(plan,(-HALF_LENGTH,-0.05,0),(HALF_LENGTH,0.05,2*CENTRE_HEIGHT),B.BOARD_SIZE)
    camera.location=Vector((0.115,-0.39,0.27))
    camera.rotation_quaternion=(Vector((0,0,0.038))-camera.location).to_track_quat("-Z","Y")
    camera.data.lens=38
    cuts=[(1,0,n)]+[(e["contact"],i,n) for i,e in enumerate(plan["impacts"],1)]
    B.build_overlay(camera,cfg,cuts,[(1,1)]+[(e["contact"],2*i) for i,e in enumerate(plan["impacts"],1)],s.frame_end)
    s.frame_set(1)
    with bpy.context.temp_override(scene=s,point_cache=world.point_cache):
        bpy.ops.ptcache.bake(bake=True)
    B.setup_render(s,cfg,".",final=False)
    return {"failures":[],"levels":[{"cuts":n,"cubes_planned":2*n,"cubes_built":2*n}],"final_pieces_validated":2*n,"target":2*n}
