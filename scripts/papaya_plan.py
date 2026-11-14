"""Papaya guide timeline. Positions are metres and Blender frames are one-based."""
import math


def radius(u):
    return 0.048 * math.sqrt(max(0, 1 - u*u)) * (1 - 0.17*u)


HALF_LENGTH = 0.1
CENTRE_HEIGHT = max(radius(i/1000) for i in range(-1000,1001))


def plan(cfg):
    count, fps = cfg["cuts_sequence"][0], cfg["fps"]
    frame = lambda t: 1 + round(t * fps / 24)
    impacts = []
    for i in range(count):
        x = 0 if i == 0 else HALF_LENGTH * (2*i/count - 1)
        contact = frame(28 + 6*i)
        impacts.append(dict(axis="lengthwise" if i == 0 else "crosswise", x=x,
                            contact=contact, exit=frame(30 + 6*i)))
    last = 28 + 6*(count-1)
    fall, end = frame(last+20), frame(last+97)
    level = dict(level=0, cuts=count, cubes=count*2, start_frame=1,
                 strokes_start=impacts[0]["contact"], strokes_end=impacts[-1]["exit"],
                 collapse_start=fall, collapse_end=frame(last+70), end_frame=end)
    blades = [dict(group_cuts=1, anchor=[e["x"],0,0], time=e) for e in impacts]
    return dict(kind="papaya_slices",config=cfg,fps=fps,levels=[level],impacts=impacts,
                strokes=[dict(level=0,axis=1 if i==0 else 0,frame=e["contact"],n_in_level=i+1) for i,e in enumerate(impacts)],
                blade_strokes=blades,total_frames=end,duration_s=end/fps,
                stats=dict(levels=1,blade_strokes=count,final_cubes=count*2))
