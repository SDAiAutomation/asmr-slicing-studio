"""Non-rendering consistency checks for SliceLoop's progressive plans.

This checks metadata and event accounting, not whether a blade actually intersects
the rendered geometry. Blender frame inspection remains mandatory.
"""


def validate_plan(plan):
    """Return a list of actionable consistency errors (empty means metadata passes)."""
    errors = []
    if plan.get("kind") == "grid":
        levels = plan.get("levels", [])
        blades = plan.get("blade_strokes", [])
        strokes = plan.get("strokes", [])
        if plan.get("stats", {}).get("blade_strokes") != len(blades):
            errors.append("grid: blade_strokes statistic differs from animated passes")
        if plan.get("stats", {}).get("planned_cuts") != len(strokes):
            errors.append("grid: planned_cuts statistic differs from stroke events")
        for i, level in enumerate(levels):
            cut_count = sum(s.get("level") == i for s in strokes)
            passes = sum(s.get("level") == i for s in blades)
            if cut_count != level.get("cuts"):
                errors.append(f"grid level {i}: cuts count does not match planned events")
            if passes != level.get("visible_blade_passes"):
                errors.append(f"grid level {i}: visible blade pass count mismatch")
            if level.get("max_cuts_per_pass", 0) < (2 if level.get("grouped_cut_count", 0) else (1 if cut_count else 0)):
                errors.append(f"grid level {i}: grouping metadata is inconsistent")
    elif "hits" in plan and "strokes" in plan:
        hits = plan["hits"]
        strokes = plan["strokes"]
        assigned = [h for s in strokes for h in s.get("hits", [])]
        if sorted(assigned) != list(range(len(hits))):
            errors.append("slices: hit indices must each occur exactly once")
        for i, stroke in enumerate(strokes):
            n = len(stroke.get("hits", []))
            if stroke.get("cut_planes") != n:
                errors.append(f"slices stroke {i}: cut_planes mismatch")
            if stroke.get("grouped_hits") != max(0, n - 1):
                errors.append(f"slices stroke {i}: grouped_hits mismatch")
        stats = plan.get("stats", {})
        if stats.get("cut_planes") != len(hits):
            errors.append("slices: cut_planes statistic mismatch")
        if stats.get("grouped_cut_planes") != sum(max(0, len(s.get("hits", [])) - 1) for s in strokes):
            errors.append("slices: grouped_cut_planes statistic mismatch")
    else:
        errors.append("unsupported plan format")
    return errors
