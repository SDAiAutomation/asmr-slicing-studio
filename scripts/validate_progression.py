"""Read-only consistency checks for SliceLoop grid and slices plans.

This module does not verify rendered blade contact or physical plausibility.
"""
from collections import Counter


def validate_plan(plan):
    errors = []
    stats = plan.get("stats", {})
    if plan.get("kind") == "grid":
        strokes = plan.get("strokes", [])
        blades = plan.get("blade_strokes", [])
        levels = plan.get("levels", [])
        if stats.get("blade_strokes") != len(blades):
            errors.append("grid: blade_strokes must count animated blade passes")
        if stats.get("planned_cuts") != len(strokes):
            errors.append("grid: planned_cuts must count planned cut events")
        by_level = Counter(s["level"] for s in strokes)
        passes = Counter(b["level"] for b in blades)
        for level in levels:
            i = level["level"]
            if by_level[i] != level["cuts"]:
                errors.append(f"grid level {i}: planned cuts mismatch")
            if passes[i] != level.get("visible_blade_passes"):
                errors.append(f"grid level {i}: visible blade passes mismatch")
            if level.get("grouped_cut_count", 0) > by_level[i]:
                errors.append(f"grid level {i}: impossible grouped count")
    elif "hits" in plan and "strokes" in plan:
        hits = plan["hits"]
        strokes = plan["strokes"]
        assigned = [idx for s in strokes for idx in s.get("hits", [])]
        if sorted(assigned) != list(range(len(hits))):
            errors.append("slices: each hit must be assigned exactly once")
        for i, stroke in enumerate(strokes):
            n = len(stroke.get("hits", []))
            if stroke.get("cut_planes") != n:
                errors.append(f"slices stroke {i}: cut_planes mismatch")
            if stroke.get("grouped_hits") != max(0, n - 1):
                errors.append(f"slices stroke {i}: grouped_hits mismatch")
        if stats.get("cut_planes") != len(hits):
            errors.append("slices: total cut_planes mismatch")
        if stats.get("grouped_cut_planes") != sum(max(0, len(s.get("hits", [])) - 1) for s in strokes):
            errors.append("slices: grouped_cut_planes mismatch")
        blades = plan.get("blade_strokes", [])
        if "blade_strokes" in stats and stats["blade_strokes"] != len(blades):
            errors.append("slices: animated blade pass statistic mismatch")
        if "planned_strokes" in stats and stats["planned_strokes"] != len(strokes):
            errors.append("slices: planned stroke statistic mismatch")
        if sum(b.get("group_cuts", 0) for b in blades) > len(hits):
            errors.append("slices: blade group coverage exceeds actual cuts")
    else:
        errors.append("unsupported plan format")
    return errors
