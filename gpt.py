"""
DXF Room Counter — Vertex Plot + Proper T‑Junction Noding
=========================================================
We **snap _first_ then node**, so a T‑stem end that’s merely *close* to a through‐
wall gets pulled onto it (gap ≤ `GAP_TOLERANCE`) **before** we insert the new
vertex.  This guarantees the resulting graph is truly planar.

Pipeline
--------
1. **Explode** every entity (blocks, curves, …) into straight `LineString`s.
2. **Snap** endpoints to nearby segments (≤ tolerance).  → endpoints land *on* the
   through‑wall in a T.
3. **Node** the snapped set → splits at every intersection (including those new
   T‑points) and returns a perfectly planar `MultiLineString`.
4. **Polygonize** & filter → rooms.
5. **Debug plot** shows grey walls, black vertices, red/orange dangles/cuts, room
   outlines.

No layer logic, no type alias fuss.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Optional

import ezdxf
from ezdxf.upright import upright_all
from ezdxf.entities import DXFGraphic, Insert
from shapely.geometry import LineString, MultiLineString
from shapely.ops import polygonize_full, snap
from shapely import node

try:
    import matplotlib.pyplot as _plt
except ImportError:  # pragma: no cover
    _plt = None  # type: ignore[assignment]

###############################################################################
# Tweakables
###############################################################################
DXF_UNIT_TO_FOOT = 1.0     # 1/12.0 if drawing is in inches
GAP_TOLERANCE   = 0.01     # ft — bridge endpoint gaps ≤ this (raised a bit)
MIN_ROOM_AREA_SQFT = 5.0  # ignore polygons smaller than this
ARC_DEG_PER_SEG = 10.0     # seg length ≈ R·π·deg/180
MIN_SEG_LEN_FT  = 0.01     # discard super-short fragments

###############################################################################
# Door-swing heuristic (works for ARC **and** SPLINE)
###############################################################################
DOOR_ARC_MIN_ANG        = 80.0      # deg   – arc sweep we expect
DOOR_ARC_MAX_ANG        = 100.0
DOOR_CHORD_MIN_FT       = 0.1       # ft    – doorway width ≈ chord length
DOOR_CHORD_MAX_FT       = 10.5
ARCLEN_CHORD_RATIO_MIN  = 0.07      # len(arc)/len(chord) for 80–100°
ARCLEN_CHORD_RATIO_MAX  = 5.25

# ---------------------------------------------------------------------------
# Geometry explosion helpers
# ---------------------------------------------------------------------------

def _drange(start, stop, step):
    while start < stop:
        yield start
        start += step


def _arc_as_lines(cx, cy, r, a0, a1):
    if a1 < a0:
        a1 += 360.0
    step = max(1e-6, ARC_DEG_PER_SEG)
    prev = (cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
    for ang in _drange(a0 + step, a1 + 1e-9, step):
        pt = (cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang)))
        ls = LineString([prev, pt])
        if ls.length >= MIN_SEG_LEN_FT:
            yield ls
        prev = pt


def _explode_entity(ent: DXFGraphic, out: List[LineString]):
    t = ent.dxftype()
    if t == "INSERT":
        blk: Insert = ent
        # m44 = blk.matrix44()
        for sub in blk.virtual_entities():
            # sub.transform(m44)
            upright_all([sub])
            _explode_entity(sub, out)
        return

    if t == "LINE":
        ls = LineString([(ent.dxf.start.x, ent.dxf.start.y), (ent.dxf.end.x, ent.dxf.end.y)])
        if ls.length >= MIN_SEG_LEN_FT:
            out.append(ls)

    elif t in {"LWPOLYLINE", "POLYLINE"}:
        pts = [(v[0], v[1]) for v in ent]
        for a, b in zip(pts, pts[1:]):
            ls = LineString([a, b])
            if ls.length >= MIN_SEG_LEN_FT:
                out.append(ls)
        if getattr(ent, "closed", False) and len(pts) > 1:
            ls = LineString([pts[-1], pts[0]])
            if ls.length >= MIN_SEG_LEN_FT:
                out.append(ls)

    elif t == "ARC":
        out.extend(_arc_as_lines(ent.dxf.center.x, ent.dxf.center.y, ent.dxf.radius,
                                 ent.dxf.start_angle, ent.dxf.end_angle))

    elif t == "CIRCLE":
        out.extend(_arc_as_lines(ent.dxf.center.x, ent.dxf.center.y, ent.dxf.radius,
                                 0.0, 360.0))

    elif t in {"ELLIPSE", "SPLINE"}:
        # 1. Grab every vertex the DXF entity produces
        verts_3d = list(ent.flattening(0.1))      # [(x, y, z), …]
        # 2. Shapely 1.x is strictly 2-D, so strip Z; with Shapely 2.x you can skip this
        coords_2d = [(x, y) for x, y, *_ in verts_3d]
        # 3. Make sure we actually have a segment, then build the geometry once
        if len(coords_2d) >= 2:                         # LineString needs ≥ 2 points
            ls = LineString(coords_2d)
            if ls.length >= MIN_SEG_LEN_FT:
                out.append(ls)

def _collect_segments(doc):
    segs: List[LineString] = []
    for e in doc.modelspace():
        _explode_entity(e, segs)
    return segs

# ---------------------------------------------------------------------------
# Door detection (returns doorway chords, not the swing arc)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Door detection  –  returns doorway chords, not the swing curve itself
# ---------------------------------------------------------------------------
def _door_chords(doc) -> List[LineString]:
    chords: List[LineString] = []

    def _push(p0, p1):
        chords.append(LineString([p0, p1]))

    def _looks_like_door_arc(chord_len_ft, ratio):
        return (DOOR_CHORD_MIN_FT < chord_len_ft < DOOR_CHORD_MAX_FT
                and ARCLEN_CHORD_RATIO_MIN < ratio < ARCLEN_CHORD_RATIO_MAX)

    def _walk(ent: DXFGraphic):
        t = ent.dxftype()

        # recurse into blocks
        if t == "INSERT":
            for sub in ent.virtual_entities():
                upright_all([sub])
                _walk(sub)
            return

        # ------------------------------------------------------------------#
        # Plain ARC swing
        # ------------------------------------------------------------------#
        if t == "ARC":
            arc = ent
            sweep = abs(arc.dxf.end_angle - arc.dxf.start_angle)
            if not (DOOR_ARC_MIN_ANG < sweep < DOOR_ARC_MAX_ANG):
                return

            cx, cy, r = arc.dxf.center.x, arc.dxf.center.y, arc.dxf.radius
            a0, a1 = math.radians(arc.dxf.start_angle), math.radians(arc.dxf.end_angle)
            p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            chord_len_ft = LineString([p0, p1]).length * DXF_UNIT_TO_FOOT
            if DOOR_CHORD_MIN_FT < chord_len_ft < DOOR_CHORD_MAX_FT:
                _push(p0, p1)
            return

        # ------------------------------------------------------------------#
        # Spline door swing
        # ------------------------------------------------------------------#
        if t in {"SPLINE", "ELLIPSE"}:
            pts3d = list(ent.flattening(0.1))
            if len(pts3d) < 3:
                return
            pts2d = [(x, y) for x, y, *_ in pts3d]
            p0, p1 = pts2d[0], pts2d[-1]
            chord = LineString([p0, p1])
            chord_len_ft = chord.length * DXF_UNIT_TO_FOOT

            # approximate arc length with polyline length
            arclen_ft = sum(
                math.hypot(bx - ax, by - ay)
                for (ax, ay), (bx, by) in zip(pts2d, pts2d[1:])
            ) * DXF_UNIT_TO_FOOT
            ratio = arclen_ft / chord_len_ft if chord_len_ft else 9e9

            if _looks_like_door_arc(chord_len_ft, ratio):
                _push(p0, p1)

    for e in doc.modelspace():
        _walk(e)
    return chords

# ---------------------------------------------------------------------------
# Graph + room extraction
# ---------------------------------------------------------------------------

def _build_network(segs):
    # 1) snap first so T‑endpoints land on through‑edges
    snapped = snap(MultiLineString(segs), MultiLineString(segs), GAP_TOLERANCE)
    # 2) node to split at every intersection (T + X)
    return node(snapped)


def _room_polys(net):
    polys, dangles, cuts, _ = polygonize_full(net)
    sqft = DXF_UNIT_TO_FOOT ** 2
    rooms = [p for p in polys.geoms if p.area * sqft >= MIN_ROOM_AREA_SQFT]
    return rooms, dangles, cuts

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def count_rooms(model) -> List[Tuple[int, float]]:
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    net = _build_network(_collect_segments(doc))
    rooms, _, _ = _room_polys(net)
    sqft = DXF_UNIT_TO_FOOT ** 2
    return [(i + 1, r.area * sqft) for i, r in enumerate(sorted(rooms, key=lambda p: p.area, reverse=True))]


def debug_plot(model, *,
               show_dangling=True,
               show_doors=False,
               doors_only=False,
               save: Optional[str | Path] = None):
    """
    show_doors=True   → overlay doorway chords in blue
    doors_only=True   → plot ONLY the doors (quick visual check)
    """
    if _plt is None:
        raise RuntimeError("matplotlib missing — pip install matplotlib")

    doc   = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    doors = _door_chords(doc)

    # -- fast path: doors-only ---------------------------------------------
    if doors_only:
        fig, ax = _plt.subplots()
        for d in doors:
            ax.plot(*d.xy, color="blue", linewidth=2)
        ax.set_aspect("equal", "box")
        ax.axis("off")
        (fig.savefig(save, dpi=300, bbox_inches="tight") if save else _plt.show())
        return

    # -- standard room plot -------------------------------------------------
    net   = _build_network(_collect_segments(doc))
    rooms, dangles, cuts = _room_polys(net)

    fig, ax = _plt.subplots()

    # walls
    for ls in (net.geoms if isinstance(net, MultiLineString) else [net]):
        ax.plot(*ls.xy, color="0.7", linewidth=1)

    # rooms
    for poly in rooms:
        ax.plot(*poly.exterior.xy, linewidth=2)

    # door overlay
    if show_doors:
        for d in doors:
            ax.plot(*d.xy, color="blue", linewidth=2)

    if show_dangling:
        for g in dangles.geoms:
            ax.plot(*g.xy, color="red", linestyle="--", linewidth=1)
        for g in cuts.geoms:
            ax.plot(*g.xy, color="orange", linestyle=":", linewidth=1)

    ax.set_aspect("equal", "box")
    ax.axis("off")
    (fig.savefig(save, dpi=300, bbox_inches="tight") if save else _plt.show())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap

    parser = argparse.ArgumentParser(
        prog="dxf_room_counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Count rooms & plot vertices (snap‑then‑node T‑junction aware).

            python dxf_room_counter.py plan.dxf          # list areas
            python dxf_room_counter.py plan.dxf --plot   # walls + vertices plot
            """
        ),
    )
    parser.add_argument("dxf", help="DXF file to analyse")
    parser.add_argument("--plot", nargs="?", const="_show_", help="Show/save plot")
    parser.add_argument("--doors-only", action="store_true",
                        help="Plot only detected doors (blue chords)")
    parser.add_argument("--show-doors", action="store_true",
                        help="Overlay detected doors on the normal plot")
    ns = parser.parse_args()

    if ns.plot is None and not ns.doors_only:
        for rid, area in count_rooms(ns.dxf):
            print(f"Room {rid}: {area:.2f} ft²")
    else:
        dest = None if (ns.plot == "_show_" or ns.plot is None) else ns.plot
        debug_plot(ns.dxf,
                  show_dangling=not ns.doors_only,
                  show_doors=ns.show_doors,
                  doors_only=ns.doors_only,
                  save=dest)
