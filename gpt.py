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
MIN_ROOM_AREA_SQFT = 10.0  # ignore polygons smaller than this
ARC_DEG_PER_SEG = 10.0     # seg length ≈ R·π·deg/180
MIN_SEG_LEN_FT  = 0.05     # discard super-short fragments
###############################################################################

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
        m44 = blk.matrix44()
        for sub in blk.virtual_entities():
            sub.transform(m44)
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

    # elif t in {"ELLIPSE", "SPLINE"}:
    #     for pts in ent.approximate(ARC_DEG_PER_SEG):
    #         ls = LineString(pts)
    #         if ls.length >= MIN_SEG_LEN_FT:
    #             out.append(ls)


def _collect_segments(doc):
    segs: List[LineString] = []
    for e in doc.modelspace():
        _explode_entity(e, segs)
    return segs

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


def debug_plot(model, *, show_dangling=True, save: Optional[str | Path] = None):
    if _plt is None:
        raise RuntimeError("matplotlib missing — pip install matplotlib")

    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    net = _build_network(_collect_segments(doc))
    rooms, dangles, cuts = _room_polys(net)

    # vertices for scatter plot
    verts = list({(round(x, 6), round(y, 6))
                  for ls in (net.geoms if isinstance(net, MultiLineString) else [net])
                  for x, y in ls.coords})

    fig, ax = _plt.subplots()

    # walls
    for ls in (net.geoms if isinstance(net, MultiLineString) else [net]):
        ax.plot(*ls.xy, color="0.7", linewidth=1)

    # vertices
    if verts:
        xs, ys = zip(*verts)
        ax.scatter(xs, ys, s=5, color="black", zorder=3)

    # rooms
    for poly in rooms:
        ax.plot(*poly.exterior.xy, linewidth=2)

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
    ns = parser.parse_args()

    if ns.plot is None:
        for rid, area in count_rooms(ns.dxf):
            print(f"Room {rid}: {area:.2f} ft²")
    else:
        dest = None if ns.plot == "_show_" else ns.plot
        debug_plot(ns.dxf, save=dest)
