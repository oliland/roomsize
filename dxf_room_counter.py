"""
DXF Room Counter — Vertex Plot + Proper T‑Junction Noding + Furniture Detection
===============================================================================
We **snap _first_ then node**, so a T‑stem end that's merely *close* to a through‐
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

Now also collects furniture polygons separately instead of discarding them.
"""
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np 
import matplotlib.pyplot as _plt

import ezdxf
from ezdxf.upright import upright_all
from ezdxf.entities import DXFGraphic, Insert
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize_full, snap
from shapely import node

###############################################################################
# Tweakables
###############################################################################
DXF_UNIT_TO_FOOT = 1.0     # 1/12.0 if drawing is in inches
GAP_TOLERANCE   = 0.05     # ft — bridge endpoint gaps ≤ this (raised a bit)
MIN_ROOM_AREA_SQFT = 11.0  # ignore polygons smaller than this
MAX_ROOM_AREA_SQFT = 500.0  # ignore polygons larger than this
ARC_DEG_PER_SEG = 8.0     # seg length ≈ R·π·deg/180
MIN_SEG_LEN_FT  = 0.05     # discard super-short fragments

# Furniture detection parameters
FURNITURE_MAX_AREA_SQFT = 0.5      # ft² — closed polygons smaller than this might be furniture
TABLE_MAX_AREA_SQFT = 8             # ft² — closed polygons smaller than this might be Table
FURNITURE_RECT_RATIO_THRESHOLD = 0.99  # area ratio to minimum bounding rectangle (perfect rectangle = 1.0)

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


def _explode_entity(ent: DXFGraphic, out: List[LineString], furniture_out: List[LineString]):
    """
    Explode entity into line segments, with furniture detection for closed polygons.
    
    Args:
        ent: DXF entity to explode
        out: List to append line segments to
        furniture_out: List to append detected furniture line segments to
    """
    t = ent.dxftype()
    if t == "INSERT":
        blk: Insert = ent
        # m44 = blk.matrix44()
        for sub in blk.virtual_entities():
            # sub.transform(m44)
            upright_all([sub])
            _explode_entity(sub, out, furniture_out)
        return

    if t == "LINE":
        ls = LineString([(ent.dxf.start.x, ent.dxf.start.y), (ent.dxf.end.x, ent.dxf.end.y)])
        if ls.length >= MIN_SEG_LEN_FT:
            out.append(ls)

    elif t in {"LWPOLYLINE", "POLYLINE"}:
        pts = [(v[0], v[1]) for v in ent]

        if getattr(ent, "closed", False) and len(pts) > 3:
            poly = Polygon(pts)
            area_sqft = poly.area * (DXF_UNIT_TO_FOOT ** 2)
            min_rect = poly.minimum_rotated_rectangle
            area_ratio = poly.area / min_rect.area if min_rect.area else 1
                
            # Check if it looks like furniture (small and rectangular)
            if (area_ratio > FURNITURE_RECT_RATIO_THRESHOLD and area_sqft < TABLE_MAX_AREA_SQFT) or area_sqft < FURNITURE_MAX_AREA_SQFT:
                # Add furniture polygon edges as line segments to furniture_out
                coords = list(poly.exterior.coords)
                for a, b in zip(coords, coords[1:]):
                    ls = LineString([a, b])
                    if ls.length >= MIN_SEG_LEN_FT:
                        furniture_out.append(ls)
                return  # Don't add to main line segments if it's furniture

        # Add as line segments (either not closed, or not furniture)
        for a, b in zip(pts, pts[1:]):
            ls = LineString([a, b])
            if ls.length >= MIN_SEG_LEN_FT:
                out.append(ls)

    elif t == "ARC":
        # Check if this arc is a door swing
        sweep = abs(ent.dxf.end_angle - ent.dxf.start_angle)
        is_door_arc = DOOR_ARC_MIN_ANG < sweep < DOOR_ARC_MAX_ANG
        
        if is_door_arc:
            # For door arcs, add the chord to furniture and hinge line to segments
            cx, cy, r = ent.dxf.center.x, ent.dxf.center.y, ent.dxf.radius
            a0, a1 = math.radians(ent.dxf.start_angle), math.radians(ent.dxf.end_angle)
            p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            chord_len_ft = LineString([p0, p1]).length * DXF_UNIT_TO_FOOT
            
            if DOOR_CHORD_MIN_FT < chord_len_ft < DOOR_CHORD_MAX_FT:
                # Add chord line to furniture (door opening)
                chord_ls = LineString([p0, p1])
                if chord_ls.length >= MIN_SEG_LEN_FT:
                    furniture_out.append(chord_ls)
                
                # Add hinge line from center to start point (structural element)
                hinge_ls = LineString([(cx, cy), p0])
                if hinge_ls.length >= MIN_SEG_LEN_FT:
                    out.append(hinge_ls)
        else:
            # Regular arc - convert to line segments
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
        
        # 3. Check if this might be a door spline
        if len(coords_2d) >= 3:
            p0, p1 = coords_2d[0], coords_2d[-1]
            chord = LineString([p0, p1])
            chord_len_ft = chord.length * DXF_UNIT_TO_FOOT

            # approximate arc length with polyline length
            arclen_ft = sum(
                math.hypot(bx - ax, by - ay)
                for (ax, ay), (bx, by) in zip(coords_2d, coords_2d[1:])
            ) * DXF_UNIT_TO_FOOT
            ratio = arclen_ft / chord_len_ft if chord_len_ft else 9e9

            is_door_spline = _looks_like_door_arc(chord_len_ft, ratio)
            
            if is_door_spline:
                # Estimate center of the door arc
                center = _estimate_arc_center(coords_2d)
                if center:
                    # Add door chord to furniture (door opening)
                    chord_ls = LineString([p0, p1])
                    if chord_ls.length >= MIN_SEG_LEN_FT:
                        furniture_out.append(chord_ls)
                    
                    # Add hinge line from estimated center to start point (structural)
                    hinge_ls = LineString([center, p1])
                    if hinge_ls.length >= MIN_SEG_LEN_FT:
                        out.append(hinge_ls)
            else:
                # Regular spline/ellipse - add as single line if long enough
                if len(coords_2d) >= 2:
                    ls = LineString(coords_2d)
                    if ls.length >= MIN_SEG_LEN_FT:
                        out.append(ls)

def _looks_like_door_arc(chord_len_ft, ratio):
    """Check if the geometry looks like a door arc based on chord length and arc/chord ratio"""
    return (DOOR_CHORD_MIN_FT < chord_len_ft < DOOR_CHORD_MAX_FT
            and ARCLEN_CHORD_RATIO_MIN < ratio < ARCLEN_CHORD_RATIO_MAX)


def _estimate_arc_center(coords_2d):
    """
    Estimate the center of an arc from a series of points using least squares circle fitting (Taubin method).
    Returns (center_x, center_y) or None if estimation fails.
    """
    if len(coords_2d) < 3:
        return None  # Not enough points for fitting
    
    coords = np.array(coords_2d)
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Calculate moments
    x_m = x.mean()
    y_m = y.mean()
    u = x - x_m
    v = y - y_m

    # Compute coefficients for the linear system
    Suu = np.sum(u**2)
    Suv = np.sum(u*v)
    Svv = np.sum(v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    Suvv = np.sum(u * v**2)
    Svuu = np.sum(v * u**2)

    # Solve the linear system
    A = np.array([[Suu, Suv],
                  [Suv, Svv]])
    
    B = np.array([0.5 * (Suuu + Suvv),
                  0.5 * (Svvv + Svuu)])
    
    try:
        uc, vc = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return None  # Singular matrix (e.g., all points colinear)
    
    center_x = x_m + uc
    center_y = y_m + vc
    
    return (center_x, center_y)

def _collect_segments(doc):
    """Collect line segments and furniture line segments from DXF document"""
    segs: List[LineString] = []
    furniture: List[LineString] = []
    for e in doc.modelspace():
        _explode_entity(e, segs, furniture)
    return segs, furniture

# ---------------------------------------------------------------------------
# Graph + room extraction
# ---------------------------------------------------------------------------

def _build_network(segs):
    snapped = snap(MultiLineString(segs), MultiLineString(segs), GAP_TOLERANCE)
    return node(snapped)


def _room_polys(net):
    polys, dangles, cuts, _ = polygonize_full(net)
    sqft = DXF_UNIT_TO_FOOT ** 2
    rooms = [p for p in polys.geoms if p.area * sqft >= MIN_ROOM_AREA_SQFT and p.area * sqft <= MAX_ROOM_AREA_SQFT]
    return rooms, dangles, cuts

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def count_rooms(model) -> List[Tuple[int, float]]:
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    segs, _ = _collect_segments(doc)
    net = _build_network(segs)
    rooms, _, _ = _room_polys(net)
    sqft = DXF_UNIT_TO_FOOT ** 2
    return [(i + 1, r.area * sqft) for i, r in enumerate(sorted(rooms, key=lambda p: p.area, reverse=True))]


def count_furniture(model) -> List[Tuple[int, float]]:
    """Count and return furniture line segments with their lengths"""
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    _, furniture = _collect_segments(doc)
    return [(i + 1, f.length * DXF_UNIT_TO_FOOT) for i, f in enumerate(sorted(furniture, key=lambda ls: ls.length, reverse=True))]


def debug_plot(model, *,
               show_dangling=True,
               show_furniture=True,
               save: Optional[str | Path] = None):
    """
    Debug plot showing rooms, furniture, and optionally dangling/cut segments
    
    Args:
        model: DXF file path or document
        show_dangling: Show dangling and cut segments
        show_furniture: Show detected furniture polygons
        save: File path to save plot, or None to show
    """
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model

    # Get segments and furniture
    segs, furniture = _collect_segments(doc)
    net = _build_network(segs)
    rooms, dangles, cuts = _room_polys(net)

    fig, ax = _plt.subplots()

    # rooms
    for poly in rooms:
        ax.plot(*poly.exterior.xy, linewidth=2, label='Rooms' if poly == rooms[0] else "")

    # furniture
    if show_furniture and furniture:
        for ls in furniture:
            ax.plot(*ls.xy, color="brown", linewidth=2, label='Furniture' if ls == furniture[0] else "")

    if show_dangling:
        for g in dangles.geoms:
            ax.plot(*g.xy, color="red", linestyle="--", linewidth=1, label='Dangles' if g == dangles.geoms[0] else "")
        for g in cuts.geoms:
            ax.plot(*g.xy, color="orange", linestyle=":", linewidth=1, label='Cuts' if g == cuts.geoms[0] else "")

    # Add legend if there are labeled items
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    ax.set_aspect("equal", "box")
    ax.axis("off")
    (fig.savefig(save, dpi=300, bbox_inches="tight") if save else _plt.show())

def count_rooms_with_geometry(model):
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    segs, _ = _collect_segments(doc)
    net = _build_network(segs)
    rooms, _, _ = _room_polys(net)
    sqft = DXF_UNIT_TO_FOOT ** 2

    result = []
    for i, r in enumerate(sorted(rooms, key=lambda p: p.area, reverse=True)):
        result.append({
            "id": i + 1,
            "area_ft2": r.area * sqft,
            "boundary": list(r.exterior.coords)  # list of (x, y) tuples
        })
    return result


def count_furniture_with_geometry(model):
    """Return furniture line segments with geometry information"""
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    _, furniture = _collect_segments(doc)

    result = []
    for i, ls in enumerate(sorted(furniture, key=lambda ls: ls.length, reverse=True)):
        result.append({
            "id": i + 1,
            "length_ft": ls.length * DXF_UNIT_TO_FOOT,
            "coordinates": list(ls.coords),  # list of (x, y) tuples
            "midpoint": [ls.interpolate(0.5).x, ls.interpolate(0.5).y]
        })
    return result


def get_full_analysis(model) -> Dict[str, Any]:
    """Get complete analysis including rooms and furniture"""
    doc = ezdxf.readfile(str(model)) if not hasattr(model, "modelspace") else model
    segs, furniture = _collect_segments(doc)
    net = _build_network(segs)
    rooms, dangles, cuts = _room_polys(net)
    sqft = DXF_UNIT_TO_FOOT ** 2

    return {
        "rooms": [
            {
                "id": i + 1,
                "area_ft2": r.area * sqft,
                "boundary": list(r.exterior.coords)
            }
            for i, r in enumerate(sorted(rooms, key=lambda p: p.area, reverse=True))
        ],
        "furniture": [
            {
                "id": i + 1,
                "length_ft": ls.length * DXF_UNIT_TO_FOOT,
                "coordinates": list(ls.coords),
                "midpoint": [ls.interpolate(0.5).x, ls.interpolate(0.5).y]
            }
            for i, ls in enumerate(sorted(furniture, key=lambda ls: ls.length, reverse=True))
        ],
        "summary": {
            "room_count": len(rooms),
            "furniture_count": len(furniture),
            "total_room_area_ft2": sum(r.area * sqft for r in rooms),
            "total_furniture_length_ft": sum(ls.length * DXF_UNIT_TO_FOOT for ls in furniture)
        }
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, textwrap, json

    parser = argparse.ArgumentParser(
        prog="dxf_room_counter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """Count rooms & furniture, plot vertices (snap‑then‑node T‑junction aware).

            python dxf_room_counter.py plan.dxf                    # list room areas
            python dxf_room_counter.py plan.dxf --furniture        # list furniture areas
            python dxf_room_counter.py plan.dxf --plot             # rooms + furniture plot
            python dxf_room_counter.py plan.dxf --json output.json # save complete analysis
            """
        ),
    )
    parser.add_argument("dxf", help="DXF file to analyse")
    parser.add_argument("--plot", nargs="?", const="_show_", help="Show/save plot")
    parser.add_argument("--furniture", action="store_true", help="Show furniture list instead of rooms")
    parser.add_argument("--json", help="Write complete analysis (rooms + furniture) to JSON file")
    ns = parser.parse_args()

    if ns.json:
        analysis = get_full_analysis(ns.dxf)
        with open(ns.json, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Complete analysis written to {ns.json}")
    elif ns.furniture:
        for fid, length in count_furniture(ns.dxf):
            print(f"Furniture {fid}: {length:.2f} ft")
    else:
        for rid, area in count_rooms(ns.dxf):
            print(f"Room {rid}: {area:.2f} ft²")

    if ns.plot is not None:
        dest = None if (ns.plot == "_show_") else ns.plot
        debug_plot(ns.dxf, show_dangling=False, show_furniture=True, save=dest)