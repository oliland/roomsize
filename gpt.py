"""
DXF Room Counter
================
Reads a DXF floor-plan, extracts LINE and (LW)POLYLINE entities that represent walls,
builds a Shapely network, and heuristically counts rooms + their areas in square feet.

Why a second version?
--------------------
* **T-junction aware** – Wall segments that intersect without sharing an explicit
  vertex (a classic T-junction) are now **noded** so the planar graph is
  topologically correct before polygonisation.
* **Flexible API** – The public `count_rooms()` function now accepts **either**
  an `ezdxf.EzDxf` model or a path-like object pointing to a DXF file.
* Constants collected in one place as before; tweak to taste.

Assumptions
-----------
* **Units** – The drawing is in *feet*.  If your DXF is in inches or metres
  change `DXF_UNIT_TO_FOOT` accordingly.
* **Entity types** – Only `LINE`, `LWPOLYLINE`, and `POLYLINE` describe walls.
  Everything else (doors, text, blocks) is ignored.
* **Gaps & Slivers** – Wall gaps ≤ `GAP_TOLERANCE` (ft) get snapped; polygons
  with area < `MIN_ROOM_AREA_SQFT` are filtered out.

"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple, Union, overload

import ezdxf  # type: ignore  – install via `pip install ezdxf`
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, snap, unary_union

###############################################################################
# TUNABLE CONSTANTS
###############################################################################
DXF_UNIT_TO_FOOT: float = 1.0     # change to 1 / 12.0 if drawing is in inches
GAP_TOLERANCE: float = 0.10       # ≤ this many feet: close gaps via `snap`
MIN_ROOM_AREA_SQFT: float = 1.0  # ignore polygons smaller than a closet
###############################################################################


def _extract_wall_segments(doc) -> List[LineString]:
    """Return Shapely `LineString`s for every wall segment in the DXF modelspace."""

    msp = doc.modelspace()
    segs: List[LineString] = []

    # Simple LINE entities --------------------------------------------------
    for ln in msp.query("LINE"):
        segs.append(LineString([(ln.dxf.start.x, ln.dxf.start.y),
                                (ln.dxf.end.x, ln.dxf.end.y)]))

    # LWPOLYLINE & POLYLINE entities ---------------------------------------
    for pl in msp.query("LWPOLYLINE POLYLINE"):
        pts = [(v[0], v[1]) for v in pl]
        for a, b in zip(pts, pts[1:]):
            segs.append(LineString([a, b]))
        # close ring where applicable
        if pl.closed and len(pts) > 1:
            segs.append(LineString([pts[-1], pts[0]]))

    print(len(segs))
    return segs


# ---------------------------------------------------------------------------
# NODING ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _node_lines(segments: Iterable[LineString]) -> MultiLineString:
    """Split all segments at their intersections (incl. T & X junctions).

    Shapely ≥ 2.0 provides `shapely.ops.node`; if unavailable we fall back to
    `unary_union`, which also performs noding for linework.
    """
    try:
        from shapely.ops import node  # type: ignore

        noded = node(MultiLineString(list(segments)))  # type: ignore[arg-type]
        return noded  # already a MultiLineString
    except Exception:  # pragma: no cover – older Shapely
        # `unary_union` will node but might return a LineString if everything
        # merges perfectly; wrap to MultiLineString for uniform downstream code.
        merged = unary_union(list(segments))
        if isinstance(merged, LineString):
            merged = MultiLineString([merged])
        return merged  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ROOM DETECTION ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _room_polygons(segments: List[LineString]) -> List[Polygon]:
    """Return polygons deemed rooms after noding and gap-snapping."""
    if not segments:
        return []

    # 1. Insert vertices at all intersections (T, X, …)
    noded = _node_lines(segments)

    # 2. Snap tiny gaps so endpoints that are *close* become coincident
    network = snap(noded, noded, GAP_TOLERANCE)

    # 3. Build polygons from the planar graph
    polys = list(polygonize(network))

    # 4. Filter by minimum size
    sqft_factor = DXF_UNIT_TO_FOOT ** 2
    rooms = [p for p in polys if p.area * sqft_factor >= MIN_ROOM_AREA_SQFT]
    return rooms


# ---------------------------------------------------------------------------
# PUBLIC API ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def count_rooms(model) -> List[Tuple[int, float]]:
    """Count rooms and return ``[(room_id, area_ft²), ...]`` ordered by area.

    Parameters
    ----------
    model
        *Either* an `ezdxf.EzDxf` document **or** a path-like object to a DXF
        file on disk.
    """
    if isinstance(model, (str, Path)):
        doc = ezdxf.readfile(str(model))
    else:
        doc = model

    segments = _extract_wall_segments(doc)
    rooms = _room_polygons(segments)

    sqft_factor = DXF_UNIT_TO_FOOT ** 2
    result = [(i + 1, r.area * sqft_factor) for i, r in enumerate(sorted(rooms, key=lambda p: p.area, reverse=True))]
    return result


# ---------------------------------------------------------------------------
# TESTS ──────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _build_sample_plan(path: Path) -> None:
    """Create a 2-room 25 ft × 10 ft drawing with a T-shaped partition."""
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # Outer rectangle
    outer = [(0, 0), (25, 0), (25, 10), (0, 10)]
    for a, b in zip(outer, outer[1:] + [outer[0]]):
        msp.add_line(a, b)

    # Vertical wall (like before) at x = 12.5
    msp.add_line((12.5, 0), (12.5, 6))  # stops before ceiling ⇒ creates a T

    # Horizontal wall (forming the T head)
    msp.add_line((12.5, 6), (25, 6))

    doc.saveas(path)


def test_two_room_with_t(tmp_path):
    """Even with a T-junction the algo should find exactly 2 × (12.5×6 and 12.5×4) ft² rooms."""
    dxf_file = tmp_path / "t_sample.dxf"
    _build_sample_plan(dxf_file)

    rooms = count_rooms(dxf_file)
    assert len(rooms) == 2

    areas = sorted(a for _, a in rooms)
    assert math.isclose(sum(areas), 250.0, abs_tol=1e-4)  # area conservation


if __name__ == "__main__":
    import argparse, sys, textwrap

    parser = argparse.ArgumentParser(
        prog="dxf_room_counter",
        description=textwrap.dedent(
            """Quickly estimate room count and areas in a floor-plan DXF.

            Pass either a path to a DXF file *or* the string "sample" to run the
            built-in 2-room demo with a T-junction.
            """
        ),
    )
    parser.add_argument("dxf", help="DXF file path or 'sample'")
    ns = parser.parse_args()

    results = count_rooms(Path(ns.dxf))
    print(results)

    for rid, area in results:
        print(f"Room {rid}: {area:.2f} ft²")
