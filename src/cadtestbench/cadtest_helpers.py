import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.TopAbs import TopAbs_IN
from OCP.gp import gp_Pnt, gp_Dir, gp_Lin
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.TopoDS import TopoDS_Shape
from OCP.TopoDS import TopoDS_Solid
import os
import tempfile
import numpy as np
import trimesh
from collections import deque

def distance_point_to_solid(solid: cq.Workplane, p: cq.Vector) -> float:
    """
    Computes the minimum Euclidean distance between a point and a solid.

    - Returns 0.0 if the point is inside the solid.
    - Returns 0.0 if the point lies on the surface.
    - Returns a positive value if the point is outside.
    - Distance is measured in the model's units.

    """
    def _to_topods_shape(obj) -> TopoDS_Shape:
        if isinstance(obj, cq.Workplane):
            obj = obj.val()

        if obj is None:
            raise ValueError("_to_topods_shape: got None")

        if hasattr(obj, "wrapped"):
            obj = obj.wrapped

        if not isinstance(obj, TopoDS_Shape):
            raise TypeError(f"_to_topods_shape: expected TopoDS_Shape, got {type(obj)}")

        # Force base-class view (some bindings are strict)
        return TopoDS_Shape(obj)
    shape = _to_topods_shape(solid)
    vtx = BRepBuilderAPI_MakeVertex(p.toPnt()).Vertex()

    dist = BRepExtrema_DistShapeShape(shape, vtx)
    if hasattr(dist, "Perform"):
        dist.Perform()
    if not dist.IsDone():
        raise RuntimeError("OCCT distance computation failed.")

    return float(dist.Value())

def is_point_inside(solid: cq.Workplane, p: cq.Vector) -> bool:
    """
    Tests whether a point lies strictly inside a solid.

    Question answered:
        "Is there material at this point?"

    Guarantees:
        - Returns True only if the point is strictly inside.
        - Returns False for points on the surface.
        - Returns False for points outside.
    """
    tol: float = 1e-6
    if isinstance(solid, cq.Workplane):
        solid = solid.val()
    if not hasattr(solid, "wrapped"):
        raise TypeError(f"Unsupported type: {type(solid)}")

    topods = solid.wrapped
    classifier = BRepClass3d_SolidClassifier(topods, p.toPnt(), tol)
    return classifier.State() == TopAbs_IN


def bbox_contains_point(shape, p: cq.Vector) -> bool:
    """
    Performs a fast axis-aligned bounding box (AABB) containment test.

    Question answered:
        "Could this point possibly be inside the shape?"

    Guarantees:
        - Returns True if the point lies within the shape's bounding box.
        - Returns False if the point is definitely outside the shape.
        - May return True for points that are not inside the actual geometry.
    """
    tol: float = 0.0
    if isinstance(shape, cq.Workplane):
        shape = shape.val()
    if not hasattr(shape, "BoundingBox"):
        raise TypeError(f"Unsupported type: {type(shape)}")

    bb = shape.BoundingBox()
    return (
        bb.xmin - tol <= p.x <= bb.xmax + tol
        and bb.ymin - tol <= p.y <= bb.ymax + tol
        and bb.zmin - tol <= p.z <= bb.zmax + tol
    )


def raycast_shape(
    solid: cq.Workplane | cq.Shape,
    p: cq.Vector,
    d: cq.Vector,

) -> list[float]:
    """
    Cast a ray and return sorted hit distances along the ray.

    Parameters
    ----------
    solid
        CadQuery Workplane or Shape.
    p
        Ray origin.
    d
        Ray direction (need not be normalized).

    Returns
    -------
    ts
        Sorted distances t >= 0 where the ray intersects the shape boundary.
        Distances are along the normalized ray direction: P(t) = p + t * d_unit.
    """

    def _to_topods_shape(obj) -> TopoDS_Shape:
        if isinstance(obj, cq.Workplane):
            obj = obj.val()

        # CadQuery Shape
        if hasattr(obj, "wrapped"):
            obj = obj.wrapped

        # Already a TopoDS_Shape
        if isinstance(obj, TopoDS_Shape):
            return obj

        # Some bindings return a TopoDS_Solid that doesn't isinstance TopoDS_Shape cleanly
        if isinstance(obj, TopoDS_Solid):
            # Upcast: TopoDS_Solid is-a TopoDS_Shape in C++, but bindings can be strict
            return TopoDS_Shape(obj)


    tol = 1e-6
    shape = _to_topods_shape(solid)

    if d.Length <= 0:
        raise ValueError("Ray direction must be non-zero")

    d_unit = d.normalized()

    origin = gp_Pnt(p.x, p.y, p.z)
    direction = gp_Dir(d_unit.x, d_unit.y, d_unit.z)
    line = gp_Lin(origin, direction)

    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(shape, float(tol))
    inter.Perform(line, 0.0, 1e9)

    ts: list[float] = []
    for i in range(1, inter.NbPnt() + 1):
        hp = inter.Pnt(i)
        hit = cq.Vector(hp.X(), hp.Y(), hp.Z())
        t = (hit - p).dot(d_unit)
        if t >= -tol:
            ts.append(float(t))

    ts.sort()

    merged: list[float] = []
    for t in ts:
        if not merged or abs(t - merged[-1]) > tol:
            merged.append(t)

    return merged


def intersection_volume_with_bbox(
    solid: cq.Workplane | cq.Shape,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
    zmin: float, zmax: float,
) -> float:
    """
    Compute volume of (solid ∩ axis-aligned bbox).

    Notes:
    - Returns 0.0 if there is no overlap (or the boolean yields an empty result).
    - Prefer bbox boundaries that are not exactly coincident with faces, add a small margin.
    """
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("Invalid bbox, expected min < max on all axes")

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cz = (zmin + zmax) / 2.0

    # Make a CadQuery box at the bbox location
    box = cq.Workplane("XY").box(dx, dy, dz, centered=(True, True, True)).translate((cx, cy, cz))

    # Normalize input to a Workplane for boolean ops
    if isinstance(solid, cq.Shape):
        a = cq.Workplane("XY").newObject([solid])
    else:
        a = solid

    # Boolean intersection
    common = a.intersect(box)

    # common may be empty, `.val()` would raise
    vals = common.vals()
    if not vals:
        return 0.0

    # If multiple solids returned, sum their volumes
    return float(sum(v.Volume() for v in vals))



def count_through_holes(
    solid: cq.Workplane,
) -> int:
    """
    Count through-holes (topological genus) of a CadQuery solid via mesh topology.

    Concept
    -------
    We tessellate the CAD solid to a triangle mesh, then use the Euler characteristic:
        genus = (2 - euler_number) / 2
    For a closed orientable surface, genus equals the number of independent through-holes:
        0 -> no through-hole (solid block)
        1 -> one through-hole (ring/pipe)
        2 -> two through-holes, etc.

    Parameters
    ----------
    solid
        CadQuery Workplane or Shape representing a 3D solid (or multiple solids).

    Returns
    -------
    int
        Total through-hole count summed over connected components.

    Notes
    -----
    - This detects through-holes only. Blind holes, pockets, internal cavities that do not
      connect through, do not increase the count.
    - If the model has multiple disconnected solids, this sums the genus per component.
    - Reliability depends on mesh quality. If results look wrong, reduce deflection values.
    """

    linear_deflection = 0.001
    angular_deflection = 0.1
    require_watertight = True
    # Normalize to a Workplane for export
    if isinstance(solid, cq.Shape):
        wp = cq.Workplane("XY").newObject([solid])
    else:
        wp = solid

    # Export to a temporary STL, then load with trimesh
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".stl")
        os.close(fd)

        # CadQuery exporters use OCC tessellation under the hood
        cq.exporters.export(
            wp,
            tmp_path,
            exportType="STL",
            tolerance=linear_deflection,
            angularTolerance=angular_deflection,
        )

        mesh = trimesh.load(tmp_path, force="mesh")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # If you export a shape with multiple solids, trimesh may load a Scene, guard anyway
    if isinstance(mesh, trimesh.Scene):
        # Merge all geometry into one mesh
        mesh = trimesh.util.concatenate(tuple(mesh.dump().geometry.values()))

    # Split into connected components and sum genus per component
    components = mesh.split(only_watertight=False)
    total = 0

    for comp in components:
        if require_watertight and (not comp.is_watertight or not comp.is_volume):
            raise RuntimeError(
                "Mesh component is not a valid closed volume, through-hole count is unreliable. "
                "Try adjusting tessellation tolerances or fixing the CAD solid."
            )

        genus = (2 - comp.euler_number) / 2
        total += int(round(genus))

    return total