from __future__ import annotations


VIEWER_GEOMETRY_EPSILON = 1e-6
GEOMETRY_IDENTITY_FIELDS = (
    "source_document_id",
    "source_sha256",
    "page_number",
    "coordinate_space",
    "coordinate_origin",
    "page_rotation",
    "page_box_basis",
    "transform_version",
)
VIEWER_RECTANGLE_FIELDS = (
    *GEOMETRY_IDENTITY_FIELDS,
    "page_width",
    "page_height",
    "bbox_precision",
    "viewer_highlightable",
    "x0",
    "y0",
    "x1",
    "y1",
)


def bbox_is_accepted(row: dict) -> bool:
    return row.get("status") == "accepted" and all(row.get(key) is not None for key in ("page_number", "x0", "y0", "x1", "y1"))


def positive_area_intersection(left: dict, right: dict, *, epsilon: float = VIEWER_GEOMETRY_EPSILON) -> bool:
    """Treat sub-micro-point coordinate noise as contact, not visible overlap."""
    width = min(float(left["x1"]), float(right["x1"])) - max(float(left["x0"]), float(right["x0"]))
    height = min(float(left["y1"]), float(right["y1"])) - max(float(left["y0"]), float(right["y0"]))
    return width > epsilon and height > epsilon


def subtract_rectangle(rectangle: dict, excluded: dict) -> tuple[dict, ...]:
    if not positive_area_intersection(rectangle, excluded):
        return (rectangle,)
    x0 = max(float(rectangle["x0"]), float(excluded["x0"]))
    y0 = max(float(rectangle["y0"]), float(excluded["y0"]))
    x1 = min(float(rectangle["x1"]), float(excluded["x1"]))
    y1 = min(float(rectangle["y1"]), float(excluded["y1"]))
    pieces = (
        rectangle | {"y1": y0},
        rectangle | {"y0": y1},
        rectangle | {"x1": x0, "y0": y0, "y1": y1},
        rectangle | {"x0": x1, "y0": y0, "y1": y1},
    )
    return tuple(
        piece
        for piece in pieces
        if float(piece["x1"]) - float(piece["x0"]) > VIEWER_GEOMETRY_EPSILON
        and float(piece["y1"]) - float(piece["y0"]) > VIEWER_GEOMETRY_EPSILON
    )


def viewer_overlay_rectangles(proposition: dict, characters_by_id: dict[str, dict] | None = None) -> tuple[dict, ...]:
    """Return persisted selected-character overlay rectangles."""
    overlay = proposition.get("viewer_overlay") or {}
    if overlay.get("status") != "complete":
        return ()
    rectangles = tuple(overlay.get("rectangles") or ())
    selected = set(proposition.get("bbox_refs") or ())
    covered = {
        character_id
        for rectangle in rectangles
        for character_id in rectangle.get("character_bbox_ids") or ()
    }
    return rectangles if rectangles and covered == selected else ()
