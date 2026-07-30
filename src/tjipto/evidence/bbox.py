from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


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
VIEWER_GEOMETRY_SPACE_FIELDS = tuple(
    field for field in VIEWER_RECTANGLE_FIELDS if field not in {"x0", "y0", "x1", "y1"}
)


def bbox_is_accepted(row: dict) -> bool:
    return row.get("status") == "accepted" and all(row.get(key) is not None for key in ("page_number", "x0", "y0", "x1", "y1"))


def positive_area_intersection(
    left: Mapping[str, Any], right: Mapping[str, Any], *, epsilon: float = VIEWER_GEOMETRY_EPSILON
) -> bool:
    """Treat sub-micro-point coordinate noise as contact, not visible overlap."""
    width = min(float(left["x1"]), float(right["x1"])) - max(float(left["x0"]), float(right["x0"]))
    height = min(float(left["y1"]), float(right["y1"])) - max(float(left["y0"]), float(right["y0"]))
    return width > epsilon and height > epsilon


def subtract_rectangle(rectangle: Mapping[str, Any], excluded: Mapping[str, Any]) -> tuple[dict, ...]:
    if not positive_area_intersection(rectangle, excluded):
        return (dict(rectangle),)
    x0 = max(float(rectangle["x0"]), float(excluded["x0"]))
    y0 = max(float(rectangle["y0"]), float(excluded["y0"]))
    x1 = min(float(rectangle["x1"]), float(excluded["x1"]))
    y1 = min(float(rectangle["y1"]), float(excluded["y1"]))
    base = dict(rectangle)
    pieces = (
        base | {"y1": y0},
        base | {"y0": y1},
        base | {"x1": x0, "y0": y0, "y1": y1},
        base | {"x0": x1, "y0": y0, "y1": y1},
    )
    return tuple(
        piece
        for piece in pieces
        if float(piece["x1"]) - float(piece["x0"]) > VIEWER_GEOMETRY_EPSILON
        and float(piece["y1"]) - float(piece["y0"]) > VIEWER_GEOMETRY_EPSILON
    )


def geometry_space_key(row: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(row.get(field) for field in GEOMETRY_IDENTITY_FIELDS)


def derive_viewer_overlay(
    proposition: Mapping[str, Any],
    characters_by_id: Mapping[str, Mapping[str, Any]],
    excluded_boxes_by_space: Mapping[tuple[object, ...], Sequence[Mapping[str, Any]]],
) -> dict:
    """Derive the compact viewer overlay from immutable selected-character geometry."""
    selected_ids = tuple(str(value) for value in proposition.get("bbox_refs") or ())
    selected = [characters_by_id[value] for value in selected_ids if value in characters_by_id]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for character in selected:
        grouped.setdefault(str(character.get("word_bbox_id") or character["character_bbox_id"]), []).append(character)

    rectangles: list[dict] = []
    clipped_rectangle_indexes: list[int] = []
    covered_ids: set[str] = set()
    clipped_ids: set[str] = set()
    selected_index = {character_id: index for index, character_id in enumerate(selected_ids)}
    geometry_spaces: list[dict] = []
    geometry_space_indexes: dict[tuple[object, ...], int] = {}
    for characters in grouped.values():
        first = characters[0]
        character_ids = tuple(str(character["character_bbox_id"]) for character in characters)
        geometry_space = {field: first.get(field) for field in VIEWER_GEOMETRY_SPACE_FIELDS}
        geometry_key = tuple(geometry_space[field] for field in VIEWER_GEOMETRY_SPACE_FIELDS)
        if geometry_key not in geometry_space_indexes:
            geometry_space_indexes[geometry_key] = len(geometry_spaces)
            geometry_spaces.append(geometry_space)
        base = {
            "x0": min(float(character["x0"]) for character in characters),
            "y0": min(float(character["y0"]) for character in characters),
            "x1": max(float(character["x1"]) for character in characters),
            "y1": max(float(character["y1"]) for character in characters),
            "selected_character_start": selected_index[character_ids[0]],
            "selected_character_end": selected_index[character_ids[-1]] + 1,
            "geometry_space_index": geometry_space_indexes[geometry_key],
        }
        pieces = [geometry_space | base]
        clipped = False
        for excluded in excluded_boxes_by_space.get(geometry_space_key(first), ()):
            clipped = clipped or any(positive_area_intersection(piece, excluded) for piece in pieces)
            pieces = [piece for candidate in pieces for piece in subtract_rectangle(candidate, excluded)]
        if pieces:
            covered_ids.update(character_ids)
        if clipped:
            clipped_ids.update(character_ids)
        for piece in pieces:
            rectangles.append({
                key: piece[key]
                for key in (
                    "selected_character_start", "selected_character_end", "geometry_space_index",
                    "x0", "y0", "x1", "y1",
                )
            })
            if clipped:
                clipped_rectangle_indexes.append(len(rectangles) - 1)

    complete = (
        len(selected) == len(selected_ids)
        and len(set(selected_ids)) == len(selected_ids)
        and covered_ids == set(selected_ids)
    )
    return {
        "status": "complete" if complete else "unavailable",
        "reason_code": None if complete else "exact_viewer_geometry_unavailable",
        "proposition_id": proposition.get("proposition_id"),
        "source_document_id": proposition.get("source_document_id"),
        "source_sha256": proposition.get("source_sha256"),
        "selector_field": "source_selectors",
        "selected_character_field": "bbox_refs",
        "geometry_spaces": geometry_spaces if complete else [],
        "clipped_character_count": len(clipped_ids),
        "rectangles": rectangles if complete else [],
        "clipped_rectangle_indexes": clipped_rectangle_indexes if complete else [],
    }


def viewer_overlay_rectangles(proposition: dict, characters_by_id: dict[str, dict] | None = None) -> tuple[dict, ...]:
    """Return persisted selected-character overlay rectangles."""
    overlay = proposition.get("viewer_overlay") or {}
    if overlay.get("status") != "complete":
        return ()
    geometry_spaces = tuple(overlay.get("geometry_spaces") or ())
    compact_rectangles = tuple(overlay.get("rectangles") or ())
    selected_ids = tuple(proposition.get("bbox_refs") or ())
    if any(
        not isinstance(rectangle.get("geometry_space_index"), int)
        or rectangle["geometry_space_index"] < 0
        or rectangle["geometry_space_index"] >= len(geometry_spaces)
        or not isinstance(rectangle.get("selected_character_start"), int)
        or not isinstance(rectangle.get("selected_character_end"), int)
        or rectangle["selected_character_start"] < 0
        or rectangle["selected_character_end"] <= rectangle["selected_character_start"]
        or rectangle["selected_character_end"] > len(selected_ids)
        for rectangle in compact_rectangles
    ):
        return ()
    rectangles = tuple(
        {
            **geometry_spaces[rectangle["geometry_space_index"]],
            **rectangle,
            "bbox_id": selected_ids[rectangle["selected_character_start"]],
            "character_bbox_ids": selected_ids[
                rectangle["selected_character_start"] : rectangle["selected_character_end"]
            ],
            "bbox_precision": "exact",
            "viewer_highlightable": True,
        }
        for rectangle in compact_rectangles
    )
    selected = set(selected_ids)
    covered = {
        character_id
        for rectangle in rectangles
        for character_id in rectangle.get("character_bbox_ids") or ()
    }
    return rectangles if rectangles and covered == selected else ()
