from __future__ import annotations

COORDINATE_SPACE = "pdf_user_space"
TRANSFORM_VERSION = "pymupdf_top_left_v1"


def coordinate_metadata(page: dict, *, highlightable: bool) -> dict:
    if not highlightable:
        return {"coordinate_space": COORDINATE_SPACE, "coordinate_origin": "top_left", "transform_version": TRANSFORM_VERSION}
    width, height = page.get("width"), page.get("height")
    if not all(isinstance(value, (int, float)) and value > 0 for value in (width, height)):
        raise ValueError("invalid_page_coordinates")
    return {
        "coordinate_space": COORDINATE_SPACE,
        "coordinate_origin": "top_left",
        "page_width": width,
        "page_height": height,
        "page_rotation": 0,
        "page_box_basis": "media_box",
        "transform_version": TRANSFORM_VERSION,
    }
