from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Json = dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    status: str
    matches: tuple[Json, ...]
