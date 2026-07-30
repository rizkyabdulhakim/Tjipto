from __future__ import annotations

from collections.abc import Callable
from threading import RLock


class BookmarkRepository:
    """Process-scoped storage for the existing session bookmark contract."""

    def __init__(self) -> None:
        self._records: dict[str, dict] = {}
        self._lock = RLock()

    def save(self, record: dict) -> None:
        with self._lock:
            self._records[str(record["bookmark_id"])] = record

    def list(self, corpus_id: str) -> tuple[dict, ...]:
        with self._lock:
            return tuple(row.copy() for row in self._records.values() if row["corpus_id"] == corpus_id)

    def delete_public(
        self,
        corpus_id: str,
        public_bookmark_id: str,
        public_id_for: Callable[[str], str],
    ) -> bool:
        with self._lock:
            bookmark_id = next(
                (
                    key
                    for key, row in self._records.items()
                    if row["corpus_id"] == corpus_id and public_id_for(key) == public_bookmark_id
                ),
                None,
            )
            if bookmark_id is None:
                return False
            del self._records[bookmark_id]
            return True
