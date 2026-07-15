from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Violation:
    code: str
    severity: str
    artifact: str
    row_id: str
    field: str
    expected: object
    actual: object
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)
