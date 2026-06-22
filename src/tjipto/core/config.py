from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tjipto.core.manifest import read_json, read_jsonl


ARTIFACT_ALIASES = {
    "bbox": "bbox_registry",
    "evidence": "evidence_registry",
}


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    manifest_path: Path
    manifest: dict
    settings: dict | None = None

    def setting(self, key: str, default=None):
        return (self.settings or {}).get(key, default)

    @property
    def query_strategy(self) -> str:
        return self.setting("query_strategy", "generic")

    @property
    def structured_strategy(self) -> str:
        return self.setting("structured_strategy", "generic")

    @property
    def preferred_source_role(self) -> str | None:
        return self.setting("preferred_source_role")

    @property
    def source_roles(self) -> tuple[str, ...]:
        return tuple(self.setting("source_roles", ()))

    @property
    def temporal_contexts(self) -> tuple[str, ...]:
        return tuple(self.setting("temporal_contexts", ()))

    def artifact_path(self, logical_key: str) -> Path:
        key = ARTIFACT_ALIASES.get(logical_key, logical_key)
        rel = self.manifest[key]
        if not isinstance(rel, str):
            raise ValueError(f"invalid artifact path:{logical_key}")
        path = Path(rel)
        if path.is_absolute():
            raise ValueError(f"invalid artifact path:{logical_key}")
        base = self.manifest_path.parent.resolve()
        resolved = (base / path).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"invalid artifact path:{logical_key}")
        return resolved

    def json(self, logical_key: str) -> dict:
        return read_json(self.artifact_path(logical_key))

    def jsonl(self, logical_key: str) -> list[dict]:
        return read_jsonl(self.artifact_path(logical_key))
