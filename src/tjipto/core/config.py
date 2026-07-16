from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence

from tjipto.core.manifest import read_json, read_jsonl


ARTIFACT_ALIASES = {
    "bbox": "bbox_registry",
    "evidence": "evidence_registry",
}


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    manifest_path: Path
    manifest: Mapping
    settings: Mapping | None = None
    repo_root: Path | None = None
    verified_artifacts: Mapping[str, object] | None = None
    manifest_digest: str | None = None
    artifact_set_digest: str | None = None

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

    def source_path(self, rel: str) -> Path:
        path = Path(rel)
        if path.is_absolute():
            raise ValueError("invalid_source_path")
        root = (self.repo_root or self.manifest_path.parents[2]).resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("invalid_source_path")
        return resolved

    def json(self, logical_key: str) -> Mapping:
        if self.verified_artifacts is not None:
            return self.verified_artifacts[self.manifest[logical_key]]  # type: ignore[return-value,index]
        return read_json(self.artifact_path(logical_key))

    def jsonl(self, logical_key: str) -> Sequence[Mapping]:
        if self.verified_artifacts is not None:
            return self.verified_artifacts[self.manifest[ARTIFACT_ALIASES.get(logical_key, logical_key)]]  # type: ignore[return-value,index]
        return read_jsonl(self.artifact_path(logical_key))
