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

    def artifact_path(self, logical_key: str) -> Path:
        key = ARTIFACT_ALIASES.get(logical_key, logical_key)
        rel = self.manifest[key]
        return self.manifest_path.parent / rel

    def json(self, logical_key: str) -> dict:
        return read_json(self.artifact_path(logical_key))

    def jsonl(self, logical_key: str) -> list[dict]:
        return read_jsonl(self.artifact_path(logical_key))
