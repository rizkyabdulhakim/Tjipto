from __future__ import annotations

import re

from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import read_json
from tjipto.core.validation import validate_text_provenance


UUD_SATU_NASKAH_HEADER_RE = re.compile(
    r"\*\)\s*:?\s*Perubahan Pertama\s*"
    r"\*\*\)\s*:?\s*Perubahan Kedua\s*"
    r"\*\*\*\)\s*:?\s*Perubahan Ketiga\s*"
    r"\*\*\*\*\)\s*:?\s*Perubahan Keempat\s*"
    r"(?:MAJELIS PERMUSYAWARATAN RAKYAT\s*"
    r"SEKRETARIAT JENDERAL\s*"
    r"UNDANGUNDANG DASAR\s*"
    r"NEGARA REPUBLIK INDONESIA TAHUN 1945\s*"
    r"DALAM SATU NASKAH)?",
    re.IGNORECASE,
)


def validate_corpus_provenance(config: CorpusConfig) -> dict:
    report = validate_text_provenance(
        config,
        header_stripper=_strip_uud_header if config.corpus_id == "uud" else None,
    )
    if config.corpus_id == "uud":
        health = read_json(config.manifest_path.parent / "validation_report.json").get("provenance_exception_health", {})
        report["provenance_exception_health"] = health
        if health.get("unresolved_needs_review_count") == 0 and health.get("runtime_loadable_needs_review_count") == 0:
            report["status"] = "pass"
            for key in ("legal_units", "chunks"):
                report[key]["status"] = "pass_with_reviewed_exceptions"
    return report


def _strip_uud_header(text: str) -> str:
    return UUD_SATU_NASKAH_HEADER_RE.sub(" ", text or "")
