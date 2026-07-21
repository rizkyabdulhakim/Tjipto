from __future__ import annotations

import re

from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import read_json


SATU_NASKAH_HEADER_RE = re.compile(
    r"(?:\*\)\s*)?:?\s*Perubahan Pertama\s*"
    r"(?:\*\*\)\s*)?:?\s*Perubahan Kedua\s*"
    r"(?:\*\*\*\)\s*)?:?\s*Perubahan Ketiga\s*"
    r"(?:\*\*\*\*\)\s*)?:?\s*Perubahan Keempat\s*"
    r"(?:MAJELIS PERMUSYAWARATAN RAKYAT\s*"
    r"SEKRETARIAT JENDERAL\s*"
    r"UNDANGUNDANG DASAR\s*"
    r"NEGARA REPUBLIK INDONESIA TAHUN 1945\s*"
    r"DALAM SATU NASKAH)?",
    re.IGNORECASE,
)


def strip_source_header(text: str) -> str:
    return SATU_NASKAH_HEADER_RE.sub(" ", text or "")


def apply_provenance_report_overrides(config: CorpusConfig, report: dict) -> dict:
    health = read_json(config.manifest_path.parent / "validation_report.json").get("provenance_exception_health", {})
    report["provenance_exception_health"] = health
    if health.get("unresolved_needs_review_count") == 0 and health.get("runtime_loadable_needs_review_count") == 0:
        report["status"] = "pass"
        for key in ("legal_units", "chunks"):
            report[key]["status"] = "pass_with_reviewed_exceptions"
    return report
