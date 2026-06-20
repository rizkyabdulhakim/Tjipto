from __future__ import annotations

import re

from tjipto.core.config import CorpusConfig
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
    return validate_text_provenance(
        config,
        header_stripper=_strip_uud_header if config.corpus_id == "uud" else None,
    )


def _strip_uud_header(text: str) -> str:
    return UUD_SATU_NASKAH_HEADER_RE.sub(" ", text or "")
