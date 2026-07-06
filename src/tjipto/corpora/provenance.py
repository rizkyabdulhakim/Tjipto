from __future__ import annotations

from tjipto.core.config import CorpusConfig
from tjipto.core.validation import validate_text_provenance
from tjipto.corpora.uud import provenance as uud_provenance


_PROVENANCE_ADAPTERS = {"uud": uud_provenance}


def validate_corpus_provenance(config: CorpusConfig) -> dict:
    adapter = _PROVENANCE_ADAPTERS.get(config.corpus_id)
    report = validate_text_provenance(
        config,
        header_stripper=getattr(adapter, "strip_source_header", None),
    )
    return adapter.apply_provenance_report_overrides(config, report) if adapter else report
