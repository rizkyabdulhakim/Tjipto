from __future__ import annotations

from tjipto.core.config import CorpusConfig
from tjipto.core.validation import validate_text_provenance
from tjipto.corpora.strategy import strategy_for


def validate_corpus_provenance(config: CorpusConfig) -> dict:
    try:
        adapter = strategy_for(config.corpus_id).provenance_adapter
    except ValueError:
        adapter = None
    report = validate_text_provenance(
        config,
        header_stripper=getattr(adapter, "strip_source_header", None),
    )
    return adapter.apply_provenance_report_overrides(config, report) if adapter else report
