"""Explicit built-in corpus composition root."""

from __future__ import annotations

from tjipto.contracts.artifacts import CURRENT_ARTIFACT_SCHEMA
from tjipto.corpora.capabilities import resolve_capability
from tjipto.corpora.strategy import CorpusContract, CorpusStrategy
from tjipto.corpora.uud import parser as uud_parser
from tjipto.corpora.uud import provenance as uud_provenance
from tjipto.corpora.uud.citation import citation_unit
from tjipto.corpora.uud.contract import CONTRACT_FINGERPRINT, CONTRACT_ID, CONTRACT_VERSION
from tjipto.corpora.uud.source_annotations import annotation_health, query_source_annotations
from tjipto.corpora.uud.validation import validate_uud_artifacts


BUILTIN_STRATEGIES = {
    "uud": CorpusStrategy(
        "uud",
        uud_parser.query_normalizer(),
        uud_parser.reference_parser(),
        uud_parser.navigation_resolver(),
        uud_parser.uud_proposition_operator,
        capability_resolver=resolve_capability,
        contract=CorpusContract(CURRENT_ARTIFACT_SCHEMA, CONTRACT_ID, CONTRACT_VERSION, CONTRACT_FINGERPRINT),
        provenance_adapter=uud_provenance,
        semantic_validator=validate_uud_artifacts,
        citation_unit_factory=citation_unit,
        source_text_query=query_source_annotations,
        source_text_health=annotation_health,
    ),
}
