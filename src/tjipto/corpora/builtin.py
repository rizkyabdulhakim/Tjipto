"""Explicit built-in corpus composition root."""

from __future__ import annotations

from tjipto.contracts.artifacts import CURRENT_ARTIFACT_SCHEMA
from tjipto.corpora.strategy import CorpusContract, CorpusStrategy
from tjipto.corpora.uud import parser as uud_parser
from tjipto.corpora.uud import provenance as uud_provenance
from tjipto.corpora.uud.contract import CONTRACT_FINGERPRINT, CONTRACT_ID, CONTRACT_VERSION
from tjipto.corpora.uud.validation import validate_uud_artifacts


BUILTIN_STRATEGIES = {
    "uud": CorpusStrategy(
        "uud",
        uud_parser.parser_adapter(),
        uud_parser.uud_proposition_operator,
        contract=CorpusContract(CURRENT_ARTIFACT_SCHEMA, CONTRACT_ID, CONTRACT_VERSION, CONTRACT_FINGERPRINT),
        provenance_adapter=uud_provenance,
        semantic_validator=validate_uud_artifacts,
    ),
}
