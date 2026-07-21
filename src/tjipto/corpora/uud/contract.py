from __future__ import annotations

from hashlib import sha256
import json

from tjipto.contracts.artifacts import ARTIFACT_ALLOWED_FIELDS, ARTIFACT_OPTIONAL_FIELDS, MINIMUM_ARTIFACT_FIELDS


CORPUS_ID = "uud"
CONTRACT_ID = "tjipto.uud.artifact-contract"
CONTRACT_VERSION = 7


def contract_definition() -> dict:
    return {
        "corpus_id": CORPUS_ID,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "artifacts": {
            name: {
                "required": list(MINIMUM_ARTIFACT_FIELDS.get(name, ())),
                "optional": sorted(ARTIFACT_OPTIONAL_FIELDS.get(name, ())),
                "allowed": sorted(ARTIFACT_ALLOWED_FIELDS.get(name, ())),
            }
            for name in sorted(MINIMUM_ARTIFACT_FIELDS)
        },
    }


CONTRACT_FINGERPRINT = sha256(
    json.dumps(contract_definition(), sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
