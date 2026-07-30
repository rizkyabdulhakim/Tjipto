from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import unittest

from tjipto.core.manifest import read_json, read_jsonl
from tjipto.corpora.verified import VerifiedCorpusRepository
from tjipto.corpora.uud.validation import validate_uud_artifact_dir
from tjipto.runtime.api import handle_request
from tjipto.runtime.service import LegalRuntimeService
from tjipto.evidence.store import EvidenceStore


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/final/uud"


class VerifiedCorpusContractTest(unittest.TestCase):
    def setUp(self) -> None:
        EvidenceStore.clear_shared_cache()
        VerifiedCorpusRepository.clear_shared_cache()

    def tearDown(self) -> None:
        EvidenceStore.clear_shared_cache()
        VerifiedCorpusRepository.clear_shared_cache()

    def test_cached_snapshot_rejects_integrity_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            service = LegalRuntimeService(root)
            service.repository.load("uud")
            artifact = root / "data/final/uud/evidence_registry.jsonl"
            artifact.write_bytes(artifact.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "artifact_size_mismatch|artifact_sha256_mismatch"):
                service.repository.load("uud")
            self.assertEqual(service.repository.load_count, 1)

    def test_cached_snapshot_rejects_semantic_mutation_with_new_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            service = LegalRuntimeService(root)
            service.repository.load("uud")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["quoted_text"] = "semantic mutation"
            data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["evidence_registry.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            with self.assertRaisesRegex(ValueError, "runtime_validation_attestation_missing"):
                service.repository.load("uud")
            self.assertEqual(service.repository.load_count, 1)

    def test_exact_quote_mutation_fails_offline_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["quoted_text"] = "fabricated exact legal quote"
            data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["evidence_registry.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            self.assertTrue(any(error.startswith("EVIDENCE_QUOTE_SOURCE_MISMATCH") for error in validate_uud_artifact_dir(final)))
            result = LegalRuntimeService(root).ask("uud", "Pasal 1")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")
            self.assertFalse(result["citations"])

    def test_parent_aggregate_span_mutation_fails_offline_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            row = next(item for item in rows if item.get("citation") == "Pasal 31" and item.get("source_role") == "current_consolidated")
            row["text_span_ids"] = row["text_span_ids"][:-1]
            data = ("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["evidence_registry.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            errors = validate_uud_artifact_dir(final)
            self.assertTrue(any(error.startswith("aggregate_text_span_sequence_incomplete") for error in errors))
            result = LegalRuntimeService(root).ask("uud", "Pasal 31")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")
            self.assertFalse(result["citations"])

    def test_individual_signatory_grounding_mutation_fails_offline_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "metadata_grounding.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            row = next(item for item in rows if "::signatories::person_" in item["metadata_grounding_id"])
            row["quoted_text"] = "wrong signatory"
            data = ("\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["metadata_grounding.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            errors = validate_uud_artifact_dir(final)
            self.assertTrue(any(error.startswith("signatory_grounding_unknown_name:") for error in errors))
            result = LegalRuntimeService(root).ask("uud", "Amien Rais Perubahan Pertama UUD")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")
            self.assertFalse(result["citations"])

    def test_source_sha_mutation_fails_lineage_before_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["source_sha256"] = "0" * 64
            data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["evidence_registry.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            self.assertTrue(any(error.startswith("EVIDENCE_SOURCE_LINEAGE_INVALID") for error in validate_uud_artifact_dir(final)))
            result = LegalRuntimeService(root).ask("uud", "Pasal 1")
            self.assertFalse(result["citations"])

    def test_manifest_contract_cannot_remove_minimum_quote_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            rows[0].pop("quoted_text")
            data = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            record = manifest["files"]["evidence_registry.jsonl"]
            record["required_fields"] = ["evidence_id"]
            record.update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            result = LegalRuntimeService(root).ask("uud", "Pasal 1")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")
            self.assertFalse(result["citations"])

    def test_trust_anchor_and_semantic_row_failures_are_typed(self) -> None:
        for expected, mutate in (
            ("trusted_manifest_missing", lambda registry, manifest, final: registry["uud"].pop("manifest_sha256")),
            ("trusted_manifest_mismatch", lambda registry, manifest, final: registry["uud"].__setitem__("manifest_sha256", "0" * 64)),
            (
                "semantic_artifact_identity_mismatch",
                lambda registry, manifest, final: manifest["files"][manifest["legal_units"]].__setitem__("artifact_kind", "chunks"),
            ),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / "data", root / "data")
                final = root / "data/final/uud"
                manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
                registry_path = root / "data/corpus_registry.json"
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                mutate(registry, manifest, final)
                if expected == "semantic_artifact_identity_mismatch":
                    _write_trusted_manifest(root, manifest)
                else:
                    registry_path.write_text(json.dumps(registry), encoding="utf-8")
                result = LegalRuntimeService(root).ask("uud", "Pasal 7")
                self.assertEqual(result["reason_code"], expected)
                self.assertFalse(result["context_pack"]["citation_payloads"])

    def test_trusted_manifest_semantic_identity_and_immutability(self) -> None:
        substitutions = {
            "legal_units": "chunks",
            "source_documents": "pages",
            "bbox_registry": "word_bboxes",
            "graph_nodes": "graph_edges",
            "retrieval_units": "evidence_registry",
        }
        for logical_key, replacement in substitutions.items():
            with self.subTest(logical_key=logical_key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / "data", root / "data")
                manifest_path = root / "data/final/uud/manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest[logical_key] = manifest[replacement]
                _write_trusted_manifest(root, manifest)
                result = LegalRuntimeService(root).ask("uud", "Pasal 7")
                self.assertEqual(result["reason_code"], "semantic_artifact_identity_mismatch")
                self.assertFalse(result["citations"])

        service = LegalRuntimeService(ROOT)
        first = service.repository.load("uud")
        second = service.repository.load("uud")
        self.assertIs(first, second)
        service.ask("uud", "Pasal 7")
        service.search("uud", "Pasal 7")
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertTrue(all(snapshot is first for snapshot in pool.map(service.repository.load, ["uud"] * 16)))
        self.assertLessEqual(service.repository.load_count, 1)
        with self.assertRaises(TypeError):
            first.artifacts["runtime_projection.json"]["artifacts"]["evidence_registry"][0]["quoted_text"] = "tampered"
        with self.assertRaises(TypeError):
            first.artifacts["runtime_projection.json"]["artifacts"]["evidence_registry"][0]["page_numbers"][0] = 999

    def test_process_publication_cache_is_manifest_path_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            first_service = LegalRuntimeService(root)
            second_service = LegalRuntimeService(root)
            first = first_service.repository.load("uud")
            second = second_service.repository.load("uud")
            self.assertIs(first, second)
            self.assertEqual(first_service.repository.load_count + second_service.repository.load_count, 1)

    def test_runtime_store_cache_has_one_owner_per_snapshot(self) -> None:
        EvidenceStore.clear_shared_cache()
        first = LegalRuntimeService(ROOT)
        second = LegalRuntimeService(ROOT)
        self.assertIs(first._store("uud"), second._store("uud"))
        self.assertEqual(len(EvidenceStore._shared_stores), 1)
        EvidenceStore.clear_shared_cache()
        self.assertEqual(len(EvidenceStore._shared_stores), 0)

    def test_duplicate_primary_id_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "evidence_registry.jsonl"
            first = artifact.read_bytes().splitlines()[0]
            artifact.write_bytes(artifact.read_bytes() + first + b"\n")
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            record = manifest["files"]["evidence_registry.jsonl"]
            data = artifact.read_bytes()
            record.update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            result = LegalRuntimeService(root).ask("uud", "Pasal 7")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")

    def test_cross_artifact_reference_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            final = root / "data/final/uud"
            artifact = final / "retrieval_units.jsonl"
            rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines() if line]
            rows[0]["evidence_id"] = "missing"
            data = ("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n").encode("utf-8")
            artifact.write_bytes(data)
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["files"]["retrieval_units.jsonl"].update({"bytes": len(data), "sha256": sha256(data).hexdigest()})
            _write_trusted_manifest(root, manifest)
            result = LegalRuntimeService(root).ask("uud", "Pasal 7")
            self.assertEqual(result["reason_code"], "runtime_validation_attestation_missing")

    def test_integrity_envelope_is_consistent_across_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "data", root / "data")
            (root / "data/final/uud/evidence_registry.jsonl").write_text("tampered", encoding="utf-8")
            service = LegalRuntimeService(root)
            responses = (
                service.ask("uud", "Pasal 7"),
                service.search("uud", "Pasal 7"),
                service.citation("uud", "Pasal 7"),
                service.viewer("uud", None),
                service.capabilities("uud"),
                service.bookmarks("uud"),
            )
            for result in responses:
                self.assertEqual(result["status"], "corpus_not_ready")
                self.assertFalse(result["readiness"])
                self.assertFalse(result["citations"])
                self.assertFalse(result["viewer_refs"])
                self.assertFalse(result["context_pack"]["citation_payloads"])
            for action, payload in (
                ("ask", {"query": "Pasal 7"}),
                ("search", {"query": "Pasal 7"}),
                ("citation", {"query": "Pasal 7"}),
                ("capabilities", {}),
                ("bookmarks", {}),
            ):
                result = handle_request("uud", action, payload, root)
                self.assertEqual(result, {"kind": "unavailable", "status": "unavailable"})

    def test_tampered_or_invalid_artifacts_cannot_answer(self) -> None:
        cases = (
            ("evidence_registry.jsonl", lambda path, manifest: path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")),
            ("graph_edges.jsonl", lambda path, manifest: path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")),
            ("bbox_registry.jsonl", lambda path, manifest: path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")),
            ("retrieval_units.jsonl", lambda path, manifest: path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")),
            ("missing", lambda path, manifest: (FINAL / "evidence_registry.jsonl").unlink()),
            ("wrong_size", lambda path, manifest: manifest["files"]["evidence_registry.jsonl"].__setitem__("bytes", 0)),
            ("wrong_sha", lambda path, manifest: manifest["files"]["evidence_registry.jsonl"].__setitem__("sha256", "0" * 64)),
            ("unsupported_schema", lambda path, manifest: manifest.__setitem__("schema_version", 1)),
            ("absolute_path", lambda path, manifest: manifest.__setitem__("evidence_registry", "C:/outside.jsonl")),
            ("path_traversal", lambda path, manifest: manifest.__setitem__("evidence_registry", "../outside.jsonl")),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                shutil.copytree(ROOT / "data", root / "data")
                final = root / "data/final/uud"
                manifest_path = final / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                target = final / "evidence_registry.jsonl"
                if name == "missing":
                    target.unlink()
                else:
                    mutate(target, manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                result = LegalRuntimeService(root).ask("uud", "Pasal 7")
                self.assertEqual(result["status"], "corpus_not_ready")
                self.assertFalse(result["readiness"])
                self.assertFalse(result["citations"])
                self.assertFalse(result["viewer_refs"])

    def test_span_graph_and_intent_contracts(self) -> None:
        spans = read_jsonl(FINAL / "page_text_spans.jsonl")
        words = {row["word_bbox_id"]: row for row in read_jsonl(FINAL / "word_bboxes.jsonl")}
        for row in spans:
            self.assertFalse(any(key.startswith("exposure_") or key == "target_bbox_ids" for key in row))
            self.assertTrue({"span_bbox_ids", "evidence_ids", "metadata_grounding_ids", "page_text_hash", "text_start", "text_end"} <= row.keys())
            self.assertNotIn("evidence_bbox_ids", row)
            self.assertNotIn("context_bbox_ids", row)
            self.assertLessEqual(row["text_start"], row["text_end"])
            for bbox_id in row["span_bbox_ids"]:
                bbox = words.get(bbox_id)
                if bbox is None:
                    continue
                self.assertEqual((row["source_document_id"], row["page_number"]), (bbox["source_document_id"], bbox["page_number"]))
                self.assertGreater(min(row["x1"], bbox["x1"]), max(row["x0"], bbox["x0"]))
                self.assertGreater(min(row["y1"], bbox["y1"]), max(row["y0"], bbox["y0"]))
        edges = read_jsonl(FINAL / "graph_edges.jsonl")
        self.assertFalse(
            any(any(key in row for key in ("edge_authority_level", "evidence_requirement", "relation_support")) for row in edges)
        )
        service = LegalRuntimeService(ROOT)
        for query in (
            "pasal berikutnya setelah Pasal 7",
            "Pasal apa berikutnya setelah Pasal 7",
            "Pasal apa berikutnya setelah Pasal 7 tentang Presiden",
            "setelah Pasal 7 pasal berapa",
            "sesudah Pasal 7 apa",
        ):
            result = service.ask("uud", query)
            self.assertEqual(
                (result["status"], result["route"], result["citations"][0]["citation"]),
                ("answer_ready", "structural_navigation", "Pasal 7A"),
            )
        for query in (
            "Siapa Presiden setelah Pasal 7?",
            "Siapa presiden berikutnya setelah Pasal 7?",
            "Setelah Pasal 7 berlaku, siapa Presiden?",
            "Siapa Presiden saat ini?",
            "Presiden Indonesia sekarang siapa?",
        ):
            result = service.ask("uud", query)
            self.assertEqual(result["route"], "current_fact_unsupported")
            self.assertFalse(result["citations"])
        telemetry = read_json(FINAL / "validation_report.json").get("amendment_context_default_boundary_health", {})
        self.assertEqual(telemetry["runtime_health_mode"], "test_suite_owned")
        self.assertNotIn("actual_elapsed_ms", telemetry)


def _write_trusted_manifest(root: Path, manifest: dict) -> None:
    manifest_path = root / "data/final/uud/manifest.json"
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(data)
    registry_path = root / "data/corpus_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["uud"]["manifest_sha256"] = sha256(data).hexdigest()
    registry_path.write_text(json.dumps(registry), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
