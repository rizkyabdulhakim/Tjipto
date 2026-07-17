from __future__ import annotations

from collections.abc import Callable

from tjipto.corpora.uud.structure_builder import slice_before, slice_between, split_effective_clause


def append_amendment_instrument_units(
    *,
    pages_by_source: dict[tuple[str, int], str],
    append_instrument_unit: Callable[..., str],
    trim_unit: Callable[..., None],
    trim_bab: Callable[[str, str, str], None],
) -> None:
    # Amendment 1
    source_id = "uud::amendment_1_historical"
    page1 = pages_by_source[(source_id, 1)]
    page3 = pages_by_source[(source_id, 3)]
    recital_text = slice_before(page1, "Setelah mempelajari").strip()
    scope_text = (
        slice_between(page1, "Setelah mempelajari", "selengkapnya menjadi berbunyi sebagai berikut :").strip()
        + " selengkapnya menjadi berbunyi sebagai berikut :"
    )
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Pertama Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Pertama Scope", scope_text, 1, 1)
    trim_unit(source_id, "Pasal 21", "Naskah perubahan ini merupakan bagian tak terpisahkan")
    closing = slice_between(page3, "Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan").strip()
    decision, effective = split_effective_clause(slice_between(page3, "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta").strip())
    determination = slice_between(page3, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page3[page3.index("MAJELIS PERMUSYAWARATAN RAKYAT") :].strip()
    append_instrument_unit(source_id, "instrument_closing_record", "Perubahan Pertama Closing", closing, 3, 3)
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Pertama Decision", decision, 3, 3)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Pertama Effective", effective, 3, 3, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Pertama Determination", determination, 3, 3)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Pertama Signatories", signatory, 3, 3)

    # Amendment 2
    source_id = "uud::amendment_2_historical"
    page1 = pages_by_source[(source_id, 1)]
    page8 = pages_by_source[(source_id, 8)]
    recital_text = slice_before(page1, "Setelah Mempelajari").strip()
    scope_text = (
        slice_between(page1, "Setelah Mempelajari", "sehingga selengkapnya berbunyi sebagai berikut :").strip()
        + " sehingga selengkapnya berbunyi sebagai berikut :"
    )
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Kedua Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Kedua Scope", scope_text, 1, 1)
    trim_bab(source_id, "BAB XV", "Ditetapkan di Jakarta")
    trim_unit(source_id, "Pasal 36C", "Ditetapkan di Jakarta")
    determination = slice_between(page8, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page8[page8.index("MAJELIS PERMUSYAWARATAN RAKYAT") :].strip()
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Kedua Determination", determination, 8, 8)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Kedua Signatories", signatory, 8, 8)

    # Amendment 3
    source_id = "uud::amendment_3_historical"
    page1 = pages_by_source[(source_id, 1)]
    page8 = pages_by_source[(source_id, 8)]
    page9 = pages_by_source[(source_id, 9)]
    recital_text = slice_before(page1, "Setelah mempelajari").strip()
    scope_text = (
        slice_between(page1, "Setelah mempelajari", "sehingga selengkapnya menjadi berbunyi sebagai berikut:").strip()
        + " sehingga selengkapnya menjadi berbunyi sebagai berikut:"
    )
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Ketiga Recital", recital_text, 1, 1)
    append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Ketiga Scope", scope_text, 1, 1)
    trim_unit(source_id, "Pasal 24C", "Naskah perubahan ini merupakan bagian tak terpisahkan")
    trim_unit(source_id, "(6)", "Naskah perubahan ini merupakan bagian tak terpisahkan", hierarchy_suffix=("Pasal 24C", "(6)"))
    closing = slice_between(page8, "Naskah perubahan ini merupakan", "Perubahan tersebut diputuskan").strip()
    decision, effective = split_effective_clause(page8[page8.index("Perubahan tersebut diputuskan") :].strip())
    determination = slice_between(page9, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page9[page9.index("MAJELIS PERMUSYAWARATAN RAKYAT") :].strip()
    append_instrument_unit(source_id, "instrument_closing_record", "Perubahan Ketiga Closing", closing, 8, 8)
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Ketiga Decision", decision, 8, 8)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Ketiga Effective", effective, 8, 8, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Ketiga Determination", determination, 9, 9)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Ketiga Signatories", signatory, 9, 9)

    # Amendment 4
    source_id = "uud::amendment_4_historical"
    page1 = pages_by_source[(source_id, 1)]
    page5 = pages_by_source[(source_id, 5)]
    page6 = pages_by_source[(source_id, 6)]
    recital_text = slice_before(page1, "(a)").strip()
    scope_text = slice_between(page1, "(a)", "berikut.").strip() + " berikut."
    append_instrument_unit(source_id, "amendment_recital_record", "Perubahan Keempat Recital", recital_text, 1, 1)
    scope_unit_id = append_instrument_unit(source_id, "amendment_scope_record", "Perubahan Keempat Scope", scope_text, 1, 1)
    for clause in ("(a)", "(b)", "(c)", "(d)", "(e)"):
        next_clause = {"(a)": "(b)", "(b)": "(c)", "(c)": "(d)", "(d)": "(e)", "(e)": "berikut."}[clause]
        clause_text = slice_between(page1, clause, next_clause).strip()
        append_instrument_unit(
            source_id,
            "instrument_clause_record",
            f"Perubahan Keempat Clause {clause}",
            clause_text,
            1,
            1,
            hierarchy=["Perubahan Keempat Scope", clause],
            parent_legal_unit_ids=[scope_unit_id],
        )
    aturan_text = slice_between(page5 + "\n" + page6, "ATURAN TAMBAHAN", "Perubahan tersebut diputuskan").strip()
    aturan_page5_text = page5[page5.index("ATURAN TAMBAHAN") :]
    pasal_i_text = aturan_page5_text[aturan_page5_text.index("Pasal I") :].strip()
    pasal_iii_text = slice_between(page6, "Pasal III", "Perubahan tersebut diputuskan").strip()
    anomaly_ref = "source_typo_reference::uud_source_typo_reference_00001"
    aturan_unit_id = append_instrument_unit(
        source_id,
        "aturan_tambahan_record",
        "ATURAN TAMBAHAN source typo reference",
        aturan_text,
        5,
        6,
        hierarchy=["ATURAN TAMBAHAN"],
        chunk_type="aturan_section_context_record",
        canonical_use_allowed=False,
        chunk_status="inactive_source_typo_reference",
        runtime_loadable=False,
        exclusion_ref=anomaly_ref,
        build_evidence=False,
    )
    append_instrument_unit(
        source_id,
        "pasal_record",
        "Pasal I",
        pasal_i_text,
        5,
        5,
        hierarchy=["ATURAN TAMBAHAN", "Pasal I"],
        parent_legal_unit_ids=[aturan_unit_id],
        chunk_type="pasal_chunk_record",
        canonical_use_allowed=False,
        chunk_status="active_historical_record",
        exclusion_ref=anomaly_ref,
    )
    append_instrument_unit(
        source_id,
        "pasal_record",
        "Pasal III",
        pasal_iii_text,
        6,
        6,
        hierarchy=["ATURAN TAMBAHAN", "Pasal III"],
        parent_legal_unit_ids=[aturan_unit_id],
        chunk_type="pasal_chunk_record",
        canonical_use_allowed=False,
        chunk_status="active_historical_record",
        exclusion_ref=anomaly_ref,
    )
    decision, effective = split_effective_clause(slice_between(page6, "Perubahan tersebut diputuskan", "Ditetapkan di Jakarta").strip())
    determination = slice_between(page6, "Ditetapkan di Jakarta", "MAJELIS PERMUSYAWARATAN RAKYAT").strip()
    signatory = page6[page6.index("MAJELIS PERMUSYAWARATAN RAKYAT") :].strip()
    append_instrument_unit(source_id, "decision_clause_record", "Perubahan Keempat Decision", decision, 6, 6)
    append_instrument_unit(source_id, "effective_clause_record", "Perubahan Keempat Effective", effective, 6, 6, build_evidence=False)
    append_instrument_unit(source_id, "determination_clause_record", "Perubahan Keempat Determination", determination, 6, 6)
    append_instrument_unit(source_id, "signatory_block_record", "Perubahan Keempat Signatories", signatory, 6, 6)
