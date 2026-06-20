from __future__ import annotations

EXPLICIT_NON_UUD = {
    "kuhp": "KUHP",
    "kuhap": "KUHAP",
    "uu pdp": "UU PDP",
    "uu ite": "UU ITE",
    "uu pers": "UU Pers",
    "uu ketenagakerjaan": "UU Ketenagakerjaan",
    "ketenagakerjaan": "UU Ketenagakerjaan",
    "uu pemilu": "UU Pemilu",
    "pemilu": "UU Pemilu",
    "uu perseroan": "UU Perseroan",
    "perseroan": "UU Perseroan",
    "uu kepolisian": "UU Kepolisian",
    "mk": "MK/MA",
    "ma": "MK/MA",
    "peraturan daerah": "peraturan daerah",
    "peraturan pemerintah": "peraturan pemerintah",
    "peraturan menteri": "peraturan menteri",
}

IMPLICIT_NON_UUD = (
    (("data pribadi", "privasi data"), "UU PDP"),
    (("pers", "wartawan", "jurnalis", "meliput"), "UU Pers"),
    (("pidana", "pencurian", "pembunuhan", "penganiayaan"), "KUHP"),
    (("penangkapan", "penyidikan", "prosedur polisi", "penahanan"), "KUHAP/UU Kepolisian"),
    (("partai", "caleg", "kampanye"), "UU Pemilu"),
    (("kerja", "buruh", "phk", "upah"), "UU Ketenagakerjaan"),
    (("perusahaan", "direksi", "komisaris", "saham"), "UU Perseroan"),
    (("transaksi elektronik", "media sosial", "pencemaran online"), "UU ITE"),
)

UUD_BASIS_TERMS = (
    "uud",
    "konstitusi",
    "undang-undang dasar",
    "undang undang dasar",
    "hak asasi",
    "hak warga",
    "pasal 28",
    "negara hukum",
    "dasar konstitusional",
)

UNKNOWN_SECTORAL_TERMS = (
    "izin usaha",
    "pajak",
    "waris",
    "perkawinan",
    "kontrak",
    "gugatan",
    "putusan",
    "sengketa",
    "sanksi administratif",
)


def required_missing_corpus(corpus_id: str, query: str) -> str | None:
    coverage = classify_coverage(corpus_id, query)
    if coverage["coverage_warning"]:
        return None
    return coverage["required_corpus"]


def classify_coverage(corpus_id: str, query: str) -> dict:
    if corpus_id != "uud":
        return _coverage(None)
    lowered = query.casefold()
    missing = _missing_corpus(lowered)
    has_uud_basis = any(term in lowered for term in UUD_BASIS_TERMS)
    if missing and has_uud_basis:
        return _coverage(missing, warning=True)
    if missing:
        return _coverage(missing)
    if any(term in lowered for term in UNKNOWN_SECTORAL_TERMS):
        return _coverage("unknown_non_uud_legal_domain")
    return _coverage(None)


def _missing_corpus(lowered: str) -> str | None:
    for term, corpus in EXPLICIT_NON_UUD.items():
        if term in lowered:
            return corpus
    for terms, corpus in IMPLICIT_NON_UUD:
        if any(term in lowered for term in terms):
            return corpus
    return None


def _coverage(required: str | None, *, warning: bool = False) -> dict:
    return {
        "required_corpus": required,
        "missing_corpus": required,
        "coverage_warning": warning,
        "answer_scope": "limited_to_uud_constitutional_basis" if warning else None,
        "no_final_sectoral_legal_conclusion": bool(required),
    }
