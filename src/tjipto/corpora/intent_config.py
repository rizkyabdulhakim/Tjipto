from __future__ import annotations

import re


_GENERIC = {
    "metadata_fields": {},
    "metadata_roles": (),
    "relation_words": (),
    "direct_relation_words": (),
    "pasal_parent_words": (),
}

_UUD_1945 = {
    "metadata_fields": {
        "penetapan": ("ditetapkan", "penetapan", "enactment"),
        "promulgation": ("diundangkan", "pengundangan", "promulgation"),
        "revocation": ("dicabut", "pencabutan", "revocation"),
        "date": ("tanggal", "kapan"),
        "institution": ("lembaga", "institusi", "majelis", "mpr", "ditetapkan oleh"),
        "place": ("tempat", "di mana", "dimana"),
        "signatories": ("penanda tangan", "ditandatangani", "ketua", "wakil ketua"),
        "decision_date": ("diputuskan",),
        "decision_session": ("rapat", "sidang"),
        "effective_rule": ("berlaku",),
        "source_publication": ("satu naskah", "source publication", "publikasi"),
        "source_anomaly_status": ("konflik sumber", "anomali sumber", "typo sumber"),
        "official_title": ("judul", "nama resmi", "naskah"),
    },
    "metadata_roles": (
        ("amendment_1_historical", re.compile(r"\b(perubahan\s*(pertama|1|i)|amendment\s*1)\b", re.IGNORECASE)),
        ("amendment_2_historical", re.compile(r"\b(perubahan\s*(kedua|2|ii)|amendment\s*2)\b", re.IGNORECASE)),
        ("amendment_3_historical", re.compile(r"\b(perubahan\s*(ketiga|3|iii)|amendment\s*3)\b", re.IGNORECASE)),
        ("amendment_4_historical", re.compile(r"\b(perubahan\s*(keempat|4|iv)|amendment\s*4)\b", re.IGNORECASE)),
        ("current_consolidated", re.compile(r"\b(satu\s+naskah|konsolidasi|current|berlaku)\b", re.IGNORECASE)),
        ("original_historical", re.compile(r"\b(naskah\s+asli|original)\b", re.IGNORECASE)),
    ),
    "relation_words": ("relasi", "hubungan", "naskah asli", "versi", "historis"),
    "direct_relation_words": ("mengubah", "diubah", "amandemen", "amended", "amends"),
    "pasal_parent_words": ("bagian dari", "berada di bab", "berada dalam bab", "masuk bab", "termasuk bab"),
}

_BY_STRATEGY = {
    "generic": _GENERIC,
    "uud_1945": _UUD_1945,
}


def intent_config_for(strategy: str | None) -> dict:
    return _BY_STRATEGY.get(strategy or "generic", _GENERIC)
