from __future__ import annotations


def required_missing_corpus(corpus_id: str, query: str) -> str | None:
    return None


def classify_coverage(corpus_id: str, query: str) -> dict:
    return {
        "required_corpus": None,
        "missing_corpus": None,
        "coverage_warning": False,
        "answer_scope": None,
        "no_final_sectoral_legal_conclusion": False,
    }
