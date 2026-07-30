from __future__ import annotations

from pathlib import Path

from tjipto.catalog import CatalogRepository
from tjipto.corpora.regulations.catalog import documents as regulation_documents
from tjipto.corpora.uud.catalog import documents as constitutional_documents


def builtin_catalog(repo_root: Path, store_provider) -> CatalogRepository:
    """Single code-owned catalog composition root."""
    return CatalogRepository((*constitutional_documents(store_provider("uud")), *regulation_documents(repo_root)))
