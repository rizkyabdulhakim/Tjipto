from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from tjipto.retrieval.dense import DENSE_DIMENSION, DENSE_DTYPE, DENSE_NORMALIZATION, INDEX_BUILDER_ID, MODEL_ID, MODEL_REVISION


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if request.get("model_id") != MODEL_ID or request.get("revision") != MODEL_REVISION:
            return _error("noncanonical_model")
        texts = request.get("texts")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            return _error("worker_request_invalid")
        vectors, tokenizer_digest, model_digest = _embed(tuple(texts))
        json.dump(
            {
                "vectors": vectors,
                "model_identity": {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "dimension": DENSE_DIMENSION,
                    "dtype": DENSE_DTYPE,
                    "normalization": DENSE_NORMALIZATION,
                    "tokenizer_sha256": tokenizer_digest,
                    "model_sha256": model_digest,
                    "builder_identity": INDEX_BUILDER_ID,
                },
            },
            sys.stdout,
            ensure_ascii=False,
        )
        return 0
    except Exception as error:  # worker boundary: never expose model internals to the API process
        return _error("dense_unavailable" if not isinstance(error, ValueError) else str(error))


def _embed(texts: tuple[str, ...]):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ValueError("worker_dependencies_missing") from error
    cache_dir = os.environ.get("TJIPTO_DENSE_MODEL_DIR")
    local_snapshot = Path(cache_dir) if cache_dir and (Path(cache_dir) / "config.json").exists() else None
    model_source = str(local_snapshot) if local_snapshot else MODEL_ID
    kwargs = {"trust_remote_code": False, "local_files_only": True}
    if local_snapshot is None:
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned revision and local-only
    model = AutoModel.from_pretrained(model_source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned revision and local-only
    model.eval()
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    if pooled.shape[1] != DENSE_DIMENSION:
        raise ValueError("embedding_dimension_invalid")
    model_path = Path(getattr(model, "name_or_path", ""))
    tokenizer_path = Path(getattr(tokenizer, "name_or_path", ""))
    if not model_path.exists() and cache_dir:
        model_path = Path(cache_dir)
    if not tokenizer_path.exists() and cache_dir:
        tokenizer_path = Path(cache_dir)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt")
    model_files = ("config.json", "pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
    return pooled.cpu().tolist(), _files_digest(tokenizer_path, tokenizer_files), _files_digest(model_path, model_files)


def _files_digest(path: Path, names: tuple[str, ...]) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    if path.is_dir():
        paths = [path / name for name in names if (path / name).is_file()]
    else:
        paths = [path]
    if not paths:
        return None
    paths = sorted(paths)
    for file_path in paths:
        digest.update(file_path.relative_to(path if path.is_dir() else path.parent).as_posix().encode())
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _error(code: str) -> int:
    json.dump({"error": code}, sys.stdout)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
