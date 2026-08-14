from __future__ import annotations

import json
import os
import sys
import ctypes
from ctypes import wintypes
from pathlib import Path

from tjipto.retrieval.dense import (
    DENSE_BATCH_SIZE,
    DENSE_ALLOWED_MAX_LENGTHS,
    DENSE_DIMENSION,
    DENSE_DTYPE,
    DENSE_MAX_LENGTH,
    DENSE_NORMALIZATION,
    DENSE_POOLING,
    DENSE_TRUNCATION_POLICY,
    INDEX_BUILDER_ID,
    MODEL_ID,
    MODEL_REVISION,
    _files_digest,
    _sha256_file,
)


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if request.get("model_id") != MODEL_ID or request.get("revision") != MODEL_REVISION:
            return _error("noncanonical_model")
        texts = request.get("texts")
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            return _error("worker_request_invalid")
        batch_size = request.get("batch_size", DENSE_BATCH_SIZE)
        max_length = request.get("max_length", DENSE_MAX_LENGTH)
        if (
            not isinstance(batch_size, int)
            or batch_size < 1
            or max_length not in DENSE_ALLOWED_MAX_LENGTHS
        ):
            return _error("worker_configuration_invalid")
        vectors, tokenizer_digest, model_digest, pooling_config_digest, truncated_indices, worker_peak_rss = _embed(
            tuple(texts), batch_size=batch_size, max_length=max_length
        )
        json.dump(
            {
                "vectors": vectors,
                "truncated_indices": truncated_indices,
                "worker_peak_rss_bytes": worker_peak_rss,
                "model_identity": {
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "dimension": DENSE_DIMENSION,
                    "dtype": DENSE_DTYPE,
                    "normalization": DENSE_NORMALIZATION,
                    "pooling": DENSE_POOLING,
                    "max_length": max_length,
                    "truncation_policy": DENSE_TRUNCATION_POLICY,
                    "tokenizer_sha256": tokenizer_digest,
                    "model_sha256": model_digest,
                    "pooling_config_sha256": pooling_config_digest,
                    "builder_identity": INDEX_BUILDER_ID,
                },
            },
            sys.stdout,
            ensure_ascii=False,
        )
        return 0
    except Exception as error:  # worker boundary: never expose model internals to the API process
        return _error("dense_unavailable" if not isinstance(error, ValueError) else str(error))


def _embed(texts: tuple[str, ...], *, batch_size: int, max_length: int):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise ValueError("worker_dependencies_missing") from error
    cache_dir = os.environ.get("TJIPTO_DENSE_MODEL_DIR")
    local_snapshot = Path(cache_dir) if cache_dir and (Path(cache_dir) / "config.json").exists() else None
    if local_snapshot is None:
        raise ValueError("model_snapshot_unavailable")
    if local_snapshot.name != MODEL_REVISION:
        raise ValueError("noncanonical_model_snapshot")
    model_source = str(local_snapshot)
    kwargs: dict[str, object] = {"trust_remote_code": False, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned revision and local-only
    model = AutoModel.from_pretrained(model_source, revision=MODEL_REVISION, **kwargs)  # nosec B615 - pinned revision and local-only
    model.eval()
    vectors: list[list[float]] = []
    truncated_indices: list[int] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        untruncated = tokenizer(
            list(batch), padding=False, truncation=False, add_special_tokens=True, return_length=True
        )
        lengths = tuple(int(length) for length in untruncated.get("length", ()))
        truncated_indices.extend(offset + index for index, length in enumerate(lengths) if length > max_length)
        inputs = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**inputs).last_hidden_state
            pooled = torch.nn.functional.normalize(_cls_pool(output), p=2, dim=1)
        if pooled.shape[1] != DENSE_DIMENSION:
            raise ValueError("embedding_dimension_invalid")
        vectors.extend(pooled.cpu().tolist())
    model_path = Path(getattr(model, "name_or_path", ""))
    tokenizer_path = Path(getattr(tokenizer, "name_or_path", ""))
    if not model_path.exists() and cache_dir:
        model_path = Path(cache_dir)
    if not tokenizer_path.exists() and cache_dir:
        tokenizer_path = Path(cache_dir)
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt")
    model_files = ("config.json", "pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
    pooling_path = model_path / "1_Pooling" / "config.json"
    if not pooling_path.is_file():
        raise ValueError("pooling_config_missing")
    try:
        pooling_config = json.loads(pooling_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("pooling_config_invalid") from error
    if (
        pooling_config.get("word_embedding_dimension") != DENSE_DIMENSION
        or pooling_config.get("pooling_mode_cls_token") is not True
        or pooling_config.get("pooling_mode_mean_tokens") is True
    ):
        raise ValueError("pooling_config_invalid")
    pooling_digest = _sha256_file(pooling_path)
    if pooling_digest is None:
        raise ValueError("pooling_config_missing")
    return (
        vectors,
        _files_digest(tokenizer_path, tokenizer_files),
        _files_digest(model_path, model_files),
        pooling_digest,
        truncated_indices,
        _peak_rss_bytes(),
    )


def _cls_pool(last_hidden_state):
    return last_hidden_state[:, 0, :]


def _peak_rss_bytes() -> int | None:
    if os.name == "nt":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        try:
            loader = getattr(ctypes, "WinDLL", None)
            if loader is None:
                return None
            kernel32 = loader("kernel32", use_last_error=True)
            psapi = loader("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            get_info = psapi.GetProcessMemoryInfo
            get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD)
            get_info.restype = wintypes.BOOL
            ok = get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError):
            return None
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _error(code: str) -> int:
    json.dump({"error": code}, sys.stdout)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
