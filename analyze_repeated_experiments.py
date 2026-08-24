from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from losd import (
    LOSDConfig,
    LOSDPipeline,
    VariantSpec,
    load_cached_candidate_texts,
    load_ontology_resources,
    normalize_surface_text,
    parse_candidate_items,
)


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    directory: str


EXPERIMENTS = (
    ExperimentSpec("DeepSeek V3", "exp_repeated_deepseek_v3_openrouter_3runs_20260821"),
    ExperimentSpec("GPT-5", "exp_repeated_gpt5_openrouter_3runs_20260821"),
    ExperimentSpec("GPT-OSS 120B", "exp_repeated_gpt_oss_120b_together_3runs_20260821"),
    ExperimentSpec("Kimi K2", "exp_repeated_kimi_k2_openrouter_3runs_4096_20260821"),
    ExperimentSpec("Llama 4 Scout", "exp_repeated_llama4_scout_openrouter_groq_3runs_20260821"),
    ExperimentSpec("Mistral Large", "exp_repeated_mistral_large_2407_openrouter_3runs_20260821"),
    ExperimentSpec("Qwen3", "exp_repeated_qwen3_openrouter_3runs_8192_20260821"),
)

METHODS = ("zero", "few", "rag")
METHOD_LABELS = {"zero": "Zero-shot", "few": "Few-shot", "rag": "RAG"}
VARIANTS = (
    VariantSpec("baseline"),
    VariantSpec("validation_rerank", validation_mode="hard", use_rerank=True),
    VariantSpec("soft_validation", validation_mode="soft"),
    VariantSpec("soft_validation_rerank", validation_mode="soft", use_rerank=True),
)
METRICS = ("semantic_f1", "hier_f1")
PAPER_EXPECTED_EMBEDDING_KEY_SHA256 = (
    "65f9144faff284caebca2702951dbb053b60ca352b63deee8e541394bbed18c5"
)


class CachingSentenceTransformerEmbedder:
    """Sentence-transformer wrapper that avoids re-encoding repeated strings."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 64,
        backend: str = "onnx",
        revision: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = backend
        self.revision = revision
        model_kwargs = None
        if backend == "onnx":
            model_kwargs = {
                "provider": "CPUExecutionProvider",
                "file_name": "onnx/model.onnx",
            }
        self.model = SentenceTransformer(
            model_name,
            backend=backend,
            model_kwargs=model_kwargs,
            revision=revision,
        )
        self.cache: dict[str, np.ndarray] = {}
        get_dimension = (
            self.model.get_embedding_dimension
            if hasattr(self.model, "get_embedding_dimension")
            else self.model.get_sentence_embedding_dimension
        )
        self.dimension = int(get_dimension())
        if int(self.model.max_seq_length) != 128:
            raise RuntimeError(
                f"Unexpected embedding maximum sequence length: {self.model.max_seq_length}"
            )

    def _encode_missing(self, texts: Sequence[str], show_progress: bool = False) -> None:
        missing = list(dict.fromkeys(str(text) for text in texts if str(text) not in self.cache))
        if not missing:
            return
        vectors = self.model.encode(
            missing,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        for text, vector in zip(missing, vectors):
            self.cache[text] = np.asarray(vector, dtype=np.float32)

    def preload(self, texts: Iterable[str]) -> None:
        unique = list(dict.fromkeys(str(text) for text in texts))
        self._encode_missing(unique, show_progress=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        normalized = [str(text) for text in texts]
        self._encode_missing(normalized)
        return np.vstack([self.cache[text] for text in normalized])

    def load_cache(self, path: Path) -> None:
        if not path.exists():
            return
        print(f"Loading persistent embedding cache: {path}", flush=True)
        with path.open("rb") as stream:
            stored = pickle.load(stream)
        if not isinstance(stored, dict):
            raise RuntimeError(f"Invalid embedding cache payload: {path}")
        for text, vector in stored.items():
            self.cache[str(text)] = np.asarray(vector, dtype=np.float32)

    def save_cache(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        print(f"Saving {len(self.cache):,} embeddings to {path}", flush=True)
        with temporary.open("wb") as stream:
            pickle.dump(self.cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all repeated model caches and compute parent-paired statistics."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ttl", type=Path, default=Path("esco_cmo_binding.ttl"))
    parser.add_argument("--output-dir", type=Path, default=Path("repeated_analysis"))
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument(
        "--embedding-revision",
        default="4328cf26390c98c5e3c738b4460a05b95f4911f5",
        help="Pinned Hugging Face commit for the evaluation encoder.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-backend",
        choices=("onnx", "torch"),
        default="onnx",
        help="Inference backend for the unchanged sentence-transformer checkpoint.",
    )
    parser.add_argument(
        "--long-text-torch-cutoff",
        type=int,
        default=80,
        help=(
            "When using ONNX, encode strings longer than this character count "
            "with the equivalent PyTorch backend before the ONNX short-text pass."
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    parser.add_argument("--embedding-cache", type=Path, default=Path("embedding_cache.pkl"))
    parser.add_argument(
        "--expected-embedding-key-sha256",
        default=None,
        help=(
            "Optional exact corpus-hash assertion. Use the archived-paper hash only "
            "with the archived response caches; fresh generations legitimately differ. "
            f"Archived-paper value: {PAPER_EXPECTED_EMBEDDING_KEY_SHA256}."
        ),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_text_keys(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(str(item) for item in values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_embedding_cache_provenance(
    cache_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
) -> None:
    if not cache_path.exists():
        return
    if not metadata_path.exists():
        raise RuntimeError(
            f"Embedding cache exists without provenance metadata: {cache_path}"
        )
    metadata = read_json(metadata_path)
    expected = {
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "maximum_sequence_length": 128,
        "normalization": "l2",
        "primary_backend": args.embedding_backend,
        "embedding_batch_size": args.embedding_batch_size,
        "long_text_torch_cutoff_characters": args.long_text_torch_cutoff,
    }
    mismatches = {
        key: {"expected": value, "recorded": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Embedding cache provenance mismatch: {mismatches}")
    actual_hash = sha256_file(cache_path)
    if actual_hash.lower() != str(metadata.get("cache_sha256") or "").lower():
        raise RuntimeError(
            f"Embedding cache checksum mismatch for {cache_path}: {actual_hash}"
        )


def write_embedding_cache_provenance(
    cache_path: Path,
    metadata_path: Path,
    embedder: CachingSentenceTransformerEmbedder,
    args: argparse.Namespace,
) -> None:
    previous = read_json(metadata_path) if metadata_path.exists() else {}
    first_vector = next(iter(embedder.cache.values()), np.zeros(0, dtype=np.float32))
    payload = {
        "schema_version": 1,
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "onnx_artifact_sha256": previous.get("onnx_artifact_sha256"),
        "pytorch_weights_sha256": previous.get("pytorch_weights_sha256"),
        "maximum_sequence_length": 128,
        "normalization": "l2",
        "primary_backend": args.embedding_backend,
        "embedding_batch_size": args.embedding_batch_size,
        "long_text_torch_cutoff_characters": args.long_text_torch_cutoff,
        "vector_count": len(embedder.cache),
        "vector_dimension": int(first_vector.shape[0]) if first_vector.ndim == 1 else None,
        "vector_dtype": str(first_vector.dtype),
        "cache_sha256": sha256_file(cache_path),
        "backend_equivalence_audit": previous.get("backend_equivalence_audit"),
    }
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_loaded_embedding_cache(
    embedder: CachingSentenceTransformerEmbedder,
    metadata_path: Path,
) -> None:
    metadata = read_json(metadata_path)
    expected_count = int(metadata.get("vector_count", -1))
    expected_dimension = int(metadata.get("vector_dimension", -1))
    expected_dtype = str(metadata.get("vector_dtype") or "")
    if len(embedder.cache) != expected_count:
        raise RuntimeError(
            f"Embedding vector-count mismatch: {len(embedder.cache)} != {expected_count}"
        )
    for text, vector in embedder.cache.items():
        if vector.shape != (expected_dimension,):
            raise RuntimeError(f"Invalid embedding shape for {text!r}: {vector.shape}")
        if str(vector.dtype) != expected_dtype:
            raise RuntimeError(f"Invalid embedding dtype for {text!r}: {vector.dtype}")
        if not np.isfinite(vector).all():
            raise RuntimeError(f"Non-finite embedding values for {text!r}")
        if not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-5):
            raise RuntimeError(f"Non-unit-normalized embedding for {text!r}")


def write_analysis_environment(
    args: argparse.Namespace,
    output_dir: Path,
    ttl: Path,
) -> None:
    packages = {}
    for name in (
        "sentence-transformers",
        "transformers",
        "torch",
        "onnxruntime",
        "optimum",
        "optimum-onnx",
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "rdflib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "embedding_model": args.embedding_model,
        "embedding_revision": args.embedding_revision,
        "embedding_backend": args.embedding_backend,
        "embedding_batch_size": args.embedding_batch_size,
        "long_text_torch_cutoff_characters": args.long_text_torch_cutoff,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "expected_embedding_key_sha256": args.expected_embedding_key_sha256,
        "ontology_path": str(ttl),
        "ontology_sha256": sha256_file(ttl),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "pipeline_script_sha256": sha256_file(
            Path(__file__).resolve().with_name("losd.py")
        ),
        "command_line": [sys.executable, *sys.argv],
        "packages": packages,
    }
    (output_dir / "analysis_environment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def complete_run_dirs(root: Path, spec: ExperimentSpec) -> list[Path]:
    experiment_dir = root / spec.directory
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")
    run_dirs = sorted(path for path in experiment_dir.glob("run_*" ) if path.is_dir())
    if len(run_dirs) != 3:
        raise RuntimeError(f"{spec.label}: expected 3 run directories, found {len(run_dirs)}")
    for run_dir in run_dirs:
        metadata = read_json(run_dir / "run_metadata.json")
        cache_count = len(list((run_dir / "cache").glob("*.json")))
        if metadata.get("status") != "complete" or cache_count != 864:
            raise RuntimeError(
                f"{spec.label}/{run_dir.name}: status={metadata.get('status')!r}, "
                f"cache_count={cache_count}; expected complete/864"
            )
    return run_dirs


def manifest_index(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads((run_dir / "prompt_manifest.json").read_text(encoding="utf-8"))
    index = {(str(row["parent_uri"]), str(row["method"])): row for row in rows}
    if len(index) != 864:
        raise RuntimeError(f"{run_dir}: prompt manifest has {len(index)} unique tasks, expected 864")
    return index


def context_from_manifest(row: dict[str, Any]) -> list[str]:
    method = str(row["method"])
    if method == "rag":
        return [str(item) for item in row.get("context_items", [])]
    if method == "few":
        output: list[str] = []
        for example in row.get("fewshot_examples", []):
            if isinstance(example, dict):
                output.extend(str(item) for item in example.get("children", []))
            elif isinstance(example, (list, tuple)) and len(example) == 2:
                output.extend(str(item) for item in example[1])
        return output
    return []


def collect_candidate_texts(root: Path, run_map: dict[str, list[Path]]) -> set[str]:
    texts: set[str] = set()
    for spec in EXPERIMENTS:
        for run_dir in run_map[spec.label]:
            for path in (run_dir / "cache").glob("*.json"):
                payload = read_json(path)
                parsed_items = parse_candidate_items(
                    str(payload.get("raw_text") or ""), max_items=12
                )
                source_items = parsed_items or payload.get("items", [])
                for item in source_items:
                    text = str(item)
                    key = normalize_surface_text(text)[1] or text
                    texts.add(key)
            manifest = manifest_index(run_dir)
            for row in manifest.values():
                for item in context_from_manifest(row):
                    key = normalize_surface_text(item)[1] or item
                    texts.add(key)
    return texts


def validate_prompt_identity(run_map: dict[str, list[Path]]) -> pd.DataFrame:
    reference: dict[tuple[str, str], str] | None = None
    rows: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        model_reference: dict[tuple[str, str], str] | None = None
        for run_dir in run_map[spec.label]:
            index = manifest_index(run_dir)
            hashes = {key: str(row["prompt_hash"]) for key, row in index.items()}
            if model_reference is None:
                model_reference = hashes
            elif hashes != model_reference:
                raise RuntimeError(f"Prompt hashes differ across runs for {spec.label}")
            if reference is None:
                reference = hashes
            elif hashes != reference:
                raise RuntimeError(f"Prompt hashes differ across models for {spec.label}")
            rows.append(
                {
                    "model_label": spec.label,
                    "run_id": run_dir.name,
                    "task_count": len(hashes),
                    "prompt_hashes_identical_across_runs_and_models": True,
                }
            )
    return pd.DataFrame(rows)


def configuration_rows(run_map: dict[str, list[Path]]) -> pd.DataFrame:
    fields = (
        "model",
        "api_service",
        "api_base_url",
        "openrouter_provider",
        "allow_provider_fallbacks",
        "require_parameters",
        "temperature",
        "top_p",
        "max_tokens",
        "token_limit_parameter",
        "reasoning",
        "reasoning_effort",
        "api_seed",
        "nominal_run_seed",
        "prompt_seed",
        "prompt_contract",
        "candidate_count",
        "embedding_model",
        "started_at_utc",
        "completed_at_utc",
        "config_hash",
        "ttl_sha256",
    )
    output: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for run_dir in run_map[spec.label]:
            metadata = read_json(run_dir / "run_metadata.json")
            row = {"model_label": spec.label, "experiment_dir": spec.directory, "run_id": run_dir.name}
            row.update({field: metadata.get(field) for field in fields})
            output.append(row)
    return pd.DataFrame(output)


def generation_metadata_rows(run_map: dict[str, list[Path]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for run_dir in run_map[spec.label]:
            for path in (run_dir / "cache").glob("*.json"):
                payload = read_json(path)
                usage = payload.get("usage") or {}
                output.append(
                    {
                        "model_label": spec.label,
                        "run_id": run_dir.name,
                        "prompt_mode": payload.get("method"),
                        "parent_uri": payload.get("parent_uri"),
                        "resolved_model": payload.get("resolved_model"),
                        "resolved_provider": payload.get("resolved_provider"),
                        "finish_reason": payload.get("finish_reason"),
                        "elapsed_ms": payload.get("elapsed_ms"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                            "reasoning_tokens"
                        ),
                        "total_tokens": usage.get("total_tokens"),
                        "cost_usd": usage.get("cost"),
                        "api_attempt": payload.get("attempt", 1),
                        "response_attempt": payload.get("response_attempt", 1),
                    }
                )
    return pd.DataFrame(output)


def cache_quality_rows(
    run_map: dict[str, list[Path]],
    candidate_count: int = 12,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    """Audit visible outputs and identify unusable incomplete responses.

    A normal ``stop`` response is retained even if the model returned fewer
    than the requested number of distinct items. A ``length`` response is
    retained only when all requested candidates are visibly parseable. Other
    non-stop responses are excluded from scoring. Raw evidence remains intact.
    """
    rows: list[dict[str, Any]] = []
    run_fingerprints: dict[tuple[str, str], str] = {}
    for spec in EXPERIMENTS:
        for run_dir in run_map[spec.label]:
            run_digest = hashlib.sha256()
            for path in sorted((run_dir / "cache").glob("*.json")):
                raw_bytes = path.read_bytes()
                cache_sha256 = hashlib.sha256(raw_bytes).hexdigest()
                run_digest.update(path.name.encode("utf-8"))
                run_digest.update(cache_sha256.encode("ascii"))
                payload = json.loads(raw_bytes)
                stored_items = [
                    str(item) for item in payload.get("items", []) if isinstance(item, str)
                ]
                parsed_items = parse_candidate_items(
                    str(payload.get("raw_text") or ""), max_items=candidate_count
                )
                finish_reason = str(payload.get("finish_reason") or "")
                eligible = finish_reason == "stop" or (
                    finish_reason == "length" and len(parsed_items) == candidate_count
                )
                rows.append(
                    {
                        "model_label": spec.label,
                        "run_id": run_dir.name,
                        "prompt_mode": payload.get("method"),
                        "parent_uri": payload.get("parent_uri"),
                        "finish_reason": finish_reason,
                        "stored_item_count": len(stored_items),
                        "reparsed_item_count": len(parsed_items),
                        "semicolon_fallback_changed_count": len(parsed_items) != len(stored_items),
                        "eligible_for_analysis": eligible,
                        "exclusion_reason": "" if eligible else "incomplete_nonstop_response",
                        "cache_sha256": cache_sha256,
                        "cache_path": str(path),
                    }
                )
            run_fingerprints[(spec.label, run_dir.name)] = run_digest.hexdigest()
    return pd.DataFrame(rows), run_fingerprints


def summarize_generation_metadata(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = records.copy()
    for column in (
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cost_usd",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    run_summary = (
        numeric.groupby(
            ["model_label", "resolved_model", "resolved_provider", "run_id", "prompt_mode"],
            as_index=False,
            dropna=False,
        )
        .agg(
            calls=("parent_uri", "count"),
            elapsed_seconds_mean=("elapsed_ms", lambda values: values.mean() / 1000.0),
            elapsed_seconds_median=("elapsed_ms", lambda values: values.median() / 1000.0),
            elapsed_seconds_p95=("elapsed_ms", lambda values: values.quantile(0.95) / 1000.0),
            prompt_tokens_mean=("prompt_tokens", "mean"),
            completion_tokens_mean=("completion_tokens", "mean"),
            reasoning_tokens_mean=("reasoning_tokens", "mean"),
            total_tokens_mean=("total_tokens", "mean"),
            cost_usd_total=("cost_usd", "sum"),
        )
    )
    repeated_rows: list[dict[str, Any]] = []
    for keys, group in run_summary.groupby(
        ["model_label", "resolved_model", "resolved_provider", "prompt_mode"], dropna=False
    ):
        latency = group["elapsed_seconds_mean"].to_numpy(dtype=float)
        ci_low, ci_high = t_interval(latency)
        row = {
                "model_label": keys[0],
                "resolved_model": keys[1],
                "resolved_provider": keys[2],
                "prompt_mode": keys[3],
                "runs": len(group),
                "elapsed_seconds_mean": float(latency.mean()),
                "elapsed_seconds_sd": float(latency.std(ddof=1)),
                "elapsed_seconds_ci_low": ci_low,
                "elapsed_seconds_ci_high": ci_high,
                "cost_usd_total_all_runs": float(group["cost_usd_total"].sum()),
            }
        for token_column in (
            "prompt_tokens_mean",
            "completion_tokens_mean",
            "reasoning_tokens_mean",
            "total_tokens_mean",
        ):
            values = group[token_column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[token_column] = float(finite.mean()) if len(finite) else math.nan
            row[f"{token_column}_sd"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else math.nan
            )
        repeated_rows.append(row)
    return run_summary, pd.DataFrame(repeated_rows)


def plot_generation_latency(summary: pd.DataFrame, output_dir: Path) -> None:
    model_order = [spec.label for spec in EXPERIMENTS]
    prompt_order = list(METHODS)
    colors = {"zero": "#4C72B0", "few": "#DD8452", "rag": "#55A868"}
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    positions = np.arange(len(model_order), dtype=float)
    width = 0.24
    for offset, prompt in enumerate(prompt_order):
        subset = (
            summary[summary["prompt_mode"] == prompt]
            .set_index("model_label")
            .reindex(model_order)
        )
        axis.bar(
            positions + (offset - 1) * width,
            subset["elapsed_seconds_mean"],
            width,
            yerr=subset["elapsed_seconds_sd"],
            capsize=2,
            color=colors[prompt],
            label=METHOD_LABELS[prompt],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Mean API latency per parent (seconds, log scale)")
    axis.set_xticks(positions, model_order, rotation=20, ha="right")
    axis.grid(axis="y", which="both", linestyle="--", alpha=0.3)
    axis.legend(ncols=3, frameon=False)
    figure.savefig(output_dir / "repeated_latency_logscale.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "repeated_latency_logscale.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_runs(
    root: Path,
    run_map: dict[str, list[Path]],
    pipeline: LOSDPipeline,
    partition_dir: Path,
    excluded_tasks: set[tuple[str, str, str, str]],
    evaluation_fingerprint: str,
    run_cache_fingerprints: dict[tuple[str, str], str],
) -> pd.DataFrame:
    partition_dir.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, Any]] = []
    for spec in EXPERIMENTS:
        for run_number, run_dir in enumerate(run_map[spec.label], start=1):
            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", spec.label).strip("_").lower()
            partition_path = partition_dir / f"{safe_label}__{run_dir.name}.csv"
            metadata_path = partition_path.with_suffix(".meta.json")
            run_exclusions = {
                (method, parent_uri)
                for model_label, run_id, method, parent_uri in excluded_tasks
                if model_label == spec.label and run_id == run_dir.name
            }
            expected_tasks = 288 * len(METHODS) - len(run_exclusions)
            expected_rows = expected_tasks * len(VARIANTS)
            partition_fingerprint = sha256_payload(
                {
                    "evaluation_fingerprint": evaluation_fingerprint,
                    "model_label": spec.label,
                    "run_id": run_dir.name,
                    "run_config_hash": read_json(run_dir / "run_metadata.json").get("config_hash"),
                    "run_cache_fingerprint": run_cache_fingerprints[(spec.label, run_dir.name)],
                    "gold_pairs_sha256": sha256_file(run_dir / "gold_pairs.csv"),
                    "prompt_manifest_sha256": sha256_file(run_dir / "prompt_manifest.json"),
                    "excluded_tasks": sorted(run_exclusions),
                }
            )
            if partition_path.exists():
                if not metadata_path.exists():
                    raise RuntimeError(
                        f"Evaluation checkpoint lacks provenance metadata: {partition_path}"
                    )
                partition_metadata = read_json(metadata_path)
                if partition_metadata.get("fingerprint") != partition_fingerprint:
                    raise RuntimeError(
                        f"Stale evaluation checkpoint fingerprint: {partition_path}"
                    )
                partition = pd.read_csv(partition_path)
                key_columns = ["parent_skill_id", "prompt_mode", "variant"]
                if len(partition) != expected_rows or partition.duplicated(key_columns).any():
                    raise RuntimeError(
                        f"Invalid evaluation checkpoint {partition_path}: "
                        f"{len(partition)} rows, expected {expected_rows} unique rows"
                    )
                print(f"Reusing evaluation checkpoint: {partition_path}", flush=True)
                output.extend(partition.to_dict(orient="records"))
                continue
            print(f"Evaluating {spec.label} {run_dir.name} ...", flush=True)
            run_output: list[dict[str, Any]] = []
            gold = pd.read_csv(run_dir / "gold_pairs.csv")
            parents = gold["parent_uri"].drop_duplicates().tolist()
            if len(parents) != 288:
                raise RuntimeError(f"{spec.label}/{run_dir.name}: expected 288 parents")
            manifest = manifest_index(run_dir)
            model_id = str(read_json(run_dir / "run_metadata.json")["model"])
            for parent_index, parent_uri in enumerate(parents, start=1):
                gold_children = (
                    gold.loc[gold["parent_uri"] == parent_uri, "child_uri"]
                    .drop_duplicates()
                    .tolist()
                )
                for method in METHODS:
                    if (method, parent_uri) in run_exclusions:
                        continue
                    task = manifest[(parent_uri, method)]
                    raw_candidates = load_cached_candidate_texts(
                        run_dir / "cache", method, parent_uri
                    )
                    if not raw_candidates:
                        raise RuntimeError(
                            f"No candidates for {spec.label}/{run_dir.name}/{method}/{parent_uri}"
                        )
                    _, parent_rows = pipeline.run_parent(
                        parent_skill_id=parent_uri,
                        prompt_mode=method,
                        raw_candidates=raw_candidates,
                        gold_children=gold_children,
                        context_texts=context_from_manifest(task),
                        variants=VARIANTS,
                    )
                    for row in parent_rows:
                        row.update(
                            {
                                "model_label": spec.label,
                                "model_id": model_id,
                                "run_id": run_dir.name,
                                "run_number": run_number,
                            }
                        )
                        run_output.append(row)
                if parent_index % 48 == 0:
                    print(
                        f"  {spec.label} {run_dir.name}: {parent_index}/288 parents",
                        flush=True,
                    )
            partition = pd.DataFrame(run_output)
            key_columns = ["parent_skill_id", "prompt_mode", "variant"]
            if len(partition) != expected_rows or partition.duplicated(key_columns).any():
                raise RuntimeError(
                    f"Generated invalid checkpoint for {spec.label}/{run_dir.name}: "
                    f"{len(partition)} rows, expected {expected_rows} unique rows"
                )
            partition.to_csv(partition_path, index=False)
            metadata_path.write_text(
                json.dumps(
                    {
                        "fingerprint": partition_fingerprint,
                        "expected_tasks": expected_tasks,
                        "expected_rows": expected_rows,
                        "excluded_tasks": sorted(run_exclusions),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            output.extend(run_output)
            print(f"Saved evaluation checkpoint: {partition_path}", flush=True)
    return pd.DataFrame(output)


def restrict_to_complete_repetition_parents(
    parent_results: pd.DataFrame,
    required_runs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = parent_results[
        ["model_label", "prompt_mode", "parent_skill_id", "run_id"]
    ].drop_duplicates()
    counts = (
        observed.groupby(
            ["model_label", "prompt_mode", "parent_skill_id"], as_index=False
        )["run_id"]
        .nunique()
        .rename(columns={"run_id": "observed_runs"})
    )
    complete_keys = counts.loc[
        counts["observed_runs"] == required_runs,
        ["model_label", "prompt_mode", "parent_skill_id"],
    ]
    filtered = parent_results.merge(
        complete_keys,
        on=["model_label", "prompt_mode", "parent_skill_id"],
        how="inner",
        validate="many_to_one",
    )
    availability = (
        counts.groupby(["model_label", "prompt_mode"], as_index=False)
        .agg(
            parents_with_any_observation=("parent_skill_id", "nunique"),
            complete_case_parents=(
                "observed_runs",
                lambda values: int((values == required_runs).sum()),
            ),
        )
    )
    availability["parents_excluded_for_incomplete_repetitions"] = (
        288 - availability["complete_case_parents"]
    )
    return filtered, availability


def t_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, mean
    sem = float(stats.sem(array))
    critical = float(stats.t.ppf(0.975, len(array) - 1))
    return mean - critical * sem, mean + critical * sem


def summarize_runs(parent_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["model_label", "model_id", "run_id", "run_number", "prompt_mode", "variant"]
    run_summary = (
        parent_results.groupby(group_columns, as_index=False)
        .agg(
            parent_count=("parent_skill_id", "nunique"),
            semantic_precision=("semantic_precision", "mean"),
            semantic_recall=("semantic_recall", "mean"),
            semantic_f1=("semantic_f1", "mean"),
            hier_precision=("hier_precision", "mean"),
            hier_recall=("hier_recall", "mean"),
            hier_f1=("hier_f1", "mean"),
            selected_count=("selected_count", "mean"),
            plausible_novel_count=("plausible_novel_count", "mean"),
            low_support_unresolved_count=("hallucination_count", "mean"),
        )
    )

    repeated_rows: list[dict[str, Any]] = []
    measure_columns = [
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "hier_precision",
        "hier_recall",
        "hier_f1",
        "selected_count",
        "plausible_novel_count",
        "low_support_unresolved_count",
    ]
    for keys, group in run_summary.groupby(["model_label", "model_id", "prompt_mode", "variant"]):
        row: dict[str, Any] = {
            "model_label": keys[0],
            "model_id": keys[1],
            "prompt_mode": keys[2],
            "variant": keys[3],
            "runs": len(group),
        }
        for measure in measure_columns:
            values = group[measure].to_numpy(dtype=float)
            ci_low, ci_high = t_interval(values)
            row[f"{measure}_mean"] = float(values.mean())
            row[f"{measure}_sd"] = float(values.std(ddof=1))
            row[f"{measure}_ci_low"] = ci_low
            row[f"{measure}_ci_high"] = ci_high
        repeated_rows.append(row)
    return run_summary, pd.DataFrame(repeated_rows)


def bootstrap_mean_ci(
    differences: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(differences)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 2_000
    cursor = 0
    while cursor < resamples:
        size = min(chunk, resamples - cursor)
        indices = rng.integers(0, n, size=(size, n))
        means[cursor : cursor + size] = differences[indices].mean(axis=1)
        cursor += size
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def stable_seed(base: int, *parts: str) -> int:
    suffix = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8]
    return (base + int(suffix, 16)) % (2**32)


def paired_test(differences: np.ndarray) -> tuple[float, float]:
    nonzero = differences[~np.isclose(differences, 0.0)]
    if len(nonzero) == 0:
        return 0.0, 1.0
    result = stats.wilcoxon(
        differences,
        zero_method="pratt",
        alternative="two-sided",
        method="asymptotic",
    )
    return float(result.statistic), float(result.pvalue)


def paired_losd_comparisons(
    parent_results: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    averaged = (
        parent_results[parent_results["variant"].isin(["baseline", "soft_validation_rerank"])]
        .groupby(["model_label", "prompt_mode", "variant", "parent_skill_id"], as_index=False)[list(METRICS)]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for (model, prompt), group in averaged.groupby(["model_label", "prompt_mode"]):
        for metric in METRICS:
            wide = group.pivot(index="parent_skill_id", columns="variant", values=metric).dropna()
            differences = (
                wide["soft_validation_rerank"] - wide["baseline"]
            ).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(
                differences,
                resamples,
                stable_seed(seed, model, prompt, metric, "losd"),
            )
            statistic, p_value = paired_test(differences)
            rows.append(
                {
                    "model_label": model,
                    "prompt_mode": prompt,
                    "metric": metric,
                    "parents": len(differences),
                    "baseline_parent_mean": float(wide["baseline"].mean()),
                    "soft_rerank_parent_mean": float(wide["soft_validation_rerank"].mean()),
                    "paired_difference": float(differences.mean()),
                    "difference_ci_low": ci_low,
                    "difference_ci_high": ci_high,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    output = pd.DataFrame(rows)
    output["p_holm"] = np.nan
    for metric, indices in output.groupby("metric").groups.items():
        output.loc[indices, "p_holm"] = holm_adjust(output.loc[indices, "p_value"])
    output["significant_holm_0_05"] = output["p_holm"] < 0.05
    return output


def paired_prompt_comparisons(
    parent_results: pd.DataFrame,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    averaged = (
        parent_results[parent_results["variant"] == "baseline"]
        .groupby(["model_label", "prompt_mode", "parent_skill_id"], as_index=False)[list(METRICS)]
        .mean()
    )
    comparisons = (("rag", "zero"), ("rag", "few"), ("few", "zero"))
    rows: list[dict[str, Any]] = []
    for model, group in averaged.groupby("model_label"):
        for left, right in comparisons:
            for metric in METRICS:
                wide = group.pivot(index="parent_skill_id", columns="prompt_mode", values=metric).dropna()
                differences = (wide[left] - wide[right]).to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(
                    differences,
                    resamples,
                    stable_seed(seed, model, left, right, metric, "prompt"),
                )
                statistic, p_value = paired_test(differences)
                rows.append(
                    {
                        "model_label": model,
                        "comparison": f"{left}_minus_{right}",
                        "metric": metric,
                        "parents": len(differences),
                        "left_parent_mean": float(wide[left].mean()),
                        "right_parent_mean": float(wide[right].mean()),
                        "paired_difference": float(differences.mean()),
                        "difference_ci_low": ci_low,
                        "difference_ci_high": ci_high,
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )
    output = pd.DataFrame(rows)
    output["p_holm"] = np.nan
    for metric, indices in output.groupby("metric").groups.items():
        output.loc[indices, "p_holm"] = holm_adjust(output.loc[indices, "p_value"])
    output["significant_holm_0_05"] = output["p_holm"] < 0.05
    return output


def plot_effect_heatmaps(comparisons: pd.DataFrame, output_dir: Path) -> None:
    model_order = [spec.label for spec in EXPERIMENTS]
    prompt_order = list(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        METRICS,
        ("Semantic F1: Soft rerank - baseline", "Hierarchy F1: Soft rerank - baseline"),
    ):
        subset = comparisons[comparisons["metric"] == metric]
        matrix = (
            subset.pivot(index="model_label", columns="prompt_mode", values="paired_difference")
            .reindex(index=model_order, columns=prompt_order)
            .to_numpy()
        )
        limit = max(0.01, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(prompt_order)), [METHOD_LABELS[item] for item in prompt_order])
        axis.set_yticks(range(len(model_order)), model_order)
        axis.set_title(title, fontsize=10)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=axis, shrink=0.75, label="Paired mean difference")
    fig.savefig(output_dir / "losd_effect_heatmaps.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "losd_effect_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "$<.001$"
    return f"${value:.3f}$".replace("0.", ".")


def write_latex_tables(
    repeated: pd.DataFrame,
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_order = [spec.label for spec in EXPERIMENTS]
    repeated = repeated.copy()
    repeated["model_label"] = pd.Categorical(repeated["model_label"], model_order, ordered=True)
    repeated["prompt_mode"] = pd.Categorical(repeated["prompt_mode"], METHODS, ordered=True)
    baseline = repeated[repeated["variant"] == "baseline"].sort_values(["model_label", "prompt_mode"])
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Prompt & Semantic F1, mean $\pm$ SD [95\% CI] & Hier-F1, mean $\pm$ SD [95\% CI] \\",
        r"\midrule",
    ]
    for row in baseline.itertuples():
        sem = (
            f"{row.semantic_f1_mean:.4f} $\\pm$ {row.semantic_f1_sd:.4f} "
            f"[{row.semantic_f1_ci_low:.4f}, {row.semantic_f1_ci_high:.4f}]"
        )
        hier = (
            f"{row.hier_f1_mean:.4f} $\\pm$ {row.hier_f1_sd:.4f} "
            f"[{row.hier_f1_ci_low:.4f}, {row.hier_f1_ci_high:.4f}]"
        )
        lines.append(
            f"{row.model_label} & {METHOD_LABELS[str(row.prompt_mode)]} & "
            f"{sem} & {hier} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "table_baseline_repeated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for metric, short_name, caption_metric in (
        ("hier_f1", "hier", "Hier-F1"),
        ("semantic_f1", "semantic", "Semantic F1"),
    ):
        soft = repeated[repeated["variant"] == "soft_validation_rerank"].copy()
        base = repeated[repeated["variant"] == "baseline"].copy()
        merge_columns = ["model_label", "prompt_mode"]
        table = base.merge(soft, on=merge_columns, suffixes=("_base", "_soft"))
        tests = comparisons[comparisons["metric"] == metric]
        table = table.merge(tests, on=merge_columns)
        table = table.sort_values(merge_columns)
        lines = [
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            f"Model & Prompt & Baseline {caption_metric} & Soft-rerank {caption_metric} & "
            r"Paired $\Delta$ [95\% CI] & Holm $p$ \\",
            r"\midrule",
        ]
        mean_col = f"{metric}_mean"
        sd_col = f"{metric}_sd"
        for _, row in table.iterrows():
            baseline_cell = f"{row[f'{mean_col}_base']:.4f} $\\pm$ {row[f'{sd_col}_base']:.4f}"
            soft_cell = f"{row[f'{mean_col}_soft']:.4f} $\\pm$ {row[f'{sd_col}_soft']:.4f}"
            delta_cell = (
                f"{row['paired_difference']:+.4f} "
                f"[{row['difference_ci_low']:+.4f}, {row['difference_ci_high']:+.4f}]"
            )
            lines.append(
                f"{row['model_label']} & {METHOD_LABELS[str(row['prompt_mode'])]} & "
                f"{baseline_cell} & {soft_cell} & {delta_cell} & "
                f"{fmt_p(row['p_holm'])} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}"])
        (output_dir / f"table_losd_{short_name}_effects.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def write_validation_report(
    parent_results: pd.DataFrame,
    repeated: pd.DataFrame,
    losd: pd.DataFrame,
    prompt: pd.DataFrame,
    output_dir: Path,
    ttl: Path,
    args: argparse.Namespace,
    quality: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    significant_losd = int(losd["significant_holm_0_05"].sum())
    significant_prompt = int(prompt["significant_holm_0_05"].sum())
    negative_semantic = int(
        ((losd["metric"] == "semantic_f1") & (losd["paired_difference"] < 0)).sum()
    )
    positive_hierarchy = int(
        ((losd["metric"] == "hier_f1") & (losd["paired_difference"] > 0)).sum()
    )
    eligible_cells = int(quality["eligible_for_analysis"].sum())
    excluded_cells = int((~quality["eligible_for_analysis"]).sum())
    min_parents = int(availability["complete_case_parents"].min())
    max_parents = int(availability["complete_case_parents"].max())
    analyzed_generation_cells = int(
        len(parent_results) // parent_results["variant"].nunique()
    )
    min_prompt_parents = int(prompt["parents"].min())
    max_prompt_parents = int(prompt["parents"].max())
    report = f"""# Repeated-Experiment Validation Report

## Material Passport

- Verification Status: ANALYZED
- Models: 7
- Independent repetitions: 3 per fixed model-prompt configuration
- Benchmark parents: 288
- Complete-case parents per model-prompt cell: {min_parents}--{max_parents}
- Prompting methods: 3
- Recorded API responses: {len(quality):,}
- Eligible response cells: {eligible_cells:,}
- Incomplete non-stop response cells excluded: {excluded_cells:,}
- Complete-case analyzed generation cells: {analyzed_generation_cells:,}
- Evaluated parent-variant rows: {len(parent_results):,}
- Common complete-case parents per model for prompt contrasts: {min_prompt_parents}--{max_prompt_parents}
- Ontology SHA-256: `{sha256_file(ttl)}`

## Statistical procedure

- Run summaries use the same complete-case parent set in all three repetitions of each model-prompt cell.
- Mean, sample SD, and t-based 95% CIs use the three run-level macro-means.
- Inferential comparisons first average the three repetitions for each parent.
- Soft Validation Rerank is paired against the shared-preprocessing baseline within model and prompt.
- Prompt comparisons use that baseline and the same paired-parent procedure.
- Two-sided Wilcoxon signed-rank tests use Pratt handling for zero differences.
- Holm correction is applied in four prespecified 21-test families: LOSD Semantic F1, LOSD Hier-F1, prompting Semantic F1, and prompting Hier-F1.
- Paired mean-difference CIs use {args.bootstrap_resamples:,} parent-level bootstrap resamples.
- Visible response text is reparsed uniformly; only unmistakable boilerplate headings are removed, and a single-line semicolon fallback is accepted only when it yields exactly 12 items.
- Responses ending in `stop` are retained; `length` is retained only with all 12 visible items; other non-stop responses are excluded before complete-case restriction.

## Headline checks

- LOSD comparisons surviving Holm correction: {significant_losd}/42.
- Prompt comparisons surviving Holm correction: {significant_prompt}/42.
- Semantic-F1 LOSD differences below zero: {negative_semantic}/21 model-prompt cells.
- Hier-F1 LOSD differences above zero: {positive_hierarchy}/21 model-prompt cells.

## Fallacy scan (11/11 checked)

1. Simpson's paradox: aggregate results are not used in place of model-stratified results.
2. Ecological fallacy: claims remain at parent/model configuration level.
3. Berkson's paradox: not applicable to the fixed ontology benchmark sample.
4. Collider bias: no covariate adjustment is performed.
5. Base-rate neglect: not applicable to the reported paired F1 comparisons.
6. Regression to the mean: no selection by extreme run performance.
7. Survivorship bias: exclusions are enumerated in `excluded_generation_tasks.csv`, and each affected parent is removed from all three repetitions of that model-prompt cell.
8. Look-elsewhere effect: all prespecified model-prompt comparisons are reported with Holm correction.
9. Garden of forking paths: fixed LOSD parameters and a fixed analysis script are used; no threshold search is performed.
10. Correlation/causation: comparisons concern controlled pipeline configurations, not population causal effects.
11. Reverse causality: not applicable to the controlled configuration comparisons.

## Interpretation boundary

With only three repetitions, run-level SD estimates and t intervals are imprecise. Parent-paired tests quantify consistency across the benchmark parents after averaging repetitions; they do not make the three runs behave like 864 independent stochastic replications. Provider/model identifiers and omitted decoding parameters must be disclosed exactly as recorded in `experiment_configuration.csv`.
"""
    (output_dir / "validation_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    ttl = (root / args.ttl).resolve() if not args.ttl.is_absolute() else args.ttl.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_analysis_environment(args, output_dir, ttl)
    embedding_cache = (
        output_dir / args.embedding_cache
        if not args.embedding_cache.is_absolute()
        else args.embedding_cache.resolve()
    )
    embedding_cache_metadata = embedding_cache.with_suffix(".meta.json")
    validate_embedding_cache_provenance(
        embedding_cache, embedding_cache_metadata, args
    )

    run_map = {spec.label: complete_run_dirs(root, spec) for spec in EXPERIMENTS}
    prompt_validation = validate_prompt_identity(run_map)
    prompt_validation.to_csv(output_dir / "prompt_identity_check.csv", index=False)
    configuration_rows(run_map).to_csv(output_dir / "experiment_configuration.csv", index=False)
    generation_records = generation_metadata_rows(run_map)
    generation_run_summary, generation_repeated_summary = summarize_generation_metadata(
        generation_records
    )
    generation_records.to_csv(output_dir / "generation_call_records.csv", index=False)
    generation_run_summary.to_csv(output_dir / "generation_run_summary.csv", index=False)
    generation_repeated_summary.to_csv(
        output_dir / "generation_repeated_summary.csv", index=False
    )
    plot_generation_latency(generation_repeated_summary, output_dir)

    quality_rows, run_cache_fingerprints = cache_quality_rows(run_map, candidate_count=12)
    quality_rows.to_csv(output_dir / "generation_output_quality_audit.csv", index=False)
    excluded_quality = quality_rows[~quality_rows["eligible_for_analysis"]].copy()
    excluded_quality.to_csv(output_dir / "excluded_generation_tasks.csv", index=False)
    excluded_tasks = {
        (
            str(row.model_label),
            str(row.run_id),
            str(row.prompt_mode),
            str(row.parent_uri),
        )
        for row in excluded_quality.itertuples()
    }

    parent_results_path = output_dir / "repeated_parent_results.csv"
    available_results_path = output_dir / "available_parent_results.csv"
    if args.skip_evaluation:
        if not parent_results_path.exists():
            raise FileNotFoundError(f"Cannot skip evaluation; missing {parent_results_path}")
        parent_results = pd.read_csv(parent_results_path)
        availability = pd.read_csv(output_dir / "complete_case_parent_counts.csv")
    else:
        config = LOSDConfig(
            final_k=5,
            candidate_pool_size=12,
            duplicate_threshold=0.90,
            alignment_threshold=0.70,
            lexical_weight=0.35,
            depth_tolerance=1,
            novelty_threshold=0.55,
            random_seed=42,
        )
        evaluation_fingerprint_payload = {
                "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
                "pipeline_script_sha256": sha256_file(
                    Path(__file__).resolve().with_name("losd.py")
                ),
                "ontology_sha256": sha256_file(ttl),
                "embedding_model": args.embedding_model,
                "embedding_revision": args.embedding_revision,
                "embedding_backend": args.embedding_backend,
                "long_text_torch_cutoff": args.long_text_torch_cutoff,
                "expected_embedding_key_sha256": args.expected_embedding_key_sha256,
                "config": asdict(config),
                "variants": [asdict(variant) for variant in VARIANTS],
                "candidate_parser": (
                    "visible-lines-heading-safe-or-exactly-12-semicolon-fields-v2"
                ),
                "incomplete_response_policy": (
                    "retain stop; retain length only with 12 visible items; "
                    "exclude other non-stop responses"
                ),
            }
        embedder = CachingSentenceTransformerEmbedder(
            args.embedding_model,
            batch_size=args.embedding_batch_size,
            backend=args.embedding_backend,
            revision=args.embedding_revision,
        )
        embedder.load_cache(embedding_cache)
        validate_loaded_embedding_cache(embedder, embedding_cache_metadata)

        def persist_embeddings(active_embedder: CachingSentenceTransformerEmbedder) -> None:
            active_embedder.save_cache(embedding_cache)
            write_embedding_cache_provenance(
                embedding_cache,
                embedding_cache_metadata,
                active_embedder,
                args,
            )

        benchmark_gold = pd.read_csv(run_map[EXPERIMENTS[0].label][0] / "gold_pairs.csv")
        closure_nodes = set(benchmark_gold["parent_uri"].astype(str)) | set(
            benchmark_gold["child_uri"].astype(str)
        )
        print(
            f"Loading ontology (materializing closures for {len(closure_nodes):,} "
            f"benchmark nodes): {ttl}",
            flush=True,
        )
        resources = load_ontology_resources(
            ttl,
            embedder,
            config,
            closure_nodes=closure_nodes,
            include_retrieval_embeddings=False,
        )
        persist_embeddings(embedder)
        candidate_texts = collect_candidate_texts(root, run_map)
        print(f"Pre-encoding {len(candidate_texts):,} unique candidate/context strings ...", flush=True)
        if args.embedding_backend == "onnx" and args.long_text_torch_cutoff > 0:
            long_texts = {
                text for text in candidate_texts if len(text) > args.long_text_torch_cutoff
            }
            short_texts = candidate_texts - long_texts
            print(
                f"Hybrid embedding pass: {len(long_texts):,} strings longer than "
                f"{args.long_text_torch_cutoff} characters with PyTorch; "
                f"{len(short_texts):,} shorter strings with ONNX.",
                flush=True,
            )
            missing_long = {text for text in long_texts if text not in embedder.cache}
            if missing_long:
                torch_embedder = CachingSentenceTransformerEmbedder(
                    args.embedding_model,
                    batch_size=args.embedding_batch_size,
                    backend="torch",
                    revision=args.embedding_revision,
                )
                torch_embedder.cache = embedder.cache
                torch_embedder.preload(missing_long)
                persist_embeddings(torch_embedder)
                embedder.cache = torch_embedder.cache
            missing_short = sorted(
                (text for text in short_texts if text not in embedder.cache),
                key=len,
                reverse=True,
            )
            checkpoint_size = 10_000
            for start in range(0, len(missing_short), checkpoint_size):
                chunk = missing_short[start : start + checkpoint_size]
                print(
                    f"ONNX short-text checkpoint "
                    f"{start // checkpoint_size + 1}/"
                    f"{math.ceil(len(missing_short) / checkpoint_size)}: "
                    f"{len(chunk):,} strings",
                    flush=True,
                )
                embedder.preload(chunk)
                persist_embeddings(embedder)
        else:
            embedder.preload(candidate_texts)
        persist_embeddings(embedder)
        ontology_embedding_keys = {
            normalize_surface_text(resources.uri2label[uri])[1]
            or resources.uri2label[uri]
            for uri in resources.all_uris
        }
        expected_embedding_keys = set(candidate_texts) | ontology_embedding_keys
        missing_expected_keys = expected_embedding_keys - set(embedder.cache)
        if missing_expected_keys:
            raise RuntimeError(
                f"Embedding preload incomplete: {len(missing_expected_keys)} expected keys missing"
            )
        expected_key_hash = sha256_text_keys(expected_embedding_keys)
        if (
            args.expected_embedding_key_sha256
            and expected_key_hash != args.expected_embedding_key_sha256
        ):
            raise RuntimeError(
                "Unexpected embedding-key corpus hash: "
                f"expected {args.expected_embedding_key_sha256}, observed {expected_key_hash}"
            )
        (output_dir / "embedding_key_manifest.json").write_text(
            json.dumps(
                {
                    "expected_key_count": len(expected_embedding_keys),
                    "expected_key_sha256": expected_key_hash,
                    "cache_key_count": len(embedder.cache),
                    "cache_key_sha256": sha256_text_keys(embedder.cache),
                    "unused_cached_key_count": len(set(embedder.cache) - expected_embedding_keys),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        evaluation_fingerprint_payload["embedding_cache_sha256"] = sha256_file(
            embedding_cache
        )
        evaluation_fingerprint = sha256_payload(evaluation_fingerprint_payload)
        pipeline = LOSDPipeline(resources, embedder, config)
        available_parent_results = evaluate_runs(
            root,
            run_map,
            pipeline,
            output_dir / "parent_partitions",
            excluded_tasks,
            evaluation_fingerprint,
            run_cache_fingerprints,
        )
        persist_embeddings(embedder)
        available_parent_results.to_csv(available_results_path, index=False)
        parent_results, availability = restrict_to_complete_repetition_parents(
            available_parent_results,
            required_runs=3,
        )
        availability.to_csv(output_dir / "complete_case_parent_counts.csv", index=False)
        parent_results.to_csv(parent_results_path, index=False)

    run_summary, repeated_summary = summarize_runs(parent_results)
    losd_comparisons = paired_losd_comparisons(
        parent_results, args.bootstrap_resamples, args.bootstrap_seed
    )
    prompt_comparisons = paired_prompt_comparisons(
        parent_results, args.bootstrap_resamples, args.bootstrap_seed
    )

    run_summary.to_csv(output_dir / "run_level_summary.csv", index=False)
    repeated_summary.to_csv(output_dir / "repeated_summary.csv", index=False)
    losd_comparisons.to_csv(output_dir / "paired_losd_comparisons.csv", index=False)
    prompt_comparisons.to_csv(output_dir / "paired_prompt_comparisons.csv", index=False)
    plot_effect_heatmaps(losd_comparisons, output_dir)
    write_latex_tables(repeated_summary, losd_comparisons, output_dir)
    write_validation_report(
        parent_results,
        repeated_summary,
        losd_comparisons,
        prompt_comparisons,
        output_dir,
        ttl,
        args,
        quality_rows,
        availability,
    )
    print(f"Saved repeated-experiment analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
