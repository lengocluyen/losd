from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from losd import (
    HashingTextEmbedder,
    LOSDConfig,
    SentenceTransformerEmbedder,
    build_fewshot_bank,
    load_ontology_resources,
    normalize_surface_text,
    parse_candidate_items,
    select_parent_pool,
)


SCHEMA_VERSION = 3
DEFAULT_METHODS = ("zero", "few", "rag")
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-chat"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_PROVIDER = "deepinfra"


@dataclass(frozen=True)
class GenerationTask:
    parent_uri: str
    parent_label: str
    method: str
    prompt: str
    prompt_hash: str
    context_items: tuple[str, ...] = ()
    fewshot_examples: tuple[tuple[str, tuple[str, ...]], ...] = ()


class RateLimiter:
    def __init__(self, rpm: float) -> None:
        if rpm <= 0:
            raise ValueError("--rpm must be greater than zero")
        self.interval_seconds = 60.0 / rpm
        self.last_started_at: float | None = None

    def wait(self) -> None:
        if self.last_started_at is not None:
            remaining = self.interval_seconds - (time.monotonic() - self.last_started_at)
            if remaining > 0:
                time.sleep(remaining)
        self.last_started_at = time.monotonic()


def exponential_retry_delay(
    base_delay: float,
    attempt: int,
    maximum_delay: float,
) -> float:
    delay = base_delay * (2 ** (attempt - 1))
    return min(delay, maximum_delay) if maximum_delay > 0 else delay


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def serialize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return serialize_for_json(value.model_dump())
    if hasattr(value, "dict"):
        return serialize_for_json(value.dict())
    return str(value)


def parse_items(raw_text: str, max_items: int) -> list[str]:
    return parse_candidate_items(raw_text, max_items=max_items)


def _legacy_line_parser(raw_text: str, max_items: int) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", raw_text or "", flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("```text", "").replace("```", "")
    bullet = re.compile(r"^\s*(?:(?:[-*•—–]+)|(?:\d+\s*[.)\-:]))\s*")
    intro = re.compile(
        r"^(?:voici|liste\s+des|here\s+are|les\s+sous-compétences)",
        flags=re.IGNORECASE,
    )
    items: list[str] = []
    seen: set[str] = set()
    for source_line in text.splitlines():
        line = bullet.sub("", source_line).strip()
        if not line or intro.match(line):
            continue
        line = line.strip(" \t;,")
        _, comparison_key = normalize_surface_text(line)
        if not comparison_key or comparison_key in seen:
            continue
        seen.add(comparison_key)
        items.append(line)
        if len(items) >= max_items:
            break
    return items


def prompt_zero(parent_label: str, contract: str, candidate_count: int) -> str:
    if contract == "notebook":
        return (
            f"Compétence: '{parent_label}'.\n"
            "Liste 5–10 sous-compétences concrètes en français, une par ligne, "
            "au même niveau de granularité, sans doublons ni synonymes.\n"
        )
    return (
        f'Compétence: "{parent_label}".\n'
        f"Génère exactement {candidate_count} sous-compétences candidates fines en français, "
        "une par ligne.\n"
        "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
        "un seul niveau plus spécifique, sans doublons, sans synonymes, sans outils, "
        "logiciels, plateformes ou métiers."
    )


def prompt_few(
    parent_label: str,
    examples: Sequence[tuple[str, Sequence[str]]],
    contract: str,
    candidate_count: int,
) -> str:
    example_text = "\n\n".join(
        f'Exemple — "{label}" → ' + "; ".join(children)
        for label, children in examples
    )
    if contract == "notebook":
        return (
            f"{example_text}\n\n"
            f"Compétence: '{parent_label}'.\n"
            "Donne 5–10 sous-compétences (une par ligne), sans synonymes ni doublons, "
            "au bon niveau de granularité.\n"
        )
    return (
        f"{example_text}\n\n"
        f'Compétence: "{parent_label}".\n'
        f"Génère exactement {candidate_count} sous-compétences candidates fines en français, "
        "une par ligne, avec la même granularité que les exemples.\n"
        "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
        "un seul niveau plus spécifique, sans doublons, sans synonymes, sans outils, "
        "logiciels, plateformes ou métiers."
    )


def prompt_rag(
    parent_label: str,
    context_items: Sequence[str],
    contract: str,
    candidate_count: int,
) -> str:
    context = "\n".join(f"- {item}" for item in context_items)
    if contract == "notebook":
        return (
            "Contexte (éléments liés d'une ontologie, sans les vrais enfants):\n"
            f"{context}\n\n"
            f"Compétence: '{parent_label}'.\n"
            "Propose 5–10 sous-compétences précises (une par ligne), pertinentes et "
            "non redondantes. Pas de synonymes.\n"
        )
    return (
        "Contexte ontologique masqué (les vrais enfants et leurs libellés sont exclus):\n"
        f"{context}\n\n"
        f'Compétence: "{parent_label}".\n'
        f"Génère exactement {candidate_count} sous-compétences candidates fines en français, "
        "une par ligne.\n"
        "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
        "un seul niveau plus spécifique, sans doublons, sans synonymes, sans outils, "
        "logiciels, plateformes ou métiers."
    )


def select_fewshot_examples(
    bank: Sequence[tuple[str, list[str]]],
    parent_uri: str,
    prompt_seed: int,
    count: int,
) -> list[tuple[str, list[str]]]:
    indices = list(range(len(bank)))
    rng = random.Random(f"{prompt_seed}:{parent_uri}")
    rng.shuffle(indices)
    return [bank[index] for index in indices[:count]]


def normalized_labels(resources: Any, uris: Iterable[str]) -> set[str]:
    masked: set[str] = set()
    for uri in uris:
        values = [resources.uri2label.get(uri, ""), *resources.alt_map.get(uri, [])]
        for value in values:
            _, key = normalize_surface_text(value or "")
            if key:
                masked.add(key)
    return masked


def nearest_context(
    resources: Any,
    embedder: Any,
    parent_uri: str,
    excluded_uris: set[str],
    count: int,
    contract: str,
) -> list[str]:
    query_text = resources.uri2text.get(parent_uri, resources.uri2label[parent_uri])
    _, query_key = normalize_surface_text(query_text)
    query_embedding = np.asarray(embedder.encode([query_key or query_text])[0], dtype=np.float32)
    scores = resources.all_text_embeddings @ query_embedding
    forbidden_labels = normalized_labels(resources, excluded_uris)
    context: list[str] = []
    for index in np.argsort(-scores):
        uri = resources.all_uris[int(index)]
        if uri in excluded_uris:
            continue
        label = resources.uri2label.get(uri, uri)
        _, label_key = normalize_surface_text(label)
        alternative_keys = normalized_labels(resources, [uri])
        if contract == "revised" and (
            label_key in forbidden_labels or alternative_keys.intersection(forbidden_labels)
        ):
            continue
        if contract == "notebook":
            item = label
        else:
            definition = (resources.def_map.get(uri) or "").strip().replace("\n", " ")
            if len(definition) > 220:
                definition = definition[:217].rstrip() + "..."
            item = f"voisin_sémantique | {label}"
            if definition:
                item += f" | {definition}"
        context.append(item)
        if len(context) >= count:
            break
    return context


def build_tasks(
    resources: Any,
    embedder: Any,
    parent_pool: Sequence[str],
    methods: Sequence[str],
    prompt_contract: str,
    candidate_count: int,
    rag_items: int,
    fewshot_count: int,
    prompt_seed: int,
) -> list[GenerationTask]:
    bank = build_fewshot_bank(
        resources.gold,
        resources.uri2label,
        excluded_parents=parent_pool,
        min_children=5,
        bank_size=8,
        children_per_example=6,
        random_seed=prompt_seed,
    )
    tasks: list[GenerationTask] = []
    for parent_uri in parent_pool:
        parent_label = resources.uri2label[parent_uri]
        children = set(resources.parent_children.get(parent_uri, []))
        excluded = {parent_uri, *children}
        selected_examples = select_fewshot_examples(
            bank, parent_uri, prompt_seed, fewshot_count
        )
        context_items: list[str] | None = None
        for method in methods:
            if method == "zero":
                prompt = prompt_zero(parent_label, prompt_contract, candidate_count)
                examples: list[tuple[str, list[str]]] = []
                context: list[str] = []
            elif method == "few":
                examples = selected_examples
                context = []
                prompt = prompt_few(
                    parent_label, examples, prompt_contract, candidate_count
                )
            elif method == "rag":
                examples = []
                if context_items is None:
                    context_items = nearest_context(
                        resources,
                        embedder,
                        parent_uri,
                        excluded,
                        rag_items,
                        prompt_contract,
                    )
                context = context_items
                prompt = prompt_rag(
                    parent_label, context, prompt_contract, candidate_count
                )
            else:  # guarded by argparse
                raise ValueError(f"Unsupported method: {method}")
            tasks.append(
                GenerationTask(
                    parent_uri=parent_uri,
                    parent_label=parent_label,
                    method=method,
                    prompt=prompt,
                    prompt_hash=sha256_text(prompt),
                    context_items=tuple(context),
                    fewshot_examples=tuple(
                        (label, tuple(children_list)) for label, children_list in examples
                    ),
                )
            )
    return tasks


def cache_path(run_dir: Path, task: GenerationTask, model: str) -> Path:
    parent_tail = safe_name(task.parent_uri.rsplit("/", 1)[-1])
    return run_dir / "cache" / f"{task.method}__{parent_tail}__{safe_name(model)}.json"


def failed_response_path(run_dir: Path, task: GenerationTask, model: str) -> Path:
    parent_tail = safe_name(task.parent_uri.rsplit("/", 1)[-1])
    return (
        run_dir
        / "failed_responses"
        / f"{task.method}__{parent_tail}__{safe_name(model)}.json"
    )


def retry_response_path(
    run_dir: Path,
    task: GenerationTask,
    model: str,
    response_attempt: int,
) -> Path:
    parent_tail = safe_name(task.parent_uri.rsplit("/", 1)[-1])
    return (
        run_dir
        / "retry_responses"
        / (
            f"{task.method}__{parent_tail}__{safe_name(model)}"
            f"__attempt_{response_attempt:02d}.json"
        )
    )


def generation_config(
    args: argparse.Namespace,
    run_number: int,
    api_seed: int,
    ttl_hash: str,
) -> dict[str, Any]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"run_{run_number:02d}",
        "api_seed": api_seed,
        "prompt_seed": args.prompt_seed,
        "api_service": "openrouter",
        "api_base_url": args.api_base_url,
        "openrouter_provider": args.openrouter_provider,
        "allow_provider_fallbacks": args.allow_provider_fallbacks,
        "require_parameters": args.require_parameters,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning,
        "rpm": args.rpm,
        "methods": list(args.methods),
        "prompt_contract": args.prompt_contract,
        "candidate_count": args.candidate_count,
        "rag_items": args.rag_items,
        "fewshot_examples": args.fewshot_examples,
        "parent_filter": args.parent_filter,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "parent_limit": args.parent_limit,
        "embedding_model": args.embedding_model,
        "embedder": args.embedder,
        "ttl_path": str(args.ttl.resolve()),
        "ttl_sha256": ttl_hash,
    }
    # Keep the historical/default configuration shape unchanged so existing
    # DeepSeek/Qwen/Kimi caches remain resumable after adding GPT-5 support.
    if args.omit_temperature:
        config["temperature"] = None
        config["omit_temperature"] = True
    if args.omit_top_p:
        config["top_p"] = None
        config["omit_top_p"] = True
    if args.omit_seed:
        config["api_seed"] = None
        config["nominal_run_seed"] = api_seed
        config["omit_seed"] = True
    if args.token_limit_parameter != "max_tokens":
        config["token_limit_parameter"] = args.token_limit_parameter
    if args.reasoning_effort != "default":
        config["reasoning_effort"] = args.reasoning_effort
    return config


def config_hash(config: dict[str, Any]) -> str:
    return sha256_text(json.dumps(config, sort_keys=True, ensure_ascii=False))


def read_complete_cache(path: Path, expected_hash: str, expected_prompt_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid cache file {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return False
    if payload.get("config_hash") != expected_hash:
        raise RuntimeError(
            f"Configuration mismatch in {path}. Use a new --output-root instead of mixing runs."
        )
    if payload.get("prompt_hash") != expected_prompt_hash:
        raise RuntimeError(
            f"Prompt mismatch in {path}. Use a new --output-root instead of mixing prompts."
        )
    return isinstance(payload.get("items"), list) and bool(payload["items"])


def extract_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    serialized = serialize_for_json(usage)
    return serialized if isinstance(serialized, dict) else {"value": serialized}


def call_openrouter(
    client: Any,
    task: GenerationTask,
    args: argparse.Namespace,
    api_seed: int,
    limiter: RateLimiter,
) -> tuple[str, str, dict[str, Any]]:
    attempts = args.retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        limiter.wait()
        started = time.perf_counter()
        try:
            request: dict[str, Any] = {
                "model": args.model,
                "messages": [{"role": "user", "content": task.prompt}],
            }
            if not args.omit_seed:
                request["seed"] = api_seed
            if not args.omit_temperature:
                request["temperature"] = args.temperature
            if not args.omit_top_p:
                request["top_p"] = args.top_p
            request[args.token_limit_parameter] = args.max_tokens
            extra_body: dict[str, Any] = {
                "provider": {
                    "only": [args.openrouter_provider],
                    "allow_fallbacks": args.allow_provider_fallbacks,
                    "require_parameters": args.require_parameters,
                }
            }
            if args.reasoning != "default" or args.reasoning_effort != "default":
                reasoning_config: dict[str, Any] = {}
                if args.reasoning != "default":
                    reasoning_config["enabled"] = args.reasoning == "enabled"
                if args.reasoning_effort != "default":
                    reasoning_config["effort"] = args.reasoning_effort
                extra_body["reasoning"] = reasoning_config
            request["extra_body"] = extra_body
            response = client.chat.completions.create(
                **request,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            choice = response.choices[0]
            message = choice.message
            raw_text = (getattr(message, "content", None) or "").strip()
            message_extra = getattr(message, "model_extra", None) or {}
            reasoning_value = getattr(message, "reasoning", None)
            if reasoning_value is None and isinstance(message_extra, dict):
                reasoning_value = message_extra.get("reasoning")
            if isinstance(reasoning_value, str):
                reasoning = reasoning_value.strip()
            elif reasoning_value is None:
                reasoning = ""
            else:
                reasoning = json.dumps(
                    serialize_for_json(reasoning_value),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            response_extra = getattr(response, "model_extra", None) or {}
            metadata = {
                "response_id": getattr(response, "id", None),
                "resolved_model": getattr(response, "model", None),
                "resolved_provider": (
                    response_extra.get("provider")
                    if isinstance(response_extra, dict)
                    else None
                ),
                "created": getattr(response, "created", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "response_seed": getattr(choice, "seed", None),
                "elapsed_ms": elapsed_ms,
                "usage": extract_usage(response),
                "attempt": attempt,
            }
            return raw_text, reasoning, metadata
        except Exception as exc:  # OpenAI-compatible SDK exposes version-specific error classes
            last_error = exc
            if attempt >= attempts:
                break
            delay = exponential_retry_delay(
                args.retry_delay, attempt, args.max_retry_delay
            )
            print(
                f"API call failed on attempt {attempt}/{attempts}: {exc}. "
                f"Waiting {delay:.1f}s before the user-enabled retry.",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    args: argparse.Namespace,
    run_number: int,
    resources: Any,
    tasks: Sequence[GenerationTask],
    parent_pool: Sequence[str],
    gold_eval: pd.DataFrame,
    ttl_hash: str,
    limiter: RateLimiter,
    client: Any | None,
    remaining_call_budget: list[int | None],
) -> str:
    run_id = f"run_{run_number:02d}"
    api_seed = args.seed_base + run_number - 1
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cache").mkdir(parents=True, exist_ok=True)

    config = generation_config(args, run_number, api_seed, ttl_hash)
    current_config_hash = config_hash(config)
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        old_hash = existing.get("config_hash")
        if old_hash and old_hash != current_config_hash:
            raise RuntimeError(
                f"{run_dir} already contains a different configuration. "
                "Choose another --output-root."
            )

    gold_eval.to_csv(run_dir / "gold_pairs.csv", index=False)
    pd.DataFrame(
        {
            "parent_uri": parent_pool,
            "parent_skill_id": parent_pool,
        }
    ).to_csv(run_dir / "parents.csv", index=False)

    task_manifest = []
    for task in tasks:
        row = asdict(task)
        row["context_items"] = list(task.context_items)
        row["fewshot_examples"] = [
            {"parent_label": label, "children": list(children)}
            for label, children in task.fewshot_examples
        ]
        task_manifest.append(row)
    atomic_write_json(run_dir / "prompt_manifest.json", task_manifest)

    run_metadata = {
        **config,
        "config_hash": current_config_hash,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "expected_api_calls": len(tasks),
        "unparseable_retries": args.unparseable_retries,
        "retries": args.retries,
        "retry_delay": args.retry_delay,
        "max_retry_delay": args.max_retry_delay,
        "started_at_utc": existing.get("started_at_utc", utc_now()) if metadata_path.exists() else utc_now(),
        "updated_at_utc": utc_now(),
        "status": "dry_run" if args.dry_run else "running",
    }
    atomic_write_json(metadata_path, run_metadata)

    if args.dry_run:
        print(f"[{run_id}] dry run: {len(parent_pool)} parents, {len(tasks)} API calls planned")
        return "dry_run"

    completed_before = 0
    pending: list[GenerationTask] = []
    for task in tasks:
        path = cache_path(run_dir, task, args.model)
        if args.resume and read_complete_cache(path, current_config_hash, task.prompt_hash):
            completed_before += 1
        else:
            pending.append(task)

    print(
        f"[{run_id}] total={len(tasks)} cached={completed_before} pending={len(pending)} "
        f"seed={api_seed}"
    )
    if pending and client is None:
        raise RuntimeError("OpenRouter client is not initialized")

    event_path = run_dir / "generation_events.jsonl"
    error_path = run_dir / "generation_errors.jsonl"
    completed_new = 0
    for task in pending:
        if remaining_call_budget[0] is not None and remaining_call_budget[0] <= 0:
            status = "paused_by_max_api_calls"
            break
        ordinal = completed_before + completed_new + 1
        print(
            f"[{run_id}] {ordinal}/{len(tasks)} {task.method} "
            f"{task.parent_uri.rsplit('/', 1)[-1]}"
        )
        started_at = utc_now()
        raw_text = ""
        reasoning = ""
        response_meta: dict[str, Any] = {}
        diagnostic_path: Path | None = None
        try:
            items: list[str] = []
            response_attempts = args.unparseable_retries + 1
            for response_attempt in range(1, response_attempts + 1):
                raw_text, reasoning, response_meta = call_openrouter(
                    client, task, args, api_seed, limiter
                )
                response_meta["response_attempt"] = response_attempt
                items = parse_items(raw_text, args.candidate_count)
                if items:
                    diagnostic_path = None
                    break

                is_final_response_attempt = response_attempt >= response_attempts
                diagnostic_path = (
                    failed_response_path(run_dir, task, args.model)
                    if is_final_response_attempt
                    else retry_response_path(
                        run_dir, task, args.model, response_attempt
                    )
                )
                diagnostic = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "unparseable_response",
                    "run_id": run_id,
                    "parent_uri": task.parent_uri,
                    "parent_label": task.parent_label,
                    "method": task.method,
                    "model": args.model,
                    "api_seed": None if args.omit_seed else api_seed,
                    "nominal_run_seed": api_seed if args.omit_seed else None,
                    "prompt_hash": task.prompt_hash,
                    "config_hash": current_config_hash,
                    "raw_text": raw_text,
                    "raw_text_characters": len(raw_text),
                    "reasoning": reasoning,
                    "reasoning_characters": len(reasoning),
                    "response_attempt": response_attempt,
                    "response_attempts_allowed": response_attempts,
                    "captured_at_utc": utc_now(),
                    **response_meta,
                }
                atomic_write_json(diagnostic_path, diagnostic)
                if is_final_response_attempt:
                    raise RuntimeError(
                        "The response produced no parseable candidate items "
                        f"after {response_attempts} response attempt(s) "
                        f"(content_chars={len(raw_text)}, "
                        f"reasoning_chars={len(reasoning)}, "
                        f"finish_reason={response_meta.get('finish_reason')!r}). "
                        f"Full response evidence was saved to {diagnostic_path}."
                    )

                delay = exponential_retry_delay(
                    args.retry_delay, response_attempt, args.max_retry_delay
                )
                print(
                    "Response produced no parseable candidate items on "
                    f"attempt {response_attempt}/{response_attempts}; evidence saved "
                    f"to {diagnostic_path}. Waiting {delay:.1f}s before retrying.",
                    file=sys.stderr,
                )
                append_jsonl(
                    run_dir / "generation_retry_events.jsonl",
                    {
                        "event": "unparseable_response_retry",
                        "run_id": run_id,
                        "parent_uri": task.parent_uri,
                        "method": task.method,
                        "response_attempt": response_attempt,
                        "response_attempts_allowed": response_attempts,
                        "failed_response_file": str(diagnostic_path),
                        "finish_reason": response_meta.get("finish_reason"),
                        "usage": response_meta.get("usage", {}),
                        "timestamp_utc": utc_now(),
                    },
                )
                time.sleep(delay)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "run_id": run_id,
                "parent_uri": task.parent_uri,
                "parent_label": task.parent_label,
                "method": task.method,
                "model": args.model,
                "api_seed": None if args.omit_seed else api_seed,
                "nominal_run_seed": api_seed if args.omit_seed else None,
                "prompt_seed": args.prompt_seed,
                "temperature": None if args.omit_temperature else args.temperature,
                "top_p": None if args.omit_top_p else args.top_p,
                "max_tokens": args.max_tokens,
                "token_limit_parameter": args.token_limit_parameter,
                "reasoning_mode": args.reasoning,
                "reasoning_effort": args.reasoning_effort,
                "prompt_contract": args.prompt_contract,
                "prompt": task.prompt,
                "prompt_hash": task.prompt_hash,
                "config_hash": current_config_hash,
                "items": items,
                "raw_text": raw_text,
                "reasoning": reasoning,
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                **response_meta,
            }
            destination = cache_path(run_dir, task, args.model)
            atomic_write_json(destination, payload)
            append_jsonl(
                event_path,
                {
                    "event": "generation_complete",
                    "cache_file": destination.name,
                    **payload,
                },
            )
            completed_new += 1
            if remaining_call_budget[0] is not None:
                remaining_call_budget[0] -= 1
            atomic_write_json(
                run_dir / "progress.json",
                {
                    "run_id": run_id,
                    "total": len(tasks),
                    "completed_before": completed_before,
                    "completed_new": completed_new,
                    "completed_total": completed_before + completed_new,
                    "updated_at_utc": utc_now(),
                },
            )
        except Exception as exc:
            error_record: dict[str, Any] = {
                "event": "generation_error",
                "run_id": run_id,
                "parent_uri": task.parent_uri,
                "method": task.method,
                "prompt_hash": task.prompt_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timestamp_utc": utc_now(),
            }
            if response_meta:
                error_record.update(
                    {
                        "response_id": response_meta.get("response_id"),
                        "resolved_model": response_meta.get("resolved_model"),
                        "resolved_provider": response_meta.get("resolved_provider"),
                        "finish_reason": response_meta.get("finish_reason"),
                        "usage": response_meta.get("usage", {}),
                        "raw_text_characters": len(raw_text),
                        "reasoning_characters": len(reasoning),
                    }
                )
            if diagnostic_path is not None:
                error_record["failed_response_file"] = str(diagnostic_path)
            append_jsonl(
                error_path,
                error_record,
            )
            run_metadata.update(
                {
                    "status": "failed",
                    "updated_at_utc": utc_now(),
                    "failed_parent_uri": task.parent_uri,
                    "failed_method": task.method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            atomic_write_json(metadata_path, run_metadata)
            raise RuntimeError(
                f"Generation failed for {task.method}/{task.parent_uri}. "
                f"Cause: {exc}. "
                f"The run stopped without an automatic retry. Re-run with --resume after checking {error_path}."
            ) from exc
    else:
        status = "complete"

    completed_total = completed_before + completed_new
    run_metadata.update(
        {
            "status": status,
            "updated_at_utc": utc_now(),
            "completed_api_calls": completed_total,
            "pending_api_calls": len(tasks) - completed_total,
        }
    )
    if status == "complete":
        run_metadata["completed_at_utc"] = utc_now()
    atomic_write_json(metadata_path, run_metadata)
    print(f"[{run_id}] status={status} completed={completed_total}/{len(tasks)}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated model predictions through OpenRouter using the ontology, "
            "parent selection, few-shot examples, and RAG design shared across models."
        )
    )
    parser.add_argument("--ttl", type=Path, default=Path("esco_cmo_binding.ttl"))
    parser.add_argument("--output-root", type=Path, default=Path("exp_repeated_deepseek_v3"))
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENROUTER_MODEL,
        help=(
            "Exact OpenRouter model identifier. The generic runner defaults to "
            f"{DEFAULT_OPENROUTER_MODEL}; model-specific wrappers may override it."
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1001)
    parser.add_argument("--prompt-seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--prompt-contract", choices=["notebook", "revised"], default="revised")
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature (needed by models/endpoints that reject it).",
    )
    parser.add_argument(
        "--omit-top-p",
        action="store_true",
        help="Do not send top_p (needed by models/endpoints that reject it).",
    )
    parser.add_argument(
        "--omit-seed",
        action="store_true",
        help="Do not send seed when the pinned provider does not support it.",
    )
    parser.add_argument(
        "--token-limit-parameter",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
        help="API field used for the output-token limit (default: max_tokens).",
    )
    parser.add_argument(
        "--reasoning",
        choices=["default", "enabled", "disabled"],
        default="default",
        help=(
            "OpenRouter reasoning mode. The original DeepSeek V3 endpoint is non-reasoning, "
            "so use 'default' to omit reasoning controls; the value is stored in metadata."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["default", "minimal", "low", "medium", "high"],
        default="default",
        help="Optional OpenRouter reasoning effort; 'default' omits the field.",
    )
    parser.add_argument("--rpm", type=float, default=15.0)
    parser.add_argument("--rag-items", type=int, default=30)
    parser.add_argument("--fewshot-examples", type=int, default=2)
    parser.add_argument("--parent-filter", choices=["balanced", "all"], default="balanced")
    parser.add_argument("--min-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--embedder", choices=["sentence-transformer", "hash"], default="sentence-transformer")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-base-url", default=DEFAULT_OPENROUTER_BASE_URL)
    parser.add_argument(
        "--openrouter-provider",
        default=DEFAULT_OPENROUTER_PROVIDER,
        help=(
            "OpenRouter provider slug to pin for every request (default: deepinfra). "
            "Provider pinning prevents backend changes across repetitions."
        ),
    )
    parser.add_argument(
        "--allow-provider-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow OpenRouter to fall back to another endpoint (default: disabled).",
    )
    parser.add_argument(
        "--require-parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the endpoint to support every requested parameter (default: enabled).",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument(
        "--unparseable-retries",
        type=int,
        default=0,
        help=(
            "Retry completed responses that contain no parseable candidate items "
            "(default: 0)."
        ),
    )
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=0.0,
        help=(
            "Cap exponential retry waits at this many seconds; 0 leaves them "
            "uncapped (default: 0)."
        ),
    )
    parser.add_argument("--max-api-calls", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume only when cached prompt and configuration hashes match (default: enabled).",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.start_run < 1:
        parser.error("--start-run must be at least 1")
    if args.candidate_count < 1:
        parser.error("--candidate-count must be at least 1")
    if args.parent_limit < 0:
        parser.error("--parent-limit cannot be negative")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.unparseable_retries < 0:
        parser.error("--unparseable-retries cannot be negative")
    if args.retry_delay < 0:
        parser.error("--retry-delay cannot be negative")
    if args.max_retry_delay < 0:
        parser.error("--max-retry-delay cannot be negative")
    if args.max_api_calls < 0:
        parser.error("--max-api-calls cannot be negative")
    if not args.openrouter_provider.strip():
        parser.error("--openrouter-provider cannot be empty")
    if args.model.lower() == DEFAULT_OPENROUTER_MODEL and args.reasoning != "default":
        parser.error(
            f"{DEFAULT_OPENROUTER_MODEL} does not expose reasoning controls through the "
            "pinned DeepInfra endpoint; use --reasoning default"
        )
    args.api_base_url = args.api_base_url.rstrip("/")
    args.openrouter_provider = args.openrouter_provider.strip().lower()
    args.ttl = args.ttl.resolve()
    args.output_root = args.output_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    if not args.ttl.exists():
        print(f"TTL file not found: {args.ttl}", file=sys.stderr)
        return 2
    embedder = (
        SentenceTransformerEmbedder(args.embedding_model)
        if args.embedder == "sentence-transformer"
        else HashingTextEmbedder()
    )
    config = LOSDConfig(
        candidate_pool_size=args.candidate_count,
        rag_context_items=args.rag_items,
        fewshot_examples=args.fewshot_examples,
        random_seed=args.prompt_seed,
    )
    print(f"Loading ontology: {args.ttl}")
    resources = load_ontology_resources(args.ttl, embedder, config)
    parent_pool = sorted(
        select_parent_pool(
            resources.gold,
            parent_filter=args.parent_filter,
            min_k=args.min_k,
            max_k=args.max_k,
            parent_limit=args.parent_limit or None,
            random_seed=args.prompt_seed,
        )
    )
    gold_eval = resources.gold.loc[resources.gold["parent_uri"].isin(parent_pool)].copy()
    tasks = build_tasks(
        resources,
        embedder,
        parent_pool,
        args.methods,
        args.prompt_contract,
        args.candidate_count,
        args.rag_items,
        args.fewshot_examples,
        args.prompt_seed,
    )
    print(
        f"Plan: {len(parent_pool)} parents × {len(args.methods)} methods × "
        f"{args.runs} runs = {len(tasks) * args.runs} possible API calls"
    )

    client = None
    if not args.dry_run:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            print(
                f"Environment variable {args.api_key_env} is not set. "
                "Set it before starting paid generation.",
                file=sys.stderr,
            )
            return 2
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "The OpenAI SDK is missing. Install it with: python -m pip install openai",
                file=sys.stderr,
            )
            return 2
        client = OpenAI(api_key=api_key, base_url=args.api_base_url)

    limiter = RateLimiter(args.rpm)
    ttl_hash = sha256_file(args.ttl)
    remaining_call_budget: list[int | None] = [
        args.max_api_calls if args.max_api_calls else None
    ]
    statuses: list[str] = []
    for run_number in range(args.start_run, args.start_run + args.runs):
        status = run_one(
            args,
            run_number,
            resources,
            tasks,
            parent_pool,
            gold_eval,
            ttl_hash,
            limiter,
            client,
            remaining_call_budget,
        )
        statuses.append(status)
        if status == "paused_by_max_api_calls":
            break

    print(f"Finished with statuses: {', '.join(statuses)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
