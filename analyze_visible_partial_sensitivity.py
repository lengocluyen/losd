from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import analyze_repeated_experiments as primary
from losd import (
    LOSDConfig,
    LOSDPipeline,
    load_cached_candidate_texts,
    load_ontology_resources,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity analysis that scores only the visible parsed candidates "
            "from generation cells excluded by the primary complete-case analysis."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--analysis-dir", type=Path, default=Path("repeated_analysis"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repeated_analysis/sensitivity_visible_partial"),
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required completed-primary artifact is missing: {path}")
    return path


def primary_config() -> LOSDConfig:
    return LOSDConfig(
        final_k=5,
        candidate_pool_size=12,
        duplicate_threshold=0.90,
        alignment_threshold=0.70,
        lexical_weight=0.35,
        depth_tolerance=1,
        novelty_threshold=0.55,
        random_seed=42,
    )


def verify_primary_provenance(
    analysis_dir: Path,
    environment: dict[str, Any],
    ttl: Path,
) -> None:
    analysis_script = Path(primary.__file__).resolve()
    pipeline_script = analysis_script.with_name("losd.py")
    checks = {
        "analysis_script_sha256": primary.sha256_file(analysis_script),
        "pipeline_script_sha256": primary.sha256_file(pipeline_script),
        "ontology_sha256": primary.sha256_file(ttl),
    }
    mismatches = {
        key: {"recorded": environment.get(key), "current": value}
        for key, value in checks.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Primary code/ontology provenance no longer matches the completed analysis: "
            f"{mismatches}"
        )

    partition_dir = require_file(analysis_dir / "parent_partitions")
    partitions = sorted(partition_dir.glob("*.csv"))
    metadata = sorted(partition_dir.glob("*.meta.json"))
    if len(partitions) != 21 or len(metadata) != 21:
        raise RuntimeError(
            "Primary analysis is not fully checkpointed: "
            f"{len(partitions)} CSV and {len(metadata)} metadata partitions found"
        )


def build_run_lookup(
    root: Path,
) -> tuple[
    dict[str, list[Path]],
    dict[tuple[str, str], tuple[int, Path]],
]:
    run_map = {
        spec.label: primary.complete_run_dirs(root, spec)
        for spec in primary.EXPERIMENTS
    }
    lookup: dict[tuple[str, str], tuple[int, Path]] = {}
    for spec in primary.EXPERIMENTS:
        for run_number, run_dir in enumerate(run_map[spec.label], start=1):
            lookup[(spec.label, run_dir.name)] = (run_number, run_dir)
    return run_map, lookup


def score_excluded_visible_outputs(
    excluded: pd.DataFrame,
    run_lookup: dict[tuple[str, str], tuple[int, Path]],
    pipeline: LOSDPipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    per_run_gold: dict[Path, pd.DataFrame] = {}
    per_run_manifest: dict[Path, dict[tuple[str, str], dict[str, Any]]] = {}
    per_run_model: dict[Path, str] = {}

    for quality_row in excluded.itertuples(index=False):
        model_label = str(quality_row.model_label)
        run_id = str(quality_row.run_id)
        prompt_mode = str(quality_row.prompt_mode)
        parent_uri = str(quality_row.parent_uri)
        run_number, run_dir = run_lookup[(model_label, run_id)]

        if run_dir not in per_run_gold:
            per_run_gold[run_dir] = pd.read_csv(run_dir / "gold_pairs.csv")
            per_run_manifest[run_dir] = primary.manifest_index(run_dir)
            per_run_model[run_dir] = str(
                primary.read_json(run_dir / "run_metadata.json")["model"]
            )

        raw_candidates = load_cached_candidate_texts(
            run_dir / "cache",
            prompt_mode,
            parent_uri,
            max_items=12,
        )
        expected_visible_count = int(quality_row.reparsed_item_count)
        if not raw_candidates or len(raw_candidates) != expected_visible_count:
            raise RuntimeError(
                f"Visible-candidate mismatch for {model_label}/{run_id}/"
                f"{prompt_mode}/{parent_uri}: loaded {len(raw_candidates)}, "
                f"quality audit recorded {expected_visible_count}"
            )

        gold = per_run_gold[run_dir]
        gold_children = (
            gold.loc[gold["parent_uri"] == parent_uri, "child_uri"]
            .drop_duplicates()
            .tolist()
        )
        if not gold_children:
            raise RuntimeError(f"No gold children found for excluded parent {parent_uri}")
        manifest_row = per_run_manifest[run_dir][(parent_uri, prompt_mode)]
        _, parent_rows = pipeline.run_parent(
            parent_skill_id=parent_uri,
            prompt_mode=prompt_mode,
            raw_candidates=raw_candidates,
            gold_children=gold_children,
            context_texts=primary.context_from_manifest(manifest_row),
            variants=primary.VARIANTS,
        )
        for row in parent_rows:
            row.update(
                {
                    "model_label": model_label,
                    "model_id": per_run_model[run_dir],
                    "run_id": run_id,
                    "run_number": run_number,
                }
            )
            score_rows.append(row)
        trace_rows.append(
            {
                "model_label": model_label,
                "run_id": run_id,
                "prompt_mode": prompt_mode,
                "parent_uri": parent_uri,
                "finish_reason": str(quality_row.finish_reason),
                "visible_candidate_count": len(raw_candidates),
                "visible_candidates": json.dumps(raw_candidates, ensure_ascii=False),
                "cache_path": str(quality_row.cache_path),
                "hidden_reasoning_used": False,
            }
        )

    return pd.DataFrame(score_rows), pd.DataFrame(trace_rows)


def ci_excludes_zero(low: pd.Series, high: pd.Series) -> pd.Series:
    return (low > 0.0) | (high < 0.0)


def effect_sign(values: pd.Series, tolerance: float = 1e-12) -> pd.Series:
    array = values.to_numpy(dtype=float)
    signs = np.where(array > tolerance, 1, np.where(array < -tolerance, -1, 0))
    return pd.Series(signs, index=values.index, dtype=int)


def compare_inference_tables(
    primary_table: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    key_columns: Sequence[str],
) -> pd.DataFrame:
    columns = [
        *key_columns,
        "parents",
        "paired_difference",
        "difference_ci_low",
        "difference_ci_high",
        "p_value",
        "p_holm",
        "significant_holm_0_05",
    ]
    merged = primary_table[columns].merge(
        sensitivity_table[columns],
        on=list(key_columns),
        how="outer",
        suffixes=("_primary", "_sensitivity"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("Primary and sensitivity comparison keys do not match")
    merged = merged.drop(columns="_merge")
    merged["paired_difference_change"] = (
        merged["paired_difference_sensitivity"]
        - merged["paired_difference_primary"]
    )
    merged["effect_sign_primary"] = effect_sign(merged["paired_difference_primary"])
    merged["effect_sign_sensitivity"] = effect_sign(
        merged["paired_difference_sensitivity"]
    )
    merged["effect_sign_changed"] = (
        merged["effect_sign_primary"] != merged["effect_sign_sensitivity"]
    )
    merged["ci_excludes_zero_primary"] = ci_excludes_zero(
        merged["difference_ci_low_primary"], merged["difference_ci_high_primary"]
    )
    merged["ci_excludes_zero_sensitivity"] = ci_excludes_zero(
        merged["difference_ci_low_sensitivity"],
        merged["difference_ci_high_sensitivity"],
    )
    merged["ci_zero_decision_changed"] = (
        merged["ci_excludes_zero_primary"]
        != merged["ci_excludes_zero_sensitivity"]
    )
    merged["holm_decision_changed"] = (
        merged["significant_holm_0_05_primary"].astype(bool)
        != merged["significant_holm_0_05_sensitivity"].astype(bool)
    )
    merged["headline_conclusion_changed"] = (
        merged["effect_sign_changed"]
        | merged["ci_zero_decision_changed"]
        | merged["holm_decision_changed"]
    )
    return merged


def comparison_summary(table: pd.DataFrame) -> dict[str, Any]:
    changed = table[table["headline_conclusion_changed"]]
    key_candidates = [
        column
        for column in ("model_label", "prompt_mode", "comparison", "metric")
        if column in table.columns
    ]
    return {
        "comparisons": len(table),
        "primary_holm_significant": int(
            table["significant_holm_0_05_primary"].astype(bool).sum()
        ),
        "sensitivity_holm_significant": int(
            table["significant_holm_0_05_sensitivity"].astype(bool).sum()
        ),
        "effect_sign_changes": int(table["effect_sign_changed"].sum()),
        "ci_zero_decision_changes": int(table["ci_zero_decision_changed"].sum()),
        "holm_decision_changes": int(table["holm_decision_changed"].sum()),
        "headline_conclusion_changes": int(table["headline_conclusion_changed"].sum()),
        "maximum_absolute_paired_difference_change": float(
            table["paired_difference_change"].abs().max()
        ),
        "changed_comparisons": changed[key_candidates].to_dict(orient="records"),
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    analysis_dir = (
        (root / args.analysis_dir).resolve()
        if not args.analysis_dir.is_absolute()
        else args.analysis_dir.resolve()
    )
    output_dir = (
        (root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = primary.read_json(require_file(analysis_dir / "analysis_environment.json"))
    ttl = Path(str(environment["ontology_path"])).resolve()
    verify_primary_provenance(analysis_dir, environment, ttl)

    primary_available = pd.read_csv(
        require_file(analysis_dir / "available_parent_results.csv")
    )
    primary_complete = pd.read_csv(
        require_file(analysis_dir / "repeated_parent_results.csv")
    )
    primary_repeated = pd.read_csv(require_file(analysis_dir / "repeated_summary.csv"))
    primary_losd = pd.read_csv(
        require_file(analysis_dir / "paired_losd_comparisons.csv")
    )
    primary_prompt = pd.read_csv(
        require_file(analysis_dir / "paired_prompt_comparisons.csv")
    )
    quality = pd.read_csv(
        require_file(analysis_dir / "generation_output_quality_audit.csv")
    )
    excluded = quality.loc[~quality["eligible_for_analysis"].astype(bool)].copy()
    if len(quality) != 18_144 or len(excluded) != 14:
        raise RuntimeError(
            f"Expected 18,144 audited cells and 14 exclusions; found "
            f"{len(quality):,} and {len(excluded):,}"
        )
    if len(primary_available) != int(quality["eligible_for_analysis"].sum()) * len(
        primary.VARIANTS
    ):
        raise RuntimeError("Primary available-row count does not match the quality audit")
    if len(primary_available) != 72_520 or len(primary_complete) != 72_408:
        raise RuntimeError(
            "Completed primary row counts differ from the audited expectations: "
            f"available={len(primary_available):,}, complete={len(primary_complete):,}"
        )

    run_map, run_lookup = build_run_lookup(root)
    embedding_cache = require_file(analysis_dir / "embedding_cache.pkl")
    embedding_metadata = require_file(analysis_dir / "embedding_cache.meta.json")
    embedding_args = argparse.Namespace(
        embedding_model=str(environment["embedding_model"]),
        embedding_revision=str(environment["embedding_revision"]),
        embedding_backend=str(environment["embedding_backend"]),
        embedding_batch_size=int(environment["embedding_batch_size"]),
        long_text_torch_cutoff=int(environment["long_text_torch_cutoff_characters"]),
    )
    primary.validate_embedding_cache_provenance(
        embedding_cache, embedding_metadata, embedding_args
    )
    embedder = primary.CachingSentenceTransformerEmbedder(
        embedding_args.embedding_model,
        batch_size=embedding_args.embedding_batch_size,
        backend=embedding_args.embedding_backend,
        revision=embedding_args.embedding_revision,
    )
    embedder.load_cache(embedding_cache)
    primary.validate_loaded_embedding_cache(embedder, embedding_metadata)
    cache_key_count_before = len(embedder.cache)
    cache_key_hash_before = primary.sha256_text_keys(embedder.cache)

    benchmark_gold = pd.read_csv(run_map[primary.EXPERIMENTS[0].label][0] / "gold_pairs.csv")
    closure_nodes = set(benchmark_gold["parent_uri"].astype(str)) | set(
        benchmark_gold["child_uri"].astype(str)
    )
    config = primary_config()
    resources = load_ontology_resources(
        ttl,
        embedder,
        config,
        closure_nodes=closure_nodes,
        include_retrieval_embeddings=False,
    )
    pipeline = LOSDPipeline(resources, embedder, config)
    added_rows, trace = score_excluded_visible_outputs(
        excluded, run_lookup, pipeline
    )

    cache_key_count_after = len(embedder.cache)
    cache_key_hash_after = primary.sha256_text_keys(embedder.cache)
    if (
        cache_key_count_after != cache_key_count_before
        or cache_key_hash_after != cache_key_hash_before
    ):
        raise RuntimeError(
            "Sensitivity scoring requested an embedding absent from the frozen cache"
        )

    expected_added_rows = len(excluded) * len(primary.VARIANTS)
    if len(added_rows) != expected_added_rows:
        raise RuntimeError(
            f"Sensitivity produced {len(added_rows)} rows; expected {expected_added_rows}"
        )
    key_columns = [
        "model_label",
        "run_id",
        "prompt_mode",
        "parent_skill_id",
        "variant",
    ]
    if added_rows.duplicated(key_columns).any():
        raise RuntimeError("Sensitivity-added rows contain duplicate task-variant keys")
    missing_columns = set(primary_available.columns) - set(added_rows.columns)
    if missing_columns:
        raise RuntimeError(f"Sensitivity rows lack primary columns: {sorted(missing_columns)}")
    added_rows = added_rows.reindex(columns=primary_available.columns)
    combined_available = pd.concat(
        [primary_available, added_rows], ignore_index=True, sort=False
    )
    if len(combined_available) != 18_144 * len(primary.VARIANTS):
        raise RuntimeError(
            f"Sensitivity full grid has {len(combined_available):,} rows; expected 72,576"
        )
    if combined_available.duplicated(key_columns).any():
        raise RuntimeError("Sensitivity full grid contains duplicate task-variant keys")

    sensitivity_complete, sensitivity_availability = (
        primary.restrict_to_complete_repetition_parents(
            combined_available, required_runs=3
        )
    )
    if len(sensitivity_complete) != 72_576:
        raise RuntimeError(
            "Visible-partial sensitivity did not restore the complete 72,576-row grid"
        )
    if not (sensitivity_availability["complete_case_parents"] == 288).all():
        raise RuntimeError("Visible-partial sensitivity did not restore all 288 parents")

    bootstrap_resamples = int(environment["bootstrap_resamples"])
    bootstrap_seed = int(environment["bootstrap_seed"])
    run_summary, repeated_summary = primary.summarize_runs(sensitivity_complete)
    losd = primary.paired_losd_comparisons(
        sensitivity_complete, bootstrap_resamples, bootstrap_seed
    )
    prompt = primary.paired_prompt_comparisons(
        sensitivity_complete, bootstrap_resamples, bootstrap_seed
    )
    expected_output_rows = {
        "run_summary": (len(run_summary), 252),
        "repeated_summary": (len(repeated_summary), 84),
        "losd_comparisons": (len(losd), 42),
        "prompt_comparisons": (len(prompt), 42),
    }
    bad_output_rows = {
        name: {"observed": observed, "expected": expected}
        for name, (observed, expected) in expected_output_rows.items()
        if observed != expected
    }
    if bad_output_rows:
        raise RuntimeError(f"Unexpected sensitivity output row counts: {bad_output_rows}")
    inference_numeric = [
        "paired_difference",
        "difference_ci_low",
        "difference_ci_high",
        "p_value",
        "p_holm",
    ]
    if not np.isfinite(losd[inference_numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite LOSD sensitivity inference result")
    if not np.isfinite(prompt[inference_numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite prompt sensitivity inference result")
    losd_comparison = compare_inference_tables(
        primary_losd,
        losd,
        ["model_label", "prompt_mode", "metric"],
    )
    prompt_comparison = compare_inference_tables(
        primary_prompt,
        prompt,
        ["model_label", "comparison", "metric"],
    )

    primary_baseline = primary_repeated.loc[
        primary_repeated["variant"] == "baseline"
    ].copy()
    sensitivity_baseline = repeated_summary.loc[
        repeated_summary["variant"] == "baseline"
    ].copy()
    baseline_columns = [
        "model_label",
        "prompt_mode",
        "semantic_f1_mean",
        "semantic_f1_sd",
        "semantic_f1_ci_low",
        "semantic_f1_ci_high",
        "hier_f1_mean",
        "hier_f1_sd",
        "hier_f1_ci_low",
        "hier_f1_ci_high",
    ]
    baseline_comparison = primary_baseline[baseline_columns].merge(
        sensitivity_baseline[baseline_columns],
        on=["model_label", "prompt_mode"],
        suffixes=("_primary", "_sensitivity"),
        validate="one_to_one",
    )
    for metric in ("semantic_f1", "hier_f1"):
        baseline_comparison[f"{metric}_mean_change"] = (
            baseline_comparison[f"{metric}_mean_sensitivity"]
            - baseline_comparison[f"{metric}_mean_primary"]
        )

    trace.to_csv(output_dir / "visible_partial_input_audit.csv", index=False)
    added_rows.to_csv(output_dir / "visible_partial_added_parent_rows.csv", index=False)
    sensitivity_availability.to_csv(
        output_dir / "complete_case_parent_counts_visible_partial.csv", index=False
    )
    run_summary.to_csv(output_dir / "run_level_summary_visible_partial.csv", index=False)
    repeated_summary.to_csv(
        output_dir / "repeated_summary_visible_partial.csv", index=False
    )
    losd.to_csv(output_dir / "paired_losd_visible_partial.csv", index=False)
    prompt.to_csv(output_dir / "paired_prompt_visible_partial.csv", index=False)
    losd_comparison.to_csv(
        output_dir / "losd_primary_vs_visible_partial.csv", index=False
    )
    prompt_comparison.to_csv(
        output_dir / "prompt_primary_vs_visible_partial.csv", index=False
    )
    baseline_comparison.to_csv(
        output_dir / "baseline_primary_vs_visible_partial.csv", index=False
    )

    summary = {
        "status": "ANALYZED",
        "interpretation": (
            "Secondary available-output sensitivity only; the primary complete-case "
            "analysis remains unchanged. Only visible response content was used."
        ),
        "audited_generation_cells": len(quality),
        "primary_excluded_cells": len(excluded),
        "visible_candidate_count_minimum": int(trace["visible_candidate_count"].min()),
        "visible_candidate_count_maximum": int(trace["visible_candidate_count"].max()),
        "restored_task_variant_rows": len(added_rows),
        "sensitivity_parent_variant_rows": len(sensitivity_complete),
        "sensitivity_complete_case_parents_minimum": int(
            sensitivity_availability["complete_case_parents"].min()
        ),
        "sensitivity_complete_case_parents_maximum": int(
            sensitivity_availability["complete_case_parents"].max()
        ),
        "hidden_reasoning_used": False,
        "embedding_cache_mutated": False,
        "embedding_cache_key_count": cache_key_count_before,
        "embedding_cache_key_sha256": cache_key_hash_before,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "losd": comparison_summary(losd_comparison),
        "prompt": comparison_summary(prompt_comparison),
        "maximum_absolute_baseline_semantic_f1_mean_change": float(
            baseline_comparison["semantic_f1_mean_change"].abs().max()
        ),
        "maximum_absolute_baseline_hier_f1_mean_change": float(
            baseline_comparison["hier_f1_mean_change"].abs().max()
        ),
        "primary_complete_parent_variant_rows": len(primary_complete),
        "primary_config": asdict(config),
        "primary_analysis_script_sha256": environment["analysis_script_sha256"],
        "primary_pipeline_script_sha256": environment["pipeline_script_sha256"],
        "sensitivity_script_sha256": primary.sha256_file(Path(__file__).resolve()),
        "python": platform.python_version(),
        "command_line": [sys.executable, *sys.argv],
    }
    summary["all_headline_conclusions_stable"] = (
        summary["losd"]["headline_conclusion_changes"] == 0
        and summary["prompt"]["headline_conclusion_changes"] == 0
    )
    (output_dir / "sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        """# Visible-partial-output sensitivity analysis

This is a secondary sensitivity analysis. It does not replace or modify the
primary complete-case analysis. The 14 primary exclusions are rescored using
only candidate items parsed from visible response content (`raw_text`). Hidden
reasoning is never parsed or scored. The same frozen ontology, embedding cache,
pipeline parameters, variants, parent averaging, bootstrap, Wilcoxon--Pratt
tests, and Holm families are reused.

The central audit is `sensitivity_summary.json`. The two `*_primary_vs_*` CSV
files report changes in effect sign, confidence-interval exclusion of zero, and
Holm-adjusted significance decisions for LOSD and prompting comparisons.
""",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
