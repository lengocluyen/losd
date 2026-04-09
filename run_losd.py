from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from losd import (
    DEFAULT_VARIANTS,
    HashingTextEmbedder,
    LOSDConfig,
    LOSDPipeline,
    SentenceTransformerEmbedder,
    VariantSpec,
    build_fewshot_bank,
    load_cached_candidate_texts,
    load_ontology_resources,
    select_parent_pool,
)


VARIANT_MAP = {variant.name: variant for variant in DEFAULT_VARIANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LOSD neuro-symbolic refinement pipeline on cached notebook outputs."
    )
    parser.add_argument("--ttl", default="esco_cmo_binding.ttl")
    parser.add_argument("--source-outdir", default="exp_out")
    parser.add_argument("--target-outdir", default="")
    parser.add_argument("--methods", nargs="+", default=["zero", "few", "rag"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[variant.name for variant in DEFAULT_VARIANTS],
        choices=list(VARIANT_MAP),
    )
    parser.add_argument("--embedder", choices=["sentence-transformer", "hash"], default="hash")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--candidate-pool-size", type=int, default=12)
    parser.add_argument("--duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--alignment-threshold", type=float, default=0.70)
    parser.add_argument("--lexical-weight", type=float, default=0.35)
    parser.add_argument("--depth-tolerance", type=int, default=1)
    parser.add_argument("--novelty-threshold", type=float, default=0.55)
    parser.add_argument("--max-plausible-novel-in-final", type=int, default=1)
    parser.add_argument("--rerank-lexical-weight", type=float, default=0.20)
    parser.add_argument("--rerank-generation-weight", type=float, default=0.10)
    parser.add_argument("--rerank-novelty-weight", type=float, default=0.10)
    parser.add_argument("--rerank-weak-unresolved-penalty", type=float, default=0.15)
    parser.add_argument("--rag-context-items", type=int, default=30)
    parser.add_argument("--fewshot-examples", type=int, default=2)
    parser.add_argument("--parent-filter", default="balanced", choices=["balanced", "all"])
    parser.add_argument("--min-k", type=int, default=5)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--parent-limit", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LOSDConfig:
    return LOSDConfig(
        final_k=args.final_k,
        candidate_pool_size=args.candidate_pool_size,
        duplicate_threshold=args.duplicate_threshold,
        alignment_threshold=args.alignment_threshold,
        lexical_weight=args.lexical_weight,
        depth_tolerance=args.depth_tolerance,
        novelty_threshold=args.novelty_threshold,
        rerank_lexical_weight=args.rerank_lexical_weight,
        rerank_generation_weight=args.rerank_generation_weight,
        rerank_novelty_weight=args.rerank_novelty_weight,
        rerank_weak_unresolved_penalty=args.rerank_weak_unresolved_penalty,
        max_plausible_novel_in_final=args.max_plausible_novel_in_final,
        rag_context_items=args.rag_context_items,
        fewshot_examples=args.fewshot_examples,
        random_seed=args.random_seed,
    )


def build_embedder(args: argparse.Namespace):
    if args.embedder == "sentence-transformer":
        return SentenceTransformerEmbedder(args.embedding_model)
    return HashingTextEmbedder()


def resolve_variants(names: list[str]) -> list[VariantSpec]:
    return [VARIANT_MAP[name] for name in names]


def resolve_eval_gold(
    resources,
    source_outdir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    gold_path = source_outdir / "gold_pairs.csv"
    if gold_path.exists():
        return pd.read_csv(gold_path)
    parent_pool = select_parent_pool(
        resources.gold,
        parent_filter=args.parent_filter,
        min_k=args.min_k,
        max_k=args.max_k,
        parent_limit=args.parent_limit or None,
        random_seed=args.random_seed,
    )
    return resources.gold.loc[resources.gold["parent_uri"].isin(parent_pool)].copy()


def fewshot_context_for_parent(
    parent_skill_id: str,
    bank: list[tuple[str, list[str]]],
    example_count: int,
    random_seed: int,
) -> list[str]:
    if not bank:
        return []
    rng = random.Random(f"{random_seed}:{parent_skill_id}")
    indices = list(range(len(bank)))
    rng.shuffle(indices)
    selected = [bank[index] for index in indices[:example_count]]
    context: list[str] = []
    for _, children in selected:
        context.extend(children)
    return context


def main() -> None:
    args = parse_args()
    config = build_config(args)
    source_outdir = Path(args.source_outdir)
    target_outdir = Path(args.target_outdir) if args.target_outdir else source_outdir / "losd"
    target_outdir.mkdir(parents=True, exist_ok=True)

    embedder = build_embedder(args)
    resources = load_ontology_resources(args.ttl, embedder, config)
    pipeline = LOSDPipeline(resources, embedder, config)

    gold_eval = resolve_eval_gold(resources, source_outdir, args)
    parent_pool = gold_eval["parent_uri"].drop_duplicates().tolist()
    gold_eval.to_csv(target_outdir / "gold_pairs.csv", index=False)
    pd.DataFrame({"parent_skill_id": parent_pool}).to_csv(target_outdir / "parents.csv", index=False)

    fewshot_bank = build_fewshot_bank(
        resources.gold,
        resources.uri2label,
        excluded_parents=parent_pool,
        min_children=max(5, args.min_k),
        bank_size=8,
        children_per_example=6,
        random_seed=args.random_seed,
    )
    variants = resolve_variants(args.variants)

    candidate_rows: list[dict] = []
    parent_rows: list[dict] = []
    cache_dir = source_outdir / "cache"

    for method in args.methods:
        for parent_skill_id in parent_pool:
            gold_children = (
                gold_eval.loc[gold_eval["parent_uri"] == parent_skill_id, "child_uri"]
                .drop_duplicates()
                .tolist()
            )
            raw_candidates = load_cached_candidate_texts(cache_dir, method, parent_skill_id)
            if method == "few":
                context_texts = fewshot_context_for_parent(
                    parent_skill_id,
                    fewshot_bank,
                    config.fewshot_examples,
                    config.random_seed,
                )
            elif method == "rag":
                exclude = set(gold_children)
                exclude.add(parent_skill_id)
                context_texts = pipeline.nearest_neighbor_context(
                    parent_skill_id,
                    k=config.rag_context_items,
                    exclude_uris=exclude,
                )
            else:
                context_texts = []

            candidate_batch, parent_batch = pipeline.run_parent(
                parent_skill_id=parent_skill_id,
                prompt_mode=method,
                raw_candidates=raw_candidates,
                gold_children=gold_children,
                context_texts=context_texts,
                variants=variants,
            )
            candidate_rows.extend(candidate_batch)
            parent_rows.extend(parent_batch)

    candidate_df = pd.DataFrame(candidate_rows)
    parent_df = pd.DataFrame(parent_rows)
    parent_summary_df = pipeline.summarize_parent_results(parent_df)
    candidate_summary_df = pipeline.summarize_candidate_results(candidate_df)

    candidate_path = target_outdir / "losd_candidate_records.csv"
    parent_path = target_outdir / "losd_parent_results.csv"
    parent_summary_path = target_outdir / "losd_method_summary.csv"
    candidate_summary_path = target_outdir / "losd_candidate_summary.csv"

    candidate_df.to_csv(candidate_path, index=False)
    parent_df.to_csv(parent_path, index=False)
    parent_summary_df.to_csv(parent_summary_path, index=False)
    candidate_summary_df.to_csv(candidate_summary_path, index=False)

    print(f"Saved: {candidate_path}")
    print(f"Saved: {parent_path}")
    print(f"Saved: {parent_summary_path}")
    print(f"Saved: {candidate_summary_path}")


if __name__ == "__main__":
    main()
