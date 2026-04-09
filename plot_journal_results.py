from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "baseline",
    "validation",
    "validation_rerank",
    "soft_validation",
    "soft_validation_rerank",
]
METHOD_ORDER = ["zero", "few", "rag"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visualizations from journal extension result CSV files."
    )
    parser.add_argument(
        "--results-dir",
        default="exp_out/journal_extension",
        help="Directory containing journal_candidate_records.csv and journal_parent_results.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where plots will be written. Defaults to <results-dir>/plots.",
    )
    return parser.parse_args()


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["variant"] = pd.Categorical(working["variant"], categories=VARIANT_ORDER, ordered=True)
    working["prompt_mode"] = pd.Categorical(working["prompt_mode"], categories=METHOD_ORDER, ordered=True)
    return working.sort_values(["prompt_mode", "variant"]).reset_index(drop=True)


def _display_labels(frame: pd.DataFrame) -> list[str]:
    return [f"{row.prompt_mode}\n{row.variant}" for row in frame.itertuples()]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_result_frames(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_path = results_dir / "journal_candidate_records.csv"
    parent_path = results_dir / "journal_parent_results.csv"
    parent_summary_path = results_dir / "journal_method_summary.csv"
    candidate_summary_path = results_dir / "journal_candidate_summary.csv"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing file: {candidate_path}")
    if not parent_path.exists():
        raise FileNotFoundError(f"Missing file: {parent_path}")

    candidate_df = pd.read_csv(candidate_path)
    parent_df = pd.read_csv(parent_path)

    if parent_summary_path.exists():
        parent_summary_df = pd.read_csv(parent_summary_path)
    else:
        parent_summary_df = (
            parent_df.groupby(["prompt_mode", "variant"], dropna=False)
            .agg(
                parent_count=("parent_skill_id", "nunique"),
                semantic_f1_mean=("semantic_f1", "mean"),
                hier_f1_mean=("hier_f1", "mean"),
                selected_count_mean=("selected_count", "mean"),
                duplicate_count_mean=("duplicate_count", "mean"),
                invalid_count_mean=("invalid_count", "mean"),
                unaligned_count_mean=("unaligned_count", "mean"),
                plausible_novel_count_mean=("plausible_novel_count", "mean"),
                hallucination_count_mean=("hallucination_count", "mean"),
            )
            .reset_index()
        )

    if candidate_summary_path.exists():
        candidate_summary_df = pd.read_csv(candidate_summary_path)
    else:
        candidate_summary_df = (
            candidate_df.groupby(["prompt_mode", "variant"], dropna=False)
            .agg(
                candidate_count=("candidate_id", "count"),
                duplicate_rate=("is_duplicate", "mean"),
                parent_repeat_rate=("viol_parent_repeat", "mean"),
                type_violation_rate=("viol_type", "mean"),
                unaligned_rate=("viol_unaligned", "mean"),
                depth_violation_rate=("viol_depth", "mean"),
                valid_rate=("is_valid", "mean"),
                selected_rate=("selected_final", "mean"),
            )
            .reset_index()
        )

    return candidate_df, parent_df, parent_summary_df, candidate_summary_df


def _main_variant_frame(parent_summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred_variants = ["baseline", "validation_rerank", "soft_validation_rerank"]
    frame = parent_summary_df[parent_summary_df["variant"].isin(preferred_variants)].copy()
    frame["variant"] = pd.Categorical(frame["variant"], categories=preferred_variants, ordered=True)
    frame["prompt_mode"] = pd.Categorical(frame["prompt_mode"], categories=METHOD_ORDER, ordered=True)
    return frame.sort_values(["prompt_mode", "variant"]).reset_index(drop=True)


def _prompt_only_frame(parent_summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred_order = ["soft_validation_rerank", "baseline", "validation_rerank", "soft_validation", "validation"]
    rows: list[pd.Series] = []
    for prompt_mode in METHOD_ORDER:
        prompt_rows = parent_summary_df[parent_summary_df["prompt_mode"] == prompt_mode]
        if prompt_rows.empty:
            continue
        chosen = None
        for variant in preferred_order:
            matched = prompt_rows[prompt_rows["variant"] == variant]
            if not matched.empty:
                chosen = matched.iloc[0]
                break
        if chosen is None:
            chosen = prompt_rows.iloc[0]
        rows.append(chosen)
    if not rows:
        return parent_summary_df.iloc[0:0].copy()
    frame = pd.DataFrame(rows).copy()
    frame["prompt_mode"] = pd.Categorical(frame["prompt_mode"], categories=METHOD_ORDER, ordered=True)
    return frame.sort_values("prompt_mode").reset_index(drop=True)


def plot_method_performance(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _sort_frame(parent_summary_df)
    labels = _display_labels(frame)
    x = np.arange(len(frame))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(frame) * 1.2), 5.5))
    ax.bar(x - width / 2, frame["semantic_f1_mean"], width=width, label="Semantic F1", color="#1f77b4")
    ax.bar(x + width / 2, frame["hier_f1_mean"], width=width, label="Hierarchy F1", color="#ff7f0e")
    ax.set_title("Journal Extension Performance by Prompt Mode and Variant")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, max(1.0, float(frame[["semantic_f1_mean", "hier_f1_mean"]].max().max()) * 1.1))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "method_performance.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_losd_ablation_scores(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _main_variant_frame(parent_summary_df)
    prompts = [prompt for prompt in METHOD_ORDER if prompt in frame["prompt_mode"].astype(str).tolist()]
    variants = ["baseline", "validation_rerank", "soft_validation_rerank"]
    colors = {
        "baseline": "#4c72b0",
        "validation_rerank": "#c44e52",
        "soft_validation_rerank": "#55a868",
    }
    display_prompt = {"zero": "ZS", "few": "FS", "rag": "RAG"}
    x = np.arange(len(prompts))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=False)
    for axis, column, title in zip(
        axes,
        ["semantic_f1_mean", "hier_f1_mean"],
        ["Semantic F1", "Hier-F1"],
    ):
        for offset, variant in enumerate(variants):
            values = []
            for prompt in prompts:
                matched = frame[(frame["prompt_mode"].astype(str) == prompt) & (frame["variant"].astype(str) == variant)]
                values.append(float(matched.iloc[0][column]) if not matched.empty else 0.0)
            axis.bar(
                x + (offset - 1) * width,
                values,
                width=width,
                color=colors[variant],
                label=variant.replace("_", r"\_"),
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([display_prompt.get(prompt, prompt.upper()) for prompt in prompts])
        axis.set_ylim(0.0, max(0.25, float(frame[column].max()) * 1.2))
        axis.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "losd_ablation_scores.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_losd_selected_count(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _main_variant_frame(parent_summary_df)
    prompts = [prompt for prompt in METHOD_ORDER if prompt in frame["prompt_mode"].astype(str).tolist()]
    variants = ["baseline", "validation_rerank", "soft_validation_rerank"]
    colors = {
        "baseline": "#4c72b0",
        "validation_rerank": "#c44e52",
        "soft_validation_rerank": "#55a868",
    }
    display_prompt = {"zero": "ZS", "few": "FS", "rag": "RAG"}
    x = np.arange(len(prompts))
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for offset, variant in enumerate(variants):
        values = []
        for prompt in prompts:
            matched = frame[(frame["prompt_mode"].astype(str) == prompt) & (frame["variant"].astype(str) == variant)]
            values.append(float(matched.iloc[0]["selected_count_mean"]) if not matched.empty else 0.0)
        ax.bar(
            x + (offset - 1) * width,
            values,
            width=width,
            color=colors[variant],
            label=variant.replace("_", r"\_"),
        )
    ax.set_title("Average Selected Count")
    ax.set_ylabel("Items per Parent")
    ax.set_xticks(x)
    ax.set_xticklabels([display_prompt.get(prompt, prompt.upper()) for prompt in prompts])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "losd_selected_count.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_dashboard(candidate_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _sort_frame(candidate_summary_df)
    labels = _display_labels(frame)
    metrics = [
        ("duplicate_rate", "Duplicate Rate", "#c44e52"),
        ("parent_repeat_rate", "Parent Repeat Rate", "#8172b2"),
        ("type_violation_rate", "Type Violation Rate", "#55a868"),
        ("unaligned_rate", "Unaligned Rate", "#dd8452"),
        ("depth_violation_rate", "Depth Violation Rate", "#937860"),
        ("valid_rate", "Valid Rate", "#4c72b0"),
    ]
    if "soft_valid_rate" in frame.columns:
        metrics.append(("soft_valid_rate", "Soft Valid Rate", "#64b5cd"))

    ncols = 3 if len(metrics) <= 6 else 4
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(16, ncols * 4.8), max(8, nrows * 3.8)), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(len(frame))

    for axis, (column, title, color) in zip(axes, metrics):
        axis.bar(x, frame[column], color=color)
        axis.set_title(title)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")

    for axis in axes[len(metrics):]:
        axis.axis("off")

    fig.suptitle("Candidate-Level Error and Validity Rates", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / "candidate_error_dashboard.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_novelty_breakdown(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _sort_frame(parent_summary_df)
    labels = _display_labels(frame)
    x = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(max(10, len(frame) * 1.2), 5.5))
    plausible = frame["plausible_novel_count_mean"]
    hallucination = frame["hallucination_count_mean"]
    ax.bar(x, plausible, label="Plausible Novel", color="#2ca02c")
    ax.bar(x, hallucination, bottom=plausible, label="Hallucination", color="#d62728")
    ax.set_title("Novelty Breakdown for Unresolved Outputs")
    ax.set_ylabel("Average Count per Parent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = output_dir / "novelty_breakdown.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_novelty_breakdown_counts(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _prompt_only_frame(parent_summary_df)
    display_prompt = {"zero": "ZS", "few": "FS", "rag": "RAG"}
    x = np.arange(len(frame))
    plausible = frame["plausible_novel_count_mean"].astype(float)
    hallucination = frame["hallucination_count_mean"].astype(float)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, plausible, color="#2ca02c", label="Plausible Novel")
    ax.bar(x, hallucination, bottom=plausible, color="#d62728", label="Hallucination")
    ax.set_title("Novelty Breakdown by Prompt Regime")
    ax.set_ylabel("Average Count per Parent")
    ax.set_xticks(x)
    ax.set_xticklabels([display_prompt.get(str(prompt), str(prompt).upper()) for prompt in frame["prompt_mode"]])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "novelty_breakdown_counts.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_novelty_breakdown_share(parent_summary_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _prompt_only_frame(parent_summary_df).copy()
    display_prompt = {"zero": "ZS", "few": "FS", "rag": "RAG"}
    x = np.arange(len(frame))
    plausible = frame["plausible_novel_count_mean"].astype(float)
    hallucination = frame["hallucination_count_mean"].astype(float)
    total = plausible + hallucination
    plausible_share = np.divide(plausible, total, out=np.zeros_like(plausible, dtype=float), where=total > 0)
    hallucination_share = np.divide(hallucination, total, out=np.zeros_like(hallucination, dtype=float), where=total > 0)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x, plausible_share, color="#2ca02c", label="Plausible Novel")
    ax.bar(x, hallucination_share, bottom=plausible_share, color="#d62728", label="Hallucination")
    ax.set_title("Share of Unresolved Output Types")
    ax.set_ylabel("Share")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([display_prompt.get(str(prompt), str(prompt).upper()) for prompt in frame["prompt_mode"]])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    output_path = output_dir / "novelty_breakdown_share.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_parent_distributions(parent_df: pd.DataFrame, output_dir: Path) -> Path:
    frame = _sort_frame(parent_df)
    frame["group"] = frame["prompt_mode"].astype(str) + "\n" + frame["variant"].astype(str)
    ordered_groups = list(dict.fromkeys(frame["group"].tolist()))

    semantic_series = [frame.loc[frame["group"] == group, "semantic_f1"].dropna().values for group in ordered_groups]
    hier_series = [frame.loc[frame["group"] == group, "hier_f1"].dropna().values for group in ordered_groups]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    axes[0].boxplot(semantic_series, tick_labels=ordered_groups, patch_artist=True)
    axes[0].set_title("Semantic F1 Distribution")
    axes[0].set_ylabel("Per-Parent Score")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].boxplot(hier_series, tick_labels=ordered_groups, patch_artist=True)
    axes[1].set_title("Hierarchy F1 Distribution")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle("Per-Parent Score Distributions")
    fig.tight_layout()
    output_path = output_dir / "parent_score_distributions.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_plot_index(output_dir: Path, plot_paths: list[Path]) -> Path:
    index_path = output_dir / "plot_index.txt"
    lines = ["Generated plots:"]
    lines.extend(str(path) for path in plot_paths)
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = _ensure_dir(Path(args.output_dir) if args.output_dir else results_dir / "plots")

    candidate_df, parent_df, parent_summary_df, candidate_summary_df = load_result_frames(results_dir)
    plot_paths = [
        plot_method_performance(parent_summary_df, output_dir),
        plot_losd_ablation_scores(parent_summary_df, output_dir),
        plot_losd_selected_count(parent_summary_df, output_dir),
        plot_error_dashboard(candidate_summary_df, output_dir),
        plot_novelty_breakdown(parent_summary_df, output_dir),
        plot_novelty_breakdown_counts(parent_summary_df, output_dir),
        plot_novelty_breakdown_share(parent_summary_df, output_dir),
        plot_parent_distributions(parent_df, output_dir),
    ]
    index_path = write_plot_index(output_dir, plot_paths)

    for path in plot_paths:
        print(f"Saved: {path}")
    print(f"Saved: {index_path}")


if __name__ == "__main__":
    main()
