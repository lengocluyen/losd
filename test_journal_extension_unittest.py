from __future__ import annotations

import json
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from journal_extension import (
    HashingTextEmbedder,
    JournalExtensionConfig,
    JournalExtensionPipeline,
    OntologyResources,
    VariantSpec,
    normalize_surface_text,
)


def build_test_pipeline() -> JournalExtensionPipeline:
    embedder = HashingTextEmbedder(dim=512)
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("parent", "child_descriptive"),
            ("parent", "child_regression"),
            ("child_descriptive", "grandchild_timeseries"),
        ]
    )
    label_map = {
        "parent": "statistical analysis",
        "child_descriptive": "descriptive statistics",
        "child_regression": "regression analysis",
        "grandchild_timeseries": "time series analysis",
    }
    def_map = {uri: None for uri in label_map}
    alt_map = {uri: [] for uri in label_map}
    uri2label = dict(label_map)
    uri2text = dict(label_map)
    gold = pd.DataFrame(
        [
            {
                "parent_uri": "parent",
                "parent_label": uri2label["parent"],
                "child_uri": "child_descriptive",
                "child_label": uri2label["child_descriptive"],
                "child_def": None,
            },
            {
                "parent_uri": "parent",
                "parent_label": uri2label["parent"],
                "child_uri": "child_regression",
                "child_label": uri2label["child_regression"],
                "child_def": None,
            },
            {
                "parent_uri": "child_descriptive",
                "parent_label": uri2label["child_descriptive"],
                "child_uri": "grandchild_timeseries",
                "child_label": uri2label["grandchild_timeseries"],
                "child_def": None,
            },
        ]
    )
    ancestors = {node: nx.ancestors(graph, node) for node in graph.nodes()}
    descendants = {node: nx.descendants(graph, node) for node in graph.nodes()}
    depths = {
        "parent": 0,
        "child_descriptive": 1,
        "child_regression": 1,
        "grandchild_timeseries": 2,
    }
    parent_children = {
        "parent": ["child_descriptive", "child_regression"],
        "child_descriptive": ["grandchild_timeseries"],
    }
    all_uris = list(label_map)
    all_text_embeddings = embedder.encode(
        [normalize_surface_text(uri2text[uri])[1] or uri2text[uri] for uri in all_uris]
    )
    label_embeddings_matrix = embedder.encode(
        [normalize_surface_text(uri2label[uri])[1] or uri2label[uri] for uri in all_uris]
    )
    label_embeddings = {
        uri: label_embeddings_matrix[index] for index, uri in enumerate(all_uris)
    }
    resources = OntologyResources(
        graph=graph,
        gold=gold,
        label_map=label_map,
        def_map=def_map,
        alt_map=alt_map,
        uri2label=uri2label,
        uri2text=uri2text,
        ancestors=ancestors,
        descendants=descendants,
        depths=depths,
        parent_children=parent_children,
        all_uris=all_uris,
        all_text_embeddings=all_text_embeddings,
        label_embeddings=label_embeddings,
    )
    config = JournalExtensionConfig(
        final_k=3,
        candidate_pool_size=6,
        duplicate_threshold=0.90,
        alignment_threshold=0.74,
        lexical_weight=0.45,
        depth_tolerance=1,
        novelty_threshold=0.22,
        fewshot_examples=2,
        rag_context_items=5,
    )
    return JournalExtensionPipeline(resources, embedder, config)


def run_test_parent():
    pipeline = build_test_pipeline()
    raw_candidates = [
        "Descriptive statistics",
        "descriptive-statistics",
        "Time series analysis",
        "Presentation software",
        "Regression analysis",
        "Statistical profiling",
    ]
    candidate_rows, parent_rows = pipeline.run_parent(
        parent_skill_id="parent",
        prompt_mode="rag",
        raw_candidates=raw_candidates,
        gold_children=["child_descriptive", "child_regression"],
        context_texts=["descriptive statistics", "regression analysis", "time series analysis"],
    )
    return pipeline, pd.DataFrame(candidate_rows), pd.DataFrame(parent_rows)


class JournalExtensionTests(unittest.TestCase):
    def test_prompts_use_over_generation_pool_size(self) -> None:
        pipeline = build_test_pipeline()
        zero_prompt = pipeline.build_zero_prompt("statistical analysis")
        few_prompt = pipeline.build_few_prompt(
            "statistical analysis",
            [("data management", ["data cleaning", "data validation"])],
        )
        rag_prompt = pipeline.build_rag_prompt("statistical analysis", ["descriptive statistics"])

        self.assertIn("exactement 6", zero_prompt)
        self.assertIn("exactement 6", few_prompt)
        self.assertIn("exactement 6", rag_prompt)
        self.assertIn("outils", rag_prompt)

    def test_duplicate_filter_and_validation_flags(self) -> None:
        _, candidate_df, _ = run_test_parent()
        refined = candidate_df[candidate_df["variant"] == "validation_rerank"].set_index("raw_text")

        self.assertTrue(refined.loc["descriptive-statistics", "is_duplicate"])
        self.assertEqual(refined.loc["descriptive-statistics", "duplicate_of"], "parent__rag__01")
        self.assertTrue(refined.loc["Presentation software", "viol_type"])
        self.assertFalse(refined.loc["Presentation software", "is_valid"])
        self.assertTrue(refined.loc["Regression analysis", "is_aligned"])

    def test_reranking_prefers_direct_children_and_tracks_counts(self) -> None:
        _, candidate_df, parent_df = run_test_parent()
        refined_candidates = candidate_df[candidate_df["variant"] == "validation_rerank"]
        selected = refined_candidates[refined_candidates["selected_final"]]["raw_text"].tolist()
        parent_row = parent_df[parent_df["variant"] == "validation_rerank"].iloc[0]

        self.assertIn("Descriptive statistics", selected)
        self.assertIn("Regression analysis", selected)
        self.assertNotIn("descriptive-statistics", selected)
        self.assertEqual(parent_row["duplicate_count"], 1)
        self.assertEqual(parent_row["candidate_pool_size"], 6)
        self.assertGreaterEqual(parent_row["hier_f1"], parent_row["semantic_f1"])

        payload = json.loads(parent_row["selected_candidates"])
        self.assertEqual(len(payload), 3)

    def test_unresolved_outputs_receive_novelty_labels(self) -> None:
        pipeline, candidate_df, parent_df = run_test_parent()
        refined = candidate_df[candidate_df["variant"] == "validation_rerank"].set_index("raw_text")

        self.assertEqual(refined.loc["Statistical profiling", "novelty_status"], "plausible_novel_skill")
        self.assertEqual(refined.loc["Presentation software", "novelty_status"], "hallucination")

        parent_summary = pipeline.summarize_parent_results(parent_df)
        candidate_summary = pipeline.summarize_candidate_results(candidate_df)

        self.assertFalse(parent_summary.empty)
        self.assertFalse(candidate_summary.empty)
        duplicate_rate = float(
            candidate_summary.loc[candidate_summary["variant"] == "validation_rerank", "duplicate_rate"].iloc[0]
        )
        self.assertTrue(np.isclose(duplicate_rate, 1 / 6))

    def test_soft_validation_rerank_backfills_non_hard_invalid_candidates(self) -> None:
        pipeline = build_test_pipeline()
        candidate_rows, parent_rows = pipeline.run_parent(
            parent_skill_id="parent",
            prompt_mode="rag",
            raw_candidates=[
                "Regression analysis",
                "Statistical profiling",
                "Presentation software",
            ],
            gold_children=["child_descriptive", "child_regression"],
            context_texts=["regression analysis", "descriptive statistics"],
            variants=[
                VariantSpec("validation_rerank", validation_mode="hard", use_rerank=True),
                VariantSpec("soft_validation_rerank", validation_mode="soft", use_rerank=True),
            ],
        )
        candidate_df = pd.DataFrame(candidate_rows)
        parent_df = pd.DataFrame(parent_rows).set_index("variant")

        hard_selected = json.loads(parent_df.loc["validation_rerank", "selected_candidates"])
        soft_selected = json.loads(parent_df.loc["soft_validation_rerank", "selected_candidates"])

        self.assertEqual(parent_df.loc["validation_rerank", "selected_count"], 1)
        self.assertEqual(parent_df.loc["soft_validation_rerank", "selected_count"], 2)
        self.assertEqual(hard_selected, ["Regression analysis"])
        self.assertIn("Regression analysis", soft_selected)
        self.assertIn("Statistical profiling", soft_selected)

        soft_records = candidate_df[candidate_df["variant"] == "soft_validation_rerank"].set_index("raw_text")
        self.assertEqual(soft_records.loc["Statistical profiling", "selection_bucket"], "plausible_novel")
        self.assertFalse(soft_records.loc["Presentation software", "is_soft_valid"])


if __name__ == "__main__":
    unittest.main()
