from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import SKOS

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback only used in thin environments
    fuzz = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - fallback only used in thin environments
    linear_sum_assignment = None


DEFAULT_NEGATIVE_PATTERNS = (
    r"\boutil(?:s)?\b",
    r"\bmachine(?:s)?\b",
    r"\béquipement(?:s)?\b",
    r"\bmatériels?\b",
    r"\blogiciel(?:s)?\b",
    r"\bsoftware\b",
    r"\bapplication(?:s)?\b",
    r"\bapps?\b",
    r"\bplateforme(?:s)?\b",
    r"\bframework(?:s)?\b",
    r"\bplugin(?:s)?\b",
    r"\bingénieur(?:e)?\b",
    r"\btechnicien(?:ne)?s?\b",
    r"\bconsultant(?:e)?s?\b",
    r"\bmanager(?:s)?\b",
    r"\bopérateur(?:trice)?s?\b",
    r"\borganisation(?:s)?\b",
    r"\bentreprise(?:s)?\b",
    r"\bsociété(?:s)?\b",
    r"\bassociation(?:s)?\b",
    r"\buniversité(?:s)?\b",
)

TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
PUNCT_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
        "‐": "-",
        "‑": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
    }
)


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class HashingTextEmbedder:
    """Small deterministic fallback embedder for tests and offline execution."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                matrix[row, idx] += 1.0
        return _normalize_rows(matrix)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    validation_mode: str = "none"
    use_rerank: bool = False


DEFAULT_VARIANTS = (
    VariantSpec("baseline"),
    VariantSpec("validation", validation_mode="hard"),
    VariantSpec("validation_rerank", validation_mode="hard", use_rerank=True),
    VariantSpec("soft_validation", validation_mode="soft"),
    VariantSpec("soft_validation_rerank", validation_mode="soft", use_rerank=True),
)


@dataclass
class JournalExtensionConfig:
    final_k: int = 5
    candidate_pool_size: int = 12
    duplicate_threshold: float = 0.90
    alignment_threshold: float = 0.70
    lexical_weight: float = 0.35
    depth_tolerance: int = 1
    novelty_threshold: float = 0.55
    rerank_alignment_weight: float = 0.50
    rerank_hierarchy_weight: float = 0.25
    rerank_context_weight: float = 0.15
    rerank_lexical_weight: float = 0.20
    rerank_generation_weight: float = 0.10
    rerank_novelty_weight: float = 0.10
    rerank_redundancy_weight: float = 0.20
    rerank_weak_unresolved_penalty: float = 0.15
    max_plausible_novel_in_final: int = 1
    fewshot_examples: int = 2
    rag_context_items: int = 30
    random_seed: int = 42
    cmo_namespace: str = "http://www.example.com/cmo#"
    relation_name: str = "hasImmediateSubCompetence"
    negative_patterns: tuple[str, ...] = field(default_factory=lambda: DEFAULT_NEGATIVE_PATTERNS)


@dataclass
class OntologyResources:
    graph: nx.DiGraph
    gold: pd.DataFrame
    label_map: dict[str, str | None]
    def_map: dict[str, str | None]
    alt_map: dict[str, list[str]]
    uri2label: dict[str, str]
    uri2text: dict[str, str]
    ancestors: dict[str, set[str]]
    descendants: dict[str, set[str]]
    depths: dict[str, int]
    parent_children: dict[str, list[str]]
    all_uris: list[str]
    all_text_embeddings: np.ndarray
    label_embeddings: dict[str, np.ndarray]


def strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_surface_text(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    normalized = unicodedata.normalize("NFKC", raw.translate(PUNCT_TRANSLATION)).lower().strip()
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" ,;:.")
    comparison = strip_accents(normalized)
    comparison = comparison.replace("-", " ")
    comparison = re.sub(r"[^\w\s]", " ", comparison)
    comparison = " ".join(comparison.split())
    return normalized, comparison


def lexical_similarity(left: str, right: str) -> float:
    _, left_key = normalize_surface_text(left)
    _, right_key = normalize_surface_text(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    left_tokens = set(tokenize(left_key))
    right_tokens = set(tokenize(right_key))
    jaccard = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    containment = 0.95 if left_key in right_key or right_key in left_key else 0.0
    if fuzz is not None:
        token_ratio = fuzz.token_sort_ratio(left_key, right_key) / 100.0
        partial_ratio = fuzz.partial_ratio(left_key, right_key) / 100.0
    else:  # pragma: no cover - normal path uses rapidfuzz
        token_ratio = jaccard
        partial_ratio = containment
    return max(jaccard, containment, token_ratio, partial_ratio)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)
    left_norm = _normalize_rows(left.astype(np.float32, copy=False))
    right_norm = _normalize_rows(right.astype(np.float32, copy=False))
    return left_norm @ right_norm.T


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")


def load_cached_candidate_texts(cache_dir: str | Path, method: str, parent_skill_id: str) -> list[str]:
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return []
    parent_tail = _safe_name(parent_skill_id.rsplit("/", 1)[-1])
    matches = sorted(cache_path.glob(f"{method}__{parent_tail}*.json"))
    for match in matches:
        try:
            payload = json.loads(match.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return [str(item) for item in payload if isinstance(item, str)]
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            return [str(item) for item in payload["items"] if isinstance(item, str)]
    return []


def select_parent_pool(
    gold: pd.DataFrame,
    parent_filter: str = "balanced",
    min_k: int = 5,
    max_k: int = 12,
    parent_limit: int | None = None,
    random_seed: int = 42,
) -> list[str]:
    counts = gold.groupby("parent_uri")["child_uri"].nunique()
    if parent_filter == "balanced":
        parent_pool = counts[(counts >= min_k) & (counts <= max_k)].index.tolist()
    else:
        parent_pool = counts.index.tolist()
    if parent_limit:
        rng = random.Random(random_seed)
        rng.shuffle(parent_pool)
        parent_pool = parent_pool[:parent_limit]
    return list(parent_pool)


def build_fewshot_bank(
    gold: pd.DataFrame,
    uri2label: dict[str, str],
    excluded_parents: Iterable[str] = (),
    min_children: int = 5,
    bank_size: int = 8,
    children_per_example: int = 6,
    random_seed: int = 42,
) -> list[tuple[str, list[str]]]:
    excluded = set(excluded_parents)
    counts = gold.groupby("parent_uri")["child_uri"].nunique()
    parent_ids = [parent for parent, count in counts.items() if count >= min_children and parent not in excluded]
    rng = random.Random(random_seed)
    rng.shuffle(parent_ids)
    bank: list[tuple[str, list[str]]] = []
    for parent_id in parent_ids:
        child_uris = (
            gold.loc[gold["parent_uri"] == parent_id, "child_uri"].drop_duplicates().tolist()
        )
        child_labels = [uri2label.get(uri, uri) for uri in child_uris][:children_per_example]
        if child_labels:
            bank.append((uri2label.get(parent_id, parent_id), child_labels))
        if len(bank) >= bank_size:
            break
    return bank


def load_ontology_resources(
    ttl_path: str | Path,
    embedder: TextEmbedder,
    config: JournalExtensionConfig | None = None,
) -> OntologyResources:
    cfg = config or JournalExtensionConfig()
    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")

    cmo = Namespace(cfg.cmo_namespace)
    relation = getattr(cmo, cfg.relation_name)
    edges = [(str(parent), str(child)) for parent, _, child in graph.triples((None, relation, None))]
    uris = set()
    for parent, child in edges:
        uris.add(parent)
        uris.add(child)

    def first(uri: URIRef, predicate: Any) -> str | None:
        value = next(graph.objects(uri, predicate), None)
        return str(value) if value is not None else None

    label_map: dict[str, str | None] = {}
    def_map: dict[str, str | None] = {}
    alt_map: dict[str, list[str]] = {}
    for uri in uris:
        ref = URIRef(uri)
        label_map[uri] = first(ref, SKOS.prefLabel)
        def_map[uri] = first(ref, SKOS.definition)
        alt_map[uri] = [str(value) for value in graph.objects(ref, SKOS.altLabel)]

    def fallback_label(uri: str) -> str:
        label = label_map.get(uri)
        if label:
            return label
        return uri.rsplit("/", 1)[-1].replace("-", " ")

    uri2label = {uri: fallback_label(uri) for uri in uris}
    uri2text = {}
    for uri in uris:
        parts = [uri2label[uri], *alt_map.get(uri, [])]
        if def_map.get(uri):
            parts.append(def_map[uri] or "")
        uri2text[uri] = ". ".join(part for part in parts if part)

    rows = []
    for parent, child in edges:
        rows.append(
            {
                "parent_uri": parent,
                "parent_label": uri2label[parent],
                "child_uri": child,
                "child_label": uri2label[child],
                "child_def": def_map.get(child),
            }
        )
    gold = pd.DataFrame(rows).drop_duplicates()

    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(edges)
    ancestors = {node: nx.ancestors(nx_graph, node) for node in nx_graph.nodes()}
    descendants = {node: nx.descendants(nx_graph, node) for node in nx_graph.nodes()}
    parent_children = {
        parent: gold.loc[gold["parent_uri"] == parent, "child_uri"].drop_duplicates().tolist()
        for parent in gold["parent_uri"].drop_duplicates().tolist()
    }
    depths = compute_depths(nx_graph)

    all_uris = sorted(uris)
    all_texts = [normalize_surface_text(uri2text[uri])[1] or uri2text[uri] for uri in all_uris]
    all_text_embeddings = embedder.encode(all_texts)

    label_texts = [normalize_surface_text(uri2label[uri])[1] or uri2label[uri] for uri in all_uris]
    label_embeddings_matrix = embedder.encode(label_texts)
    label_embeddings = {
        uri: label_embeddings_matrix[index] for index, uri in enumerate(all_uris)
    }

    return OntologyResources(
        graph=nx_graph,
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


def compute_depths(graph: nx.DiGraph) -> dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    roots = [node for node, indegree in graph.in_degree() if indegree == 0]
    depths: dict[str, int] = {}
    try:
        for node in nx.topological_sort(graph):
            parents = list(graph.predecessors(node))
            if not parents:
                depths[node] = 0
            else:
                depths[node] = min(depths[parent] + 1 for parent in parents)
        return depths
    except nx.NetworkXUnfeasible:  # pragma: no cover - ontology is expected to be a DAG
        if not roots:
            return {node: 0 for node in graph.nodes()}
        bfs = nx.multi_source_dijkstra_path_length(graph.to_undirected(), roots)
        return {node: int(bfs.get(node, 0)) for node in graph.nodes()}


class JournalExtensionPipeline:
    def __init__(
        self,
        resources: OntologyResources,
        embedder: TextEmbedder,
        config: JournalExtensionConfig | None = None,
    ) -> None:
        self.resources = resources
        self.embedder = embedder
        self.config = config or JournalExtensionConfig()
        self._negative_regexes = [
            re.compile(pattern, flags=re.IGNORECASE) for pattern in self.config.negative_patterns
        ]
        self._descendant_cache: dict[str, dict[str, Any]] = {}

    def build_zero_prompt(self, parent_label: str) -> str:
        m = self.config.candidate_pool_size
        return (
            f'Compétence: "{parent_label}".\n'
            f"Génère exactement {m} sous-compétences candidates fines en français, "
            "une par ligne.\n"
            "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
            "un seul niveau plus spécifique, sans doublons, sans outils, logiciels, plateformes "
            "ou métiers."
        )

    def build_few_prompt(self, parent_label: str, examples: Sequence[tuple[str, Sequence[str]]]) -> str:
        example_block = "\n\n".join(
            f'Exemple — "{parent}" → ' + "; ".join(children)
            for parent, children in list(examples)[: self.config.fewshot_examples]
        )
        m = self.config.candidate_pool_size
        return (
            f"{example_block}\n\n"
            f'Compétence: "{parent_label}".\n'
            f"Génère exactement {m} sous-compétences candidates fines en français, "
            "une par ligne, avec la même granularité que les exemples.\n"
            "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
            "un seul niveau plus spécifique, sans doublons, sans outils, logiciels, plateformes "
            "ou métiers."
        )

    def build_rag_prompt(self, parent_label: str, context_items: Sequence[str]) -> str:
        context = "\n".join(f"- {item}" for item in context_items[: self.config.rag_context_items])
        m = self.config.candidate_pool_size
        return (
            "Contexte (éléments liés d'une ontologie, sans les vrais enfants):\n"
            f"{context}\n\n"
            f'Compétence: "{parent_label}".\n'
            f"Génère exactement {m} sous-compétences candidates fines en français, "
            "une par ligne.\n"
            "Contraintes: groupes nominaux courts, directement liés à la compétence parente, "
            "un seul niveau plus spécifique, sans doublons, sans outils, logiciels, plateformes "
            "ou métiers."
        )

    def nearest_neighbor_context(
        self,
        parent_skill_id: str,
        k: int | None = None,
        exclude_uris: Iterable[str] | None = None,
    ) -> list[str]:
        limit = k if k is not None else self.config.rag_context_items
        exclude = set(exclude_uris or ())
        query_text = self.resources.uri2text.get(parent_skill_id, self.resources.uri2label[parent_skill_id])
        query_key = normalize_surface_text(query_text)[1] or query_text
        query_embedding = self.embedder.encode([query_key])[0]
        scores = cosine_similarity_matrix(query_embedding[None, :], self.resources.all_text_embeddings)[0]
        items: list[str] = []
        for index in np.argsort(-scores):
            uri = self.resources.all_uris[int(index)]
            if uri in exclude:
                continue
            label = self.resources.uri2label.get(uri, uri)
            if not label:
                continue
            items.append(label)
            if len(items) >= limit:
                break
        return items

    def run_parent(
        self,
        parent_skill_id: str,
        prompt_mode: str,
        raw_candidates: Sequence[str],
        gold_children: Sequence[str] | None = None,
        context_texts: Sequence[str] | None = None,
        variants: Sequence[VariantSpec] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        parent_label = self.resources.uri2label.get(parent_skill_id, parent_skill_id)
        selected_gold_children = list(gold_children) if gold_children is not None else list(
            self.resources.parent_children.get(parent_skill_id, [])
        )
        base_records = self._initialize_candidate_records(parent_skill_id, prompt_mode, raw_candidates)
        candidate_texts = [
            record["comparison_key"] or record["normalized_text"] for record in base_records
        ]
        candidate_embeddings = (
            self.embedder.encode(candidate_texts) if candidate_texts else np.zeros((0, 0), dtype=np.float32)
        )

        self._align_candidates(parent_skill_id, base_records, candidate_embeddings)
        self._mark_duplicates(base_records, candidate_embeddings)
        self._validate_candidates(parent_skill_id, parent_label, base_records)
        self._classify_novelty(parent_skill_id, parent_label, base_records, candidate_embeddings)
        self._assign_selection_buckets(parent_skill_id, base_records)

        all_candidate_rows: list[dict[str, Any]] = []
        all_parent_rows: list[dict[str, Any]] = []
        for variant in variants or DEFAULT_VARIANTS:
            candidate_rows, parent_row = self._run_variant(
                parent_skill_id=parent_skill_id,
                parent_label=parent_label,
                prompt_mode=prompt_mode,
                gold_children=selected_gold_children,
                base_records=base_records,
                candidate_embeddings=candidate_embeddings,
                context_texts=context_texts or (),
                variant=variant,
            )
            all_candidate_rows.extend(candidate_rows)
            all_parent_rows.append(parent_row)
        return all_candidate_rows, all_parent_rows

    def summarize_parent_results(self, parent_results: pd.DataFrame) -> pd.DataFrame:
        if parent_results.empty:
            return parent_results.copy()
        grouped = parent_results.groupby(["prompt_mode", "variant"], dropna=False)
        return (
            grouped.agg(
                parent_count=("parent_skill_id", "nunique"),
                semantic_f1_mean=("semantic_f1", "mean"),
                hier_f1_mean=("hier_f1", "mean"),
                duplicate_count_mean=("duplicate_count", "mean"),
                invalid_count_mean=("invalid_count", "mean"),
                unaligned_count_mean=("unaligned_count", "mean"),
                plausible_novel_count_mean=("plausible_novel_count", "mean"),
                hallucination_count_mean=("hallucination_count", "mean"),
                selected_count_mean=("selected_count", "mean"),
            )
            .reset_index()
            .sort_values(["semantic_f1_mean", "hier_f1_mean"], ascending=False)
        )

    def summarize_candidate_results(self, candidate_results: pd.DataFrame) -> pd.DataFrame:
        if candidate_results.empty:
            return candidate_results.copy()
        grouped = candidate_results.groupby(["prompt_mode", "variant"], dropna=False)
        return (
            grouped.agg(
                candidate_count=("candidate_id", "count"),
                duplicate_rate=("is_duplicate", "mean"),
                parent_repeat_rate=("viol_parent_repeat", "mean"),
                type_violation_rate=("viol_type", "mean"),
                unaligned_rate=("viol_unaligned", "mean"),
                depth_violation_rate=("viol_depth", "mean"),
                valid_rate=("is_valid", "mean"),
                soft_valid_rate=("is_soft_valid", "mean"),
                selected_rate=("selected_final", "mean"),
            )
            .reset_index()
        )

    def _initialize_candidate_records(
        self,
        parent_skill_id: str,
        prompt_mode: str,
        raw_candidates: Sequence[str],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        parent_tail = _safe_name(parent_skill_id.rsplit("/", 1)[-1])
        for rank, raw_text in enumerate(list(raw_candidates)[: self.config.candidate_pool_size], start=1):
            normalized_text, comparison_key = normalize_surface_text(str(raw_text))
            if not normalized_text:
                continue
            records.append(
                {
                    "candidate_id": f"{parent_tail}__{prompt_mode}__{rank:02d}",
                    "parent_skill_id": parent_skill_id,
                    "prompt_mode": prompt_mode,
                    "raw_text": str(raw_text).strip(),
                    "normalized_text": normalized_text,
                    "comparison_key": comparison_key,
                    "generation_rank": rank,
                    "is_duplicate": False,
                    "duplicate_of": "",
                    "duplicate_score": 0.0,
                    "aligned_node_id": "",
                    "aligned_label": "",
                    "alignment_score": 0.0,
                    "embedding_score": 0.0,
                    "lexical_score": 0.0,
                    "is_aligned": False,
                    "viol_parent_repeat": False,
                    "viol_duplicate": False,
                    "viol_type": False,
                    "viol_unaligned": False,
                    "viol_depth": False,
                    "is_soft_valid": False,
                    "is_valid": False,
                    "hard_violations": "",
                    "soft_violations": "",
                    "selection_bucket": "",
                    "selection_priority": np.nan,
                    "utility_score": np.nan,
                    "rank_after_rerank": np.nan,
                    "selected_final": False,
                    "novelty_status": "",
                    "novelty_score": 0.0,
                }
            )
        return records

    def _descendant_index(self, parent_skill_id: str) -> dict[str, Any]:
        cached = self._descendant_cache.get(parent_skill_id)
        if cached is not None:
            return cached
        descendant_uris = sorted(self.resources.descendants.get(parent_skill_id, set()))
        child_uris = set(self.resources.parent_children.get(parent_skill_id, []))
        label_matrix = (
            np.vstack([self.resources.label_embeddings[uri] for uri in descendant_uris])
            if descendant_uris
            else np.zeros((0, 0), dtype=np.float32)
        )
        lexical_labels = {
            uri: [self.resources.uri2label.get(uri, uri), *self.resources.alt_map.get(uri, [])]
            for uri in descendant_uris
        }
        payload = {
            "descendant_uris": descendant_uris,
            "label_matrix": label_matrix,
            "lexical_labels": lexical_labels,
            "child_uris": child_uris,
        }
        self._descendant_cache[parent_skill_id] = payload
        return payload

    def _align_candidates(
        self,
        parent_skill_id: str,
        records: list[dict[str, Any]],
        candidate_embeddings: np.ndarray,
    ) -> None:
        index = self._descendant_index(parent_skill_id)
        descendant_uris = index["descendant_uris"]
        if not descendant_uris:
            return
        label_matrix = index["label_matrix"]
        sim_matrix = cosine_similarity_matrix(candidate_embeddings, label_matrix)
        for row_idx, record in enumerate(records):
            best_uri = ""
            best_mix = -1.0
            best_embedding = 0.0
            best_lexical = 0.0
            for col_idx, uri in enumerate(descendant_uris):
                embedding_score = float(sim_matrix[row_idx, col_idx])
                lexical_score = max(
                    lexical_similarity(record["comparison_key"], label)
                    for label in index["lexical_labels"][uri]
                    if label
                )
                mix_score = (
                    self.config.lexical_weight * lexical_score
                    + (1.0 - self.config.lexical_weight) * embedding_score
                )
                if mix_score > best_mix:
                    best_uri = uri
                    best_mix = mix_score
                    best_embedding = embedding_score
                    best_lexical = lexical_score
            if best_uri:
                record["aligned_node_id"] = best_uri
                record["aligned_label"] = self.resources.uri2label.get(best_uri, best_uri)
                record["alignment_score"] = float(best_mix)
                record["embedding_score"] = float(best_embedding)
                record["lexical_score"] = float(best_lexical)
                record["is_aligned"] = best_mix >= self.config.alignment_threshold

    def _mark_duplicates(
        self,
        records: list[dict[str, Any]],
        candidate_embeddings: np.ndarray,
    ) -> None:
        if len(records) < 2:
            return
        sim_matrix = cosine_similarity_matrix(candidate_embeddings, candidate_embeddings)
        for left_idx in range(len(records)):
            if records[left_idx]["is_duplicate"]:
                continue
            for right_idx in range(left_idx + 1, len(records)):
                if records[right_idx]["is_duplicate"]:
                    continue
                similarity = float(sim_matrix[left_idx, right_idx])
                if similarity < self.config.duplicate_threshold:
                    continue
                keep_idx, drop_idx = self._choose_duplicate_keep(records, left_idx, right_idx)
                records[drop_idx]["is_duplicate"] = True
                records[drop_idx]["duplicate_of"] = records[keep_idx]["candidate_id"]
                records[drop_idx]["duplicate_score"] = similarity

    def _choose_duplicate_keep(
        self,
        records: list[dict[str, Any]],
        left_idx: int,
        right_idx: int,
    ) -> tuple[int, int]:
        left = records[left_idx]
        right = records[right_idx]
        left_score = float(left.get("alignment_score") or 0.0)
        right_score = float(right.get("alignment_score") or 0.0)
        if left_score > right_score:
            return left_idx, right_idx
        if right_score > left_score:
            return right_idx, left_idx
        if int(left["generation_rank"]) <= int(right["generation_rank"]):
            return left_idx, right_idx
        return right_idx, left_idx

    def _validate_candidates(
        self,
        parent_skill_id: str,
        parent_label: str,
        records: list[dict[str, Any]],
    ) -> None:
        _, parent_key = normalize_surface_text(parent_label)
        parent_depth = self.resources.depths.get(parent_skill_id, 0)
        for record in records:
            hard_violations: list[str] = []
            soft_violations: list[str] = []

            record["viol_parent_repeat"] = record["comparison_key"] == parent_key
            record["viol_duplicate"] = bool(record["is_duplicate"])
            record["viol_type"] = self._violates_type(record["normalized_text"])
            record["viol_unaligned"] = not bool(record["is_aligned"])

            depth_gap = None
            aligned_node = record.get("aligned_node_id")
            if aligned_node:
                aligned_depth = self.resources.depths.get(aligned_node, parent_depth)
                depth_gap = aligned_depth - parent_depth
                if depth_gap == 1:
                    record["viol_depth"] = False
                elif 1 < depth_gap <= 1 + self.config.depth_tolerance:
                    record["viol_depth"] = False
                    soft_violations.append("deep_descendant")
                else:
                    record["viol_depth"] = True
            else:
                record["viol_depth"] = False

            if record["viol_parent_repeat"]:
                hard_violations.append("parent_repeat")
            if record["viol_duplicate"]:
                hard_violations.append("duplicate")
            if record["viol_type"]:
                hard_violations.append("type")
            if record["viol_unaligned"]:
                soft_violations.append("unaligned")
            if record["viol_depth"]:
                soft_violations.append("depth")
            if depth_gap and depth_gap > 1:
                soft_violations.append(f"depth_gap_{depth_gap}")

            record["hard_violations"] = ";".join(hard_violations)
            record["soft_violations"] = ";".join(sorted(set(soft_violations)))
            record["is_soft_valid"] = not hard_violations
            record["is_valid"] = not hard_violations and not record["viol_unaligned"] and not record["viol_depth"]

    def _classify_novelty(
        self,
        parent_skill_id: str,
        parent_label: str,
        records: list[dict[str, Any]],
        candidate_embeddings: np.ndarray,
    ) -> None:
        parent_key = normalize_surface_text(parent_label)[1] or parent_label
        parent_embedding = self.embedder.encode([parent_key])[0]
        for idx, record in enumerate(records):
            if record["is_aligned"] or record["is_duplicate"]:
                continue
            parent_relevance = float(
                cosine_similarity_matrix(candidate_embeddings[idx][None, :], parent_embedding[None, :])[0, 0]
            )
            local_score = float(record.get("alignment_score") or 0.0)
            novelty_score = max(0.0, 0.55 * parent_relevance + 0.45 * local_score)
            record["novelty_score"] = novelty_score
            if (
                not record["viol_type"]
                and not record["viol_parent_repeat"]
                and novelty_score >= self.config.novelty_threshold
            ):
                record["novelty_status"] = "plausible_novel_skill"
            else:
                record["novelty_status"] = "hallucination"

    def _assign_selection_buckets(
        self,
        parent_skill_id: str,
        records: list[dict[str, Any]],
    ) -> None:
        child_uris = set(self.resources.parent_children.get(parent_skill_id, []))
        priority_map = {
            "exact_child": 0,
            "aligned_descendant": 1,
            "plausible_novel": 2,
            "weak_unresolved": 3,
            "hard_invalid": 4,
        }
        for record in records:
            if not record["is_soft_valid"]:
                bucket = "hard_invalid"
            elif record["is_aligned"] and record.get("aligned_node_id") in child_uris:
                bucket = "exact_child"
            elif record["is_aligned"]:
                bucket = "aligned_descendant"
            elif record.get("novelty_status") == "plausible_novel_skill":
                bucket = "plausible_novel"
            else:
                bucket = "weak_unresolved"
            record["selection_bucket"] = bucket
            record["selection_priority"] = priority_map[bucket]

    def _violates_type(self, text: str) -> bool:
        return any(regex.search(text) for regex in self._negative_regexes)

    def _run_variant(
        self,
        parent_skill_id: str,
        parent_label: str,
        prompt_mode: str,
        gold_children: Sequence[str],
        base_records: list[dict[str, Any]],
        candidate_embeddings: np.ndarray,
        context_texts: Sequence[str],
        variant: VariantSpec,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        records = [dict(record) for record in base_records]
        for record in records:
            record["variant"] = variant.name
            record["selected_final"] = False
            if not variant.use_rerank:
                record["utility_score"] = np.nan
                record["rank_after_rerank"] = np.nan

        eligible_indices = self._eligible_indices(records, variant)

        if variant.use_rerank:
            ordered_indices, utilities = self._rerank_candidates(
                parent_skill_id=parent_skill_id,
                records=records,
                candidate_embeddings=candidate_embeddings,
                eligible_indices=eligible_indices,
                context_texts=context_texts,
            )
            for rank, index in enumerate(ordered_indices, start=1):
                records[index]["rank_after_rerank"] = rank
                records[index]["utility_score"] = utilities.get(index, np.nan)
        else:
            ordered_indices = self._default_order(records, eligible_indices, variant)

        selected_indices = self._select_final_indices(records, ordered_indices, variant)
        for index in selected_indices:
            records[index]["selected_final"] = True

        selected_texts = [records[index]["raw_text"] for index in selected_indices]
        selected_uris = [
            records[index]["aligned_node_id"]
            for index in selected_indices
            if records[index]["is_aligned"] and records[index]["aligned_node_id"]
        ]

        sem_precision, sem_recall, sem_f1 = self._semantic_eval(selected_texts, gold_children)
        hier_precision, hier_recall, hier_f1 = self._hier_partial_scores(parent_skill_id, selected_uris, gold_children)

        parent_row = {
            "parent_skill_id": parent_skill_id,
            "parent_label": parent_label,
            "prompt_mode": prompt_mode,
            "variant": variant.name,
            "candidate_pool_size": len(records),
            "final_k": self.config.final_k,
            "selected_count": len(selected_indices),
            "selected_candidates": json.dumps(selected_texts, ensure_ascii=False),
            "duplicate_count": int(sum(1 for record in records if record["is_duplicate"])),
            "invalid_count": int(sum(1 for record in records if not record["is_valid"])),
            "unaligned_count": int(sum(1 for record in records if not record["is_aligned"])),
            "plausible_novel_count": int(
                sum(1 for record in records if record["novelty_status"] == "plausible_novel_skill")
            ),
            "hallucination_count": int(
                sum(1 for record in records if record["novelty_status"] == "hallucination")
            ),
            "semantic_precision": sem_precision,
            "semantic_recall": sem_recall,
            "semantic_f1": sem_f1,
            "hier_precision": hier_precision,
            "hier_recall": hier_recall,
            "hier_f1": hier_f1,
        }
        return records, parent_row

    def _rerank_candidates(
        self,
        parent_skill_id: str,
        records: list[dict[str, Any]],
        candidate_embeddings: np.ndarray,
        eligible_indices: Sequence[int],
        context_texts: Sequence[str],
    ) -> tuple[list[int], dict[int, float]]:
        if not eligible_indices:
            return [], {}

        context_embeddings = (
            self.embedder.encode(
                [normalize_surface_text(text)[1] or text for text in context_texts if text]
            )
            if context_texts
            else np.zeros((0, 0), dtype=np.float32)
        )
        child_uris = set(self.resources.parent_children.get(parent_skill_id, []))
        selected: list[int] = []
        utilities: dict[int, float] = {}
        remaining = list(eligible_indices)
        while remaining:
            best_idx = remaining[0]
            best_score = -math.inf
            for index in remaining:
                alignment_term = float(records[index].get("alignment_score") or 0.0)
                hierarchy_term = self._hierarchy_credit(records[index], child_uris)
                context_term = self._context_consistency(candidate_embeddings[index], context_embeddings)
                lexical_term = float(records[index].get("lexical_score") or 0.0)
                generation_term = 1.0 / max(1, int(records[index]["generation_rank"]))
                novelty_term = (
                    float(records[index].get("novelty_score") or 0.0)
                    if records[index].get("selection_bucket") == "plausible_novel"
                    else 0.0
                )
                weak_penalty = (
                    self.config.rerank_weak_unresolved_penalty
                    if records[index].get("selection_bucket") == "weak_unresolved"
                    else 0.0
                )
                redundancy_term = self._redundancy_penalty(index, selected, candidate_embeddings)
                utility = (
                    self.config.rerank_alignment_weight * alignment_term
                    + self.config.rerank_hierarchy_weight * hierarchy_term
                    + self.config.rerank_context_weight * context_term
                    + self.config.rerank_lexical_weight * lexical_term
                    + self.config.rerank_generation_weight * generation_term
                    + self.config.rerank_novelty_weight * novelty_term
                    - self.config.rerank_redundancy_weight * redundancy_term
                    - weak_penalty
                )
                if utility > best_score:
                    best_idx = index
                    best_score = utility
                elif math.isclose(utility, best_score, rel_tol=1e-9, abs_tol=1e-9):
                    current = records[index]
                    incumbent = records[best_idx]
                    current_key = (
                        -float(current.get("alignment_score") or 0.0),
                        int(current["generation_rank"]),
                    )
                    incumbent_key = (
                        -float(incumbent.get("alignment_score") or 0.0),
                        int(incumbent["generation_rank"]),
                    )
                    if current_key < incumbent_key:
                        best_idx = index
                        best_score = utility
            selected.append(best_idx)
            utilities[best_idx] = best_score
            remaining.remove(best_idx)
        return selected, utilities

    def _hierarchy_credit(self, record: dict[str, Any], child_uris: set[str]) -> float:
        aligned_node_id = record.get("aligned_node_id") or ""
        if not aligned_node_id:
            if record.get("selection_bucket") == "plausible_novel":
                return 0.2
            return 0.0
        if aligned_node_id in child_uris:
            return 1.0
        if record.get("is_aligned"):
            return 0.5
        return 0.0

    def _eligible_indices(
        self,
        records: list[dict[str, Any]],
        variant: VariantSpec,
    ) -> list[int]:
        eligible_indices: list[int] = []
        for index, record in enumerate(records):
            if variant.validation_mode == "hard":
                if record["is_valid"]:
                    eligible_indices.append(index)
            elif variant.validation_mode == "soft":
                if record["is_soft_valid"]:
                    eligible_indices.append(index)
            else:
                if not record["is_duplicate"]:
                    eligible_indices.append(index)
        return eligible_indices

    def _default_order(
        self,
        records: list[dict[str, Any]],
        eligible_indices: Sequence[int],
        variant: VariantSpec,
    ) -> list[int]:
        if variant.validation_mode == "soft":
            return sorted(
                eligible_indices,
                key=lambda idx: (
                    float(records[idx].get("selection_priority") or math.inf),
                    int(records[idx]["generation_rank"]),
                    -float(records[idx].get("alignment_score") or 0.0),
                ),
            )
        return sorted(
            eligible_indices,
            key=lambda idx: (
                int(records[idx]["generation_rank"]),
                -float(records[idx].get("alignment_score") or 0.0),
            ),
        )

    def _select_final_indices(
        self,
        records: list[dict[str, Any]],
        ordered_indices: Sequence[int],
        variant: VariantSpec,
    ) -> list[int]:
        if variant.validation_mode != "soft":
            return list(ordered_indices[: self.config.final_k])

        selected: list[int] = []
        seen: set[int] = set()
        plausible_limit = max(0, int(self.config.max_plausible_novel_in_final))
        plausible_used = 0
        bucket_order = ("exact_child", "aligned_descendant", "plausible_novel", "weak_unresolved")

        for bucket in bucket_order:
            for index in ordered_indices:
                if index in seen or records[index].get("selection_bucket") != bucket:
                    continue
                if bucket == "plausible_novel" and plausible_used >= plausible_limit:
                    continue
                selected.append(index)
                seen.add(index)
                if bucket == "plausible_novel":
                    plausible_used += 1
                if len(selected) >= self.config.final_k:
                    return selected

        for index in ordered_indices:
            if index in seen:
                continue
            bucket = records[index].get("selection_bucket")
            if bucket == "hard_invalid":
                continue
            if bucket == "plausible_novel" and plausible_used >= plausible_limit:
                continue
            selected.append(index)
            seen.add(index)
            if bucket == "plausible_novel":
                plausible_used += 1
            if len(selected) >= self.config.final_k:
                break
        return selected

    def _context_consistency(self, candidate_embedding: np.ndarray, context_embeddings: np.ndarray) -> float:
        if context_embeddings.size == 0:
            return 0.0
        return float(cosine_similarity_matrix(candidate_embedding[None, :], context_embeddings).max())

    def _redundancy_penalty(
        self,
        candidate_idx: int,
        selected_indices: Sequence[int],
        candidate_embeddings: np.ndarray,
    ) -> float:
        if not selected_indices:
            return 0.0
        selected_matrix = np.vstack([candidate_embeddings[index] for index in selected_indices])
        return float(
            cosine_similarity_matrix(candidate_embeddings[candidate_idx][None, :], selected_matrix).max()
        )

    def _semantic_eval(
        self,
        pred_texts: Sequence[str],
        gold_uris: Sequence[str],
    ) -> tuple[float, float, float]:
        pred_texts = [text for text in pred_texts if text and str(text).strip()]
        if not pred_texts or not gold_uris:
            return 0.0, 0.0, 0.0
        pred_matrix = self.embedder.encode(
            [normalize_surface_text(text)[1] or text for text in pred_texts]
        )
        gold_matrix = np.vstack(
            [self.resources.label_embeddings[uri] for uri in gold_uris if uri in self.resources.label_embeddings]
        )
        if gold_matrix.size == 0:
            return 0.0, 0.0, 0.0
        sim_matrix = cosine_similarity_matrix(pred_matrix, gold_matrix)
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(1.0 - sim_matrix)
            total_similarity = float(sim_matrix[rows, cols].sum())
        else:  # pragma: no cover - normal path uses scipy
            total_similarity = self._greedy_sum(sim_matrix)
        precision = total_similarity / max(1, len(pred_texts))
        recall = total_similarity / max(1, gold_matrix.shape[0])
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return float(precision), float(recall), float(f1)

    def _hier_partial_scores(
        self,
        parent_skill_id: str,
        pred_uris: Sequence[str],
        gold_uris: Sequence[str],
    ) -> tuple[float, float, float]:
        unique_preds = list(dict.fromkeys(pred_uris))
        gold_set = set(gold_uris)
        if not unique_preds or not gold_set:
            return 0.0, 0.0, 0.0
        scores = []
        for uri in unique_preds:
            if uri in gold_set:
                scores.append(1.0)
                continue
            if any(
                uri in self.resources.ancestors.get(gold_uri, set())
                or uri in self.resources.descendants.get(gold_uri, set())
                for gold_uri in gold_set
            ):
                scores.append(0.5)
            else:
                scores.append(0.0)
        precision = sum(scores) / max(1, len(unique_preds))
        recall = min(sum(sorted(scores, reverse=True)[: len(gold_set)]), len(gold_set)) / max(1, len(gold_set))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return float(precision), float(recall), float(f1)

    @staticmethod
    def _greedy_sum(matrix: np.ndarray) -> float:
        if matrix.size == 0:
            return 0.0
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        total = 0.0
        flat_indices = np.dstack(np.unravel_index(np.argsort(-matrix.ravel()), matrix.shape))[0]
        for row, col in flat_indices:
            row = int(row)
            col = int(col)
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            total += float(matrix[row, col])
            if len(used_rows) == min(matrix.shape):
                break
        return total
