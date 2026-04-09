# Journal Extension Implementation Specification

## Title
**Bridging the Granularity Gap in Skill Ontologies: A Neuro-Symbolic Framework for Skill Decomposition with LLMs**

## 1. Objective

Extend the current skill decomposition pipeline based on **zero-shot (ZS)**, **few-shot (FS)**, and **RAG** prompting with a **post-generation neuro-symbolic refinement layer** suitable for a journal paper.

The journal extension should add non-trivial new material through:

- over-generation of candidate sub-skills
- ontology-guided validation
- structure-aware reranking
- novelty analysis of unresolved outputs

The goal is to improve the validity, granularity, and structural coherence of generated sub-skills while keeping the task formulation consistent with the current work.

---

## 2. Current baseline pipeline

The current implementation already supports:

1. parent skill input
2. prompt construction for ZS / FS / RAG
3. candidate generation by the LLM
4. normalization and lightweight deduplication
5. ontology alignment
6. evaluation with semantic F1 and hierarchy-aware F1

The journal paper should **extend** this pipeline rather than replace it.

Current pipeline:

```text
Prompt -> Generate -> Normalize -> Align -> Evaluate
```

Proposed journal pipeline:

```text
Prompt -> Over-generate -> Normalize -> Align -> Validate -> Rerank -> Select top-k -> Evaluate
```

---

## 3. Module A: Over-generation

### Purpose
Generate more than `k` candidates so the system can filter and rank them before selecting the final top-k decomposition.

### Inputs
- `parent_skill`
- `prompt_mode` in {ZS, FS, RAG}
- `k` = final number of sub-skills
- `m` = number of generated candidates, with `m > k`

### Recommended defaults
- `k = 5` or `10`
- `m = 12` or `15`

### Prompt modification
Replace:

> generate exactly `k` fine-grained sub-skills

with:

> generate exactly `m` candidate fine-grained sub-skills

while keeping the same constraints:
- short noun phrases
- no tools / software / occupations
- directly relevant to the parent
- one level more specific than the parent
- no duplicates

### Output fields
- `candidate_id`
- `parent_skill_id`
- `prompt_mode`
- `raw_text`
- `generation_rank`

---

## 4. Module B: Normalization

### Purpose
Canonicalize generated candidates before alignment and symbolic checks.

### Processing steps
- lowercase
- trim whitespace
- normalize punctuation
- normalize hyphenation
- remove empty items
- preserve the original surface form for display

### Output fields
- `raw_text`
- `normalized_text`
- `comparison_key`

---

## 5. Module C: Duplicate / paraphrase filtering

### Purpose
Remove semantically redundant generated candidates.

### Inputs
- normalized candidates
- sentence embeddings from the same multilingual Sentence-BERT model used in evaluation

### Rule
For each pair of candidates `(s_i, s_j)`:
- compute cosine similarity
- if similarity >= `rho`, mark one candidate as duplicate

### Recommended default
- `rho = 0.90` (to be tuned if needed)

### Keep rule
Prefer:
1. the candidate with the better alignment score if already aligned
2. otherwise the one with the earlier generation rank

### Output fields
- `is_duplicate`
- `duplicate_of`
- `duplicate_score`

---

## 6. Module D: Ontology alignment

### Purpose
Map free-text generated sub-skills to ontology descendants of the parent skill.

### Inputs
- candidate text
- ontology descendant set of the parent skill
- `prefLabel` and `altLabel`
- multilingual embedding model

### Alignment function
For a candidate `s`, compute:

```math
alpha(s) = argmax_{v in Desc(S_broad)} sim(s, v)
```

### Practical similarity
Use a mixed lexical-semantic score:

```math
sim(s, v) = lambda * lex(s, v) + (1 - lambda) * cos(E(s), E(label(v)))
```

where:
- `lex(s, v)` = lexical match score
- `E(.)` = multilingual Sentence-BERT embedding
- `lambda` is a weighting coefficient

### Acceptance rule
Accept the alignment only if:

```math
sim(s, alpha(s)) >= tau
```

Otherwise set the candidate as unresolved.

### Recommended defaults
- `tau` around the current threshold region already used in the project
- `lambda` tuned on a small development split

### Output fields
- `aligned_node_id`
- `aligned_label`
- `alignment_score`
- `embedding_score`
- `lexical_score`
- `is_aligned`

---

## 7. Module E: Ontology-guided validation

### Purpose
Transform conceptual task constraints into executable symbolic checks.

### Validation checks

#### E1. Parent repetition
Reject if the normalized candidate equals the normalized parent label.

Field:
- `viol_parent_repeat`

#### E2. Duplicate
Reject if the candidate is marked as duplicate.

Field:
- `viol_duplicate`

#### E3. Type conformity
Reject if the candidate denotes a:
- tool
- software
- platform
- occupation
- organization

instead of a skill.

### Implementation options
Use one or more of:
- ontology type metadata if available
- manually curated negative lexicon
- lightweight classifier
- rule-based lexical patterns

Field:
- `viol_type`

#### E4. Alignment validity
Reject if no ontology descendant alignment exceeds threshold `tau`.

Field:
- `viol_unaligned`

#### E5. Granularity validity
Reject or penalize candidates that are:
- too broad
- too deep
- outside the expected child granularity region

### Practical depth rule
Let:
- `depth_parent` = ontology depth of the parent
- `depth_aligned` = ontology depth of the aligned node

Preferred:
- exact child depth = `depth_parent + 1`
- acceptable deeper descendant = small tolerance below an expected child subtree

Field:
- `viol_depth`

### Validity function

```math
V(s_i) = 1 if no hard constraint is violated, else 0
```

### Output fields
- `is_valid`
- `hard_violations`
- `soft_violations`

---

## 8. Module F: Structure-aware reranking

### Purpose
Rank valid candidates and keep the final top-k list.

### Inputs
- valid candidates
- alignment scores
- hierarchy credit
- context consistency
- duplicate penalties

### Utility score
For each valid candidate:

```math
U(s_i) = lambda_1 * A_i + lambda_2 * H_i + lambda_3 * C_i - lambda_4 * D_i
```

where:
- `A_i` = alignment quality
- `H_i` = hierarchy credit
- `C_i` = context consistency with FS / RAG evidence
- `D_i` = redundancy penalty

### Components

#### Alignment term `A_i`
Normalized alignment score from Module D.

#### Hierarchy term `H_i`
Use:
- `1.0` for exact child
- `0.5` for deeper descendant under a gold child
- `0.0` otherwise

#### Context term `C_i`
- in FS: consistency with exemplar granularity and style
- in RAG: consistency with masked ontology evidence

#### Redundancy term `D_i`
Penalty for semantic overlap with already selected candidates.

### Final selection
- sort valid candidates by `U(s_i)`
- keep the top `k`

### Output fields
- `utility_score`
- `rank_after_rerank`
- `selected_final`

---

## 9. Module G: Novelty analysis of unresolved outputs

### Purpose
Analyze unaligned outputs instead of treating them only as failures.

### Classes
Each unresolved candidate should be assigned one of:
- `hallucination`
- `plausible_novel_skill`

### Suggested rule
A candidate can be labeled `plausible_novel_skill` if it is:
- type-valid
- relevant to the parent skill
- semantically close to the local ontology neighborhood
- non-duplicate
- unaligned to any existing descendant above threshold

Otherwise label it as `hallucination`.

### Output fields
- `novelty_status`
- `novelty_score`

---

## 10. End-to-end algorithm

For each parent skill:

1. build the prompt according to ZS / FS / RAG
2. generate `m` candidates
3. normalize candidates
4. filter duplicates
5. align to ontology descendants
6. validate symbolically
7. rerank valid candidates
8. select top `k`
9. assign novelty labels to unresolved candidates
10. compute evaluation metrics

---

## 11. Candidate record schema

```text
candidate_id
parent_skill_id
prompt_mode
raw_text
normalized_text
generation_rank
is_duplicate
duplicate_of
duplicate_score
aligned_node_id
aligned_label
alignment_score
embedding_score
lexical_score
is_aligned
viol_parent_repeat
viol_duplicate
viol_type
viol_unaligned
viol_depth
is_valid
utility_score
rank_after_rerank
selected_final
novelty_status
novelty_score
```

---

## 12. Parent-skill result schema

```text
parent_skill_id
parent_label
prompt_mode
candidate_pool_size
final_k
selected_candidates
duplicate_count
invalid_count
unaligned_count
plausible_novel_count
hallucination_count
semantic_f1
hier_f1
```

---

## 13. Experiments for the journal paper

### Experiment 1. Core ablation
Compare:
- ZS
- FS
- RAG
- RAG + validation
- RAG + validation + reranking

This is the main experiment showing the added value of the journal extension.

### Experiment 2. Error reduction analysis
Measure:
- duplicate rate
- parent repetition rate
- type-violation rate
- unaligned rate
- granularity-violation rate

### Experiment 3. Sensitivity analysis
Vary:
- alignment threshold `tau`
- duplicate threshold `rho`
- candidate pool size `m`
- number of RAG evidence items
- reranking weights

### Experiment 4. Difficulty-aware analysis
Break performance down by:
- branching factor
- ontology depth
- domain / sector
- lexical ambiguity

### Experiment 5. Novelty analysis
On unresolved outputs:
- proportion of hallucinations
- proportion of plausible novel skills
- optional manual review on a sample

---

## 14. Suggested new paper subsections

Add these subsections to the journal version:

### 3.4 Over-Generated Candidate Pool
Describe why the system generates `m > k` candidates.

### 3.5 Ontology-Guided Validation
Formalize symbolic checks based on task constraints.

### 3.6 Structure-Aware Candidate Reranking
Define the utility score and final top-k selection.

### 3.7 Novelty Analysis of Unaligned Candidates
Separate hallucinations from plausible enrichment candidates.

---

## 15. Minimal realistic implementation plan

### Phase 1
Implement:
- over-generation
- validation
- reranking

### Phase 2
Add:
- novelty labeling
- analysis logging

### Phase 3
Run:
- ablation experiments
- sensitivity study
- error analysis
- difficulty-aware analysis

---

## 16. Short paper-ready description

> We extend the original prompting-and-alignment pipeline with a post-generation neuro-symbolic refinement stage that over-generates candidate sub-skills, validates them using ontology-derived structural constraints, reranks valid candidates according to semantic and hierarchy-aware signals, and analyzes unresolved outputs as either hallucinations or plausible ontology-enrichment candidates.
