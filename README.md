# LOSD Public Release

This directory is the public implementation of **LOSD** (LLM- and
Ontology-Guided Skill Decomposition). It includes the LOSD post-generation
pipeline, the repeated OpenRouter generation protocol, the paper-specific
repeated analysis, tests, and small sample results.

The repository intentionally excludes API keys, virtual environments, model
response caches, embedding caches, logs, and bulk experiment directories.

## Main files

- `losd.py`: ontology loading, candidate parsing, validation, reranking, and evaluation.
- `run_losd.py`: apply LOSD variants to an existing generation cache.
- `run_repetitions.py`: model-agnostic three-run OpenRouter generator.
- `run_*_repetitions.py`: model/provider defaults used by the repeated study.
- `analyze_repeated_experiments.py`: primary repeated, paired, and complete-case analysis.
- `analyze_visible_partial_sensitivity.py`: visible-output sensitivity analysis for incomplete responses.
- `plot_losd_results.py`: generate plots from LOSD result tables.
- `tests/` and `test_losd_unittest.py`: regression tests.

## Setup on Ubuntu or WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-transformer.txt
```

The repeated generator calls OpenRouter. Export the key only in the shell; do
not place it in source files or commit it:

```bash
export OPENROUTER_API_KEY='your-key-here'
```

## Smoke test the generation protocol

This command makes one paid API call for one parent:

```bash
python -u run_deepseek_repetitions.py \
  --runs 1 \
  --methods zero \
  --parent-limit 1 \
  --max-api-calls 1 \
  --output-root exp_repeated_smoke \
  --resume
```

Use `--dry-run` to build and validate the experiment plan without calling the
API. Resume is configuration- and prompt-hash guarded: an existing cache is
reused only when its recorded task identity matches the current task.

## Repeated model runs

The shared defaults are three repetitions, seeds 1001--1003, zero-shot,
few-shot and RAG, 12 requested candidates, two few-shot examples, and 30 RAG
items. The following launchers set the exact model/provider defaults used in
the study:

```bash
python -u run_deepseek_repetitions.py --resume
python -u run_gpt_repetitions.py --resume
python -u run_gpt_oss_repetitions.py --resume
python -u run_kimi_repetitions.py --resume
python -u run_llama4_repetitions.py --resume
python -u run_mistral_large_repetitions.py --resume
python -u run_qwen3_repetitions.py --resume
```

Every injected launcher default can be overridden on the command line. Model
and provider availability can change; retain the resolved model, provider,
prompt hash, seed support, token limits, and decoding fields stored in each
run's metadata when conducting a new study.

## Reproduce the repeated analysis

The primary analysis expects the seven experiment-directory names created by
the launchers above and the included `esco_cmo_binding.ttl` file:

```bash
python -u analyze_repeated_experiments.py \
  --root . \
  --ttl esco_cmo_binding.ttl \
  --output-dir repeated_analysis \
  --embedding-cache embedding_cache.pkl \
  --embedding-batch-size 128
```

The analysis reparses visible response text uniformly, checks cache identity,
computes the four LOSD variants, restricts primary inference to parents with
all three repetitions within each model--prompt condition, and writes the
run-level summaries and parent-paired comparisons. Hidden reasoning text is
never used as candidate output.

After a completed primary analysis, the visible-partial sensitivity check is:

```bash
python -u analyze_visible_partial_sensitivity.py \
  --root . \
  --analysis-dir repeated_analysis
```

The analysis pins the multilingual MPNet embedding revision and always records
the observed input-corpus hash. With the archived paper caches, additionally
use the following option to assert the exact paper corpus:

```bash
--expected-embedding-key-sha256 65f9144faff284caebca2702951dbb053b60ca352b63deee8e541394bbed18c5
```

Do not use that assertion for fresh API generations, whose candidate text will
legitimately differ. If prompts, models, or parents change, use a new output
directory and provenance record.

## Apply LOSD to an existing cache

```bash
python run_losd.py --source-outdir exp_out_gpt_5
```

If model caches are not distributed, the included summaries can still be
inspected under `sample_results/losd/` and plotted with:

```bash
python plot_losd_results.py --results-dir sample_results/losd
```

## Tests

```bash
python -m pytest tests test_losd_unittest.py -q
```

Before publishing, review [PUBLIC_RELEASE_NOTES.md](./PUBLIC_RELEASE_NOTES.md).
