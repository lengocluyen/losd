# LOSD Public Release

This directory is a cleaned public-facing subset of the original project for **LOSD** (LLM- and Ontology-Guided Skill Decomposition).

It contains:

- the core LOSD refinement pipeline
- the command-line runner
- the plotting script
- tests
- the ontology file required by the runner
- a small sample result directory with summary CSVs and publication figures

It does **not** contain:

- local virtual environments
- cached model generations
- bulk experiment folders for every model
- notebooks and local exploratory artifacts

## Directory Layout

```text
github_public/
├── README.md
├── PUBLIC_RELEASE_NOTES.md
├── .gitignore
├── esco_cmo_binding.ttl
├── losd.py
├── run_losd.py
├── plot_losd_results.py
├── run_all_losd.sh
├── losd_specification.md
├── requirements.txt
├── requirements-transformer.txt
├── test_losd_unittest.py
├── tests/
│   └── test_losd.py
└── sample_results/
    └── losd/
        ├── losd_candidate_records.csv
        ├── losd_method_summary.csv
        ├── losd_candidate_summary.csv
        ├── losd_parent_results.csv
        ├── gold_pairs.csv
        ├── parents.csv
        └── plots/
            ├── losd_ablation_scores.pdf
            ├── losd_selected_count.pdf
            ├── novelty_breakdown_counts.pdf
            ├── novelty_breakdown_share.pdf
            └── plot_index.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional Sentence-Transformer support:

```bash
pip install -r requirements-transformer.txt
```

## Run the LOSD Pipeline

The runner expects:

- an ontology file, default: `esco_cmo_binding.ttl`
- a source experiment folder containing a `cache/` directory
- optionally, `gold_pairs.csv`

Example:

```bash
python run_losd.py --source-outdir exp_out_gpt_5
```

If you do not publish cached model outputs, users can still inspect the included sample results under `sample_results/losd/`.

## Generate Figures

```bash
python plot_losd_results.py --results-dir sample_results/losd
```

## Tests

```bash
python -m unittest test_losd_unittest.py
```

or

```bash
python -m unittest tests.test_losd
```

## Before Publishing

Read [PUBLIC_RELEASE_NOTES.md](./PUBLIC_RELEASE_NOTES.md) before pushing this directory to GitHub.
