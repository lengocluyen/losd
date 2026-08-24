from __future__ import annotations

from losd import HashingTextEmbedder, load_ontology_resources
from run_deepseek_repetitions import inject_defaults as deepseek_defaults
from run_gpt_oss_repetitions import inject_defaults as gpt_oss_defaults
from run_gpt_repetitions import inject_gpt5_defaults
from run_kimi_repetitions import inject_kimi_defaults
from run_llama4_repetitions import inject_defaults as llama_defaults
from run_mistral_large_repetitions import inject_mistral_large_defaults
from run_qwen3_repetitions import inject_defaults as qwen_defaults


def option_value(argv: list[str], option: str) -> str:
    index = argv.index(option)
    return argv[index + 1]


def test_model_wrappers_inject_study_endpoints() -> None:
    cases = [
        (deepseek_defaults, "deepseek/deepseek-chat", "deepinfra"),
        (inject_gpt5_defaults, "openai/gpt-5", "openai"),
        (gpt_oss_defaults, "openai/gpt-oss-120b", "together"),
        (inject_kimi_defaults, "moonshotai/kimi-k2", "novita"),
        (llama_defaults, "meta-llama/llama-4-scout", "groq"),
        (inject_mistral_large_defaults, "mistralai/mistral-large-2407", "mistral"),
        (qwen_defaults, "qwen/qwen3-235b-a22b-thinking-2507", "deepinfra"),
    ]
    for inject, model, provider in cases:
        argv = inject([])
        assert option_value(argv, "--model") == model
        assert option_value(argv, "--openrouter-provider") == provider


def test_explicit_wrapper_values_are_not_overwritten() -> None:
    argv = qwen_defaults(["--max-tokens", "2048", "--output-root=custom"])
    assert option_value(argv, "--max-tokens") == "2048"
    assert "--output-root=custom" in argv
    assert argv.count("--max-tokens") == 1


def test_scoped_ontology_closure_and_embedding_skip(tmp_path) -> None:
    ttl = tmp_path / "tiny.ttl"
    ttl.write_text(
        """
@prefix cmo: <http://www.example.com/cmo#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
<http://example.org/p> cmo:hasImmediateSubCompetence <http://example.org/c> ;
    skos:prefLabel "Parent" .
<http://example.org/c> cmo:hasImmediateSubCompetence <http://example.org/g> ;
    skos:prefLabel "Child" .
<http://example.org/g> skos:prefLabel "Grandchild" .
""".strip(),
        encoding="utf-8",
    )
    resources = load_ontology_resources(
        ttl,
        HashingTextEmbedder(dim=64),
        closure_nodes={"http://example.org/p", "http://example.org/c"},
        include_retrieval_embeddings=False,
    )
    assert set(resources.ancestors) == {"http://example.org/p", "http://example.org/c"}
    assert resources.descendants["http://example.org/p"] == {
        "http://example.org/c",
        "http://example.org/g",
    }
    assert resources.ancestors["http://example.org/c"] == {"http://example.org/p"}
    assert resources.all_text_embeddings.shape == (3, 64)
