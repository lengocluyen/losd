"""Mistral Large entry point for the shared repeated-generation experiment.

The original notebook used ``mistral-large-latest``, which resolved to the
2411 generation during the original experiment. OpenRouter no longer exposes
a live 2411 endpoint, so this wrapper uses the closest runnable fixed snapshot,
Mistral Large 2407, and pins the Mistral provider.
"""

from __future__ import annotations

import sys

from run_repetitions import main


MISTRAL_LARGE_DEFAULTS = {
    "--model": "mistralai/mistral-large-2407",
    "--openrouter-provider": "mistral",
    "--reasoning": "default",
    "--token-limit-parameter": "max_tokens",
    "--max-tokens": "512",
    "--temperature": "0.2",
    "--top-p": "1.0",
    "--rpm": "10",
    "--retries": "20",
    "--retry-delay": "60",
    "--max-retry-delay": "300",
    "--output-root": "exp_repeated_mistral_large_2407_openrouter_3runs_20260821",
}


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def inject_mistral_large_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for option, value in MISTRAL_LARGE_DEFAULTS.items():
        if not _has_option(result, option):
            result.extend([option, value])
    return result


if __name__ == "__main__":
    sys.argv[1:] = inject_mistral_large_defaults(sys.argv[1:])
    raise SystemExit(main())
