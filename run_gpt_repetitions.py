"""GPT-5 entry point for the shared repeated-generation experiment.

This wrapper keeps the experiment implementation in ``run_repetitions.py``
while supplying GPT-5/OpenRouter-compatible defaults. Every injected default
can still be overridden explicitly on the command line.
"""

from __future__ import annotations

import sys

from run_repetitions import main


GPT5_DEFAULTS = {
    "--model": "openai/gpt-5",
    "--openrouter-provider": "openai",
    # The original main_gpt-5 notebook did not transmit a reasoning option.
    "--reasoning": "default",
    # OpenRouter's pinned OpenAI endpoint advertises max_tokens. Its Azure
    # endpoints advertise max_completion_tokens, but this wrapper pins OpenAI.
    "--token-limit-parameter": "max_tokens",
    "--max-tokens": "4096",
    "--output-root": "exp_repeated_gpt5_openrouter_3runs_20260821",
}
GPT5_SWITCH_DEFAULTS = ("--omit-temperature", "--omit-top-p")


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def inject_gpt5_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for option, value in GPT5_DEFAULTS.items():
        if not _has_option(result, option):
            result.extend([option, value])
    for option in GPT5_SWITCH_DEFAULTS:
        if not _has_option(result, option):
            result.append(option)
    return result


if __name__ == "__main__":
    sys.argv[1:] = inject_gpt5_defaults(sys.argv[1:])
    raise SystemExit(main())
