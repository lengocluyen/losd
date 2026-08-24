"""Kimi K2 entry point for the shared repeated-generation experiment.

The pinned Novita endpoint occasionally returns a completed, length-limited
response with no visible candidate text. This wrapper enables bounded retries
for that provider anomaly without changing prompts or decoding parameters.
"""

from __future__ import annotations

import sys

from run_repetitions import main


KIMI_DEFAULTS = {
    "--model": "moonshotai/kimi-k2",
    "--openrouter-provider": "novita",
    "--reasoning": "default",
    "--token-limit-parameter": "max_tokens",
    "--max-tokens": "4096",
    "--rpm": "15",
    "--retries": "5",
    "--unparseable-retries": "5",
    "--retry-delay": "30",
    "--output-root": "exp_repeated_kimi_k2_openrouter_3runs_4096_20260821",
}


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def inject_kimi_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for option, value in KIMI_DEFAULTS.items():
        if not _has_option(result, option):
            result.extend([option, value])
    return result


if __name__ == "__main__":
    sys.argv[1:] = inject_kimi_defaults(sys.argv[1:])
    raise SystemExit(main())
