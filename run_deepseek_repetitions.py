"""DeepSeek V3 defaults for the shared repeated-generation runner."""

from __future__ import annotations

import sys

from run_repetitions import main


DEFAULTS = {
    "--model": "deepseek/deepseek-chat",
    "--openrouter-provider": "deepinfra",
    "--reasoning": "default",
    "--max-tokens": "512",
    "--output-root": "exp_repeated_deepseek_v3_openrouter_3runs_20260821",
}


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def inject_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for option, value in DEFAULTS.items():
        if not _has_option(result, option):
            result.extend([option, value])
    return result


if __name__ == "__main__":
    sys.argv[1:] = inject_defaults(sys.argv[1:])
    raise SystemExit(main())
