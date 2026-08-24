"""GPT-OSS 120B defaults for the shared repeated-generation runner."""

from __future__ import annotations

import sys

from run_repetitions import main


DEFAULTS = {
    "--model": "openai/gpt-oss-120b",
    "--openrouter-provider": "together",
    "--reasoning": "enabled",
    "--reasoning-effort": "medium",
    "--max-tokens": "4096",
    "--output-root": "exp_repeated_gpt_oss_120b_together_3runs_20260821",
}
SWITCH_DEFAULTS = ("--omit-seed",)


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(item.startswith(f"{option}=") for item in argv)


def inject_defaults(argv: list[str]) -> list[str]:
    result = list(argv)
    for option, value in DEFAULTS.items():
        if not _has_option(result, option):
            result.extend([option, value])
    for option in SWITCH_DEFAULTS:
        if not _has_option(result, option):
            result.append(option)
    return result


if __name__ == "__main__":
    sys.argv[1:] = inject_defaults(sys.argv[1:])
    raise SystemExit(main())
