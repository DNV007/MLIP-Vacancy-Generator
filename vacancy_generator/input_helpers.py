"""Interactive CLI input helpers."""

from __future__ import annotations

from typing import Optional, Sequence


def ask_string(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        print("Please enter a value.")


def ask_choice(prompt: str, choices: Sequence[str], default: str) -> str:
    choices_str = "/".join(choices)
    while True:
        raw = input(f"{prompt} [{choices_str}, default: {default}]: ").strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        print(f"Please choose one of: {choices_str}")


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def ask_int(prompt: str, default: Optional[int] = None,
            min_value: Optional[int] = None,
            max_value: Optional[int] = None) -> int:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        if min_value is not None and value < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "y" if default else "n"
    while True:
        raw = input(f"{prompt} [y/n, default: {default_text}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer with y or n.")
