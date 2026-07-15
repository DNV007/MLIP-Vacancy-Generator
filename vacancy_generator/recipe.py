"""Defect recipe parsing and vacancy combination enumeration."""

from __future__ import annotations

from itertools import combinations, product
from typing import Dict, List, Tuple

from pymatgen.core import Structure

from .structure_utils import get_indices_by_species, get_species_counts


def parse_removal_recipe(recipe_text: str,
                         available_species: Dict[str, int]) -> Dict[str, int]:
    """Parse text like 'Mg:1, Se:2' into a dictionary.

    Returns
    -------
    dict
        Example: {'Mg': 1, 'Se': 2}

    Raises
    ------
    ValueError
        If a species appears more than once, is unknown, or has an invalid count.
    """
    recipe: Dict[str, int] = {}
    parts = [part.strip() for part in recipe_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("No defect recipe was provided.")

    for part in parts:
        if ":" not in part:
            raise ValueError(
                f"Could not parse '{part}'. Please use the form Species:Count, "
                f"for example Mg:1"
            )
        symbol, count_text = [item.strip() for item in part.split(":", 1)]

        if symbol in recipe:
            raise ValueError(
                f"Species '{symbol}' appears more than once in the recipe. "
                f"Combine into a single entry, e.g. '{symbol}:<total>'."
            )

        if symbol not in available_species:
            raise ValueError(
                f"Species '{symbol}' is not present. Available species are: "
                f"{', '.join(available_species)}"
            )

        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Removal count for {symbol} must be an integer, got '{count_text}'."
            ) from exc

        if count < 1:
            raise ValueError(f"Removal count for {symbol} must be at least 1.")
        if count > available_species[symbol]:
            raise ValueError(
                f"Cannot remove {count} atoms of {symbol}; "
                f"only {available_species[symbol]} are present."
            )

        recipe[symbol] = count
    return recipe


def ask_removal_recipe(structure: Structure) -> Dict[str, int]:
    """Ask the user which atoms to remove (e.g. 'Mg:1' or 'Mg:1, Se:2')."""
    available = get_species_counts(structure)
    print("\nEnter the defect recipe as Species:Count pairs separated by commas.")
    print("Examples: Mg:1   or   Mg:1, Se:2")
    while True:
        raw = input("Defect recipe: ").strip()
        try:
            return parse_removal_recipe(raw, available)
        except ValueError as err:
            print(err)


def defect_class_label(recipe: Dict[str, int]) -> str:
    """Return a short human-readable label such as 'Mg1+Se2'."""
    return "+".join(f"{symbol}{recipe[symbol]}" for symbol in sorted(recipe))


def generate_raw_removal_combinations(structure: Structure,
                                      recipe: Dict[str, int]) -> List[Tuple[int, ...]]:
    """Generate every possible atom-index combination for the requested recipe."""
    indices_by_species = get_indices_by_species(structure)
    specieswise_choices: List[List[Tuple[int, ...]]] = []

    for symbol in sorted(recipe):
        count = recipe[symbol]
        species_indices = indices_by_species[symbol]
        specieswise_choices.append(list(combinations(species_indices, count)))

    all_combos = []
    for choice_group in product(*specieswise_choices):
        flat = tuple(sorted(idx for subgroup in choice_group for idx in subgroup))
        all_combos.append(flat)

    return all_combos
