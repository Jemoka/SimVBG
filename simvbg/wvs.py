from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any

from .types import Trait, TraitVector


def load_questions(path: str | Path | None = None) -> dict[str, Any]:
    return _load_json(path, "questions.json")


def load_nature_options(path: str | Path | None = None) -> dict[str, Any]:
    return _load_json(path, "nature_options.json")


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row_number, row in enumerate(reader, start=1):
            row["Row_Number"] = row_number
            rows.append(row)
    return rows


def trait_vector_from_wvs_row(
    row: dict[str, Any],
    question_ids: list[str],
    *,
    nature_options: dict[str, Any] | None = None,
    questions: dict[str, Any] | None = None,
) -> TraitVector:
    """Convert WVS answers into a generic TraitVector."""

    nature_options = nature_options or load_nature_options()
    questions = questions or load_questions()
    traits: list[Trait] = []

    for question_id in question_ids:
        if question_id not in row:
            continue
        value = str(row[question_id])
        if value.startswith("-"):
            continue

        trait_text = _naturalized_answer(question_id, value, nature_options, questions)
        if trait_text is None:
            continue
        traits.append(
            Trait(
                name="",
                value=trait_text,
                dimension="wvs",
            )
        )

    return TraitVector(traits=traits, metadata={"source": "WVS"})


def _naturalized_answer(
    question_id: str,
    value: str,
    nature_options: dict[str, Any],
    questions: dict[str, Any],
) -> str | None:
    if question_id in nature_options:
        data = nature_options[question_id]
        options = data.get("options", {})
        if value in options:
            return str(options[value])
        template = data.get("template")
        if template:
            return str(template).format(value=value)

    if question_id in questions:
        data = questions[question_id]
        question_text = data.get("question_text", question_id)
        options = data.get("options", {})
        if value in options:
            return f"{question_text}: {options[value]}"
        return f"{question_text}: {value}"

    return None


def _load_json(path: str | Path | None, filename: str) -> dict[str, Any]:
    if path is not None:
        with Path(path).open(encoding="utf-8") as file:
            return json.load(file)

    package_data = resources.files("simvbg").joinpath("data", filename)
    if package_data.is_file():
        with package_data.open(encoding="utf-8") as file:
            return json.load(file)

    repo_data = Path(__file__).resolve().parent.parent / "WVS_dataset" / filename
    with repo_data.open(encoding="utf-8") as file:
        return json.load(file)
