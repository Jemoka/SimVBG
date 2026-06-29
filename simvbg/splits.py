from __future__ import annotations

import json
import random
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


ALL_QUESTIONS = [f"Q{i}" for i in range(1, 291)]


@dataclass(frozen=True, slots=True)
class QuestionSplit:
    """Original SimVBG profile/test question split."""

    train_set: list[str]
    test_set: list[str]
    fold: int | None = None
    raw: dict[str, Any] | None = None

    @property
    def profile_questions(self) -> list[str]:
        return self.train_set

    @property
    def scenario_questions(self) -> list[str]:
        return self.test_set


def create_cross_validation_splits(
    questions: list[str] | None = None,
    *,
    num_folds: int = 5,
    random_seed: int = 42,
) -> list[QuestionSplit]:
    """Reproduce the original question-level k-fold split behavior."""

    rng = random.Random(random_seed)
    shuffled_questions = list(questions or ALL_QUESTIONS)
    rng.shuffle(shuffled_questions)
    fold_size = len(shuffled_questions) // num_folds

    splits: list[QuestionSplit] = []
    for index in range(num_folds):
        start_idx = index * fold_size
        end_idx = (index + 1) * fold_size if index < num_folds - 1 else len(shuffled_questions)
        test_set = shuffled_questions[start_idx:end_idx]
        train_set = shuffled_questions[:start_idx] + shuffled_questions[end_idx:]
        splits.append(
            QuestionSplit(
                fold=index + 1,
                train_set=train_set,
                test_set=test_set,
                raw={"fold": index + 1, "train_set": train_set, "test_set": test_set},
            )
        )
    return splits


def load_split(path: str | Path) -> QuestionSplit:
    """Load an original SimVBG split JSON file."""

    with Path(path).open(encoding="utf-8") as file:
        return _split_from_mapping(json.load(file))


def load_packaged_split(name: str = "fromCV_SPLIT_FOLD_5.json") -> QuestionSplit:
    """Load a split packaged with the wheel.

    Available names mirror the repository's `data_split/` files, for example
    `fromCV_SPLIT_FOLD_5.json` and
    `CROSS_VAL_5_FOLDS_SEED=42/SPLIT_FOLD_1.json`.
    """

    package_path = resources.files("simvbg").joinpath("data_split", name)
    if package_path.is_file():
        return _split_from_mapping(json.loads(package_path.read_text(encoding="utf-8")))

    root = Path(__file__).resolve().parent.parent
    for repo_path in [
        root / "resources" / "data_split" / name,
        root / "data_split" / name,
    ]:
        if repo_path.is_file():
            return load_split(repo_path)
    raise FileNotFoundError(f"Could not find packaged split resource: {name}")


def _split_from_mapping(data: dict[str, Any]) -> QuestionSplit:
    return QuestionSplit(
        fold=data.get("fold"),
        train_set=list(data.get("train_set", [])),
        test_set=list(data.get("test_set", [])),
        raw=data,
    )
