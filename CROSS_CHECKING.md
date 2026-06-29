# Cross-Checking Behavior

This repository is a Codex-written packaging fork around the preserved SimVBG behavior from https://github.com/bangdedadi/SimVBG. The goal is to expose the original prompt and decision behavior through an importable Python API, not to invent new simulation logic.

Use these checks when changing the package API.

## 1. Prompt Resource Parity

The packaged prompts should match the original prompt files or documented original code strings:

```bash
cmp legacy/code/Q_prompts/storymodule.txt simvbg/prompts/storymodule.txt
cmp legacy/code/Q_prompts/prompt_cognitive.txt simvbg/prompts/prompt_cognitive.txt
cmp legacy/code/Q_prompts/prompt_affective.txt simvbg/prompts/prompt_affective.txt
cmp legacy/code/Q_prompts/prompt_behavioral.txt simvbg/prompts/prompt_behavioral.txt
```

`simvbg/prompts/coordinator.txt` and `simvbg/prompts/prompt_single.txt` come from string literals in `legacy/code/main_cv.py`, so they are checked by tests rather than by `cmp`.

## 2. Unit Equivalence Tests

Run:

```bash
uv run python -m unittest tests.test_behavior_equivalence
```

These tests compare the package API against preserved legacy behavior for:

- story prompt construction
- cognitive/affective/behavioral prompt construction
- single/direct answer prompt construction
- coordinator prompt construction
- average-ceiling decision behavior
- numeric answer parsing
- WVS row to profile text conversion

## 3. Build Resource Check

Build the wheel:

```bash
uv build
```

Then inspect the wheel resources:

```bash
uv run python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("dist").glob("simvbg-*.whl"))
required = [
    "simvbg/data/nature_options.json",
    "simvbg/data/questions.json",
    "simvbg/prompts/coordinator.txt",
    "simvbg/prompts/prompt_affective.txt",
    "simvbg/prompts/prompt_behavioral.txt",
    "simvbg/prompts/prompt_cognitive.txt",
    "simvbg/prompts/prompt_single.txt",
    "simvbg/prompts/storymodule.txt",
    "simvbg/py.typed",
]

with ZipFile(wheel) as zf:
    names = set(zf.namelist())
missing = [name for name in required if name not in names]
print("missing:", missing)
PY
```

`missing: []` means downstream repositories can import the package and load its resources.

## 4. Legacy Mapping

See `legacy/MAPPING.md` for the current mapping from original scripts/functions to public package APIs.
