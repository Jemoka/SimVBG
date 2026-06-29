# SimVBG

> This is a Codex-written packaging fork of the original SimVBG behavior from https://github.com/bangdedadi/SimVBG. The original scripts are preserved under `legacy/code/`; the `simvbg` package exposes their prompt and decision behavior through a friendlier importable API. See `CROSS_CHECKING.md` for how to verify package behavior against the preserved legacy code.

SimVBG is a small Python package for trait-conditioned actor simulation. It turns a vector of traits, beliefs, survey answers, or biographical facts into an `Actor` that can answer caller-provided scenarios/questions through the original single-agent path or cognitive/affective/behavioral (CAB) path.

The legacy experiment scripts are preserved in `legacy/code/`, but the package API is centered on building new simulations from Python.

## Installation

```bash
uv sync
```

For editable use from another project:

```bash
uv add --editable /path/to/SimVBG
```

Set provider credentials using the environment variables LiteLLM expects. For OpenAI-compatible endpoints:

```bash
export OPENAI_API_KEY="..."
```

## Quick Start

```python
from simvbg import Actor, Scenario, Trait

actor = Actor(
    traits=[
        Trait("risk tolerance", "low", dimension="behavioral"),
        Trait("political interest", "high", dimension="cognitive"),
        Trait("family orientation", "strong", dimension="affective"),
    ],
    name="sample_actor",
)

scenario = Scenario(
    "A local council proposes a tax increase to fund public transit.",
    choices={
        "1": "Strongly oppose",
        "2": "Oppose",
        "3": "Support",
        "4": "Strongly support",
    },
)

response = actor.turn(scenario, mode="cab")
print(response.answer)
print(response.analysis)
print(response.perspectives["cognitive"]["analysis"])
```

## Trait Vectors

You can pass traits as `Trait` objects, dictionaries, tuples, or a plain mapping:

```python
from simvbg import Actor

actor = Actor({
    "age": 45,
    "religion": "not religious",
    "trust in institutions": "low",
})
```

For explicit metadata and dimensions:

```python
from simvbg import Trait, TraitVector

traits = TraitVector([
    Trait("prefers stability", True, dimension="behavioral", weight=0.8),
    Trait("values personal freedom", "very high", dimension="affective"),
])
```

## Packaged Prompts

The package uses the legacy prompt text from `simvbg/prompts/` via `importlib.resources`. JSON resources live outside the module under `resources/` in the source tree, and `pyproject.toml` copies them into package resource paths at wheel build time:

- `resources/wvs/` -> `simvbg/data/`
- `resources/data_split/` -> `simvbg/data_split/`

For a mapping from original scripts/functions to public package APIs, see `legacy/MAPPING.md`.

## WVS Helpers

The original experiments get profile traits from WVS question IDs. A split's `train_set` is used as profile/trait questions, and its `test_set` is used as questions to answer. The package includes helpers for loading those splits and converting WVS rows into trait vectors:

```python
from simvbg import Actor, load_packaged_split, load_rows, trait_vector_from_wvs_row

split = load_packaged_split("fromCV_SPLIT_FOLD_5.json")
rows = load_rows("WVS_dataset/WVS_Cross-National_Wave_7_csv_v6_0.csv")
traits = trait_vector_from_wvs_row(rows[0], split.profile_questions)
actor = Actor(traits)
```

`questions.json`, `nature_options.json`, and the checked-in split JSON files are included in built wheels. The large raw WVS CSV is not bundled; pass its path explicitly when you need it.

## Models

`Actor` uses `LiteLLMBackend` by default, so local and remote models share one API. Use LiteLLM model strings directly:

```python
from simvbg import Actor, LiteLLMBackend

remote = LiteLLMBackend(model="openai/gpt-4o-mini")
local = LiteLLMBackend(model="ollama/llama3.1")
vllm = LiteLLMBackend(model="hosted_vllm/my-model", api_base="http://localhost:8000/v1")

actor = Actor({"optimism": "high"}, backend=remote)
```

You can also inject any object with a `chat(messages, temperature=None) -> str` method. For tests:

```python
from simvbg import Actor, StaticBackend

actor = Actor({"patience": "low"}, backend=StaticBackend("Answer: 2\nAnalysis: Test response."))
assert actor.turn("Choose an option.").answer == 2
```

## Legacy Experiments

The original cross-validation, TopK, ablation, and profile-impact scripts remain under `legacy/code/`. Install their extra dependencies with:

```bash
uv sync --extra experiments
```

Those scripts still expect local WVS data paths and model gateway settings. The package API is a wrapper around the original prompt behavior; it does not add scenario generation or new simulation logic that was not present in the original code.
