# Legacy Mapping

The `simvbg` package is a friendly API over the original prompt behavior from https://github.com/bangdedadi/SimVBG. The original experiment scripts are preserved under `legacy/code/` for reproducibility and reference.

## Public Package API

| New API | Original source | Notes |
| --- | --- | --- |
| `simvbg.Actor.backstory()` | `legacy/code/main_cv.py::generate_story_with_llm` and `legacy/code/Q_prompts/storymodule.txt` | Uses the same story prompt template and sends one user message. |
| `simvbg.Actor.turn(..., mode="cab")` | `process_*_perspective_question_with_analysis`, `collaborative_decision`, `coordinator_decision` in `legacy/code/main_cv.py` | Runs cognitive, affective, and behavioral prompts, then either coordinates or applies average-ceiling behavior. |
| `simvbg.Actor.turn(..., mode="single")` | `legacy/code/main_cv.py::process_original_profile_question` prompt shape | Exposes the original profile/direct answer path. It does not generate a new scenario. |
| `simvbg.Trait`, `simvbg.TraitVector` | `profile_texts` lists assembled in `main_cv.py` | Structured wrapper around the original newline-joined profile text. |
| `simvbg.Scenario` | `generate_question_text_with_options` output shape | Caller-provided question/scenario text plus optional `Options:` block. |
| `simvbg.trait_vector_from_wvs_row` | `generate_profile_text_list` | Converts WVS rows into the same natural-language profile lines used by the legacy scripts. |
| `simvbg.LiteLLMBackend` | `call_llm_api` / `init_client` | Replaces hardcoded OpenAI-compatible clients with LiteLLM while preserving the same chat-message shape. |

## Packaged Resources

The original prompt files are copied into package resources:

| Package resource | Original file |
| --- | --- |
| `simvbg/prompts/storymodule.txt` | `legacy/code/Q_prompts/storymodule.txt` |
| `simvbg/prompts/prompt_cognitive.txt` | `legacy/code/Q_prompts/prompt_cognitive.txt` |
| `simvbg/prompts/prompt_affective.txt` | `legacy/code/Q_prompts/prompt_affective.txt` |
| `simvbg/prompts/prompt_behavioral.txt` | `legacy/code/Q_prompts/prompt_behavioral.txt` |
| `simvbg/prompts/coordinator.txt` | String literal in `legacy/code/main_cv.py::coordinator_decision` |
| `simvbg/prompts/prompt_single.txt` | String literal in `legacy/code/main_cv.py::process_original_profile_question` |

The wheel also includes `questions.json` and `nature_options.json` as `simvbg/data/`.

## Still Legacy-Only

These experiment runners are intentionally not converted into the public API:

| Legacy script | Purpose |
| --- | --- |
| `legacy/code/main_cv.py` | Cross-validation experiment runner. |
| `legacy/code/topk_cv.py` | TopK/RAG baseline. |
| `legacy/code/ablation.py` | Ablation experiments. |
| `legacy/code/profile_impact.py` | Incremental profile information experiment. |

They still contain local path assumptions and model-gateway placeholders from the original codebase.
