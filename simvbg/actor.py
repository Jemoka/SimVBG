from __future__ import annotations

import json
import re
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Literal

from .backend import ChatBackend, LiteLLMBackend
from .types import ActorResponse, Scenario, TraitInput, TraitValue, TraitVector


TurnMode = Literal["single", "cab"]

PERSPECTIVE_PROMPT_FILES: Mapping[str, str] = {
    "cognitive": "prompt_cognitive.txt",
    "affective": "prompt_affective.txt",
    "behavioral": "prompt_behavioral.txt",
}


@dataclass(slots=True)
class Actor:
    """Trait-conditioned SimVBG actor using the legacy CAB prompt behavior."""

    traits: TraitVector | Mapping[str, TraitValue] | Iterable[TraitInput]
    backend: ChatBackend | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        self.traits = TraitVector.coerce(self.traits)
        if self.backend is None:
            self.backend = LiteLLMBackend()

    def system(self) -> str:
        """Render the profile text that legacy SimVBG prompts place in User Profile.

        The original scripts do not send a chat ``system`` role. This method is
        exposed for inspection and for callers that want the profile string.
        """

        return self.traits.render()

    def backstory(self, *, temperature: float | None = None) -> str:
        """Generate the legacy SimVBG story module output."""

        prompt = _load_prompt("storymodule.txt").format(profile_text=self.traits.render())
        return self._chat(prompt, temperature=temperature)

    def turn(
        self,
        scenario: str | Scenario,
        *,
        mode: TurnMode = "single",
        coordinate: bool = True,
        structured: bool = True,
        temperature: float | None = None,
    ) -> ActorResponse:
        """Respond to a scenario using legacy single-agent or CAB behavior.

        By default, `turn` asks the model for a JSON object so `response.answer`
        is easier to parse. Set `structured=False` for exact legacy prompt text.
        """

        scenario_obj = scenario if isinstance(scenario, Scenario) else Scenario(str(scenario))
        if mode == "single":
            return self._single_turn(scenario_obj, structured=structured, temperature=temperature)
        if mode == "cab":
            return self._cab_turn(
                scenario_obj,
                coordinate=coordinate,
                structured=structured,
                temperature=temperature,
            )
        raise ValueError(f"Unsupported turn mode: {mode}")

    def _single_turn(
        self,
        scenario: Scenario,
        *,
        structured: bool,
        temperature: float | None,
    ) -> ActorResponse:
        prompt = _single_prompt(self.traits.render(), scenario)
        content = self._chat(prompt, structured=structured, temperature=temperature)
        answer, analysis = parse_model_answer(content)
        return ActorResponse(content=content, prompt=prompt, answer=answer, analysis=analysis)

    def _cab_turn(
        self,
        scenario: Scenario,
        *,
        coordinate: bool,
        structured: bool,
        temperature: float | None,
    ) -> ActorResponse:
        rendered_scenario = scenario.render()
        profile = self.traits.render()
        perspective_results: dict[str, dict[str, Any]] = {}

        for perspective, prompt_file in PERSPECTIVE_PROMPT_FILES.items():
            prompt = _load_prompt(prompt_file).format(
                profile_text=profile,
                question_text_with_options=rendered_scenario,
            )
            content = self._chat(prompt, structured=structured, temperature=temperature)
            answer, analysis = parse_model_answer(content)
            perspective_results[perspective] = {
                "answer": answer,
                "analysis": analysis,
                "content": content,
                "prompt": prompt,
            }

        answers = [
            result["answer"]
            for result in perspective_results.values()
            if result["answer"] is not None
        ]
        if len(set(answers)) <= 1:
            content = _merge_without_coordinator(perspective_results, method="unanimous")
            answer = answers[0] if answers else None
            return ActorResponse(
                content=content,
                prompt=rendered_scenario,
                answer=answer,
                analysis=content,
                mode="cab",
                perspectives=perspective_results,
            )
        if not coordinate:
            answer, method, decision_data = _average_decision(perspective_results)
            content = f"{method}: {decision_data}"
            return ActorResponse(
                content=content,
                prompt=rendered_scenario,
                answer=answer,
                analysis=content,
                mode="cab",
                perspectives=perspective_results,
            )

        coordinator_prompt = _load_prompt("coordinator.txt").format(
            question_text=scenario.description,
            options_text=_format_options(scenario.choices),
            cognitive_answer=perspective_results["cognitive"]["answer"],
            cognitive_analysis=perspective_results["cognitive"]["analysis"],
            affective_answer=perspective_results["affective"]["answer"],
            affective_analysis=perspective_results["affective"]["analysis"],
            behavioral_answer=perspective_results["behavioral"]["answer"],
            behavioral_analysis=perspective_results["behavioral"]["analysis"],
        )
        content = self._chat(coordinator_prompt, structured=structured, temperature=temperature)
        answer, analysis = parse_model_answer(content)
        if answer is None:
            fallback_answer, method, decision_data = _average_decision(perspective_results)
            if fallback_answer is not None:
                answer = fallback_answer
                analysis = analysis or f"Fallback {method}: {decision_data}"
        return ActorResponse(
            content=content,
            prompt=coordinator_prompt,
            answer=answer,
            analysis=analysis,
            mode="cab",
            perspectives=perspective_results,
        )

    def _chat(self, prompt: str, *, structured: bool = False, temperature: float | None) -> str:
        assert self.backend is not None
        messages = [{"role": "user", "content": _structured_prompt(prompt) if structured else prompt}]
        if not structured:
            return self.backend.chat(messages, temperature=temperature)

        try:
            return self.backend.chat(
                messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except TypeError:
            return self.backend.chat(messages, temperature=temperature)


def parse_model_answer(text: str) -> tuple[int | None, str]:
    answer, analysis = parse_json_answer_and_analysis(text)
    if answer is not None or analysis:
        return answer, analysis
    return parse_answer_and_analysis(text)


def parse_json_answer_and_analysis(text: str) -> tuple[int | None, str]:
    data = _load_json_object(text)
    if not isinstance(data, Mapping):
        return None, ""

    raw_answer = data.get("answer")
    answer: int | None = None
    if isinstance(raw_answer, int):
        answer = raw_answer
    elif isinstance(raw_answer, str):
        match = re.search(r"\b\d+\b", raw_answer)
        if match:
            answer = int(match.group(0))

    raw_analysis = data.get("analysis", "")
    analysis = raw_analysis if isinstance(raw_analysis, str) else str(raw_analysis)
    return answer, analysis.strip()


def parse_answer_and_analysis(text: str) -> tuple[str | int | None, str]:
    answer_match = re.search(r"Answer:\s*(\d+)", text, flags=re.IGNORECASE)
    answer = int(answer_match.group(1)) if answer_match else None

    analysis_match = re.search(
        r"Analysis:(.*?)(?=Answer:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    if not analysis and answer is not None:
        parts = text.split("\n", 1)
        if len(parts) > 1:
            analysis = parts[1].strip()
    return answer, analysis


def _load_json_object(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _structured_prompt(prompt: str) -> str:
    return f"""{prompt}

Return JSON only with this schema:
{{"answer": <selected option number as an integer or null>, "analysis": <string>}}"""


def _single_prompt(profile: str, scenario: Scenario) -> str:
    return _load_prompt("prompt_single.txt").format(
        question_text_with_options=scenario.render(),
        profile_text=profile,
    )


def _merge_without_coordinator(
    perspective_results: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
) -> str:
    lines = []
    lines.append(method)
    for name, result in perspective_results.items():
        lines.append(f"{name}: answer={result.get('answer')} analysis={result.get('analysis', '')}")
    return "\n".join(lines)


def _format_options(choices: Mapping[str, str] | Sequence[str] | None) -> str:
    if not choices:
        return "No options available"
    if isinstance(choices, Mapping):
        return "\n".join(f"{key}: {value}" for key, value in choices.items())
    return "\n".join(f"{index}: {value}" for index, value in enumerate(choices, start=1))


def _load_prompt(name: str) -> str:
    text = resources.files("simvbg").joinpath("prompts", name).read_text(encoding="utf-8")
    if name in {
        "storymodule.txt",
        "prompt_cognitive.txt",
        "prompt_affective.txt",
        "prompt_single.txt",
    }:
        return text.rstrip("\n")
    return text


def _average_decision(
    perspective_results: Mapping[str, Mapping[str, Any]],
) -> tuple[int | None, str, dict[str, Any]]:
    valid_answers = [
        result["answer"]
        for result in perspective_results.values()
        if result.get("answer") is not None
    ]
    if not valid_answers:
        return None, "no_valid_answers", {}
    if len(valid_answers) == 1:
        return valid_answers[0], "single_answer", {}
    if len(set(valid_answers)) == 1:
        return valid_answers[0], "unanimous", {}

    average = sum(valid_answers) / len(valid_answers)
    return math.ceil(average), "average_ceiling", {
        "average": average,
        "valid_answers": valid_answers,
    }
