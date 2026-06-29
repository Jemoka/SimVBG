from __future__ import annotations

import math
import unittest
from pathlib import Path

from simvbg import Actor, Scenario
from simvbg.actor import parse_answer_and_analysis
from simvbg.wvs import trait_vector_from_wvs_row


ROOT = Path(__file__).resolve().parents[1]


class CaptureBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, *, temperature=None):
        self.messages.append(list(messages))
        return self.responses.pop(0)


def original_prompt(name: str) -> str:
    return (ROOT / "legacy" / "code" / "Q_prompts" / name).read_text(encoding="utf-8")


class BehaviorEquivalenceTests(unittest.TestCase):
    def test_packaged_story_prompt_matches_original_prompt_file(self):
        backend = CaptureBackend(["story"])
        actor = Actor({"age": 45, "trust": "low"}, backend=backend)

        actor.backstory()

        expected = original_prompt("storymodule.txt").format(profile_text=actor.system())
        self.assertEqual(backend.messages[0], [{"role": "user", "content": expected}])

    def test_packaged_cab_prompts_match_original_prompt_files(self):
        backend = CaptureBackend(
            [
                "Answer: 1\nAnalysis: cognitive",
                "Answer: 1\nAnalysis: affective",
                "Answer: 1\nAnalysis: behavioral",
            ]
        )
        actor = Actor({"age": 45}, backend=backend)
        scenario = Scenario("Question text?", choices={"1": "Yes", "2": "No"})

        actor.turn(scenario, mode="cab")

        rendered = scenario.render()
        for idx, name in enumerate(["cognitive", "affective", "behavioral"]):
            expected = original_prompt(f"prompt_{name}.txt").format(
                profile_text=actor.system(),
                question_text_with_options=rendered,
            )
            self.assertEqual(backend.messages[idx], [{"role": "user", "content": expected}])

    def test_single_turn_prompt_matches_original_original_profile_path(self):
        backend = CaptureBackend(["2"])
        actor = Actor({"age": 45}, backend=backend)
        scenario = Scenario("Question text?", choices={"1": "Yes", "2": "No"})

        actor.turn(scenario, mode="single")

        expected = (
            f"Question:{scenario.render()}\n"
            f"User profile: {actor.system()}\n"
            "Consider both the question context and the user's background when formulating your response. "
            "Aim for a balanced perspective that respects accuracy while reflecting the user's viewpoint.\n"
            "Answer format: 'option you selected'"
        )
        self.assertEqual(backend.messages[0], [{"role": "user", "content": expected}])

    def test_coordinator_prompt_matches_original_code_string(self):
        backend = CaptureBackend(
            [
                "Answer: 1\nAnalysis: cognitive",
                "Answer: 2\nAnalysis: affective",
                "Answer: 3\nAnalysis: behavioral",
                "Answer: 2\nAnalysis: coordinated",
            ]
        )
        actor = Actor({"age": 45}, backend=backend)
        scenario = Scenario("Question text?", choices={"1": "Yes", "2": "No", "3": "Maybe"})

        actor.turn(scenario, mode="cab", coordinate=True)

        expected = f"""
You are a coordinator in a user simulation system, and you need to synthesize analyses from three different perspectives to make a final decision.

Question: {scenario.description}
Options: 1: Yes
2: No
3: Maybe

Cognitive perspective answer: 1
Cognitive perspective analysis: cognitive

Emotional perspective answer: 2
Emotional perspective analysis: affective

Behavioral perspective answer: 3
Behavioral perspective analysis: behavioral

Consider:
• How their thoughts, feelings, and behavioral tendencies might interact in this situation
• Which aspects of their psychology seem most influential here
• Where their different perspectives align or create tension

Format your response exactly as follows:
Answer: [option number]
Analysis: [your reasoning for this decision]
"""
        self.assertEqual(backend.messages[3], [{"role": "user", "content": expected}])

    def test_coordinate_false_uses_original_average_ceiling_decision(self):
        backend = CaptureBackend(
            [
                "Answer: 1\nAnalysis: cognitive",
                "Answer: 2\nAnalysis: affective",
                "Answer: 3\nAnalysis: behavioral",
            ]
        )
        actor = Actor({"age": 45}, backend=backend)

        response = actor.turn("Question text?", mode="cab", coordinate=False)

        self.assertEqual(response.answer, math.ceil((1 + 2 + 3) / 3))
        self.assertIn("average_ceiling", response.content)

    def test_answer_parser_matches_original_numeric_answer_behavior(self):
        self.assertEqual(parse_answer_and_analysis("Answer: 7\nAnalysis: because"), (7, "because"))
        self.assertEqual(parse_answer_and_analysis("Answer: yes\nAnalysis: because"), (None, "because"))
        self.assertEqual(parse_answer_and_analysis("Answer: 2\nextra text"), (2, "extra text"))

    def test_wvs_trait_vector_matches_original_nature_profile_lines(self):
        nature_options = {
            "Q1": {"question_text": "Question one", "options": {"1": "Chosen option"}},
            "Q2": {"question_text": "Age", "options": {}, "template": "Age is {value}"},
            "Q3": {"question_text": "Raw", "options": {}},
        }
        questions = {
            "Q1": {"question_text": "Question one", "options": {"1": "Chosen option"}},
            "Q2": {"question_text": "Age", "options": {}},
            "Q3": {"question_text": "Raw", "options": {}},
        }
        row = {"Q1": "1", "Q2": "45", "Q3": "8"}

        traits = trait_vector_from_wvs_row(
            row,
            ["Q1", "Q2", "Q3"],
            nature_options=nature_options,
            questions=questions,
        )

        self.assertEqual(traits.render(), "Chosen option\nAge is 45\nRaw: 8")


if __name__ == "__main__":
    unittest.main()
