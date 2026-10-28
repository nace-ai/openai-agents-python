import pytest
from nace_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

from openai_agents_nace.decide import (
    ChoiceSelect,
    NoulSelect,
    ScoreSelect,
    judge,
    pick_route,
    probability_map,
    verdict_info,
)


def response(answer, name="check"):
    return SystemOneResponse(model="drex-v2.0", usage=Usage(), answers={name: answer})


def test_noul_trips_only_when_yes_probability_is_above_threshold():
    above = judge(response(NoulAnswer(noul=0.81)), NoulSelect(), 0.8, "check")
    equal = judge(response(NoulAnswer(noul=0.8)), NoulSelect(), 0.8, "check")

    assert above.trip is True
    assert equal.trip is False
    assert above.probabilities == {"yes": 0.81, "no": pytest.approx(0.19)}
    assert "choice" not in verdict_info(above)


def test_choice_uses_the_named_option_not_the_collapsed_label():
    answer = ChoiceAnswer(
        choice="billing",
        confidence=0.4,
        probabilities={"billing": 0.2, "technical": 0.8},
    )

    verdict = judge(response(answer), ChoiceSelect("technical"), 0.5, "check")

    assert verdict.trip is True
    assert verdict.probabilities == {"billing": 0.2, "technical": 0.8}
    assert probability_map(answer)["technical"] == 0.8


def test_choice_missing_option_raises_instead_of_treating_it_as_zero():
    answer = ChoiceAnswer(choice="billing", confidence=1, probabilities={"billing": 1.0})

    with pytest.raises(ValueError, match="technical"):
        judge(response(answer), ChoiceSelect("technical"), 0.5, "check")


def test_score_tail_sums_levels_at_or_above_on():
    answer = ScoreAnswer(
        score=2.0,
        confidence=0.4,
        legend=("Calm", "Annoyed", "Angry", "Furious"),
        probabilities={0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4},
    )

    verdict = judge(response(answer), ScoreSelect(2), 0.6, "check")

    assert verdict.probability == pytest.approx(0.7)
    assert verdict.trip is True
    assert verdict_info(verdict)["legend"] == ["Calm", "Annoyed", "Angry", "Furious"]
    assert verdict.probabilities == {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}


def test_route_ignores_collapsed_label_and_abstains_on_a_tie_or_at_the_threshold():
    disagreed = ChoiceAnswer(
        choice="billing",
        confidence=0.4,
        probabilities={"billing": 0.2, "technical": 0.8, "other": 0.0},
    )
    tied = ChoiceAnswer(
        choice="billing",
        confidence=0.5,
        probabilities={"billing": 0.5, "technical": 0.5},
    )
    routes = {"billing": "money", "technical": "bugs"}

    route, probability, probabilities = pick_route(disagreed, routes, 0.5)
    tie, _, _ = pick_route(tied, routes, 0.4)
    at_threshold, _, _ = pick_route(disagreed, routes, 0.8)

    assert route == "technical"
    assert probability == 0.8
    assert probabilities["other"] == 0.0
    assert tie is None
    assert at_threshold is None


def test_route_raises_when_a_requested_route_has_no_probability():
    answer = ChoiceAnswer(choice="billing", confidence=1, probabilities={"billing": 1.0})

    with pytest.raises(ValueError, match="technical"):
        pick_route(answer, {"billing": None, "technical": None}, 0.5)
