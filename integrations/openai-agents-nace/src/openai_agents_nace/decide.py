from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from nace_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, SystemOneResponse
from nace_sdk import NaceError

MODELS = frozenset({"drex-max", "drex-v2.0"})
CHECK = "check"
ROUTE = "route"


@dataclass(frozen=True)
class NoulSelect:
    pass


@dataclass(frozen=True)
class ChoiceSelect:
    option: str


@dataclass(frozen=True)
class ScoreSelect:
    at_least: int


Select = NoulSelect | ChoiceSelect | ScoreSelect
QuestionType = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Verdict:
    trip: bool
    probabilities: dict[str, float]
    probability: float
    threshold: float
    model: str
    request_id: str | None
    question_type: str
    legend: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RouteDecision:
    route: str | None
    probabilities: dict[str, float]
    probability: float
    model: str
    request_id: str | None


def check_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(
            "model must be 'drex-max' or 'drex-v2.0'. "
            "drex-latest does not point at drex-max, so this package does not send it."
        )


def check_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be a number from 0 to 1")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a number from 0 to 1")


def check_images(images: Sequence[Any] | None, model: str) -> list[Any] | None:
    if not images:
        return None
    if isinstance(images, (str, bytes)):
        raise ValueError("images must be a list, not a single string or bytes")
    if model != "drex-max":
        raise ValueError("images require model='drex-max'")
    for index, image in enumerate(images):
        if isinstance(image, str) and image.startswith(("http://", "https://", "file://", "/", "./", "~/")):
            raise ValueError(
                f"images[{index}] must be a data URL, base64, or a workspace file. "
                "URLs and paths are not accepted."
            )
    return list(images)


def build_check(
    question_type: QuestionType,
    instructions: str,
    criteria: Mapping[str, str | None] | Sequence[str] | None,
    on: str | int | None,
) -> tuple[Noul | Choice | Score, Select]:
    if question_type == "noul":
        if on is not None:
            raise ValueError("a noul check compares the yes probability; omit on")
        if criteria is not None:
            if not isinstance(criteria, Mapping) or set(criteria) - {"true", "false"}:
                raise ValueError("noul criteria only accepts true and false")
        return Noul(instructions=instructions, criteria=criteria), NoulSelect()
    if question_type == "choice":
        if not isinstance(criteria, Mapping) or len(criteria) < 2:
            raise ValueError("choice criteria must map at least two options to descriptions")
        if not isinstance(on, str) or on not in criteria:
            raise ValueError("choice checks need on set to one criteria key, the option whose probability trips")
        return Choice(instructions=instructions, criteria=dict(criteria)), ChoiceSelect(on)
    if question_type == "score":
        if isinstance(criteria, (str, bytes)) or not isinstance(criteria, Sequence) or not criteria:
            raise ValueError("score criteria must be an ordered list of level labels, low to high")
        if not all(isinstance(level, str) and level for level in criteria):
            raise ValueError("score criteria must be non-empty strings")
        if isinstance(on, bool) or not isinstance(on, int) or not 0 <= on < len(criteria):
            raise ValueError("score checks need on, the lowest level index whose probabilities are summed")
        return Score(instructions=instructions, criteria=list(criteria)), ScoreSelect(on)
    raise ValueError("question_type must be 'noul', 'choice', or 'score'")


def request_id_of(response: SystemOneResponse) -> str | None:
    try:
        return response.request_id
    except NaceError:
        return None


def probability_map(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> dict[str, float]:
    if isinstance(answer, NoulAnswer):
        return {"yes": answer.noul, "no": 1.0 - answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return dict(answer.probabilities)
    return {str(level): probability for level, probability in answer.probabilities.items()}


def selected_probability(answer: NoulAnswer | ChoiceAnswer | ScoreAnswer, select: Select) -> float:
    if isinstance(select, NoulSelect):
        if not isinstance(answer, NoulAnswer):
            raise ValueError("expected a noul answer")
        return answer.noul
    if isinstance(select, ChoiceSelect):
        if not isinstance(answer, ChoiceAnswer):
            raise ValueError("expected a choice answer")
        try:
            return answer.probabilities[select.option]
        except KeyError as exc:
            raise ValueError(f"choice answer has no probability for {select.option!r}") from exc
    if not isinstance(answer, ScoreAnswer):
        raise ValueError("expected a score answer")
    return sum(probability for level, probability in answer.probabilities.items() if level >= select.at_least)


def judge(response: SystemOneResponse, select: Select, threshold: float, question: str) -> Verdict:
    answer = response.answers.get(question)
    if answer is None:
        raise ValueError(f"Drex response has no answer for {question!r}")
    probability = selected_probability(answer, select)
    return Verdict(
        trip=probability > threshold,
        probabilities=probability_map(answer),
        probability=probability,
        threshold=threshold,
        model=response.model,
        request_id=request_id_of(response),
        question_type=answer.type,
        legend=answer.legend if isinstance(answer, ScoreAnswer) else None,
    )


def pick_route(
    answer: ChoiceAnswer,
    routes: Mapping[str, str | None],
    threshold: float,
) -> tuple[str | None, float, dict[str, float]]:
    missing = [name for name in routes if name not in answer.probabilities]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"choice answer is missing probabilities for {joined}")
    probabilities = dict(answer.probabilities)
    best = max(probabilities[name] for name in routes)
    winners = [name for name in routes if probabilities[name] == best]
    route = winners[0] if best > threshold and len(winners) == 1 else None
    return route, best, probabilities


def verdict_info(verdict: Verdict) -> dict[str, Any]:
    info: dict[str, Any] = {
        "probabilities": verdict.probabilities,
        "probability": verdict.probability,
        "threshold": verdict.threshold,
        "model": verdict.model,
        "request_id": verdict.request_id,
        "question_type": verdict.question_type,
        "tripwire_triggered": verdict.trip,
    }
    if verdict.legend is not None:
        info["legend"] = list(verdict.legend)
    return info
