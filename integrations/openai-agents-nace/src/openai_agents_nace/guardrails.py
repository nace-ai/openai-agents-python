from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agents import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail
from agents.tool_guardrails import ToolGuardrailFunctionOutput, ToolInputGuardrail
from nace_sdk import Choice, ChoiceAnswer, NaceClient, RetryPolicy

from openai_agents_nace.decide import (
    CHECK,
    ROUTE,
    QuestionType,
    RouteDecision,
    Select,
    build_check,
    check_images,
    check_model,
    check_questions,
    check_threshold,
    judge,
    pick_route,
    request_id_of,
    verdict_info,
)

CLIENT_INFO = "openai-agents-nace/0.1.0"


def open_client(*, api_key: str | None = None, base_url: str | None = None) -> NaceClient:
    return NaceClient(
        api_key=api_key,
        base_url=base_url,
        client_info=CLIENT_INFO,
        timeout=60,
        retry=RetryPolicy(timeout=180),
    )


def drex_input_guardrail(
    instructions: str,
    *,
    threshold: float,
    client: NaceClient | None = None,
    model: str = "drex-v2.0",
    question_type: QuestionType = "noul",
    criteria: Mapping[str, str | None] | Sequence[str] | None = None,
    on: str | int | None = None,
    name: str = "drex_input",
    run_in_parallel: bool = False,
    state_from: Callable[..., str] | None = None,
    images_from: Callable[..., Sequence[Any]] | None = None,
) -> InputGuardrail:
    question, select = _prepare(instructions, threshold, model, question_type, criteria, on)

    def function(ctx: Any, agent: Any, incoming: Any) -> GuardrailFunctionOutput:
        state = state_from(ctx, agent, incoming) if state_from else text_from(incoming)
        images = images_from(ctx, agent, incoming) if images_from else None
        return _guardrail_output(client, model, state, question, select, threshold, images)

    return InputGuardrail(guardrail_function=function, name=name, run_in_parallel=run_in_parallel)


def drex_output_guardrail(
    instructions: str,
    *,
    threshold: float,
    client: NaceClient | None = None,
    model: str = "drex-v2.0",
    question_type: QuestionType = "noul",
    criteria: Mapping[str, str | None] | Sequence[str] | None = None,
    on: str | int | None = None,
    name: str = "drex_output",
    state_from: Callable[..., str] | None = None,
    images_from: Callable[..., Sequence[Any]] | None = None,
) -> OutputGuardrail:
    question, select = _prepare(instructions, threshold, model, question_type, criteria, on)

    def function(ctx: Any, agent: Any, output: Any) -> GuardrailFunctionOutput:
        state = state_from(ctx, agent, output) if state_from else text_from(output)
        images = images_from(ctx, agent, output) if images_from else None
        return _guardrail_output(client, model, state, question, select, threshold, images)

    return OutputGuardrail(guardrail_function=function, name=name)


def drex_tool_guardrail(
    instructions: str,
    *,
    threshold: float,
    client: NaceClient | None = None,
    model: str = "drex-v2.0",
    question_type: QuestionType = "noul",
    criteria: Mapping[str, str | None] | Sequence[str] | None = None,
    on: str | int | None = None,
    name: str = "drex_tool",
    state_from: Callable[..., str] | None = None,
    images_from: Callable[..., Sequence[Any]] | None = None,
) -> ToolInputGuardrail:
    question, select = _prepare(instructions, threshold, model, question_type, criteria, on)

    def function(data: Any) -> ToolGuardrailFunctionOutput:
        state = state_from(data) if state_from else tool_state(data)
        images = images_from(data) if images_from else None
        verdict = _verdict(client, model, state, question, select, threshold, images)
        info = verdict_info(verdict)
        if verdict.trip:
            return ToolGuardrailFunctionOutput.raise_exception(output_info=info)
        return ToolGuardrailFunctionOutput.allow(output_info=info)

    return ToolInputGuardrail(guardrail_function=function, name=name)


def drex_route(
    state: Any,
    routes: Mapping[str, str | None],
    *,
    instructions: str,
    threshold: float,
    client: NaceClient | None = None,
    model: str = "drex-v2.0",
    images: Sequence[Any] | None = None,
) -> RouteDecision:
    check_model(model)
    check_threshold(threshold)
    if not isinstance(routes, Mapping) or len(routes) < 2:
        raise ValueError("routes must map at least two names to descriptions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    if not all(isinstance(name, str) and name for name in routes):
        raise ValueError("route names must be non-empty strings")
    question = Choice(instructions=instructions.strip(), criteria=dict(routes))
    response = _call(client, model=model, state=state, questions={ROUTE: question}, images=images)
    answer = response.answers.get(ROUTE)
    if not isinstance(answer, ChoiceAnswer):
        raise ValueError("Drex response did not include a choice answer for the route question")
    route, probability, probabilities = pick_route(answer, routes, threshold)
    return RouteDecision(
        route=route,
        probabilities=probabilities,
        probability=probability,
        model=response.model,
        request_id=request_id_of(response),
    )


def _prepare(
    instructions: str,
    threshold: float,
    model: str,
    question_type: QuestionType,
    criteria: Mapping[str, str | None] | Sequence[str] | None,
    on: str | int | None,
) -> tuple[Any, Select]:
    check_model(model)
    check_threshold(threshold)
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("instructions must be a non-empty string")
    return build_check(question_type, instructions.strip(), criteria, on)


def _guardrail_output(
    client: NaceClient | None,
    model: str,
    state: Any,
    question: Any,
    select: Select,
    threshold: float,
    images: Sequence[Any] | None,
) -> GuardrailFunctionOutput:
    verdict = _verdict(client, model, state, question, select, threshold, images)
    return GuardrailFunctionOutput(
        tripwire_triggered=verdict.trip,
        output_info=verdict_info(verdict),
    )


def _verdict(
    client: NaceClient | None,
    model: str,
    state: Any,
    question: Any,
    select: Select,
    threshold: float,
    images: Sequence[Any] | None,
):
    response = _call(client, model=model, state=state, questions={CHECK: question}, images=images)
    return judge(response, select, threshold, CHECK)


def _call(
    client: NaceClient | None,
    *,
    model: str,
    state: Any,
    questions: Mapping[str, Any],
    images: Sequence[Any] | None,
):
    check_questions(questions, model)
    wire_images = check_images(images, model)
    owns = client is None
    active = client if client is not None else open_client()
    try:
        return active.system_one(state, questions, model=model, images=wire_images)
    finally:
        if owns:
            active.close()


def text_from(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [part for item in value if (part := _item_text(item))]
        if parts:
            return "\n".join(parts)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), ensure_ascii=False, default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, default=str)
    return str(value)


def tool_state(data: Any) -> str:
    arguments = getattr(getattr(data, "context", None), "tool_arguments", None)
    if isinstance(arguments, str):
        return arguments
    return text_from(data)


def _item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    content = item.get("content") if isinstance(item, Mapping) else getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    bits: list[str] = []
    for part in content:
        if isinstance(part, str):
            bits.append(part)
        elif isinstance(part, Mapping) and part.get("type") in {"input_text", "output_text", "text"}:
            bits.append(str(part.get("text", "")))
    return "\n".join(bit for bit in bits if bit)
