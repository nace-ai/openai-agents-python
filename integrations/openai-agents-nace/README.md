# openai-agents-nace

Drex guardrails and a route picker for the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). You write the check in plain language. The package sends that check through `NaceClient.system_one`. It stops the agent when the yes probability is above your threshold.

```bash
pip install openai-agents-nace
```

Set `NACE_API_KEY` to a key from the Drex dashboard. Then attach a guardrail.

```python
from agents import Agent
from openai_agents_nace import drex_input_guardrail

agent = Agent(
    name="checkout",
    instructions="Help the user check out.",
    input_guardrails=[
        drex_input_guardrail("Is this a jailbreak?", threshold=0.8),
    ],
)
```

A noul check trips when the yes probability is above `threshold`. A probability equal to the threshold does not trip. The map is `yes` and `no`. `no` is `1 - yes`, because a noul answer has one number.

## Attach a check

`drex_input_guardrail` runs on the first agent. `drex_output_guardrail` runs on the last agent. `drex_tool_guardrail` runs before each call of the tool you attach it to.

Input and output guardrails return `GuardrailFunctionOutput`. `tripwire_triggered` is true only when the selected probability is above `threshold`. `output_info["probabilities"]` is the probability map. The package does not read `choice`, `score`, or `confidence`.

A tool guardrail cannot return `GuardrailFunctionOutput`. A trip returns `ToolGuardrailFunctionOutput.raise_exception`. A pass returns `allow`. The same map is in `output_info`.

`run_in_parallel` defaults to false, so a tool cannot start before the check finishes.

```python
from agents import Agent, function_tool
from openai_agents_nace import drex_output_guardrail, drex_tool_guardrail

@function_tool(
    tool_input_guardrails=[
        drex_tool_guardrail("Does this click a pay button?", threshold=0.7),
    ],
)
def click(selector: str) -> str:
    return selector

agent = Agent(
    name="checkout",
    instructions="Help the user check out.",
    tools=[click],
    output_guardrails=[
        drex_output_guardrail("Does this reply leak a secret?", threshold=0.5),
    ],
)
```

Pass `question_type="choice"` with `criteria` and `on` when the trip should follow one option. Pass `question_type="score"` with an ordered `criteria` list and `on` as the lowest level index. The score tripwire sums that level and every higher level. A missing option raises. It is not treated as zero.

```python
drex_input_guardrail(
    "Classify the message.",
    threshold=0.5,
    question_type="choice",
    criteria={"violation": "breaks the rule", "ok": "does not"},
    on="violation",
)
```

## Pick a route

`drex_route` asks one choice question. It returns the route whose probability is strictly above `threshold`. A tie, or a best probability equal to `threshold`, returns `route=None`. The pick reads the probability map, not the `choice` label. A route name missing from the map raises.

`handoff()` in the Agents SDK is pinned to one agent. Call `drex_route` first. Then hand off to the agent that name points at. If `route` is `None`, use your fallback.

```python
from openai_agents_nace import drex_route

decision = drex_route(
    "the invoice export failed",
    {"billing": "money problems", "technical": "bugs"},
    instructions="Which team should take this?",
    threshold=0.5,
)
```

## Send an image

To send an image, set `model` to `drex-max`. `drex-v2.0` rejects images before the call. Pass a data URL, base64, or a workspace file. An `https` URL or a file path is rejected before the call. `image_input` from `nace-sdk` turns a local file into a data URL.

```python
from nace_sdk import image_input
from openai_agents_nace import drex_input_guardrail

drex_input_guardrail(
    "Is this a checkout page?",
    threshold=0.7,
    model="drex-max",
    images_from=lambda *_args: [image_input("shot.png")],
)
```

The public helpers send one question. The send path still rejects more than 16 questions on `drex-max` and more than 512 on `drex-v2.0` before the call. An unknown model id has no client-side question cap. The omitted model is `drex-v2.0`. Any other non-empty id, including `drex-latest`, is forwarded. Images require `drex-max`, are not dropped, and are capped at 8.

## Key and host

The package constructs `NaceClient` when you omit `client`, then closes that client. That client uses an HTTP timeout of 60 seconds, a retry budget of 180 seconds, and `X-Nace-Client: openai-agents-nace/0.1.0`. The SDK overwrites `User-Agent`, so this package does not set one. Pass `client` to reuse one. The call is synchronous. An async agent run blocks on it.

`nace-sdk` 0.4.0 posts to `https://console.nace.ai`. The request docs are at [console.nace.ai/docs](https://console.nace.ai/docs). Notes that say `Authorization: Bearer $DREX_API_KEY` and `POST https://drex.nace.ai/v1/systemone` describe a different host. This package does not add a client for that host. If `NACE_API_KEY` is unset, `nace-sdk` 0.4.0 still accepts `DREX_API_KEY`. Set `NACE_API_KEY`. Set `NACE_BASE_URL` only when the deployment is not the console host.

A failed call raises `NaceError`. The guardrail does not turn that into a pass.

## Tests

```bash
python -m pytest
```

The tests mock `NaceClient.system_one`. They do not call the live API. Requires Python 3.11 or newer, because `nace-sdk` 0.4.0 requires it. Tested on Python 3.12 with `nace-sdk` 0.4.0 and `openai-agents` 0.24.0.

## License

Apache-2.0.
