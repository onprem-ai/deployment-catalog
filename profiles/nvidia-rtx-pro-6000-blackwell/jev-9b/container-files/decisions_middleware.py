"""Typed decisions (the OpenAI Decisions API, ``POST /v1/decisions``) inside vLLM.

A deploy profile's container file, loaded by the profile's own vLLM server as ASGI
middleware (``--middleware decisions_middleware.decisions``, with the container files
directory on ``PYTHONPATH``). A ``POST /v1/decisions`` request is answered here; every
other request passes to vLLM untouched.

A decision model answers a typed question about some input with a calibrated
probability for every answer, in one forward pass. Each question becomes one
``/v1/chat/completions`` request to the same server's decision module (a LoRA module):
a raw decision prompt in the model's protocol (``bare-v1``), the answer restricted to
the option tokens, their log-probabilities asked for with sampling off, and the
module's per-option bias and per-kind temperature (the profile's
``container-files/decision-head.json``) applied to what comes back.

The mapping, per question type:

- ``predicate``: the model's ``noul`` (options ``false``, ``true``); the answer is the
  probability of ``true``.
- ``choice``: the model's ``choice``, values labelled ``A)`` to ``P)``. The decision head
  scores 16 labels, so at most 16 values are asked; more are refused with the reason.
- ``score``: the model's ``score``, a fixed scale of the points ``0`` to ``5``. A question
  with N levels (2 to 6) is asked with its levels as the first N points, and the
  probabilities are renormalized over them; the score is their expected level index.

Configuration, from the container's environment:

- ``DECISIONS_MODULE``: the decision module's served name (``--lora-modules``).
- ``DECISIONS_PARAMETERS_FILE``: the decision parameters file.
- ``DECISIONS_ENGINE_URL``: this server's own address (default ``http://127.0.0.1:8080``).

Only what the vLLM image already carries is imported: pydantic, httpx, starlette.
"""


from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# --- the decision parameters a route carries ------------------------------------------


class DecisionKindParameters(BaseModel):
    """One question kind's slots in the decision head, and its calibration temperature."""

    model_config = ConfigDict(extra="forbid", strict=True)

    first_slot: int = Field(ge=0)
    slot_count: int = Field(ge=2)
    temperature: float = Field(gt=0)


class DecisionParameters(BaseModel):
    """What the gateway needs to turn a decision module's log-probabilities into answers.

    Read from the deploy profile's ``container-files/decision-head.json``: the decision
    LoRA module the requests address, the prompt protocol, the verbalizer tokens with
    their ids and biases (one per slot), and the slots and temperature of each kind.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    module: str = Field(min_length=1)
    prompt_protocol: Literal["bare-v1"]
    verbalizers: list[str]
    verbalizer_token_ids: list[int]
    bias: list[float]
    kinds: dict[Literal["noul", "choice", "score"], DecisionKindParameters]

    @model_validator(mode="after")
    def _consistent(self) -> DecisionParameters:
        slots = len(self.verbalizers)
        if len(self.verbalizer_token_ids) != slots or len(self.bias) != slots:
            raise ValueError(
                "verbalizers, verbalizer_token_ids and bias must have one entry per slot "
                f"({slots}, {len(self.verbalizer_token_ids)}, {len(self.bias)})"
            )
        for kind in ("noul", "choice", "score"):
            if kind not in self.kinds:
                raise ValueError(f"the decision parameters name no slots for {kind!r}")
            parameters = self.kinds[kind]
            if parameters.first_slot + parameters.slot_count > slots:
                raise ValueError(f"the {kind!r} slots run past the {slots} verbalizers")
        return self


# --- the OpenAI Decisions request -----------------------------------------------------

_REQUEST = ConfigDict(extra="forbid")

#: The decision head scores the labels A to P.
MAXIMUM_CHOICES = 16
#: The model rates on the points 0 to 5.
MAXIMUM_SCORE_LEVELS = 6
#: OpenAI's own bound on inline images per request.
MAXIMUM_IMAGES = 128


class DecisionInputText(BaseModel):
    model_config = _REQUEST
    type: Literal["input_text"]
    text: str


class DecisionInputImage(BaseModel):
    model_config = _REQUEST
    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto", "original"] | None = None

    @field_validator("image_url")
    @classmethod
    def _inline_only(cls, value: str) -> str:
        if not value.startswith("data:image/") or ";base64," not in value:
            raise ValueError(
                "must be an inline image: a base64 data URL such as 'data:image/png;base64,...'; "
                "external URLs are not fetched"
            )
        try:
            base64.b64decode(value.split(";base64,", 1)[1], validate=True)
        except ValueError as error:
            raise ValueError(f"the data URL's base64 payload does not decode: {error}") from error
        return value


DecisionInputPart = Annotated[DecisionInputText | DecisionInputImage, Field(discriminator="type")]


class DecisionInputMessage(BaseModel):
    model_config = _REQUEST
    role: Literal["user"]
    type: Literal["message"] | None = None
    content: str | list[DecisionInputPart]


class PredicateQuestion(BaseModel):
    model_config = _REQUEST
    type: Literal["predicate"]
    instructions: str = Field(min_length=1)
    name: str | None = None


class ChoiceValue(BaseModel):
    model_config = _REQUEST
    value: str | bool
    description: str | None = None


class ChoiceQuestion(BaseModel):
    model_config = _REQUEST
    type: Literal["choice"]
    instructions: str = Field(min_length=1)
    choices: list[ChoiceValue] = Field(min_length=2)
    name: str | None = None

    @field_validator("choices")
    @classmethod
    def _bounded_and_distinct(cls, choices: list[ChoiceValue]) -> list[ChoiceValue]:
        if len(choices) > MAXIMUM_CHOICES:
            raise ValueError(
                f"this model chooses among at most {MAXIMUM_CHOICES} values; "
                f"{len(choices)} were sent. Split the question, or narrow the values first"
            )
        # A string and a boolean with the same text are distinct values.
        keys = [(type(choice.value).__name__, choice.value) for choice in choices]
        if len(set(keys)) != len(keys):
            raise ValueError("every choice value must be distinct")
        for choice in choices:
            text = _choice_line(choice)
            if "\n" in text:
                raise ValueError("a choice value or description must be one line: no newlines")
        return choices


class ScoreLevel(BaseModel):
    model_config = _REQUEST
    label: str = Field(min_length=1)
    description: str | None = None


class ScoreQuestion(BaseModel):
    model_config = _REQUEST
    type: Literal["score"]
    instructions: str = Field(min_length=1)
    levels: list[ScoreLevel] = Field(min_length=2)
    name: str | None = None

    @field_validator("levels")
    @classmethod
    def _bounded(cls, levels: list[ScoreLevel]) -> list[ScoreLevel]:
        if len(levels) > MAXIMUM_SCORE_LEVELS:
            raise ValueError(
                f"this model rates on at most {MAXIMUM_SCORE_LEVELS} levels; {len(levels)} were sent"
            )
        return levels


Question = Annotated[
    PredicateQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class DecisionRequest(BaseModel):
    """``POST /v1/decisions``: the evidence once, and the questions about it."""

    model_config = _REQUEST

    model: str = Field(min_length=1)
    input: str | list[DecisionInputMessage]
    questions: list[Question] = Field(min_length=1, max_length=32)
    safety_identifier: str | None = None

    @field_validator("input")
    @classmethod
    def _bounded_images(cls, value: str | list[DecisionInputMessage]) -> Any:
        if isinstance(value, str):
            return value
        images = sum(
            1
            for message in value
            if not isinstance(message.content, str)
            for part in message.content
            if isinstance(part, DecisionInputImage)
        )
        if images > MAXIMUM_IMAGES:
            raise ValueError(f"at most {MAXIMUM_IMAGES} images per request; {images} were sent")
        return value


def parse_decision_request(payload: Any) -> DecisionRequest:
    """The request, or ``ValueError`` naming every field that does not fit."""
    try:
        return DecisionRequest.model_validate(payload)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'body'}: {issue['msg']}"
            for issue in error.errors()
        )
        raise ValueError(f"the decision request does not fit the Decisions API: {problems}") from error


# --- the prompt -----------------------------------------------------------------------

#: The chat template a decision request is rendered with: the content parts in order,
#: text verbatim and each image as the model's vision placeholder. No role markers and
#: no generation prompt -- the decision prompt is the whole input.
RAW_DECISION_TEMPLATE = (
    "{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}"
    "{{ c['text'] }}{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}"
    "{%- endfor -%}{%- endfor -%}"
)

_LABELS = "ABCDEFGHIJKLMNOP"


def _choice_line(choice: ChoiceValue) -> str:
    text = str(choice.value).lower() if isinstance(choice.value, bool) else choice.value
    return f"{text}: {choice.description}" if choice.description else text


def _input_parts(decision_input: str | list[DecisionInputMessage]) -> list[dict[str, Any]]:
    """The evidence as chat content parts, in order; messages are separated by a newline."""
    if isinstance(decision_input, str):
        return [{"type": "text", "text": decision_input}]
    parts: list[dict[str, Any]] = []
    for index, message in enumerate(decision_input):
        if index:
            parts.append({"type": "text", "text": "\n"})
        if isinstance(message.content, str):
            parts.append({"type": "text", "text": message.content})
            continue
        for part in message.content:
            if isinstance(part, DecisionInputText):
                parts.append({"type": "text", "text": part.text})
            else:
                parts.append({"type": "image_url", "image_url": {"url": part.image_url}})
    return parts


def model_kind(question: PredicateQuestion | ChoiceQuestion | ScoreQuestion) -> str:
    """The model's own name for the question's kind."""
    if isinstance(question, PredicateQuestion):
        return "noul"
    if isinstance(question, ChoiceQuestion):
        return "choice"
    return "score"


def option_lines(question: PredicateQuestion | ChoiceQuestion | ScoreQuestion) -> list[str]:
    """The ``[options]`` lines of the decision prompt."""
    if isinstance(question, PredicateQuestion):
        return ["false", "true"]
    if isinstance(question, ChoiceQuestion):
        return [f"{_LABELS[index]}) {_choice_line(choice)}" for index, choice in enumerate(question.choices)]
    lines = []
    for index, level in enumerate(question.levels):
        text = f"{level.label}: {level.description}" if level.description else level.label
        lines.append(f"{index}) {text}")
    return lines


def slot_count(question: PredicateQuestion | ChoiceQuestion | ScoreQuestion) -> int:
    """How many of the kind's slots the question uses."""
    if isinstance(question, PredicateQuestion):
        return 2
    if isinstance(question, ChoiceQuestion):
        return len(question.choices)
    return len(question.levels)


def engine_request(
    request: DecisionRequest,
    question: PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    parameters: DecisionParameters,
) -> dict[str, Any]:
    """The ``/v1/chat/completions`` request that asks one question of the decision module.

    The answer is one token restricted to the question's option tokens, with their
    log-probabilities; ``top_k`` 0 and ``top_p`` 1.0 keep the engine's request defaults
    from cutting any option off, and ``temperature`` 1.0 leaves the log-probabilities as
    the module produced them.
    """
    kind = model_kind(question)
    kind_parameters = parameters.kinds[kind]
    first = kind_parameters.first_slot
    token_ids = parameters.verbalizer_token_ids[first : first + slot_count(question)]
    header = f"[kind] {kind}\n[state] "
    footer = (
        f"\n[question] {question.instructions}\n[options]\n"
        + "\n".join(option_lines(question))
        + "\n[decision]:"
    )
    content = [{"type": "text", "text": header}, *_input_parts(request.input), {"type": "text", "text": footer}]
    return {
        "model": parameters.module,
        "messages": [{"role": "user", "content": content}],
        "chat_template": RAW_DECISION_TEMPLATE,
        "add_generation_prompt": False,
        "add_special_tokens": False,
        "max_tokens": 1,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "logprobs": True,
        "top_logprobs": len(token_ids),
        "allowed_token_ids": token_ids,
        "return_tokens_as_token_ids": True,
    }


# --- the answer -----------------------------------------------------------------------


class EngineAnswerError(ValueError):
    """The engine's answer to a decision request lacks the log-probabilities it must carry."""


def _option_logprobs(engine_answer: dict[str, Any]) -> dict[int, float]:
    """The log-probability of every token the engine listed, by token id."""
    try:
        listed = engine_answer["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise EngineAnswerError(
            "the model's answer carries no log-probabilities for the options "
            f"({type(error).__name__}: {error}); the engine must run with "
            "--logprobs-mode processed_logprobs and the decision module loaded"
        ) from error
    logprobs: dict[int, float] = {}
    for entry in listed:
        token = entry.get("token", "")
        if not isinstance(token, str) or not token.startswith("token_id:"):
            raise EngineAnswerError(
                f"the model's answer names an option token as {token!r}, not as 'token_id:<id>'"
            )
        logprobs[int(token.split(":", 1)[1])] = float(entry["logprob"])
    return logprobs


def option_probabilities(
    question: PredicateQuestion | ChoiceQuestion | ScoreQuestion,
    parameters: DecisionParameters,
    engine_answer: dict[str, Any],
) -> list[float]:
    """The calibrated probability of each option, in the question's order, summing to 1.

    Each option's log-probability plus its slot's bias, divided by the kind's
    temperature, then a softmax over the question's options. An option the engine did
    not list has no mass.
    """
    kind = model_kind(question)
    kind_parameters = parameters.kinds[kind]
    first = kind_parameters.first_slot
    count = slot_count(question)
    logprobs = _option_logprobs(engine_answer)
    logits = []
    for offset in range(count):
        token_id = parameters.verbalizer_token_ids[first + offset]
        if token_id not in logprobs:
            logits.append(-math.inf)
            continue
        logits.append((logprobs[token_id] + parameters.bias[first + offset]) / kind_parameters.temperature)
    peak = max(logits)
    if peak == -math.inf:
        raise EngineAnswerError("the model's answer lists none of the question's option tokens")
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def answer(
    question: PredicateQuestion | ChoiceQuestion | ScoreQuestion, probabilities: list[float]
) -> dict[str, Any]:
    """One answer in the Decisions API's shape."""
    named: dict[str, Any] = {"name": question.name} if question.name is not None else {}
    if isinstance(question, PredicateQuestion):
        return {**named, "type": "predicate", "probability": probabilities[1]}
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if isinstance(question, ChoiceQuestion):
        return {
            **named,
            "type": "choice",
            "choice": question.choices[best].value,
            "confidence": probabilities[best],
            "probabilities": [
                {"value": choice.value, "probability": probability}
                for choice, probability in zip(question.choices, probabilities, strict=True)
            ],
        }
    return {
        **named,
        "type": "score",
        "score": sum(index * probability for index, probability in enumerate(probabilities)),
        "confidence": probabilities[best],
        "probabilities": [
            {"label": level.label, "value": index, "probability": probability}
            for index, (level, probability) in enumerate(zip(question.levels, probabilities, strict=True))
        ],
    }


# --- asking the questions ---------------------------------------------------------------

class EngineReply:
    """One engine answer: its HTTP status and its body, as the engine sent them."""

    def __init__(self, status_code: int, body: bytes) -> None:
        self.status_code = status_code
        self.body = body




class DecisionFailure(Exception):
    """A question could not be answered; ``status`` is the HTTP status to refuse with.

    The message is complete: which question, and the engine's own status and body when
    the engine refused.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


async def answer_decisions(
    request: DecisionRequest, parameters: DecisionParameters, ask: Any
) -> dict[str, Any]:
    """The Decisions API answer to ``request``: every question asked in parallel.

    Fail-fast on purpose (``asyncio.gather`` without ``return_exceptions``): an answer
    with one question missing is not a Decisions API answer, so the first failure
    refuses the request and the remaining questions are cancelled.
    """

    async def one(index: int) -> tuple[dict[str, Any], dict[str, int]]:
        question = request.questions[index]
        label = f"question {index}" + (f" ({question.name!r})" if question.name else "")
        reply = await ask(engine_request(request, question, parameters))
        if reply.status_code >= 400:
            raise DecisionFailure(
                502,
                f"The model refused {label} with HTTP {reply.status_code}: "
                f"{reply.body.decode('utf-8', errors='replace')[:2000]}",
            )
        try:
            engine_answer = json.loads(reply.body)
        except ValueError as error:
            raise DecisionFailure(502, f"The model's answer to {label} is not JSON ({error})") from error
        try:
            probabilities = option_probabilities(question, parameters, engine_answer)
        except EngineAnswerError as error:
            raise DecisionFailure(502, f"The model's answer to {label} is unusable: {error}") from error
        usage = engine_answer.get("usage") if isinstance(engine_answer, dict) else None
        return answer(question, probabilities), usage if isinstance(usage, dict) else {}

    results = await asyncio.gather(*(one(index) for index in range(len(request.questions))))
    input_tokens = sum(int(usage.get("prompt_tokens", 0)) for _, usage in results)
    output_tokens = sum(int(usage.get("completion_tokens", 0)) for _, usage in results)
    return {
        "model": request.model,
        "answers": [result for result, _ in results],
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        },
    }


# --- the middleware -------------------------------------------------------------------

logger = logging.getLogger("decisions_middleware")

DECISIONS_PATH = "/v1/decisions"
#: One question's engine request: a single forward pass, so a minute is generous.
ENGINE_TIMEOUT_SECONDS = 60.0

_parameters: DecisionParameters | None = None
_client: httpx.AsyncClient | None = None


def load_decision_parameters(module: str, parameters_file: Path) -> DecisionParameters:
    """The decision parameters: the profile's parameters file and the module name."""
    document = json.loads(parameters_file.read_text(encoding="utf-8"))
    return DecisionParameters.model_validate(
        {
            "module": module,
            "prompt_protocol": document.get("prompt_protocol"),
            "verbalizers": document.get("verbalizers"),
            "verbalizer_token_ids": document.get("verbalizer_token_ids"),
            "bias": document.get("bias"),
            "kinds": document.get("kinds"),
        }
    )


def _configured_parameters() -> DecisionParameters:
    global _parameters
    if _parameters is None:
        module = os.environ.get("DECISIONS_MODULE", "")
        parameters_file = os.environ.get("DECISIONS_PARAMETERS_FILE", "")
        if not module or not parameters_file:
            raise RuntimeError(
                "DECISIONS_MODULE and DECISIONS_PARAMETERS_FILE must name the decision module "
                "and the decision parameters file in the container's environment"
            )
        _parameters = load_decision_parameters(module, Path(parameters_file))
    return _parameters


def _engine_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=os.environ.get("DECISIONS_ENGINE_URL", "http://127.0.0.1:8080"),
            timeout=httpx.Timeout(ENGINE_TIMEOUT_SECONDS),
        )
    return _client


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    """An error in the OpenAI shape, the message complete."""
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "param": None, "code": error_type}},
        status_code=status,
    )


async def _ask_engine(body: dict[str, Any]) -> EngineReply:
    response = await _engine_client().post("/v1/chat/completions", json=body)
    return EngineReply(response.status_code, response.content)


async def answer_decision_request(raw_body: bytes) -> Response:
    """One ``POST /v1/decisions`` request, answered or refused with the reason."""
    try:
        payload = json.loads(raw_body)
    except ValueError as error:
        return _error(400, f"The request body is not JSON ({error})", "invalid_request_error")
    try:
        request = parse_decision_request(payload)
    except ValueError as error:
        return _error(400, str(error), "invalid_request_error")
    try:
        parameters = _configured_parameters()
    except (OSError, ValueError, RuntimeError) as error:
        logger.error("The decision parameters cannot be loaded (%s): %s", type(error).__name__, error)
        return _error(
            500,
            f"This model's decision parameters cannot be loaded ({type(error).__name__}): {error}",
            "server_error",
        )
    try:
        result = await answer_decisions(request, parameters, _ask_engine)
    except DecisionFailure as error:
        logger.warning("A decision request failed: %s", error)
        return _error(error.status, str(error), "server_error")
    except httpx.HTTPError as error:
        logger.warning("A decision request could not reach the engine (%s): %s", type(error).__name__, error)
        return _error(
            502,
            f"The decision request could not reach the model's engine ({type(error).__name__}): {error}",
            "server_error",
        )
    return JSONResponse(result)


async def decisions(request: Request, call_next: Any) -> Response:
    """vLLM's ``--middleware`` hook: ``POST /v1/decisions`` here, everything else to vLLM."""
    if request.url.path == DECISIONS_PATH and request.method == "POST":
        return await answer_decision_request(await request.body())
    return await call_next(request)
