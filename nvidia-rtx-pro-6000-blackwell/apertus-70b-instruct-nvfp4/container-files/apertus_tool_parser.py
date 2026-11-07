# Apertus tool-call parser plugin for vLLM.
# Based on https://github.com/vllm-project/vllm/pull/41154 (Infomaniak).
# Loaded at runtime via --tool-parser-plugin; registers as "apertus".
#
# Apertus tool call format:
#   <|tools_prefix|>[{"function_name": {"arg": "value"}}]<|tools_suffix|>

import json
from collections.abc import Sequence

import regex as re
from partial_json_parser.core.options import Allow
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
    ToolParserManager,
)
from vllm.tool_parsers.utils import (
    find_common_prefix,
    partial_json_loads,
)

logger = init_logger(__name__)

TOOL_CALLS_PREFIX = "<|tools_prefix|>"
TOOL_CALLS_SUFFIX = "<|tools_suffix|>"


class ApertusToolParser(ToolParser):

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_regex = re.compile(
            rf"{re.escape(TOOL_CALLS_PREFIX)}"
            rf"(.*?)"
            rf"(?:{re.escape(TOOL_CALLS_SUFFIX)}|$)",
            re.DOTALL,
        )
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self.buffered_delta_text = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool: list[str] = []

    def adjust_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Surface special tokens so we can detect tool call boundaries
            request.skip_special_tokens = False
        return request

    def _buffer_delta_text(self, delta_text: str) -> str:
        self.buffered_delta_text += delta_text
        text = self.buffered_delta_text

        for tag in (TOOL_CALLS_PREFIX, TOOL_CALLS_SUFFIX):
            if text.endswith(tag):
                self.buffered_delta_text = ""
                return text

            for i in range(len(tag) - 1, 0, -1):
                if text.endswith(tag[:i]):
                    self.buffered_delta_text = text[-i:]
                    return text[:-i]

        self.buffered_delta_text = ""
        return text

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        match = self.tool_call_regex.search(model_output)
        if not match:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matched_text = match.group(1)
            stripped_text = matched_text.lstrip()

            try:
                parsed_json, idx = json.JSONDecoder().raw_decode(stripped_text)
                trailing_in_group = stripped_text[idx:]
            except json.JSONDecodeError:
                parsed_json, _ = partial_json_loads(matched_text, Allow.ALL)
                trailing_in_group = ""

            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json] if parsed_json else []

            tool_calls: list[ToolCall] = []
            for obj in parsed_json:
                if isinstance(obj, dict) and obj:
                    name, args = next(iter(obj.items()))
                    tool_calls.append(
                        ToolCall(
                            type="function",
                            id=make_tool_call_id(),
                            function=FunctionCall(
                                name=name,
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    )

            content_str = model_output[: match.start()].strip()

            if trailing_in_group.strip():
                trailing = trailing_in_group.replace(TOOL_CALLS_SUFFIX, "").strip()
                if trailing:
                    content_str = (content_str + "\n" + trailing).strip()

            after_suffix = (
                model_output[match.end() :].replace(TOOL_CALLS_SUFFIX, "").strip()
            )
            if after_suffix:
                content_str = (content_str + "\n" + after_suffix).strip()

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content_str if content_str else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Apertus response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        delta_text = self._buffer_delta_text(delta_text)
        if not delta_text:
            return None

        if TOOL_CALLS_PREFIX not in current_text:
            return DeltaMessage(content=delta_text)

        try:
            return self._extract_streaming(current_text, delta_text)
        except Exception:
            logger.exception("Error in Apertus streaming tool call extraction")
            return None

    def _extract_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        prefix_idx = current_text.rfind(TOOL_CALLS_PREFIX)
        suffix_idx = current_text.rfind(TOOL_CALLS_SUFFIX)

        is_inside_tools = prefix_idx > suffix_idx

        json_completed = False
        json_end_idx: int | None = None

        if is_inside_tools:
            json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
            s = current_text[json_start:].lstrip()
            try:
                _, idx = json.JSONDecoder().raw_decode(s)
                json_end_idx = len(current_text) - len(s) + idx
                json_completed, is_inside_tools = True, False
            except Exception:
                pass

        just_finished = (TOOL_CALLS_SUFFIX in delta_text) or json_completed

        if not is_inside_tools and not just_finished:
            text = delta_text.replace(TOOL_CALLS_PREFIX, "").replace(
                TOOL_CALLS_SUFFIX, ""
            )
            return DeltaMessage(content=text) if text else None

        content_str = ""
        if TOOL_CALLS_PREFIX in delta_text:
            content_str += delta_text.split(TOOL_CALLS_PREFIX)[0].replace(
                TOOL_CALLS_SUFFIX, ""
            )

        if just_finished:
            if json_completed and json_end_idx is not None:
                delta_start_idx = len(current_text) - len(delta_text)
                content_start = max(json_end_idx, delta_start_idx)
                if content_start < len(current_text):
                    content_str += current_text[content_start:].replace(
                        TOOL_CALLS_SUFFIX, ""
                    )
            else:
                content_str += delta_text.split(TOOL_CALLS_SUFFIX)[-1]

        json_start = prefix_idx + len(TOOL_CALLS_PREFIX)
        json_end = suffix_idx if suffix_idx > prefix_idx else json_end_idx
        json_str = current_text[json_start:json_end]

        tool_calls = self._parse_and_diff_json(json_str, is_final=not is_inside_tools)

        if tool_calls or content_str:
            return DeltaMessage(
                content=content_str if content_str else None,
                tool_calls=tool_calls if tool_calls else None,
            )

        return None

    def _parse_and_diff_json(
        self, json_str: str, is_final: bool
    ) -> list[DeltaToolCall]:
        try:
            parsed, _ = partial_json_loads(json_str, Allow.ALL)
            if not isinstance(parsed, list):
                parsed = [parsed] if parsed else []
        except Exception:
            return []

        if not parsed:
            return []

        tool_calls: list[DeltaToolCall] = []
        latest_index = len(parsed) - 1

        while self.current_tool_id < latest_index:
            if self.current_tool_id >= 0:
                if not self.current_tool_name_sent:
                    self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

                delta = self._get_tool_diff(
                    parsed, self.current_tool_id, is_final=True
                )
                if delta:
                    tool_calls.append(delta)

            self.current_tool_id += 1
            self.current_tool_name_sent = False
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

        if self.current_tool_id >= 0:
            if not self.current_tool_name_sent:
                self._emit_tool_name(parsed, self.current_tool_id, tool_calls)

            delta = self._get_tool_diff(parsed, self.current_tool_id, is_final)
            if delta:
                tool_calls.append(delta)

        return tool_calls

    def _emit_tool_name(
        self, parsed: list, index: int, tool_calls: list[DeltaToolCall]
    ) -> None:
        obj = parsed[index]
        if isinstance(obj, dict) and obj:
            name = next(iter(obj))
            self.current_tool_name_sent = True
            tool_calls.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(name=name, arguments="").model_dump(
                        exclude_none=True
                    ),
                )
            )

    def _get_tool_diff(
        self, parsed: list, index: int, is_final: bool
    ) -> DeltaToolCall | None:
        obj = parsed[index]
        if not isinstance(obj, dict) or not obj:
            return None

        _name, args = next(iter(obj.items()))
        if args is None:
            return None

        args_json = json.dumps(args, ensure_ascii=False)

        if not is_final:
            while args_json and args_json[-1] in ("}", '"', "]", " ", ","):
                args_json = args_json[:-1]

        prev_sent = self.streamed_args_for_tool[index]
        if args_json == prev_sent:
            return None

        prefix = find_common_prefix(prev_sent, args_json)
        if len(prefix) < len(prev_sent):
            self.streamed_args_for_tool[index] = prefix
            return None

        diff = args_json[len(prev_sent):]
        if diff:
            self.streamed_args_for_tool[index] = args_json
            return DeltaToolCall(
                index=index,
                function=DeltaFunctionCall(arguments=diff).model_dump(
                    exclude_none=True
                ),
            )

        return None


# Eager registration -- makes "apertus" available to --tool-call-parser
ToolParserManager.register_module(module=ApertusToolParser, name="apertus")
