"""Turn a sampled token stream into OpenAI-style SSE chunks.

A reply is streamed in three stages, dispatched by :meth:`ChatStreamer.stream`:

* :meth:`ChatStreamer.streaming_think` - ``reasoning_content`` deltas until the thinking
  block closes;
* :meth:`ChatStreamer.streaming_content` - ``content`` deltas until a tool call starts;
* :meth:`ChatStreamer.streaming_tools` - ``tool_calls`` deltas until the call closes.

Decoded pieces are folded into the visible content by :func:`fold_body`, which skips
complete tool-call blocks and holds back a trailing partial marker, so a client never
sees half a tag.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field


def sse(payload: dict) -> str:
    """Serialise one chunk as a server-sent event."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def chunk(request, session_id, delta: dict, finish_reason=None, usage=None) -> dict:
    """Build one OpenAI-style streaming chunk."""
    payload = {
        "id": str(session_id),
        "object": "chat.completion.chunk",
        "created": 0,
        "model": request.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def parse_tool_calls(text: str, tool_pattern: re.Pattern) -> tuple[str, list[dict]]:
    matches = list(tool_pattern.finditer(text))
    if not matches:
        return text, []

    tool_calls = []
    try:
        for index, match in enumerate(matches):
            payload = json.loads(match.group(1))
            name = payload["name"]
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(name, str) or not isinstance(arguments, (dict, list)):
                raise ValueError("Invalid tool call payload")  # noqa: TRY004
            arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            # deterministic id: the streaming path and the session bookkeeping parse the same
            # reply separately and must agree on the ids they report
            fingerprint = hashlib.sha1(f"{index}:{name}:{arguments_json}".encode()).hexdigest()[:24]
            tool_calls.append(
                {
                    "id": f"call_{fingerprint}",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_json},
                }
            )
        content = tool_pattern.sub("", text).strip()
        return content, tool_calls
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return text, []


def parse_thinking(text: str, thinking_pattern: re.Pattern) -> tuple[str | None, str | None]:
    try:
        match = next(thinking_pattern.finditer(text))
    except StopIteration:
        return "", text

    payload = match.group(1)
    if payload is not None:
        return thinking_pattern.sub("", text, count=1), payload
    else:
        # thinking pattern is not closed or not properly formatted, assume all content is reasoning content to make sure prefix consistency for chat template
        return "", text


def tool_pattern_markers(tool_pattern: re.Pattern | None) -> tuple[str, str]:
    """The literal open/close markers around the capture group of a tool
    pattern.

    "<tool_call>(.*?)</tool_call>" -> ("<tool_call>", "</tool_call>")
    """
    if not tool_pattern:
        return "", ""
    open_marker, _, rest = tool_pattern.pattern.partition("(")
    close_marker = rest.rpartition(")")[2] if ")" in rest else ""
    return open_marker, close_marker


def decode_tokens(tokenizer, ids: list[int]) -> str:
    """Decode ids, dropping special tokens so the stop marker never leaks into
    the stream."""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:  # a tokenizer without skip_special_tokens
        return tokenizer.decode(ids)


def take_text(tokenizer, pending: list[int]) -> str:
    """Decode the tokens buffered since the last step and clear the buffer.

    Only the new tokens are decoded on each step, so streaming stays linear in the number of tokens instead of re-
    decoding the whole answer.
    """
    if not pending:
        return ""
    text = decode_tokens(tokenizer, pending)
    pending.clear()
    return text


def split_thinking(text: str, enable_thinking: bool, thinking_pattern: re.Pattern | None) -> tuple[str, str]:
    """Split raw text into (body, reasoning) with the configured thinking
    pattern."""
    if not enable_thinking or not thinking_pattern:
        return text, ""
    body, reasoning = parse_thinking(text, thinking_pattern)
    return body, reasoning or ""


@dataclass
class StreamState:
    """Mutable text state of one SSE stream."""

    emitted: str = ""
    pending: list[int] = field(default_factory=list)
    body: str = ""
    content: str = ""
    folded: int = 0
    tool_call_start: int = 0
    in_tool_call: bool = False
    thinking_end: bool = False
    reasoning_sent: int = 0
    content_sent: int = 0
    tool_calls_sent: int = 0


def tool_block_is_valid(block: str, tool_pattern: re.Pattern | None) -> bool:
    """True when every tool-call block in ``block`` is well-formed JSON.

    Parsing is fail-open elsewhere: a malformed block must stay in the visible content
    (markers included) instead of being dropped, so the client still sees what the model
    said and can retry.
    """
    if not tool_pattern:
        return False
    matches = list(tool_pattern.finditer(block))
    if not matches:
        return False
    for match in matches:
        try:
            payload = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return False
    return True


def fold_body(
    state: StreamState, tool_open: str, tool_close: str, hold: int, tool_pattern: re.Pattern | None = None
) -> bool:
    """Fold new body text into content, skipping tool-call blocks.

    Holds back a trailing partial marker so no half tag reaches the client. Returns True when a tool call just closed.
    """
    closed_a_call = False
    while True:
        if state.in_tool_call:
            end_at = state.body.find(tool_close, state.folded) if tool_close else -1
            if end_at < 0:
                break  # not find
            else:
                block_end = end_at + len(tool_close)
                block = state.body[state.tool_call_start : block_end]
                if tool_block_is_valid(block, tool_pattern):
                    state.folded = block_end
                    state.in_tool_call = False
                    closed_a_call = True
                    continue
                # malformed block: keep it verbatim in the content, markers included
                state.content += block
                state.folded = block_end
                state.in_tool_call = False
                continue
        open_at = state.body.find(tool_open, state.folded) if tool_open else -1
        if open_at >= 0:
            state.content += state.body[state.folded : open_at]
            state.tool_call_start = open_at
            state.folded = open_at + len(tool_open)
            state.in_tool_call = True
            continue
        safe = max(state.folded, len(state.body) - hold)
        state.content += state.body[state.folded : safe]
        state.folded = safe
        break
    return closed_a_call


class ChatStreamer:
    """Stream one request's assistant turn as OpenAI-style SSE chunks."""

    def __init__(self, tokenizer, *, thinking_pattern, tool_pattern, enable_thinking):
        self.tokenizer = tokenizer
        self.thinking_pattern = thinking_pattern
        self.tool_pattern = tool_pattern
        self.tool_open, self.tool_close = tool_pattern_markers(tool_pattern)
        self.hold = max(len(self.tool_open) - 1, 0)
        self.state = StreamState(thinking_end=not enable_thinking)

    # chunks

    def _content_delta(self) -> dict | None:
        """Delta with the content that arrived since the last one."""
        state = self.state
        if len(state.content) <= state.content_sent:
            return None
        delta = state.content[state.content_sent :]
        state.content_sent = len(state.content)
        return {"content": delta}

    def _tool_call_deltas(self) -> list[dict]:
        """Deltas for the tool calls that closed since the last call."""
        state = self.state
        _, tool_calls = parse_tool_calls(state.body, self.tool_pattern)
        if len(tool_calls) <= state.tool_calls_sent:
            return []
        delta_calls = [
            {
                "index": index,
                "id": call["id"],
                "type": "function",
                "function": call["function"],
            }
            for index, call in enumerate(tool_calls[state.tool_calls_sent :], start=state.tool_calls_sent)
        ]
        state.tool_calls_sent = len(tool_calls)
        return [{"tool_calls": delta_calls}]

    # stages

    async def _next_piece(self, stream_queue, task) -> str | None:
        """Next decoded text piece, or None once the producer is finished."""
        state = self.state
        while True:
            if task.done() and stream_queue.empty():
                return None
            try:
                token = await asyncio.wait_for(stream_queue.get(), timeout=0.02)
            except TimeoutError:
                continue
            state.pending.append(token)
            piece = take_text(self.tokenizer, state.pending)
            if piece:
                return piece

    def streaming_think(self, piece: str):
        """Reasoning stage: consume one piece as reasoning_content."""
        state = self.state
        state.emitted += piece
        body, reasoning = split_thinking(state.emitted, True, self.thinking_pattern)
        delta = reasoning[state.reasoning_sent :]
        state.reasoning_sent = len(reasoning)
        if delta:
            yield {"reasoning_content": delta}
        if body:
            # thinking closed: the body is a suffix of the raw text from now on
            state.body = body
            state.thinking_end = True

    def streaming_content(self, piece: str):
        """Content stage: consume one piece of text outside any tool call."""
        state = self.state
        state.body += piece
        closed_a_call = fold_body(state, self.tool_open, self.tool_close, self.hold, self.tool_pattern)
        content_delta = self._content_delta()
        if content_delta:
            yield content_delta
        if closed_a_call:
            yield from self._tool_call_deltas()

    def streaming_tools(self, piece: str):
        """Tool-call stage: consume one piece from the opening to the closing marker."""
        state = self.state
        state.body += piece
        closed_a_call = fold_body(state, self.tool_open, self.tool_close, self.hold, self.tool_pattern)
        if closed_a_call:
            yield from self._tool_call_deltas()
        content_delta = self._content_delta()  # text released after the call
        if content_delta:
            yield content_delta

    async def stream(self, stream_queue, task):
        """Pull pieces and dispatch each one to the stage it currently belongs
        to."""
        state = self.state
        while True:
            piece = await self._next_piece(stream_queue, task)
            if piece is None:
                break
            if not state.thinking_end:
                stage = self.streaming_think
            elif state.in_tool_call:
                stage = self.streaming_tools
            else:
                stage = self.streaming_content
            for payload in stage(piece):
                yield payload

        # flush the text held back as a possible partial marker; a tool call that never
        # closed is not a call at all, so its text is emitted as content
        tail = take_text(self.tokenizer, state.pending)
        if tail:
            state.body += tail
        if state.in_tool_call:
            state.content += state.body[state.tool_call_start :]
        else:
            state.content += state.body[state.folded :]
        state.folded = len(state.body)
        state.in_tool_call = False
        content_delta = self._content_delta()
        if content_delta:
            yield content_delta
