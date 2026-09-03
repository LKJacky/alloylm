"""OpenAI-compatible tool support for agent environments.

:class:`BaseEnv` is the base class for tool-bearing environments. Methods
marked with :meth:`BaseEnv.node_tool` are exposed as function-calling tools:
:meth:`BaseEnv.tools` builds the OpenAI ``tools=`` payload (each JSON schema
is inferred from the method's type hints unless given explicitly in the
decorator), :meth:`BaseEnv.call_tool` dispatches a returned tool call, and
:meth:`BaseEnv.execute` runs the tool calls of an assistant message.
"""

import inspect
import json
from collections.abc import Callable
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from openai.types.chat import ChatCompletionMessage

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation) -> dict:
    """Map a type hint to a JSON-schema ``{"type": ...}`` entry."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        types = []
        for part in get_args(annotation):
            if part is type(None):
                continue
            json_type = _json_type(part)["type"]
            types.extend(json_type if isinstance(json_type, list) else [json_type])
        types = sorted(set(types))
        return {"type": types[0] if len(types) == 1 else types}
    if origin is list:
        return {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    if annotation in (Any, object):
        return {"type": ["string", "number", "boolean", "object", "array"]}
    return {"type": _JSON_TYPES.get(annotation, "string")}


def _infer_parameters(fn) -> dict:
    """Build a JSON-schema object for a method from its signature and type
    hints.

    Parameters without a default are required; ``self``/``cls`` and
    var-args are skipped.
    """
    try:
        hints = get_type_hints(fn)
    except (NameError, TypeError):
        hints = {}
    properties = {}
    required = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _json_type(hints.get(name, param.empty))
        if param.default is param.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class BaseEnv:
    """Base class for tool-bearing environments.

    Subclasses mark their operations with :meth:`node_tool`; the tools are
    then driveable through :meth:`tools`, :meth:`call_tool`, and
    :meth:`execute`.
    """

    def __init__(self, expose_get_tool_definitions: bool = False):
        if expose_get_tool_definitions:
            self.get_tool_definitions = BaseEnv.node_tool(description="Get tool definitions")(
                self.get_tool_definitions
            )

    @classmethod
    async def create_env(cls, **kwargs):
        """Async factory returning ``cls(**kwargs)``."""
        return cls(**kwargs)

    async def execute(self, messages: ChatCompletionMessage) -> str:
        """Run the tool calls of an assistant message against this env.

        Each ``tool_calls`` entry is dispatched through :meth:`execute_call`
        and the results are joined into one string; a message without tool
        calls returns its content.
        """
        results = [self.execute_call(call) for call in messages.tool_calls or []]
        return "\n".join(results) if results else (messages.content or "")

    def execute_call(self, call) -> str:
        """Dispatch a single tool call and return its result string."""
        name = call.function.name
        arguments = call.function.arguments
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return f"Invalid arguments for {name}: {arguments}"
        if not isinstance(parsed, dict):
            return f"Invalid arguments for {name}: expected a JSON object"
        return self.call_tool(name, parsed)

    async def close(self):
        pass

    @staticmethod
    def node_tool(description: str | Callable[[Any], str], parameters: dict | None = None):
        """Mark a method as a node tool.

        The capability schema is inferred from the method's type hints unless
        ``parameters`` (a JSON-schema object) is given explicitly.
        ``description`` may be a callable taking the env instance and
        returning the description string, resolved by :meth:`tools` (useful
        for mentioning per-instance values such as a unit size).
        """

        def deco(fn):
            fn._node_tool = {"description": description, "parameters": parameters}
            return fn

        return deco

    def tools(self) -> list[dict]:
        """Return OpenAI-compatible tool definitions for every method of this
        env registered with :meth:`node_tool`.

        Parameter schemas are inferred from the methods' type hints unless
        given explicitly in the decorator; callable descriptions are
        resolved against ``self``. The class MRO is walked so subclasses
        inherit tools and can override them.
        """
        decorated = {}
        for base in reversed(type(self).__mro__):
            for name, value in vars(base).items():
                tool = getattr(value, "_node_tool", None)
                if tool is not None:
                    decorated[name] = tool
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        tool["description"](self) if callable(tool["description"]) else tool["description"]
                    ),
                    "parameters": tool["parameters"] or _infer_parameters(getattr(type(self), name)),
                },
            }
            for name, tool in decorated.items()
        ]

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-compatible tool definitions for this env.

        This is a convenience wrapper around :meth:`tools` that can be
        overridden to add extra tools or modify the definitions.
        """
        return self.tools()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a returned OpenAI tool call (function ``name`` + parsed
        ``arguments``) to the matching operation and return its result string.

        Unknown tools or bad arguments return an error string instead of raising, matching the tool-facing contract.
        """
        tool = getattr(self, name, None)
        if tool is None:
            return f"Unknown tool {name}"
        if not isinstance(arguments, dict):
            return f"Invalid arguments for {name}: expected a JSON object"
        try:
            result = tool(**arguments)
        except Exception as e:  # noqa
            return f"Error calling {name}: {e}"
        return result if isinstance(result, str) else str(result)
