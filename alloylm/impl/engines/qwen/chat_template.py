"""Qwen engine helpers and chat templates."""

import json
from collections.abc import Mapping

QWEN_TOOL_PATTERN = r"<tool_call>\s*(.*?)\s*</tool_call>"


def _get(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def json2str(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class Qwen3ChatTemplate:
    # this is a modified qwen3 chat template. The main difference is that the template do not drop past reasoning content, and it will keep all the reasoning content in the final output.
    @classmethod
    def render(
        cls,
        messages,
        tools=None,
        add_generation_prompt=False,
        enable_thinking=False,
        **_kwargs,
    ):
        output = []

        if tools:
            output.append("<|im_start|>system\n")
            if _get(messages[0], "role") == "system":
                output.append(_get(messages[0], "content") + "\n\n")
            output.append(
                "# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n"
                "<tools>"
            )
            for tool in tools:
                output.append(f"\n{json2str(tool)}")
            output.append(
                "\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags:\n"
                '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call><|im_end|>\n"
            )
        elif _get(messages[0], "role") == "system":
            output.append(f"<|im_start|>system\n{_get(messages[0], 'content')}<|im_end|>\n")

        # last_query_index = len(messages) - 1
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            content = _get(message, "content")
            if (
                _get(message, "role") == "user"
                and isinstance(content, str)
                and not (content.startswith("<tool_response>") and content.endswith("</tool_response>"))
            ):
                # last_query_index = index
                break

        for index, message in enumerate(messages):
            role = _get(message, "role")
            raw_content = _get(message, "content")
            content = raw_content if isinstance(raw_content, str) else ""

            if role == "user" or (role == "system" and index != 0):
                output.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            elif role == "assistant":
                reasoning_content = _get(message, "reasoning_content")
                ###############################################################################
                # if not isinstance(reasoning_content, str):
                #     reasoning_content = ""
                #     if "</think>" in content:
                #         reasoning_content = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                #         content = content.split("</think>")[-1].lstrip("\n")

                # if index > last_query_index and (index == len(messages) - 1 or reasoning_content):
                #     output.append(
                #         "<|im_start|>assistant\n<think>\n"
                #         f"{reasoning_content.strip(chr(10))}\n</think>\n\n{content.lstrip(chr(10))}"
                #     )
                # else:
                #     output.append(f"<|im_start|>assistant\n{content}")

                if reasoning_content:
                    output.append(f"<|im_start|>assistant\n<think>\n\n{reasoning_content}</think>{content}")
                else:
                    output.append(f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}")

                ###############################################################################

                tool_calls = _get(message, "tool_calls")
                if tool_calls:
                    for call_index, tool_call in enumerate(tool_calls):
                        if (call_index == 0 and content) or call_index != 0:
                            output.append("\n")
                        function = _get(tool_call, "function")
                        if function:
                            tool_call = function
                        arguments = _get(tool_call, "arguments")
                        if not isinstance(arguments, str):
                            arguments = json2str(arguments)
                        output.append(
                            f'<tool_call>\n{{"name": "{_get(tool_call, "name")}", '
                            f'"arguments": {arguments}}}\n</tool_call>'
                        )
                output.append("<|im_end|>\n")
            elif role == "tool":
                if index == 0 or _get(messages[index - 1], "role") != "tool":
                    output.append("<|im_start|>user")
                output.append(f"\n<tool_response>\n{content}\n</tool_response>")
                if index == len(messages) - 1 or _get(messages[index + 1], "role") != "tool":
                    output.append("<|im_end|>\n")

        if add_generation_prompt:
            output.append("<|im_start|>assistant\n")
            if enable_thinking is False:
                output.append("<think>\n\n</think>\n\n")
            # + ###########################################################
            else:
                output.append("<think>\n\n")
            ###############################################################

        return "".join(output)


"""
qwen3 official chat template
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
"""
