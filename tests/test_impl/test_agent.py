import asyncio
import time
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from alloylm.impl.agent.env import BaseEnv


def tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class FakeEnv(BaseEnv):
    @BaseEnv.node_tool(description="Return a value after a delay")
    def delayed(self, value: int, delay: float = 0):
        time.sleep(delay)
        return value

    @BaseEnv.node_tool(description="Return a value asynchronously")
    async def async_value(self, value: str):
        await asyncio.sleep(0)
        return value


class TestAgent(IsolatedAsyncioTestCase):
    async def test_execute_sync_and_async_tools(self):
        env = FakeEnv()

        self.assertEqual(await env.execute_call(tool_call("delayed", '{"value": 3}')), "3")
        self.assertEqual(await env.execute_call(tool_call("async_value", {"value": "ok"})), "ok")

    async def test_execute_call_reports_invalid_requests(self):
        env = FakeEnv()

        self.assertEqual(await env.execute_call(tool_call("missing", {})), "Unknown tool missing")
        self.assertEqual(
            await env.execute_call(tool_call("delayed", "not-json")),
            "Invalid arguments for delayed: not-json",
        )
        self.assertEqual(
            await env.execute_call(tool_call("delayed", "[]")),
            "Invalid arguments for delayed: expected a JSON object",
        )

    async def test_long_sync_tool_warns_without_being_cancelled(self):
        env = FakeEnv(exec_time_warning_threshold=0.01)

        with patch("alloylm.impl.agent.env.get_logger") as get_logger:
            result = await env.execute_call(tool_call("delayed", {"value": 7, "delay": 0.03}))

        self.assertEqual(result, "7")
        get_logger.return_value.warning.assert_called()

    def test_tools_infer_parameters(self):
        tools = {tool["function"]["name"]: tool["function"] for tool in FakeEnv().tools()}

        self.assertEqual(tools["delayed"]["parameters"]["required"], ["value"])
        self.assertEqual(tools["delayed"]["parameters"]["properties"]["value"], {"type": "integer"})
        self.assertEqual(tools["async_value"]["parameters"]["properties"]["value"], {"type": "string"})
