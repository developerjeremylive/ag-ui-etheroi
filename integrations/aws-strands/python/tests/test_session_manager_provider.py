"""Tests for session_manager_provider behavior in StrandsAgent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ag_ui_strands.agent import StrandsAgent
from ag_ui_strands.config import StrandsAgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_input(thread_id: str | None = "thread-1", run_id: str = "run-1"):
    """Create a minimal mock RunAgentInput."""
    mock = MagicMock()
    mock.thread_id = thread_id
    mock.run_id = run_id
    mock.messages = []
    mock.tools = []
    mock.state = None
    return mock


def _make_template_agent():
    """Create a mock template Strands agent."""
    agent = MagicMock()
    agent.model = MagicMock()
    agent.system_prompt = "You are a helpful assistant."
    agent.tool_registry = MagicMock()
    agent.tool_registry.registry = {}
    agent.record_direct_tool_call = True
    return agent


async def _empty_stream(_msg):
    """Async generator that yields nothing."""
    if False:
        yield


async def _collect(agen):
    """Drain an async generator into a list."""
    return [e async for e in agen]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSessionManagerProvider:

    def _make_agent(self, provider):
        config = StrandsAgentConfig(session_manager_provider=provider)
        return StrandsAgent(agent=_make_template_agent(), name="test-agent", config=config)

    @pytest.mark.asyncio
    async def test_provider_called_for_new_thread(self):
        """Provider is called when a thread_id is first seen, and result is passed to StrandsAgentCore."""
        mock_sm = MagicMock()
        provider = MagicMock(return_value=mock_sm)

        strands_agent = self._make_agent(provider)
        input_data = _make_run_input(thread_id="thread-new")

        with pytest.MonkeyPatch.context() as mp:
            MockCore = MagicMock()
            instance = MagicMock()
            instance.stream_async.side_effect = _empty_stream
            MockCore.return_value = instance
            mp.setattr("ag_ui_strands.agent.StrandsAgentCore", MockCore)

            await _collect(strands_agent.run(input_data))

        provider.assert_called_once_with(input_data)
        _, kwargs = MockCore.call_args
        assert kwargs.get("session_manager") is mock_sm

    @pytest.mark.asyncio
    async def test_provider_not_called_for_existing_thread(self):
        """Provider is NOT called on subsequent requests for the same thread_id."""
        mock_sm = MagicMock()
        provider = MagicMock(return_value=mock_sm)

        strands_agent = self._make_agent(provider)
        input_data = _make_run_input(thread_id="thread-repeat")

        with pytest.MonkeyPatch.context() as mp:
            MockCore = MagicMock()
            instance = MagicMock()
            instance.stream_async.side_effect = _empty_stream
            MockCore.return_value = instance
            mp.setattr("ag_ui_strands.agent.StrandsAgentCore", MockCore)

            await _collect(strands_agent.run(input_data))
            await _collect(strands_agent.run(input_data))

        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_async_provider_is_awaited(self):
        """Async providers are properly awaited via maybe_await."""
        mock_sm = MagicMock()

        async def async_provider(_input):
            return mock_sm

        strands_agent = self._make_agent(async_provider)
        input_data = _make_run_input(thread_id="thread-async")

        with pytest.MonkeyPatch.context() as mp:
            MockCore = MagicMock()
            instance = MagicMock()
            instance.stream_async.side_effect = _empty_stream
            MockCore.return_value = instance
            mp.setattr("ag_ui_strands.agent.StrandsAgentCore", MockCore)

            await _collect(strands_agent.run(input_data))

        _, kwargs = MockCore.call_args
        assert kwargs.get("session_manager") is mock_sm

    @pytest.mark.asyncio
    async def test_provider_exception_yields_run_error(self):
        """Provider exceptions produce RunStartedEvent then RunErrorEvent."""
        from ag_ui.core import EventType

        def failing_provider(_input):
            raise RuntimeError("connection refused")

        strands_agent = self._make_agent(failing_provider)
        input_data = _make_run_input(thread_id="thread-err")

        events = await _collect(strands_agent.run(input_data))
        types = [e.type for e in events]

        assert EventType.RUN_STARTED in types
        assert EventType.RUN_ERROR in types
        assert types.index(EventType.RUN_STARTED) < types.index(EventType.RUN_ERROR)

    @pytest.mark.asyncio
    async def test_no_provider_passes_none_session_manager(self):
        """Without a provider, session_manager=None is passed to StrandsAgentCore."""
        strands_agent = self._make_agent(None)
        input_data = _make_run_input(thread_id="thread-none")

        with pytest.MonkeyPatch.context() as mp:
            MockCore = MagicMock()
            instance = MagicMock()
            instance.stream_async.side_effect = _empty_stream
            MockCore.return_value = instance
            mp.setattr("ag_ui_strands.agent.StrandsAgentCore", MockCore)

            await _collect(strands_agent.run(input_data))

        _, kwargs = MockCore.call_args
        assert kwargs.get("session_manager") is None

    @pytest.mark.asyncio
    async def test_none_thread_id_maps_to_default(self):
        """thread_id=None falls back to 'default', so two calls share the same agent."""
        mock_sm = MagicMock()
        provider = MagicMock(return_value=mock_sm)

        strands_agent = self._make_agent(provider)

        with pytest.MonkeyPatch.context() as mp:
            MockCore = MagicMock()
            instance = MagicMock()
            instance.stream_async.side_effect = _empty_stream
            MockCore.return_value = instance
            mp.setattr("ag_ui_strands.agent.StrandsAgentCore", MockCore)

            await _collect(strands_agent.run(_make_run_input(thread_id=None)))
            await _collect(strands_agent.run(_make_run_input(thread_id=None)))

        # Provider only called once; both calls mapped to "default"
        assert provider.call_count == 1
