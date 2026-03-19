#!/usr/bin/env python
"""Tests for serialization of non-JSON-serializable args (e.g. enums).

Reproduces the bug in https://github.com/ag-ui-protocol/ag-ui/issues/1331
where SecuritySchemeType enum in adk_request_credential args caused
TypeError: Object of type SecuritySchemeType is not JSON serializable.
"""

import asyncio
import enum
import json
from unittest.mock import MagicMock

from ag_ui.core import EventType
from ag_ui_adk import EventTranslator
from ag_ui_adk.event_translator import _translate_function_calls_to_tool_calls


class SecuritySchemeType(str, enum.Enum):
    """Simulates the google.adk SecuritySchemeType enum."""
    oauth2 = "oauth2"
    apiKey = "apiKey"
    http = "http"


async def test_lro_function_call_with_enum_args():
    """translate_lro_function_calls should handle enum values in args."""
    translator = EventTranslator()

    adk_event = MagicMock()
    adk_event.author = "assistant"
    adk_event.partial = False
    adk_event.content = MagicMock()

    lro_call = MagicMock()
    lro_call.id = "cred-call-1"
    lro_call.name = "adk_request_credential"
    lro_call.args = {
        "auth_config": {
            "auth_scheme": {
                "type_": SecuritySchemeType.oauth2,
                "flows": {"implicit": {"authorizationUrl": "https://example.com/auth"}},
            }
        }
    }

    part = MagicMock()
    part.function_call = lro_call

    adk_event.content.parts = [part]
    adk_event.get_function_calls = lambda: [lro_call]
    adk_event.long_running_tool_ids = ["cred-call-1"]

    events = []
    async for e in translator.translate_lro_function_calls(adk_event):
        events.append(e)

    event_types = [e.type for e in events]
    assert EventType.TOOL_CALL_START in event_types
    assert EventType.TOOL_CALL_ARGS in event_types
    assert EventType.TOOL_CALL_END in event_types

    # Verify the args were serialized without error and contain the enum value
    args_event = [e for e in events if e.type == EventType.TOOL_CALL_ARGS][0]
    parsed = json.loads(args_event.delta)
    assert parsed["auth_config"]["auth_scheme"]["type_"] == "oauth2"


async def test_translate_function_calls_with_enum_args():
    """_translate_function_calls should handle enum values in args."""
    translator = EventTranslator()

    func_call = MagicMock()
    func_call.id = "fc-enum-1"
    func_call.name = "some_tool"
    func_call.args = {
        "scheme_type": SecuritySchemeType.apiKey,
        "name": "my-key",
    }

    events = []
    async for e in translator._translate_function_calls([func_call]):
        events.append(e)

    args_event = [e for e in events if e.type == EventType.TOOL_CALL_ARGS][0]
    parsed = json.loads(args_event.delta)
    assert parsed["scheme_type"] == "apiKey"
    assert parsed["name"] == "my-key"


def test_translate_function_calls_to_tool_calls_with_enum_args():
    """_translate_function_calls_to_tool_calls should handle enum values."""
    fc = MagicMock()
    fc.id = "fc-enum-2"
    fc.name = "credential_tool"
    fc.args = {"type": SecuritySchemeType.http, "count": 42}

    tool_calls = _translate_function_calls_to_tool_calls([fc])
    assert len(tool_calls) == 1
    parsed = json.loads(tool_calls[0].function.arguments)
    assert parsed["type"] == "http"
    assert parsed["count"] == 42
