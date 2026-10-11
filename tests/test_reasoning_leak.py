"""Pre-tool reasoning text must never become the spoken/visible reply."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from homeassistant.components import conversation
from homeassistant.components.conversation.chat_log import ChatLog
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.grok_conversation.api_helpers import (
    async_chat_completion,
    reasoning_effort_for_model,
)
from custom_components.grok_conversation.const import (
    CONF_CHAT_MODEL,
    CONF_FALLBACK_MODEL,
)
from custom_components.grok_conversation.entity import strip_reasoning_markup

from test_conversation import (  # noqa: E402
    _completion_response,
    _conversation_entity,
    _mock_llm_api,
    _RecordingTool,
    _tool_call,
)

LEAKED = (
    "No response needed, just the tool call. But since it's voice, after tool "
    "it will respond. I need to make the tool call. Call: area=Keittiö, "
    "domain=light, name=tiskialtaan valo Yes"
)


async def _run(hass, entry, client, responses, user_text="Sytytä tiskialtaan valo"):
    tool = _RecordingTool()
    entity = _conversation_entity(hass, entry)
    client.chat.completions.create = AsyncMock(side_effect=responses)
    chat_log = ChatLog(hass=hass, conversation_id="conv-reasoning")
    chat_log.llm_api = _mock_llm_api(hass, tool)
    chat_log.async_add_user_content(conversation.UserContent(content=user_text))
    await entity._async_handle_chat_log(  # noqa: SLF001
        chat_log,
        model="grok-4.3",
        options={CONF_CHAT_MODEL: "grok-4.3", CONF_FALLBACK_MODEL: "grok-4.3"},
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": user_text},
        ],
        agent_id=entity.entity_id,
        service="conversation",
        fallback_model="grok-4.3",
    )
    return entity, chat_log, tool


async def test_pre_tool_reasoning_not_spoken_when_final_empty(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Empty final turn must not fall back to the tool-call turn's text."""
    entity, chat_log, tool = await _run(
        hass,
        mock_config_entry,
        mock_openai_client,
        [
            _completion_response(
                LEAKED, tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            _completion_response(""),
        ],
    )
    assert tool.calls == [{"action": "turn_on"}]
    # Successful action without final text: empty speech, never the
    # pre-tool reasoning and no hardcoded confirmation text.
    assert entity._pick_speech_content(chat_log, "fi") == ""  # noqa: SLF001

    # Kept only as thinking content, never as visible content
    tool_turns = [
        c
        for c in chat_log.content
        if isinstance(c, conversation.AssistantContent) and c.tool_calls
    ]
    assert tool_turns and tool_turns[0].content is None
    assert tool_turns[0].thinking_content == LEAKED

    # Not fed back to the model as something it "said"
    second = mock_openai_client.chat.completions.create.await_args_list[1].kwargs
    for msg in second["messages"]:
        if msg.get("role") == "assistant":
            assert not msg.get("content")


async def test_final_text_is_spoken_not_pre_tool_reasoning(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """With a final reply, only the final reply is spoken."""
    entity, chat_log, _ = await _run(
        hass,
        mock_config_entry,
        mock_openai_client,
        [
            _completion_response(
                LEAKED, tool_calls=[_tool_call("test_light", '{"action":"turn_on"}')]
            ),
            _completion_response("<think>light is on now</think>Valo on päällä."),
        ],
    )
    assert entity._pick_speech_content(chat_log, "fi") == "Valo on päällä."  # noqa: SLF001


async def test_no_speech_from_previous_turn(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """A reply from an earlier user turn is never reused for the current one."""
    entity = _conversation_entity(hass, mock_config_entry)
    chat_log = ChatLog(hass=hass, conversation_id="conv-prev")
    chat_log.async_add_user_content(conversation.UserContent(content="Hei"))
    chat_log.content.append(
        conversation.AssistantContent(agent_id="x", content="Hei vaan!")
    )
    chat_log.async_add_user_content(conversation.UserContent(content="Mitä?"))
    assert entity._pick_speech_content(chat_log, "fi") is None  # noqa: SLF001


def test_strip_reasoning_markup() -> None:
    assert strip_reasoning_markup("<think>a\nb</think> Hello") == "Hello"
    assert strip_reasoning_markup("<THINKING>x</THINKING>Hi") == "Hi"
    assert strip_reasoning_markup("Plain text") == "Plain text"
    assert strip_reasoning_markup("<think>unterminated") == ""
    assert strip_reasoning_markup(None) == ""


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("grok-4.3", "low", "low"),
        ("grok-4.3-latest", "none", "none"),
        ("grok-4.7", "high", "high"),
        ("grok-4.7", "none", None),
        ("grok-4.20-0309-reasoning", "medium", "medium"),
        ("grok-4.20-0309-reasoning", "none", None),
        ("grok-4.20-0309-non-reasoning", "low", None),
        ("grok-4.20-non-reasoning", "low", None),
        ("grok-4.20-multi-agent-0309", "low", None),
        ("grok-4.3", None, None),
    ],
)
def test_reasoning_effort_for_model(model, effort, expected) -> None:
    assert reasoning_effort_for_model(model, effort) == expected


async def test_chat_completion_sends_effort_for_grok_43() -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value="ok")
    await async_chat_completion(
        client, model="grok-4.3", messages=[], reasoning_effort="none"
    )
    assert client.chat.completions.create.await_args.kwargs["reasoning_effort"] == "none"
    await async_chat_completion(
        client,
        model="grok-4.20-non-reasoning",
        messages=[],
        reasoning_effort="low",
    )
    assert "reasoning_effort" not in client.chat.completions.create.await_args.kwargs


def test_tool_result_outcome_and_read_only() -> None:
    from custom_components.grok_conversation.action_guard import (
        is_read_only_tool,
        tool_result_outcome,
    )

    ok = {
        "response_type": "action_done",
        "data": {"success": [{"name": "Greenroom", "type": "area", "id": "g"}],
                 "failed": [], "targets": []},
    }
    assert tool_result_outcome(ok)[0] is False
    assert tool_result_outcome(ok)[1][0]["id"] == "g"
    err = {"error": "MatchFailedError", "error_text": "No area named vierahuone"}
    assert tool_result_outcome(err) == (True, [], [], "No area named vierahuone")
    assert tool_result_outcome({"response_type": "error", "data": {"code": "no_valid_targets"}})[0]
    assert is_read_only_tool("homeassistant__GetLiveContext")
    assert is_read_only_tool("GetLiveContext")
    assert not is_read_only_tool("intent__HassTurnOn")



def test_parse_final_response() -> None:
    from custom_components.grok_conversation.action_guard import (
        parse_final_response,
    )

    assert parse_final_response('{"speech": "Hi", "action_claimed": true}') == (
        "Hi",
        True,
    )
    assert parse_final_response("plain text") == ("plain text", None)
    assert parse_final_response("{not json") == ("{not json", None)
    assert parse_final_response('{"speech": 1}') == ('{"speech": 1}', None)


def test_language_instruction_uses_code_only() -> None:
    from custom_components.grok_conversation.action_guard import (
        language_instruction,
    )

    assert "language code: fi" in language_instruction("fi")
    assert "language code: pt-BR" in language_instruction("pt-BR")
