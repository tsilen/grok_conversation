"""End-to-end: request contents and the language-neutral action guard.

No test depends on the language of the model's text: claims are signalled by
the structured ``action_claimed`` flag and success by tool results.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_expose_entity,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er, intent
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.grok_conversation.action_guard import FINAL_RESPONSE_FORMAT
from test_conversation import _completion_response, _tool_call  # noqa: E402


def _reply(speech: str, claimed: bool = False) -> MagicMock:
    """Structured final answer as the model returns it."""
    return _completion_response(
        json.dumps({"speech": speech, "action_claimed": claimed})
    )


def _tool(suffix: str, args: str, call_id: str = "call_1"):
    """Step: call the tool whose (possibly namespaced) name ends with suffix."""

    def _step(kwargs):
        name = next(
            t["function"]["name"]
            for t in kwargs["tools"]
            if t["function"]["name"].endswith(suffix)
        )
        return _completion_response(tool_calls=[_tool_call(name, args, call_id)])

    return _step


TURN_ON_OK = '{"area":"viherhuone","domain":["light"]}'
TURN_ON_BAD = '{"area":"vierahuone","domain":["light"]}'


def _scripted(*steps):
    """side_effect: each step is a response or a callable(kwargs)->response."""
    it = iter(steps)

    async def _create(**kwargs):
        step = next(it)
        return step if isinstance(step, MagicMock) else step(kwargs)

    return _create


async def _setup_greenroom(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "intent", {})
    if not hass.services.has_service("light", "turn_on"):
        async_mock_service(hass, "light", "turn_on")
    area = ar.async_get(hass).async_create("greenroom", aliases={"viherhuone"})
    entry = er.async_get(hass).async_get_or_create(
        "light", "test", "greenroom_1", suggested_object_id="greenroom"
    )
    er.async_get(hass).async_update_entity(entry.entity_id, area_id=area.id)
    hass.states.async_set(entry.entity_id, "off", {"friendly_name": "Greenroom"})
    async_expose_entity(hass, conversation.DOMAIN, entry.entity_id, True)


async def _converse(hass, entry, text="Laita viherhuoneen valot päälle."):
    client = entry.runtime_data
    pkg = "custom_components.grok_conversation"
    with (
        patch(f"{pkg}.openai.AsyncOpenAI", return_value=client),
        patch(f"{pkg}.async_validate_voice_access", return_value=(True, "ok")),
        patch(f"{pkg}.get_async_client", return_value=None),
    ):
        # Updating options reloads the entry (update listener)
        hass.config_entries.async_update_entry(
            entry, options={**entry.options, CONF_LLM_HASS_API: ["assist"]}
        )
        await hass.async_block_till_done()
    entry.runtime_data = client
    agent = conversation.async_get_agent(hass, entry.entry_id)
    return await conversation.async_converse(
        hass,
        text=text,
        conversation_id=None,
        context=Context(),
        language="fi",
        agent_id=agent.entity_id,
    )


async def test_request_has_system_prompt_tools_and_language(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _tool("HassTurnOn", TURN_ON_OK), _reply("Valot ovat päällä.", True)
        )
    )
    result = await _converse(hass, mock_config_entry)

    calls = mock_openai_client.chat.completions.create.await_args_list
    kwargs = calls[0].kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    assert "Static Context" in system
    assert "greenroom" in system.lower() and "viherhuone" in system
    assert "- greenroom (aliases: viherhuone)" in system
    assert "language code: fi" in system
    assert any(t["function"]["name"].endswith("HassTurnOn") for t in kwargs["tools"])
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["response_format"] == FINAL_RESPONSE_FORMAT
    assert messages[-1] == {"role": "user", "content": "Laita viherhuoneen valot päälle."}

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Valot ovat päällä."
    assert "light.greenroom" in [t.id for t in result.response.success_results]


async def test_claim_without_tool_call_gets_one_corrective_retry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _reply("I'll turn on the greenhouse lights now.", True),
            _tool("HassTurnOn", TURN_ON_OK),
            _reply("Valot ovat päällä.", True),
        )
    )
    result = await _converse(hass, mock_config_entry)

    calls = mock_openai_client.chat.completions.create.await_args_list
    assert len(calls) == 3
    retry = calls[1].kwargs["messages"]
    assert retry[-2] == {
        "role": "assistant",
        "content": "I'll turn on the greenhouse lights now.",
    }
    assert retry[-1]["role"] == "system"
    assert "No Home Assistant action succeeded" in retry[-1]["content"]
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Valot ovat päällä."


async def test_claim_twice_without_action_reports_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_reply("x", True), _reply("y", True))
    )
    result = await _converse(hass, mock_config_entry)

    assert mock_openai_client.chat.completions.create.await_count == 2
    assert result.response.response_type == intent.IntentResponseType.ERROR
    speech = result.response.speech["plain"]["speech"]
    assert speech not in ("x", "y")
    assert speech  # from translations (en fallback)


async def test_answer_without_claim_is_not_retried(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_reply("Pariisi.", False))
    )
    result = await _converse(hass, mock_config_entry, text="Mikä on Ranskan pääkaupunki?")
    assert mock_openai_client.chat.completions.create.await_count == 1
    assert result.response.speech["plain"]["speech"] == "Pariisi."


async def test_failed_tool_triggers_retry_with_errors_then_success(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Misheard area fails; reply claims success; retry includes the error."""
    turn_on_calls = async_mock_service(hass, "light", "turn_on")
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _tool("HassTurnOn", TURN_ON_BAD, "c1"),
            _reply("Selvä Tuomas.", True),
            _tool("HassTurnOn", TURN_ON_OK, "c2"),
            _reply("Valot ovat päällä.", True),
        )
    )
    result = await _converse(hass, mock_config_entry, text="Laita vierahuoneen valot päälle.")
    calls = mock_openai_client.chat.completions.create.await_args_list
    assert len(calls) == 4
    assert "Tool errors so far" in calls[2].kwargs["messages"][-1]["content"]
    assert len(turn_on_calls) == 1
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert "light.greenroom" in [t.id for t in result.response.success_results]


async def test_failed_tool_even_without_claim_gets_retry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """A failed action triggers the retry; honest reply -> error response."""
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _tool("HassTurnOn", TURN_ON_BAD),
            _reply("a", False),
            _reply("b", False),
        )
    )
    result = await _converse(hass, mock_config_entry)
    assert mock_openai_client.chat.completions.create.await_count == 3
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert result.response.speech["plain"]["speech"] == "b"


async def test_failed_tool_with_empty_replies_reports_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _tool("HassTurnOn", TURN_ON_BAD), _reply("", False), _reply("", False)
        )
    )
    result = await _converse(hass, mock_config_entry)
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert result.response.speech["plain"]["speech"]


async def test_read_only_tool_then_claim_gets_retry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """GetLiveContext alone is not an action."""
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _tool("GetLiveContext", "{}"), _reply("z", True), _reply("z", True)
        )
    )
    result = await _converse(hass, mock_config_entry)
    assert mock_openai_client.chat.completions.create.await_count == 3
    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_status_answer_after_read_tool_is_not_retried(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_tool("GetLiveContext", "{}"), _reply("Off.", False))
    )
    result = await _converse(hass, mock_config_entry, text="Onko valo päällä?")
    assert mock_openai_client.chat.completions.create.await_count == 2
    assert result.response.speech["plain"]["speech"] == "Off."


async def test_successful_action_with_empty_reply_has_empty_speech(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_tool("HassTurnOn", TURN_ON_OK), _reply("", True))
    )
    result = await _converse(hass, mock_config_entry)
    assert mock_openai_client.chat.completions.create.await_count == 2
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == ""


async def test_structured_format_rejected_falls_back_to_text(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)

    async def _create(**kwargs):
        if kwargs.get("response_format"):
            raise openai.BadRequestError(
                "response_format unsupported", response=MagicMock(), body=None
            )
        return _completion_response("Plain answer.")

    mock_openai_client.chat.completions.create = AsyncMock(side_effect=_create)
    result = await _converse(hass, mock_config_entry, text="Hei")
    assert result.response.speech["plain"]["speech"] == "Plain answer."


async def test_no_llm_api_claim_is_not_passed_on(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    client = mock_config_entry.runtime_data
    pkg = "custom_components.grok_conversation"
    options = {
        k: v for k, v in mock_config_entry.options.items() if k != CONF_LLM_HASS_API
    }
    with (
        patch(f"{pkg}.openai.AsyncOpenAI", return_value=client),
        patch(f"{pkg}.async_validate_voice_access", return_value=(True, "ok")),
        patch(f"{pkg}.get_async_client", return_value=None),
    ):
        hass.config_entries.async_update_entry(mock_config_entry, options=options)
        await hass.async_block_till_done()
    mock_config_entry.runtime_data = client
    client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_reply("Selvä Tuomas.", True))
    )
    agent = conversation.async_get_agent(hass, mock_config_entry.entry_id)
    result = await conversation.async_converse(
        hass,
        text="Laita valot päälle.",
        conversation_id=None,
        context=Context(),
        language="fi",
        agent_id=agent.entity_id,
    )
    kwargs = client.chat.completions.create.await_args.kwargs
    assert "tools" not in kwargs
    assert "Static Context" not in kwargs["messages"][0]["content"]
    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "Control Home Assistant" in result.response.speech["plain"]["speech"]


def test_recommended_options_enable_assist() -> None:
    from homeassistant.helpers import llm

    from custom_components.grok_conversation.config_flow import RECOMMENDED_OPTIONS

    assert RECOMMENDED_OPTIONS[CONF_LLM_HASS_API] == [llm.LLM_API_ASSIST]
