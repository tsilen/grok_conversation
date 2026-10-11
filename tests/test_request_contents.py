"""End-to-end: the xAI request carries HA's system prompt, tools and language."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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

from test_conversation import _completion_response, _tool_call  # noqa: E402


def _turn_on_call(kwargs) -> MagicMock:
    """Tool call for HassTurnOn using the name HA exposes (may be namespaced)."""
    name = next(
        t["function"]["name"]
        for t in kwargs["tools"]
        if t["function"]["name"].endswith("HassTurnOn")
    )
    return _completion_response(
        tool_calls=[_tool_call(name, '{"area":"viherhuone","domain":["light"]}')]
    )


def _scripted(*steps):
    """side_effect: each step is a response or a callable(kwargs)->response."""
    it = iter(steps)

    async def _create(**kwargs):
        step = next(it)
        return step if isinstance(step, MagicMock) else step(kwargs)

    return _create


async def _setup_greenroom(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, "intent", {})
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
            _turn_on_call,
            _completion_response("Viherhuoneen valot ovat päällä."),
        )
    )
    result = await _converse(hass, mock_config_entry)

    kwargs = mock_openai_client.chat.completions.create.await_args_list[0].kwargs
    messages = kwargs["messages"]
    assert messages[0]["role"] == "system"
    system = messages[0]["content"]
    # HA's Assist API prompt with the exposed entity, area and its alias
    assert "Static Context" in system
    assert "greenroom" in system.lower() and "viherhuone" in system
    # Language pinned to the user's Assist language
    assert "Finnish (fi)" in system
    # Tools are sent, tool_choice auto
    names = {t["function"]["name"] for t in kwargs["tools"]}
    assert any(n.endswith("HassTurnOn") for n in names)
    assert kwargs["tool_choice"] == "auto"
    assert messages[-1] == {"role": "user", "content": "Laita viherhuoneen valot päälle."}

    assert result.response.speech["plain"]["speech"] == "Viherhuoneen valot ovat päällä."


async def test_claim_without_tool_call_gets_one_corrective_retry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """'I'll turn on the lights now.' with no tool call -> retry -> tool runs."""
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _completion_response("I'll turn on the greenhouse lights now."),
            _turn_on_call,
            _completion_response("Viherhuoneen valot ovat päällä."),
        )
    )
    result = await _converse(hass, mock_config_entry)

    calls = mock_openai_client.chat.completions.create.await_args_list
    assert len(calls) == 3
    retry_messages = calls[1].kwargs["messages"]
    assert retry_messages[-2] == {
        "role": "assistant",
        "content": "I'll turn on the greenhouse lights now.",
    }
    assert retry_messages[-1]["role"] == "system"
    assert "without calling any tool" in retry_messages[-1]["content"]
    assert calls[1].kwargs["tools"]
    assert result.response.speech["plain"]["speech"] == "Viherhuoneen valot ovat päällä."
    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE


async def test_claim_without_tool_call_twice_reports_failure(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    """Still no tool call after the retry -> honest Finnish error, not action_done."""
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(
            _completion_response("I'll turn on the greenhouse lights now."),
            _completion_response("Laitan viherhuoneen valot päälle."),
        )
    )
    result = await _converse(hass, mock_config_entry)

    assert mock_openai_client.chat.completions.create.await_count == 2
    assert result.response.response_type == intent.IntentResponseType.ERROR
    speech = result.response.speech["plain"]["speech"]
    assert "en saanut" in speech


async def test_plain_question_is_not_retried(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: MagicMock,
) -> None:
    await _setup_greenroom(hass)
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=_scripted(_completion_response("Pariisi on Ranskan pääkaupunki."))
    )
    result = await _converse(
        hass, mock_config_entry, text="Mikä on Ranskan pääkaupunki?"
    )
    assert mock_openai_client.chat.completions.create.await_count == 1
    assert result.response.speech["plain"]["speech"] == "Pariisi on Ranskan pääkaupunki."
