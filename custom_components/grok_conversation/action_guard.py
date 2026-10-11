"""Guard against replies that claim a smart-home action without a tool call.

Reasoning models sometimes answer a command with "I'll turn on the lights now."
and end the turn without calling any Home Assistant tool, so nothing happens.
These helpers detect that case so the tool loop can ask the model once more
and, if it still does not call a tool, tell the user honestly.
"""

from __future__ import annotations

from typing import Any
import re

# User text that looks like a device command (English / Finnish / Swedish).
_COMMAND_RE = re.compile(
    r"\b("
    # English
    r"turn|switch|dim|brighten|open|close|lock|unlock|start|stop|activate|"
    r"deactivate|enable|disable|pause|resume|set"
    r")\b"
    r"|\b("
    # Finnish imperatives / common phrasing
    r"laita|laittakaa|sytytä|sammuta|kytke|käynnistä|pysäytä|avaa|sulje|"
    r"lukitse|avaa lukko|himmennä|kirkasta|säädä|aseta|nosta|laske|vaihda"
    r")\w*\b"
    r"|\bpäälle\b|\bpois päältä\b"
    r"|\b(tänd|släck|slå på|stäng av|öppna|stäng|lås)\b",
    re.IGNORECASE,
)

# Assistant text that claims (or announces) having performed an action.
_CLAIM_RE = re.compile(
    r"\b(i'?ll|i will|i'?m going to|i am going to|let me|i'?ve|i have|now)\b"
    r"[^.?!]{0,60}?\b("
    r"turn(ed|ing)?|switch(ed|ing)?|set(ting)?|dimm?(ed|ing)?|brighten(ed|ing)?|"
    r"open(ed|ing)?|clos(e|ed|ing)|lock(ed|ing)?|unlock(ed|ing)?|"
    r"start(ed|ing)?|stop(ped|ping)?|activat(e|ed|ing)|run(ning)?|"
    r"paus(e|ed|ing)|resum(e|ed|ing)"
    r")\b"
    r"|\b(turn(ed|ing)|switch(ed|ing))\s+(the\s+)?\w*\s*(on|off)\b"
    r"|\b(done|ok(ay)?)[,!.]"
    # Finnish: first-person / passive forms of common actions
    r"|\b(sytytän|sytytin|sytytetty|sammutan|sammutin|sammutettu|laitan|laitoin|"
    r"laitettu|kytken|kytkin|kytketty|käynnistän|käynnistin|käynnistetty|"
    r"pysäytän|pysäytin|avaan|avasin|avattu|suljen|suljin|suljettu|lukitsen|"
    r"lukitsin|lukittu|himmennän|himmensin|säädän|säädin|asetan|asetin|"
    r"asetettu|selvä|tehty|valmis)\b",
    re.IGNORECASE,
)

CORRECTION_PROMPT = (
    "You replied without calling any tool, so nothing was actually done in "
    "the home. If the user asked to control or check a device, area or "
    "entity, call the appropriate Home Assistant tool now. Use the exact "
    "entity or area name, or one of its aliases, from the smart home "
    "context; map inflected or misheard names (e.g. Finnish genitive forms) "
    "to the closest exposed name. If a tool reports no match, try again with "
    "the closest corrected name. If you cannot perform the action, say so "
    "honestly. Never claim an action you did not perform. Reply briefly in "
    "the user's language."
)

_NOT_DONE: dict[str, str] = {
    "en": "Sorry, I couldn't do that — no device command was sent.",
    "fi": "Valitettavasti en saanut sitä tehtyä, eikä laitteille lähetetty komentoa.",
    "sv": "Tyvärr kunde jag inte göra det – inget kommando skickades.",
    "de": "Das konnte ich leider nicht ausführen – es wurde kein Befehl gesendet.",
}

_NO_CONTROL: dict[str, str] = {
    "en": (
        "I can't control devices: Home Assistant control is not enabled for "
        "this assistant. Select Assist under 'Control Home Assistant' in the "
        "xAI Grok integration options."
    ),
    "fi": (
        "En pysty ohjaamaan laitteita, koska Home Assistantin ohjaus ei ole "
        "käytössä tälle avustajalle. Valitse Assist kohtaan 'Control Home "
        "Assistant' xAI Grok -integraation asetuksissa."
    ),
}

_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fi": "Finnish",
    "sv": "Swedish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "da": "Danish",
    "nb": "Norwegian",
    "hu": "Hungarian",
}


def _primary(language: str | None) -> str:
    return (language or "en").replace("_", "-").split("-")[0].lower()


def looks_like_command(text: str | None) -> bool:
    """Return True if the user text looks like a device command."""
    return bool(text and _COMMAND_RE.search(text))


def claims_action(text: str | None) -> bool:
    """Return True if assistant text announces or claims an action."""
    return bool(text and _CLAIM_RE.search(text))


def not_done_message(language: str | None) -> str:
    """Honest reply when the model claimed an action but called no tool."""
    return _NOT_DONE.get(_primary(language), _NOT_DONE["en"])


def no_control_message(language: str | None) -> str:
    """Reply when a device command arrives but no LLM API/tools are enabled."""
    return _NO_CONTROL.get(_primary(language), _NO_CONTROL["en"])


def is_not_done_message(text: str | None) -> bool:
    return bool(text) and (
        text in _NOT_DONE.values() or text in _NO_CONTROL.values()
    )


def language_instruction(language: str | None) -> str:
    """System-prompt line that pins the reply language."""
    code = (language or "").strip()
    if not code or code == "*":
        return "Always reply in the language of the user's message."
    name = _LANGUAGE_NAMES.get(_primary(code))
    label = f"{name} ({code})" if name else code
    return (
        "Always reply in the language of the user's message. The user's "
        f"Assist language is {label}; reply in it unless the user writes in "
        "another language."
    )


# Tools that only read state. HA 2026.10+ marks these via annotations; the
# names cover older HA versions.
_READ_ONLY_TOOL_NAMES = frozenset(
    {
        "GetLiveContext",
        "GetDateTime",
        "HassGetState",
        "HassGetWeather",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "todo_get_items",
        "calendar_get_events",
    }
)


def is_read_only_tool(tool_name: str, llm_api: Any = None) -> bool:
    """Return True if the named tool only reads state (does not act)."""
    for tool in getattr(llm_api, "tools", None) or []:
        if getattr(tool, "name", None) == tool_name:
            annotations = getattr(tool, "annotations", None)
            if annotations is not None and getattr(annotations, "read_only", False):
                return True
            break
    return tool_name.split("__")[-1] in _READ_ONLY_TOOL_NAMES


def tool_result_outcome(result: Any) -> tuple[bool, list[dict], list[dict], str]:
    """Return (error, success_targets, failed_targets, error_text) for a result.

    Handles llm.ToolResult (HA 2026.10+) and plain dicts (older HA), including
    intent responses ({"response_type": ..., "data": {"success": [...]}}) and
    errors wrapped by the chat log ({"error": "MatchFailedError", ...}).
    """
    data = result
    error = False
    if result is not None and not isinstance(result, dict) and hasattr(result, "data"):
        data = result.data
        error = bool(getattr(result, "error", False))
    if not isinstance(data, dict):
        return error, [], [], ""
    error_text = ""
    if "error" in data:
        error = True
        error_text = str(data.get("error_text") or data.get("error") or "")
    if data.get("response_type") == "error":
        error = True
        error_text = error_text or str(
            (data.get("data") or {}).get("code") or "error"
        )
    inner = data.get("data") if isinstance(data.get("data"), dict) else {}
    success = [t for t in inner.get("success") or [] if isinstance(t, dict)]
    failed = [t for t in inner.get("failed") or [] if isinstance(t, dict)]
    if failed and not success:
        error = True
        error_text = error_text or "no target succeeded"
    return error, success, failed, error_text


def summarize_turn_tools(chat_log: Any) -> dict[str, Any]:
    """Summarize tool results of the current turn (after the last user message)."""
    from homeassistant.components import conversation  # noqa: PLC0415

    summary: dict[str, Any] = {
        "ran_tools": False,
        "action_ok": False,
        "action_failed": False,
        "errors": [],
        "success": [],
        "failed": [],
    }
    llm_api = getattr(chat_log, "llm_api", None)
    for content in reversed(chat_log.content):
        if isinstance(content, conversation.UserContent):
            break
        if not isinstance(content, conversation.ToolResultContent):
            continue
        summary["ran_tools"] = True
        name = getattr(content, "tool_name", "") or ""
        result = getattr(content, "result", None)
        if result is None:
            result = getattr(content, "tool_result", None)
        error, success, failed, error_text = tool_result_outcome(result)
        if is_read_only_tool(name, llm_api):
            if error:
                summary["errors"].append(f"{name}: {error_text}")
            continue
        summary["success"].extend(success)
        summary["failed"].extend(failed)
        if error:
            summary["action_failed"] = True
            summary["errors"].append(f"{name}: {error_text}")
        else:
            summary["action_ok"] = True
    return summary
