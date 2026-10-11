"""Language-neutral guard for smart-home actions.

Reasoning models sometimes answer a device command with "I'll turn on the
lights now." and end the turn without a successful tool call, so nothing
happens. Nothing here parses the model's or the user's text:

- Success/failure is decided only from structured tool results of the turn
  (which non-read-only tools ran, and whether they returned an error).
- Whether the final reply claims an action comes from the model itself via a
  structured final answer (``FINAL_RESPONSE_FORMAT``: ``speech`` plus
  ``action_claimed``), so it works in any language.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Structured final answer for conversation turns. Tool-call turns are not
# affected; only a text reply is constrained to this schema.
FINAL_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "assist_reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "speech": {
                    "type": "string",
                    "description": (
                        "The reply spoken/shown to the user, in the user's "
                        "language, plain text."
                    ),
                },
                "action_claimed": {
                    "type": "boolean",
                    "description": (
                        "true if the speech says that you performed, are "
                        "performing or will perform an action that changes "
                        "something in the home (turn on/off, set, open, "
                        "start, ...); false for answers, questions, status "
                        "reports and refusals."
                    ),
                },
            },
            "required": ["speech", "action_claimed"],
            "additionalProperties": False,
        },
    },
}


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


def parse_final_response(text: str | None) -> tuple[str, bool | None]:
    """Return (speech, action_claimed) from a structured final answer.

    Tolerates surrounding whitespace, ```json fences, text before or after
    the object and pretty-printed JSON. If no object with a string
    ``speech`` is found, return the text unchanged and None (unknown).
    """
    raw = (text or "").strip()
    if "{" not in raw:
        return raw, None
    candidate = _FENCE_RE.sub("", raw).strip()
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    while start != -1:
        try:
            data, _end = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            start = candidate.find("{", start + 1)
            continue
        if isinstance(data, dict) and isinstance(data.get("speech"), str):
            claimed = data.get("action_claimed")
            return (
                data["speech"].strip(),
                claimed if isinstance(claimed, bool) else None,
            )
        start = candidate.find("{", start + 1)
    return raw, None


def speech_from_reply(text: str | None) -> str:
    """Speech for a stored reply; never returns a structured JSON answer."""
    return parse_final_response(text)[0]


CORRECTION_PROMPT = (
    "No Home Assistant action succeeded in this turn, so nothing was changed "
    "in the home. If the user asked to control a device, area or entity, "
    "call the appropriate Home Assistant tool now, using an exact entity or "
    "area name, or one of its aliases, from the smart home context; map "
    "misheard or inflected names to the closest listed name. If a tool "
    "reports no match, try again with the closest corrected name. If you "
    "cannot perform the action, say so honestly. Never claim an action you "
    "did not perform."
)


def language_instruction(language: str | None) -> str:
    """System-prompt line that pins the reply language by its code."""
    code = (language or "").strip()
    if not code or code == "*":
        return "Respond in the language of the user's message."
    return (
        f"Respond in the user's language (language code: {code}) unless the "
        "user writes in another language."
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
