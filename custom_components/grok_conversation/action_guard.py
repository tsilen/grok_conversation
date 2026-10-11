"""Guard against replies that claim a smart-home action without a tool call.

Reasoning models sometimes answer a command with "I'll turn on the lights now."
and end the turn without calling any Home Assistant tool, so nothing happens.
These helpers detect that case so the tool loop can ask the model once more
and, if it still does not call a tool, tell the user honestly.
"""

from __future__ import annotations

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
    r"|\b(is|are)\s+now\s+(on|off|open|closed|locked|unlocked)\b"
    r"|\b(done|ok(ay)?)[,!.]"
    # Finnish: first-person / passive forms of common actions
    r"|\b(sytytän|sytytin|sytytetty|sammutan|sammutin|sammutettu|laitan|laitoin|"
    r"laitettu|kytken|kytkin|kytketty|käynnistän|käynnistin|käynnistetty|"
    r"pysäytän|pysäytin|avaan|avasin|avattu|suljen|suljin|suljettu|lukitsen|"
    r"lukitsin|lukittu|himmennän|himmensin|säädän|säädin|asetan|asetin|"
    r"asetettu|selvä|tehty|valmis)\b"
    r"|\b(nyt\s+)?(päällä|pois päältä|päälle)\b",
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


def is_not_done_message(text: str | None) -> bool:
    return bool(text) and text in _NOT_DONE.values()


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
