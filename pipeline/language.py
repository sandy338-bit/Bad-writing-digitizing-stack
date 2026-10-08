from __future__ import annotations

from collections import Counter
import re


SUPPORTED_LANGUAGES = ("English", "Hindi", "Tamil", "Malayalam")


def detect_language(text: str) -> str:
    counts = Counter()
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            counts["Hindi"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["Tamil"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            counts["Malayalam"] += 1
        elif char.isalpha() and char.isascii():
            counts["English"] += 1
    return counts.most_common(1)[0][0] if counts else "Unknown"


def enforce_language(text: str, language: str) -> str:
    """Prevent Latin hallucinations when a non-English script is selected."""
    if language.casefold() in {"auto", "english"}:
        return text
    if language.casefold() not in {"hindi", "tamil", "malayalam"}:
        return text
    # Keep punctuation, digits, and the selected Indic script; replace Latin
    # word fragments so they cannot become trusted translated output.
    return re.sub(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", "[[illegible]]", text)
