from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from .schemas import Box, Candidate, Region, WordResult

TOKEN = re.compile(r"\[\[illegible\]\]|\S+", re.IGNORECASE)
NUMBER = re.compile(r"^(?:\d+(?:[./-]\d+)*|\d+(?:\.\d+)?%?)$")
UNITS = {"mg", "mcg", "g", "kg", "ml", "l", "mmhg", "bpm", "tablet", "tablets", "tab", "capsule", "capsules"}
NEGATIONS = {"no", "not", "never", "none", "without", "denies", "negative"}
MEDICAL = {"allergy", "allergies", "pain", "patient", "prescribed", "diagnosis", "dose", "daily", "twice",
           "amlodipine", "paracetamol", "metformin", "aspirin", "insulin", "antibiotic", "hypertension"}


def clean(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'").casefold()


def risk_tag(token: str, position: int) -> str:
    value = clean(token)
    if NUMBER.match(value):
        return "number"
    if value in UNITS:
        return "unit"
    if value in NEGATIONS:
        return "negation"
    if value in MEDICAL or value.endswith(("cillin", "pril", "olol", "statin", "formin", "dipine")):
        return "medical_term"
    if position > 0 and token[:1].isupper():
        return "name"
    return "normal"


def aligned_tokens(reference: list[str], candidate: str) -> list[str | None]:
    tokens = TOKEN.findall(candidate)
    aligned: list[str | None] = [None] * len(reference)
    matcher = SequenceMatcher(None, [clean(t) for t in reference], [clean(t) for t in tokens])
    for kind, a1, a2, b1, b2 in matcher.get_opcodes():
        if kind == "equal":
            aligned[a1:a2] = tokens[b1:b2]
        elif kind == "replace":
            for offset in range(min(a2 - a1, b2 - b1)):
                aligned[a1 + offset] = tokens[b1 + offset]
    return aligned


def word_box(box: Box, tokens: list[str], index: int) -> Box:
    weights = [max(1, len(clean(token))) for token in tokens]
    total = sum(weights) + max(0, len(weights) - 1)
    start = sum(weights[:index]) + index
    end = start + weights[index]
    width = box.x2 - box.x1
    x1 = box.x1 + int(width * start / total)
    x2 = box.x1 + max(int(width * end / total), int(width * start / total) + 1)
    return Box(x1=x1, y1=box.y1, x2=min(box.x2, x2), y2=box.y2)


def analyze_region(region: Region, line_index: int) -> list[WordResult]:
    output = TOKEN.findall(region.text) or ["[[illegible]]"]
    sources = [(candidate.source, aligned_tokens(output, candidate.text), candidate.confidence)
               for candidate in region.candidates]
    words = []
    for position, text in enumerate(output):
        candidates = [Candidate(source=source, text=tokens[position], confidence=confidence)
                      for source, tokens, confidence in sources if tokens[position] is not None]
        counts = Counter(clean(candidate.text) for candidate in candidates)
        agreement = max(counts.values(), default=0)
        agreement_ratio = agreement / max(1, len(candidates))
        reader_quality = (sum(candidate.confidence if candidate.confidence is not None else 0.7
                              for candidate in candidates) / len(candidates)) if candidates else 0
        word_confidence = min(1.0, 0.5 + 0.35 * agreement_ratio + 0.15 * reader_quality) if candidates else 0
        if len(candidates) < 2:
            word_confidence = min(word_confidence, 0.35)
        tag = risk_tag(text, position)
        high_risk = tag != "normal"
        illegible = "[[illegible]]" in text.casefold() or region.status == "illegible"
        region_blocker = any(reason in {"Low image sharpness", "Correction changed both reader outputs"}
                             for reason in region.reasons)
        if region.status == "crossed_out":
            status, eligibility, legibility = "crossed_out", "excluded", "struck"
        elif illegible:
            status, eligibility, legibility = "illegible", "illegible", "illegible"
        elif region.status == "verified":
            status, eligibility, legibility = "verified", "verified", "legible"
        elif agreement >= 2 and agreement == len(candidates) and not region_blocker:
            status, eligibility, legibility = "accepted", "eligible", "legible"
        else:
            status, eligibility, legibility = "review", "needs_review", "uncertain"
        disagreement = 1 - agreement_ratio
        risk = 1.0 if illegible else min(1.0, max(1 - word_confidence, disagreement) *
                                         (1.25 if high_risk else 1.0))
        reasons = list(region.reasons)
        if high_risk:
            reasons.append(f"High-impact token: {tag.replace('_', ' ')}")
        if candidates and agreement < len(candidates):
            reasons.append("Readers disagree on this word")
        words.append(WordResult(id=f"{region.id}-w{position + 1}", region_id=region.id,
                                line_index=line_index, position=position,
                                box=word_box(region.box, output, position), text=text,
                                status=status, eligibility=eligibility, legibility=legibility,
                                confidence=word_confidence if status != "illegible" else 0,
                                risk=round(risk, 3), risk_tag=tag, candidates=candidates,
                                readers_agree=agreement, readers_total=len(candidates),
                                reasons=list(dict.fromkeys(reasons))))
    return words


def summarize_words(regions: list[Region]) -> dict[str, int | float]:
    words = [word for region in regions for word in region.words]
    counts = Counter(word.eligibility for word in words)
    return {"total_words": len(words), "eligible": counts["eligible"],
            "needs_review": counts["needs_review"], "illegible": counts["illegible"],
            "excluded": counts["excluded"], "verified": counts["verified"],
            "review_completion": round((counts["eligible"] + counts["verified"] + counts["excluded"]) /
                                       max(1, len(words)), 3)}


def populate_words(regions: list[Region]) -> dict[str, int | float]:
    for line_index, region in enumerate(regions):
        region.words = analyze_region(region, line_index)
    return summarize_words(regions)
