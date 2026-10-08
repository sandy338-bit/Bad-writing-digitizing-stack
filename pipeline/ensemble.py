"""Evidence preserving fusion of independent handwriting readers."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher

from PIL import Image

from .images import crop, enhance, quality_score
from .readers import DetectedLine, ReaderUnavailable
from .schemas import BaselineResult, Box, Candidate, Region, Transcription
from .words import populate_words, summarize_words
from .language import detect_language, enforce_language
import os


def _key(text: str) -> str:
    return " ".join(text.casefold().split())


def _choose(candidates: list[Candidate]) -> str:
    readable = [c for c in candidates if c.text.strip() and "[[illegible]]" not in c.text.casefold()]
    if not readable:
        return "[[illegible]]"
    counts = Counter(_key(c.text) for c in readable)
    winner, votes = counts.most_common(1)[0]
    if votes >= 2:
        return next(c.text.strip() for c in readable if _key(c.text) == winner)
    # No majority: use the reading closest to the others, while requiring review.
    priority = {"qwen": 3, "qwen_page": 3, "gemini": 2, "gemini_page": 2, "trocr": 1}
    return max(readable, key=lambda c: (sum(SequenceMatcher(None, _key(c.text), _key(other.text)).ratio()
                                           for other in readable if other is not c),
                                        priority.get(c.source, 0))).text.strip()


def _page_region(image: Image.Image, candidates: list[Candidate], warnings: list[str],
                 filename: str, models: dict[str, str],
                 audit_candidates: list[Candidate] | None = None) -> Transcription:
    text = _choose(candidates)
    region = Region(id="r1", box=Box(x1=0, y1=0, x2=image.width, y2=image.height),
                    text=text, status="illegible" if text == "[[illegible]]" else "review",
                    confidence=0.0 if text == "[[illegible]]" else 0.2,
                    candidates=candidates, reasons=["Text location could not be verified; review the full page",
                                                    "TrOCR has no aligned word reading on this page"])
    populate_words([region])
    for word in region.words:
        word.source_scope = "page"
        if word.status == "accepted":
            word.status, word.eligibility, word.legibility = "review", "needs_review", "uncertain"
    warnings.append("Word positions are page-level because line localization was unreliable")
    return Transcription(filename=filename, width=image.width, height=image.height,
                         text=text, language=detect_language(text), page_status="review", page_candidate=candidates[0].text if candidates else None,
                         page_candidates=audit_candidates or candidates, regions=[region], warnings=warnings,
                         summary=summarize_words([region]), model_info=models)


def transcribe_ensemble(image: Image.Image, filename: str, readers: dict[str, object],
                        warnings: list[str], max_regions: int, crop_limit: int,
                        language_hint: str = "auto") -> Transcription:
    """Read page and line crops; missing providers never fabricate agreement."""
    models = {name: str(getattr(reader, "model", getattr(reader, "model_name", name)))
              for name, reader in readers.items()}
    page_candidates: list[Candidate] = []
    available_vision = {}
    for name in ("gemini", "qwen"):
        reader = readers.get(name)
        if reader is None:
            continue
        try:
            text = enforce_language(reader.baseline(image).strip(), language_hint)
            if text:
                page_candidates.append(Candidate(source=name + "_page", text=text))
            available_vision[name] = reader
        except ReaderUnavailable as exc:
            warnings.append(f"{name} page reading unavailable: {exc}")
        except Exception as exc:
            warnings.append(f"{name} page reading was unusable; continuing with the other readers")

    lines: list[DetectedLine] = []
    if readers.get("trocr") is not None:
        try:
            lines = readers["trocr"].read(enhance(image))[:max_regions]
        except Exception as exc:
            warnings.append(f"TrOCR reading failed: {type(exc).__name__}")

    if not lines:
        if not page_candidates:
            if readers.get("trocr") is not None:
                warnings.append("No reader could isolate readable writing; verify the page manually")
                return _page_region(image, [Candidate(source="trocr", text="[[illegible]]")],
                                    warnings, filename, models)
            raise ReaderUnavailable("provider", "TrOCR")
        return _page_region(image, page_candidates, warnings, filename, models)

    trocr_words = sum(len(line.candidate.text.split()) for line in lines)
    page_words = max((len(candidate.text.split()) for candidate in page_candidates), default=0)
    for line in lines:
        line.candidate.text = enforce_language(line.candidate.text, language_hint)
    trocr_audit = Candidate(source="trocr_page", text="\n".join(line.candidate.text for line in lines))
    audits = page_candidates + [trocr_audit]
    if page_words >= 12 and (trocr_words < page_words * 0.5 or page_words / len(lines) > 8):
        warnings.append("TrOCR lines do not cover the page reliably; its raw reading is shown for comparison")
        return _page_region(image, page_candidates, warnings, filename, models, audits)

    regions: list[Region] = []
    for index, line in enumerate(lines):
        box = line.box
        snippet = crop(image, box)
        candidates = [line.candidate] if line.candidate.text.strip() else []
        crossed = []
        if index < crop_limit:
            for name, reader in tuple(available_vision.items()):
                try:
                    text, kind = reader.read_crop(snippet)
                    if text.strip():
                        candidates.append(Candidate(source=name, text=enforce_language(text.strip(), language_hint)))
                    crossed.append(kind == "crossed_out")
                except ReaderUnavailable as exc:
                    warnings.append(f"{name} crop reading unavailable: {exc}")
                    available_vision.pop(name, None)
                    continue
                except Exception as exc:
                    warnings.append(f"{name} crop reading failed at line {index + 1}: {type(exc).__name__}")
        elif index == crop_limit:
            warnings.append("Crop request limit reached; remaining TrOCR lines require review")

        text = _choose(candidates)
        # Constrained correction: ask one vision reader to resolve a disagreement,
        # but accept the response only when it exactly matches a reader candidate.
        if len(candidates) >= 2 and "qwen" in available_vision:
            alternate = next((c.text for c in candidates if _key(c.text) != _key(text)), None)
            if alternate:
                try:
                    corrected = available_vision["qwen"].correct(snippet, text, alternate, "")
                    allowed = {_key(candidate.text): candidate.text.strip() for candidate in candidates}
                    if _key(corrected) in allowed:
                        text = allowed[_key(corrected)]
                    else:
                        warnings.append(f"Constrained correction rejected at line {index + 1}")
                except Exception as exc:
                    warnings.append(f"Constrained correction unavailable at line {index + 1}: {type(exc).__name__}")
        unanimous = (len(candidates) >= 2 and all(_key(c.text) == _key(text) for c in candidates)
                     and "[[illegible]]" not in text.casefold())
        quality = quality_score(snippet)
        strike = len(crossed) >= 2 and all(crossed)
        kind = "crossed_out" if strike else ("margin" if box.x2 < image.width * .2 or
                                                box.x1 > image.width * .8 else "body")
        reasons = []
        if len(candidates) < 2:
            reasons.append("Only one reader was available")
        elif not unanimous:
            reasons.append("Readers disagree on this line")
        if any(crossed) and not strike:
            reasons.append("Possible crossed-out writing requires review")
        if quality < .2:
            reasons.append("Low image sharpness")
        if text == "[[illegible]]":
            status, confidence = "illegible", 0.0
        elif strike:
            status, confidence = "crossed_out", 0.0
        elif unanimous and quality >= .2:
            status, confidence = "accepted", .85
        else:
            status, confidence = "review", .25
        regions.append(Region(id=f"r{index + 1}", box=box, kind=kind, text=text,
                              status=status, confidence=confidence, candidates=candidates,
                              reasons=reasons))

    output = "\n".join("[crossed out]" if r.status == "crossed_out" else r.text for r in regions)
    for candidate in page_candidates:
        if SequenceMatcher(None, _key(candidate.text), _key(output)).ratio() < .8:
            warnings.append(f"{candidate.source} differs from the line transcript; check for omissions")
    summary = populate_words(regions)
    return Transcription(filename=filename, width=image.width, height=image.height,
                         text=output, language=detect_language(output), page_status="review", page_candidate=page_candidates[0].text if page_candidates else None,
                         page_candidates=audits, regions=regions, warnings=warnings,
                         summary=summary, model_info=models)


def baseline_ensemble(image: Image.Image, filename: str, readers: dict[str, object]) -> BaselineResult:
    """Keep the comparison baseline a single reader, preferring Qwen page reading."""
    for name in ("qwen", "gemini"):
        reader = readers.get(name)
        if reader is not None:
            try:
                return BaselineResult(filename=filename, text=reader.baseline(image),
                                      model=str(getattr(reader, "model", name)))
            except Exception:
                continue
    reader = readers.get("trocr")
    if reader is not None:
        lines = reader.read(image)
        if lines:
            return BaselineResult(filename=filename,
                                  text="\n".join(line.candidate.text for line in lines),
                                  model=str(getattr(reader, "model_name", "TrOCR")))
    raise ReaderUnavailable("provider", "TrOCR")
