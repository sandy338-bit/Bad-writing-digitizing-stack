from __future__ import annotations

from difflib import SequenceMatcher
from functools import wraps
from threading import RLock
from PIL import Image

from .config import Settings
from .ensemble import baseline_ensemble, transcribe_ensemble
from .images import crop, detect_lines, enhance, quality_score
from .readers import DetectedLine, GeminiReader, QwenReader, QwenLocalReader, PaddleReader, TrOCRReader, ReaderUnavailable
from .schemas import BaselineResult, Box, Candidate, Region, Transcription
from .words import populate_words


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold().strip(), b.casefold().strip()).ratio()


def _normalized(text: str) -> str:
    return " ".join(text.split())


def serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def _margin(box: Box, width: int) -> bool:
    return box.x2 < width * 0.2 or box.x1 > width * 0.8


def _decide(candidates: list[Candidate], proposed: str, quality: float,
            crossed_out: bool, threshold: float = 0.75) -> tuple[str, float, list[str]]:
    reasons: list[str] = []
    if crossed_out:
        return "crossed_out", 0.0, ["Detected as crossed-out writing"]
    if not proposed.strip() or "[[illegible]]" in proposed.lower():
        return "illegible", 0.0, ["Unreadable characters were detected"]
    if len(candidates) < 2:
        return "review", 0.25, ["Only one reader was available"]

    a, b = candidates[:2]
    agreement = _similarity(a.text, b.text)
    score = min(1.0, 0.35 + 0.35 * agreement + 0.2 * (a.confidence if a.confidence is not None else 0.5)
                + 0.1 * quality)
    if _normalized(a.text) != _normalized(b.text):
        reasons.append("Readers disagree")
    if a.confidence is not None and a.confidence < 0.6:
        reasons.append("Low OCR confidence")
    if proposed.strip().casefold() not in {a.text.strip().casefold(), b.text.strip().casefold()}:
        reasons.append("Correction changed both reader outputs")
        score = min(score, 0.65)
    if any(ch.isdigit() for ch in proposed) and agreement < 1.0:
        reasons.append("Number requires verification")
    if quality < 0.2:
        reasons.append("Low image sharpness")
    if (agreement >= 0.85 and not reasons and quality >= 0.2 and score >= threshold):
        return "accepted", round(score, 3), reasons
    return "review", round(score, 3), reasons or ["Evidence is insufficient for automatic acceptance"]


class Digitizer:
    def __init__(self, settings: Settings | None = None,
                 ocr: PaddleReader | TrOCRReader | None = None,
                 vision: GeminiReader | QwenReader | QwenLocalReader | None = None, correction: bool = True,
                 ensemble_readers: dict[str, object] | None = None) -> None:
        self.settings = settings or Settings()
        self._lock = RLock()
        self.correction = correction
        self.ocr = ocr
        self.vision = vision
        self._ocr_attempted = ocr is not None
        self._ocr_error: ReaderUnavailable | None = None
        self._vision_attempted = vision is not None
        self._vision_error: ReaderUnavailable | None = None
        self._init_warnings: list[str] = []
        self._ensemble_readers = ensemble_readers

    def _get_ensemble(self) -> dict[str, object]:
        if self._ensemble_readers is not None:
            return self._ensemble_readers
        readers: dict[str, object] = {}
        try:
            readers["trocr"] = TrOCRReader(self.settings.trocr_model, self.settings.max_regions)
        except Exception as exc:
            self._init_warnings.append(f"TrOCR unavailable: {type(exc).__name__}")
        if self.settings.gemini_api_key:
            try:
                readers["gemini"] = GeminiReader(self.settings.gemini_api_key,
                                                  self.settings.gemini_model,
                                                  self.settings.request_timeout_seconds,
                                                  [m.strip() for m in self.settings.gemini_fallback_models.split(",") if m.strip()])
            except Exception as exc:
                self._init_warnings.append(f"Gemini unavailable: {type(exc).__name__}")
        else:
            self._init_warnings.append("Gemini key is not configured")
        try:
            if self.settings.qwen_backend == "ollama":
                readers["qwen"] = QwenLocalReader(self.settings.qwen_local_model,
                                                   self.settings.request_timeout_seconds)
            elif self.settings.qwen_api_key:
                readers["qwen"] = QwenReader(self.settings.qwen_api_key,
                                              self.settings.qwen_model,
                                              self.settings.qwen_base_url,
                                              self.settings.request_timeout_seconds)
            else:
                self._init_warnings.append("Qwen key is not configured")
        except Exception as exc:
            self._init_warnings.append(f"Qwen unavailable: {type(exc).__name__}")
        self._ensemble_readers = readers
        return readers

    def _get_ocr(self) -> PaddleReader | TrOCRReader | None:
        if not self._ocr_attempted:
            self._ocr_attempted = True
            if self.settings.vision_reader == "trocr":
                try:
                    self.ocr = TrOCRReader(self.settings.trocr_model, self.settings.max_regions)
                except Exception as exc:
                    self._ocr_error = ReaderUnavailable("model", "TrOCR")
                    self._init_warnings.append(f"TrOCR unavailable: {type(exc).__name__}")
            elif self.settings.enable_paddle:
                try:
                    self.ocr = PaddleReader()
                except Exception as exc:
                    self._init_warnings.append(f"PaddleOCR unavailable: {type(exc).__name__}")
        return self.ocr

    def _get_vision(self) -> GeminiReader | QwenReader | QwenLocalReader | None:
        if not self._vision_attempted:
            self._vision_attempted = True
            if self.settings.vision_reader == "qwen" and self.settings.qwen_backend == "ollama":
                try:
                    self.vision = QwenLocalReader(self.settings.qwen_local_model,
                                                  self.settings.request_timeout_seconds)
                except ReaderUnavailable as exc:
                    self._vision_error = exc
                    self._init_warnings.append(str(exc))
            elif self.settings.vision_reader == "qwen" and self.settings.qwen_api_key:
                try:
                    self.vision = QwenReader(self.settings.qwen_api_key,
                                             self.settings.qwen_model,
                                             self.settings.qwen_base_url,
                                             self.settings.request_timeout_seconds)
                except Exception as exc:
                    self._init_warnings.append(f"Qwen-VL unavailable: {type(exc).__name__}")
            elif self.settings.vision_reader == "gemini" and self.settings.gemini_api_key:
                try:
                    self.vision = GeminiReader(self.settings.gemini_api_key,
                                               self.settings.gemini_model,
                                               self.settings.request_timeout_seconds,
                                               [m.strip() for m in self.settings.gemini_fallback_models.split(",") if m.strip()])
                except Exception as exc:
                    self._init_warnings.append(f"Gemini unavailable: {type(exc).__name__}")
        return self.vision

    def capabilities(self) -> dict:
        if self.settings.vision_reader == "ensemble":
            readers = self._get_ensemble()
            return {"vision_configured": bool(readers), "vision_provider": "Gemini + Qwen-VL + TrOCR",
                    "vision_model": ", ".join(readers), "readers": {name: name in readers for name in ("gemini", "qwen", "trocr")},
                    "paddle_enabled": False, "paddle_loaded": False,
                    "reader_warnings": list(self._init_warnings), "geometry_fallback": "opencv"}
        qwen = self.settings.vision_reader == "qwen"
        trocr = self.settings.vision_reader == "trocr"
        if qwen and self.settings.qwen_backend == "ollama" and not self._vision_attempted:
            self._get_vision()
        if trocr and not self._ocr_attempted:
            self._get_ocr()
        return {"vision_configured": self.ocr is not None if trocr else ((self.vision is not None if self.settings.qwen_backend == "ollama" else bool(self.settings.qwen_api_key)) if qwen else bool(self.settings.gemini_api_key)),
                "vision_provider": "TrOCR" if trocr else (("Qwen-VL local" if self.settings.qwen_backend == "ollama" else "Qwen-VL") if qwen else "Gemini"),
                "vision_model": self.settings.trocr_model if trocr else ((self.settings.qwen_local_model if self.settings.qwen_backend == "ollama" else self.settings.qwen_model) if qwen else self.settings.gemini_model),
                "paddle_enabled": self.settings.enable_paddle,
                "paddle_loaded": isinstance(self.ocr, PaddleReader),
                "reader_warnings": list(self._init_warnings),
                "geometry_fallback": "opencv"}

    @serialized
    def transcribe(self, image: Image.Image, filename: str = "image",
                   page_audit_text: str | None = None,
                   correction: bool | None = None) -> Transcription:
        if self.settings.vision_reader == "ensemble":
            readers = self._get_ensemble()
            if not readers:
                raise RuntimeError("No handwriting readers are available; check TrOCR, Gemini, and Qwen setup")
            return transcribe_ensemble(image, filename, readers, list(self._init_warnings),
                                       self.settings.max_regions, self.settings.ensemble_crop_limit,
                                       self.settings.language_hint)
        ocr = self._get_ocr()
        vision = self._get_vision()
        if self.settings.vision_reader == "trocr" and not ocr:
            raise self._ocr_error or ReaderUnavailable("config", "TrOCR")
        if self.settings.vision_reader == "qwen" and not vision:
            raise self._vision_error or ReaderUnavailable("config", "Qwen-VL local" if self.settings.qwen_backend == "ollama" else "Qwen-VL")
        if not ocr and not vision:
            raise ReaderUnavailable("config", "Qwen-VL local" if self.settings.vision_reader == "qwen" and self.settings.qwen_backend == "ollama" else ("Qwen-VL" if self.settings.vision_reader == "qwen" else "Gemini"))
        warnings = list(self._init_warnings)
        page_candidate = page_audit_text
        page_error: Exception | None = None
        if vision and page_candidate is None:
            try:
                page_candidate = vision.baseline(image)
            except Exception as exc:
                page_error = exc
                warnings.append(f"Page audit failed: {type(exc).__name__}")
        lines: list[DetectedLine] = []
        if ocr:
            try:
                lines = ocr.read(enhance(image))
            except Exception as exc:
                warnings.append(f"OCR failed: {type(exc).__name__}")
        if isinstance(page_error, ReaderUnavailable):
            if not lines:
                raise page_error
            warnings.append("Vision reader skipped after page request failed; OCR-only text needs review")
            vision = None
        if not lines:
            if self.settings.vision_reader == "trocr":
                lines = [DetectedLine(Box(x1=0, y1=0, x2=image.width, y2=image.height),
                                      Candidate(source="trocr", text="[[illegible]]"))]
                warnings.append("TrOCR could not isolate a text line; verify this page manually")
        if not lines:
            boxes = detect_lines(image)
            lines = [DetectedLine(box, Candidate(source="geometry_only", text="")) for box in boxes]
            if boxes:
                warnings.append("OpenCV located ink regions; no independent OCR reading is available")
        if len(lines) > self.settings.max_regions:
            warnings.append("Region limit reached; remaining writing requires page verification")
            lines = lines[:self.settings.max_regions]
        if not lines:
            # A single page-wide region is honest about missing localization.
            lines = [DetectedLine(Box(x1=0, y1=0, x2=image.width, y2=image.height),
                                  Candidate(source="layout_fallback", text=""))]
            warnings.append("Text localization unavailable; page is one review region")

        if not ocr and page_candidate:
            # With only one remote reader, an extra request for every detected line
            # quickly exhausts free-tier quotas without creating independent evidence.
            page_lines = [line.strip() for line in page_candidate.splitlines() if line.strip()]
            aligned = (len(page_lines) == len(lines) and
                       all(line.candidate.source != "layout_fallback" for line in lines))
            if aligned:
                regions = [Region(id=f"r{index + 1}", box=line.box,
                                  kind="margin" if _margin(line.box, image.width) else "body",
                                  text=page_lines[index], status="review", confidence=0.25,
                                  candidates=[Candidate(source=self.settings.vision_reader + "_page", text=page_lines[index])],
                                  reasons=["Only one reader was available", "Page reading aligned to detected line by order"])
                           for index, line in enumerate(lines)]
            else:
                regions = [Region(id="r1", box=Box(x1=0, y1=0, x2=image.width, y2=image.height),
                                  text=page_candidate, status="review", confidence=0.2,
                                  candidates=[Candidate(source=self.settings.vision_reader + "_page", text=page_candidate)],
                                  reasons=["Only one reader was available", "Page reading could not be aligned to detected lines"])]
                warnings.append("Word locations are page-level because text and detected lines could not be aligned")
            summary = populate_words(regions)
            for region in regions:
                scope = "line_estimate" if aligned else "page"
                for word in region.words:
                    word.source_scope = scope
            return Transcription(filename=filename, width=image.width, height=image.height,
                                 text="\n".join(region.text for region in regions), page_status="review",
                                 page_candidate=page_candidate, regions=regions, warnings=warnings,
                                 summary=summary, model_info={"ocr": "unavailable",
                                                              "vision": getattr(vision, "model", "injected-reader")})

        regions: list[Region] = []
        for index, line in enumerate(lines):
            original_crop = crop(image, line.box)
            candidates = [] if line.candidate.source in {"layout_fallback", "geometry_only"} else [line.candidate]
            kind = "margin" if _margin(line.box, image.width) else "body"
            if vision:
                try:
                    if line.candidate.source == "layout_fallback":
                        if page_candidate:
                            vlm_text, vlm_kind = page_candidate, "unknown"
                        else:
                            vlm_text, vlm_kind = vision.read_crop(original_crop)
                    else:
                        vlm_text, vlm_kind = vision.read_crop(original_crop)
                    candidates.append(Candidate(source=self.settings.vision_reader, text=vlm_text))
                    if vlm_kind == "crossed_out":
                        kind = "crossed_out"
                except Exception as exc:
                    warnings.append(f"Vision read failed for region {index + 1}: {type(exc).__name__}")

            primary = candidates[0].text if candidates else ""
            secondary = candidates[1].text if len(candidates) > 1 else ""
            proposed = secondary or primary
            if (self.correction if correction is None else correction) and vision and len(candidates) > 1 and _normalized(primary) != _normalized(secondary):
                context = " ".join(r.text for r in regions[-2:])
                try:
                    corrected = vision.correct(original_crop, primary, secondary, context)
                    if corrected.strip():
                        proposed = corrected
                except Exception as exc:
                    warnings.append(f"Correction failed for region {index + 1}: {type(exc).__name__}")
            if not proposed.strip():
                proposed = "[[illegible]]"
            status, confidence, reasons = _decide(candidates, proposed,
                                                  quality_score(original_crop),
                                                  kind == "crossed_out",
                                                  self.settings.accept_threshold)
            regions.append(Region(id=f"r{index + 1}", box=line.box, kind=kind,
                                  candidates=candidates, text=proposed,
                                  status=status, confidence=confidence, reasons=reasons))

        output_lines = ["[crossed out]" if r.status == "crossed_out" else r.text
                        for r in regions]
        output_text = "\n".join(output_lines)
        if not any(r.candidates for r in regions):
            if isinstance(page_error, ReaderUnavailable):
                raise page_error
            raise RuntimeError("All recognition readers failed; retry the image")
        page_status = "accepted" if all(r.status == "accepted" for r in regions) else "review"
        if page_candidate and _normalized(page_candidate) != _normalized(output_text):
            page_status = "review"
            warnings.append("Page-wide reader disagrees with detected lines; check for missing text")
        if (vision and page_candidate is None) or any("Region limit" in w for w in warnings):
            page_status = "review"
        summary = populate_words(regions)
        return Transcription(filename=filename, width=image.width, height=image.height,
                             text=output_text, page_status=page_status,
                             page_candidate=page_candidate, regions=regions,
                             warnings=warnings,
                             summary=summary,
                             model_info={"ocr": getattr(ocr, "model_name", "PaddleOCR") if ocr else "unavailable",
                                         "vision": getattr(vision, "model", "injected-reader") if vision else "unavailable"})

    @serialized
    def baseline(self, image: Image.Image, filename: str = "image") -> BaselineResult:
        if self.settings.vision_reader == "ensemble":
            return baseline_ensemble(image, filename, self._get_ensemble())
        if self.settings.vision_reader == "trocr":
            ocr = self._get_ocr()
            if not ocr:
                raise self._ocr_error or ReaderUnavailable("config", "TrOCR")
            lines = ocr.read(image)
            if not lines:
                raise ReaderUnavailable("provider", "TrOCR")
            return BaselineResult(filename=filename,
                                  text="\n".join(line.candidate.text for line in lines),
                                  model=self.settings.trocr_model)
        vision = self._get_vision()
        if not vision:
            raise self._vision_error or ReaderUnavailable("config", "Qwen-VL local" if self.settings.vision_reader == "qwen" and self.settings.qwen_backend == "ollama" else ("Qwen-VL" if self.settings.vision_reader == "qwen" else "Gemini"))
        text = vision.baseline(image)
        return BaselineResult(filename=filename, text=text,
                              model=getattr(vision, "model", "injected-reader"))
