import unittest

from PIL import Image, ImageDraw

from pipeline.config import Settings
from pipeline.readers import DetectedLine, ReaderUnavailable
from pipeline.schemas import Box, Candidate
from pipeline.service import Digitizer


class FakeTrOCR:
    model_name = "test-trocr"

    def read(self, image):
        return [DetectedLine(Box(x1=10, y1=10, x2=290, y2=70),
                             Candidate(source="trocr", text="Read these notes"))]


class FakeVision:
    def __init__(self, text, fail=False):
        self.text = text
        self.fail = fail
        self.model = "test-vision"

    def baseline(self, image):
        if self.fail:
            raise ReaderUnavailable("quota")
        return self.text

    def read_crop(self, image):
        if self.fail:
            raise AssertionError("Failed page reader must not get crop calls")
        return self.text, "body"


class EnsembleTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (300, 80), "white")
        ImageDraw.Draw(self.image).text((10, 20), "Read these notes", fill="black")
        self.settings = Settings(vision_reader="ensemble", enable_paddle=False, language_hint="auto")

    def test_three_readers_preserve_evidence_and_flag_disagreement(self):
        readers = {"trocr": FakeTrOCR(), "gemini": FakeVision("Read these notes"),
                   "qwen": FakeVision("Read those notes")}
        result = Digitizer(self.settings, ensemble_readers=readers).transcribe(self.image)
        self.assertEqual(result.text, "Read these notes")
        self.assertEqual([c.source for c in result.regions[0].candidates],
                         ["trocr", "gemini", "qwen"])
        self.assertEqual(result.regions[0].words[1].eligibility, "needs_review")
        self.assertEqual(result.page_status, "review")
        self.assertEqual([c.source for c in result.page_candidates],
                         ["gemini_page", "qwen_page", "trocr_page"])

    def test_quota_failure_keeps_independent_results(self):
        readers = {"trocr": FakeTrOCR(), "gemini": FakeVision("", fail=True),
                   "qwen": FakeVision("Read these notes")}
        result = Digitizer(self.settings, ensemble_readers=readers).transcribe(self.image)
        self.assertEqual(result.text, "Read these notes")
        self.assertEqual([c.source for c in result.regions[0].candidates], ["trocr", "qwen"])
        self.assertTrue(any("gemini page reading unavailable" in warning for warning in result.warnings))

    def test_no_line_boxes_links_text_to_page_and_requires_review(self):
        class NoLines:
            model_name = "test-trocr"
            def read(self, image):
                return []

        readers = {"trocr": NoLines(), "qwen": FakeVision("Read these notes")}
        result = Digitizer(self.settings, ensemble_readers=readers).transcribe(self.image)
        self.assertTrue(all(word.source_scope == "page" for word in result.regions[0].words))
        self.assertTrue(all(word.eligibility == "needs_review" for word in result.regions[0].words))

    def test_no_readable_lines_is_explicitly_illegible(self):
        class NoLines:
            model_name = "test-trocr"
            def read(self, image):
                return []

        result = Digitizer(self.settings, ensemble_readers={"trocr": NoLines()}).transcribe(self.image)
        self.assertEqual(result.text, "[[illegible]]")
        self.assertEqual(result.regions[0].words[0].eligibility, "illegible")


if __name__ == "__main__":
    unittest.main()
