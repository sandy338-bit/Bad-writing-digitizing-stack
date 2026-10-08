from PIL import Image

from pipeline.config import Settings
from pipeline.readers import DetectedLine
from pipeline.schemas import Box, Candidate
from pipeline.service import Digitizer, _decide


class FakeOCR:
    def read(self, image):
        return [DetectedLine(Box(x1=10, y1=10, x2=190, y2=60),
                             Candidate(source="paddleocr", text="take 2 tablets", confidence=0.95))]


class FakeVision:
    def read_crop(self, image):
        return "take 2 tablets", "body"

    def correct(self, image, primary, secondary, context):
        raise AssertionError("correction should not run on agreement")

    def baseline(self, image):
        return "take 2 tablets"


def test_agreement_is_accepted():
    result = Digitizer(Settings(enable_paddle=False, vision_reader="gemini"), ocr=FakeOCR(), vision=FakeVision()).transcribe(
        Image.new("RGB", (200, 100), "white"))
    # A blank crop has poor sharpness, so review remains the safe choice.
    assert result.regions[0].status == "review"
    assert result.regions[0].text == "take 2 tablets"
    assert result.page_status == "review"


def test_disagreement_and_single_reader_are_flagged():
    a = Candidate(source="ocr", text="10 mg", confidence=0.99)
    b = Candidate(source="vision", text="20 mg")
    assert _decide([a, b], "20 mg", 1.0, False)[0] == "review"
    assert _decide([a], "10 mg", 1.0, False)[0] == "review"
    assert _decide([a, b], "[[illegible]]", 1.0, False)[0] == "illegible"


def test_no_localizer_yields_page_review():
    result = Digitizer(Settings(enable_paddle=False, vision_reader="gemini"), vision=FakeVision()).transcribe(
        Image.new("RGB", (200, 100), "white"))
    assert result.text == "take 2 tablets"
    assert result.regions[0].status == "review"
    assert result.page_candidate == "take 2 tablets"
    assert any("localization unavailable" in warning for warning in result.warnings)


def test_paddle_output_adapter_without_native_runtime():
    from pipeline.readers import PaddleReader

    class Page:
        json = {"res": {"rec_polys": [[[10, 10], [100, 10], [100, 40], [10, 40]]],
                        "rec_texts": ["hello"], "rec_scores": [0.9]}}

    class Model:
        def predict(self, image):
            return [Page()]

    reader = object.__new__(PaddleReader)
    reader.model = Model()
    lines = reader.read(Image.new("RGB", (200, 100), "white"))
    assert len(lines) == 1
    assert lines[0].candidate.text == "hello"
    assert lines[0].box.x2 == 100
