from io import BytesIO
import json
from pathlib import Path
import tempfile
import time
import unittest
from threading import Event

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from pipeline.api import create_app
from pipeline.config import Settings
from pipeline.export import render_text
from pipeline.jobs import Jobs, QueueFullError
from pipeline.schemas import Box, Candidate, CorrectionRequest, Region, RegionEdit, Transcription
from pipeline.service import Digitizer, _decide
from pipeline.store import ConflictError, Store
from pipeline.images import detect_lines, load_image
from pipeline.readers import GeminiReader, QwenReader, QwenLocalReader, ReaderUnavailable
from pipeline.words import populate_words
from tests import test_service


def picture():
    output = BytesIO()
    Image.new("RGB", (200, 100), "white").save(output, format="PNG")
    return output.getvalue()


def result(filename="sample.png"):
    value = Transcription(filename=filename, width=200, height=100, text="10 mg", regions=[
        Region(id="r1", box=Box(x1=10, y1=10, x2=150, y2=50), text="10 mg",
               status="review", confidence=0.5, candidates=[Candidate(source="test", text="10 mg")])])
    value.summary = populate_words(value.regions)
    return value


class Engine:
    fail = False
    last_correction = None

    def capabilities(self):
        return {"test": True}

    def transcribe(self, image, filename, correction=True):
        self.last_correction = correction
        if self.fail == "quota":
            raise ReaderUnavailable("quota")
        if self.fail:
            raise RuntimeError("provider down")
        return result(filename)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = Engine()
        self.settings = Settings(database_path=str(Path(self.temp.name) / "test.sqlite3"),
                                 enable_paddle=False, gemini_api_key="", max_upload_mb=1)
        self.client = TestClient(create_app(self.settings, self.engine))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def submit(self):
        response = self.client.post("/v1/documents", files={"file": ("sample.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 202)
        return response.json()["id"]

    def wait(self, identifier):
        for _ in range(200):
            document = self.client.get(f"/v1/documents/{identifier}").json()
            if document["status"] in {"completed", "failed"}:
                return document
            time.sleep(0.01)
        self.fail("job did not finish")

    def test_complete_frontend_workflow(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Upload handwriting", page.text)
        identifier = self.submit()
        document = self.wait(identifier)
        self.assertEqual(document["status"], "completed")
        base = f"/v1/documents/{identifier}"
        self.assertEqual(self.client.get("/v1/documents").json()["total"], 1)
        exported = self.client.get(base + "/export").text
        self.assertIn("[[review: 10]] [[review: mg]]", exported)
        self.assertIn("[[page requires verification]]", exported)
        for suffix in ["/image", "/image?annotated=true", "/regions/r1/image"]:
            response = self.client.get(base + suffix)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(load_image(response.content).mode, "RGB")
        queue = self.client.get(base + "/review-queue").json()
        self.assertEqual(queue["total"], 2)
        self.assertEqual(queue["items"][0]["risk_tag"], "number")
        self.assertEqual(self.client.get(base + "/review-queue?min_agreement=0").status_code, 422)
        word_id = queue["items"][0]["id"]
        self.assertEqual(load_image(self.client.get(base + f"/words/{word_id}/image").content).mode, "RGB")
        edited = self.client.patch(base + "/corrections", json={"revision": 1,
            "edits": [{"id": "r1", "text": "20 mg"}], "page_reviewed": True})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["revision"], 2)
        self.assertEqual(edited.json()["result"]["regions"][0]["status"], "verified")
        self.assertEqual(self.client.get(base + "/export").text, "20 mg")
        history = self.client.get(base + "/history").json()["items"]
        self.assertEqual(history[0]["changes"]["edits"][0]["before"], "10 mg")
        self.assertEqual(self.client.get(base + "/export?format=json").json()["result"]["regions"][0]["candidates"][0]["text"], "10 mg")
        self.assertIn("```text", self.client.get(base + "/export?format=md").text)
        self.assertEqual(self.client.delete(base).status_code, 204)
        self.assertEqual(self.client.get(base).status_code, 404)

    def test_word_correction_and_illegible_action(self):
        identifier = self.submit()
        self.wait(identifier)
        base = f"/v1/documents/{identifier}"
        queue = self.client.get(base + "/review-queue").json()["items"]
        number = next(item for item in queue if item["risk_tag"] == "number")
        edited = self.client.patch(base + "/corrections", json={"revision": 1,
            "word_edits": [{"id": number["id"], "text": "20", "action": "replace"}]})
        self.assertEqual(edited.status_code, 200)
        words = edited.json()["result"]["regions"][0]["words"]
        self.assertEqual(words[0]["text"], "20")
        self.assertEqual(words[0]["eligibility"], "verified")
        other = words[1]["id"]
        marked = self.client.patch(base + "/corrections", json={"revision": 2,
            "word_edits": [{"id": other, "text": "", "action": "illegible"}]})
        self.assertEqual(marked.status_code, 200)
        self.assertIn("[[illegible]]", marked.json()["result"]["text"])
        self.assertEqual(self.client.get(base + "/review-queue").json()["total"], 0)
        reviewed = self.client.post(base + "/review-page", json={"revision": 3})
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["result"]["summary"]["illegible"], 1)

    def test_upload_correction_option_is_used(self):
        response = self.client.post("/v1/documents?correction=false",
                                    files={"file": ("sample.png", picture(), "image/png")})
        self.assertEqual(response.status_code, 202)
        document = self.wait(response.json()["id"])
        self.assertEqual(document["status"], "completed")
        self.assertFalse(document["correction_enabled"])
        self.assertFalse(self.engine.last_correction)
        self.assertFalse(self.client.get("/v1/documents").json()["items"][0]["correction_enabled"])

    def test_old_single_reader_score_is_not_shown_as_low_risk(self):
        identifier = self.submit()
        self.wait(identifier)
        with self.client.app.state.store.connection() as db:
            raw = json.loads(db.execute("SELECT result FROM documents WHERE id=?", (identifier,)).fetchone()[0])
            raw["regions"][0]["words"][0]["confidence"] = .95
            raw["regions"][0]["words"][0]["risk"] = .06
            db.execute("UPDATE documents SET result=? WHERE id=?", (json.dumps(raw), identifier))
        word = self.client.get(f"/v1/documents/{identifier}").json()["result"]["regions"][0]["words"][0]
        self.assertLessEqual(word["confidence"], .35)
        self.assertGreaterEqual(word["risk"], .8)

    def test_page_review_requires_resolved_words(self):
        identifier = self.submit()
        self.wait(identifier)
        base = f"/v1/documents/{identifier}"
        self.assertEqual(self.client.post(base + "/review-page", json={"revision": 1}).status_code, 409)
        word_ids = [w["id"] for w in self.client.get(base + "/review-queue").json()["items"]]
        edited = self.client.patch(base + "/corrections", json={"revision": 1,
            "word_edits": [{"id": word_id, "action": "accept"} for word_id in word_ids]})
        self.assertEqual(edited.status_code, 200)
        reviewed = self.client.post(base + "/review-page", json={"revision": 2})
        self.assertEqual(reviewed.status_code, 200)
        self.assertEqual(reviewed.json()["result"]["page_status"], "accepted")

    def test_stale_revision_and_bad_region_rejected(self):
        identifier = self.submit()
        self.wait(identifier)
        url = f"/v1/documents/{identifier}/corrections"
        self.assertEqual(self.client.patch(url, json={"revision": 1, "edits": [{"id": "missing", "text": "x"}]}).status_code, 422)
        body = {"revision": 1, "edits": [{"id": "r1", "text": "x"}]}
        self.assertEqual(self.client.patch(url, json=body).status_code, 200)
        self.assertEqual(self.client.patch(url, json=body).status_code, 409)

    def test_failed_job_can_retry(self):
        self.engine.fail = True
        identifier = self.submit()
        self.assertEqual(self.wait(identifier)["status"], "failed")
        self.engine.fail = False
        self.assertEqual(self.client.post(f"/v1/documents/{identifier}/retry").status_code, 202)
        self.assertEqual(self.wait(identifier)["status"], "completed")
        self.assertEqual(self.client.post(f"/v1/documents/{identifier}/retry").status_code, 409)

    def test_quota_failure_is_actionable(self):
        self.engine.fail = "quota"
        document = self.wait(self.submit())
        self.assertEqual(document["status"], "failed")
        self.assertIn("quota is exhausted", document["error"])

    def test_failed_job_can_be_reviewed_manually(self):
        self.engine.fail = True
        identifier = self.submit()
        self.assertEqual(self.wait(identifier)["status"], "failed")
        base = f"/v1/documents/{identifier}"
        manual = self.client.post(base + "/manual-transcription",
                                  json={"revision": 1, "text": "Hello [[illegible]]"})
        self.assertEqual(manual.status_code, 200)
        self.assertEqual(manual.json()["result"]["regions"][0]["words"][0]["source_scope"], "page")
        crop = self.client.get(base + "/words/r1-w1/image")
        self.assertEqual(load_image(crop.content).size, (200, 100))
        self.assertEqual(self.client.get(base + "/review-queue").status_code, 200)
        self.assertEqual(self.client.post(base + "/manual-transcription",
                                          json={"revision": 1, "text": "again"}).status_code, 409)

    def test_invalid_uploads(self):
        for name, content, media, expected in [("bad.txt", b"x", "text/plain", 415),
                ("bad.png", b"bad", "image/png", 422),
                ("big.png", b"x" * (1024 * 1024 + 1), "image/png", 413)]:
            response = self.client.post("/v1/documents", files={"file": (name, content, media)})
            self.assertEqual(response.status_code, expected)

    def test_missing_and_invalid_export(self):
        self.assertEqual(self.client.get("/v1/documents/missing").status_code, 404)
        self.assertEqual(self.client.get("/v1/documents/missing/export?format=exe").status_code, 422)
        self.assertEqual(self.client.get("/v1/documents?limit=0").status_code, 422)


class ReliabilityTests(unittest.TestCase):
    def test_qwen_reader_request_and_json_response(self):
        import httpx
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content":
                '{"text":"Hello [[illegible]]","kind":"unknown"}'}}]})
        reader = QwenReader("test-key", "qwen3-vl-plus",
                            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                            client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.assertEqual(reader.baseline(Image.new("RGB", (20, 20), "white")),
                         "Hello [[illegible]]")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-key")
        self.assertIn("data:image/jpeg;base64,", requests[0].content.decode())

    def test_qwen_missing_key_cannot_silently_use_gemini(self):
        settings = Settings(vision_reader="qwen", qwen_backend="dashscope", qwen_api_key="", gemini_api_key="old-key",
                            enable_paddle=False)
        engine = Digitizer(settings)
        self.assertEqual(engine.capabilities()["vision_provider"], "Qwen-VL")
        self.assertFalse(engine.capabilities()["vision_configured"])
        with self.assertRaises(ReaderUnavailable) as context:
            engine.transcribe(Image.new("RGB", (20, 20), "white"))
        self.assertEqual(context.exception.reason, "config")
        self.assertIn("DASHSCOPE_API_KEY", str(context.exception))

    def test_local_qwen_reader_uses_image_and_returns_json(self):
        import httpx
        requests = []
        def respond(request):
            requests.append(request)
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:3b"}]})
            return httpx.Response(200, json={"message": {"content":
                '{"text":"Hello","kind":"body"}'}})
        reader = QwenLocalReader(client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.assertEqual(reader.baseline(Image.new("RGB", (20, 20), "white")), "Hello")
        payload = json.loads(requests[-1].content)
        self.assertEqual(payload["model"], "qwen2.5vl:3b")
        self.assertTrue(payload["messages"][0]["images"][0])

    def test_gemini_quota_error_is_specific(self):
        class ProviderError(Exception):
            def __init__(self, code):
                self.code = code
        class Models:
            def generate_content(self, model, **kwargs):
                raise ProviderError(429 if model == "current" else 404)
        reader = GeminiReader("test-key", "current", fallback_models=["old"],
                              client=type("Client", (), {"models": Models()})())
        with self.assertRaises(ReaderUnavailable) as context:
            reader.baseline(Image.new("RGB", (20, 20), "white"))
        self.assertEqual(context.exception.reason, "quota")
        self.assertEqual(reader.models, ["current"])

    def test_gemini_reader_decodes_supported_json(self):
        class Models:
            def generate_content(self, model, **kwargs):
                return type("Response", (), {"text":
                    '{"text":"Hello [[illegible]]","kind":"unknown"}'})()
        reader = GeminiReader("test-key", "gemini-3.5-flash-lite",
                              client=type("Client", (), {"models": Models()})())
        self.assertEqual(reader.baseline(Image.new("RGB", (20, 20), "white")),
                         "Hello [[illegible]]")

    def test_page_reading_avoids_extra_crop_calls_without_ocr(self):
        class Vision:
            model = "test"
            calls = 0
            def baseline(self, image):
                self.calls += 1
                return "Hello world"
            def read_crop(self, image):
                raise AssertionError("single-reader crops should not consume more API calls")
        vision = Vision()
        image = Image.new("RGB", (300, 100), "white")
        ImageDraw.Draw(image).text((20, 30), "Hello world", fill="black")
        result_value = Digitizer(Settings(enable_paddle=False, vision_reader="gemini", gemini_api_key=""), vision=vision).transcribe(image)
        self.assertEqual(vision.calls, 1)
        self.assertEqual(result_value.page_status, "review")
        self.assertIn(result_value.regions[0].words[0].source_scope, {"line_estimate", "page"})

    def test_existing_service_checks(self):
        test_service.test_agreement_is_accepted()
        test_service.test_disagreement_and_single_reader_are_flagged()
        test_service.test_no_localizer_yields_page_review()
        test_service.test_paddle_output_adapter_without_native_runtime()

    def test_subtle_word_error_never_accepted(self):
        candidates = [Candidate(source="ocr", text="patient can take tablets", confidence=0.99),
                      Candidate(source="vision", text="patient cannot take tablets")]
        self.assertEqual(_decide(candidates, candidates[1].text, 1, False)[0], "review")
        candidates[1].text = candidates[0].text
        self.assertEqual(_decide(candidates, candidates[0].text, 1, False)[0], "accepted")

    def test_word_level_agreement_accepts_only_supported_tokens(self):
        region = Region(id="r1", box=Box(x1=0, y1=0, x2=400, y2=40),
                        text="No amlodipine 5 mg", status="review", confidence=.62,
                        candidates=[Candidate(source="ocr", text="No amlodipine 5 mg", confidence=.8),
                                    Candidate(source="vision", text="No amiodarone 6 mg")])
        populate_words([region])
        self.assertEqual([word.eligibility for word in region.words],
                         ["eligible", "needs_review", "needs_review", "eligible"])

    def test_single_reader_never_has_low_review_priority(self):
        region = Region(id="r1", box=Box(x1=0, y1=0, x2=200, y2=40),
                        text="hello 20", status="review", confidence=.25,
                        candidates=[Candidate(source="gemini_page", text="hello 20")])
        populate_words([region])
        self.assertTrue(all(word.confidence <= .35 for word in region.words))
        self.assertGreaterEqual(region.words[0].risk, .65)
        self.assertGreaterEqual(region.words[1].risk, .8)

    def test_geometry_fallback(self):
        image = Image.new("RGB", (300, 100), "white")
        ImageDraw.Draw(image).text((20, 30), "hello world", fill="black")
        self.assertTrue(detect_lines(image))
        self.assertEqual(detect_lines(Image.new("RGB", (300, 100), "white")), [])

    def test_all_readers_failed_is_not_a_completed_transcription(self):
        class Broken:
            def baseline(self, image):
                raise RuntimeError("down")
            def read_crop(self, image):
                raise RuntimeError("down")
        with self.assertRaises(RuntimeError):
            Digitizer(Settings(enable_paddle=False, vision_reader="gemini", gemini_api_key=""), vision=Broken()).transcribe(Image.new("RGB", (200, 100), "white"))

    def test_restart_recovers_interrupted_work_and_preserves_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "state.sqlite3")
            store = Store(path)
            identifier = store.create("image.png", picture())
            Store(path).recover()
            self.assertEqual(store.get(identifier)["status"], "failed")
            self.assertEqual(store.image(identifier), picture())

    def test_queue_limit(self):
        started, finish = Event(), Event()
        class Blocked(Engine):
            def transcribe(self, image, filename, correction=True):
                started.set()
                finish.wait(3)
                return result(filename)
        with tempfile.TemporaryDirectory() as directory:
            jobs = Jobs(Store(str(Path(directory) / "queue.sqlite3")), Blocked(), 1)
            try:
                jobs.submit("sample.png", picture())
                self.assertTrue(started.wait(2))
                with self.assertRaises(QueueFullError):
                    jobs.submit("sample.png", picture())
            finally:
                finish.set()
                jobs.close()


if __name__ == "__main__":
    unittest.main()
