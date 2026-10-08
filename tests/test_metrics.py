import unittest

from benchmarks.metrics import region_metrics
from pipeline.schemas import Box, Region, Transcription


class MetricsTests(unittest.TestCase):
    def test_missing_regions_reduce_coverage(self):
        labels = [{"box": [0, 0, 100, 30], "text": "hello"},
                  {"box": [0, 50, 100, 80], "text": "world"}]
        result = Transcription(filename="x", width=200, height=100, text="hello", regions=[
            Region(id="r1", box=Box(x1=0, y1=0, x2=100, y2=30), text="hello",
                   status="accepted", confidence=0.9)])
        metrics = region_metrics([(labels, result)])
        self.assertEqual(metrics["detection_coverage"], 0.5)
        self.assertEqual(metrics["automatic_coverage"], 0.5)
        self.assertEqual(metrics["missing"], 1)
        self.assertEqual(metrics["missing_text_page_flag_recall"], 1)

    def test_one_prediction_cannot_match_multiple_labels(self):
        labels = [{"box": [0, 0, 100, 30], "text": "hello"}] * 2
        result = Transcription(filename="x", width=200, height=100, text="wrong", regions=[
            Region(id="r1", box=Box(x1=0, y1=0, x2=100, y2=30), text="wrong",
                   status="accepted", confidence=0.9)])
        metrics = region_metrics([(labels, result)])
        self.assertEqual(metrics["matched"], 1)
        self.assertEqual(metrics["accepted_error_rate"], 1)
        self.assertEqual(metrics["matched_error_flag_recall"], 0)


if __name__ == "__main__":
    unittest.main()
