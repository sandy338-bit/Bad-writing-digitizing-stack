# HNX26EPS04 — Extreme Bad-Handwriting Digitizing Stack

## Problem statement

Conventional OCR often fails on cramped, inconsistent, crossed-out, low-quality, and mixed-script handwriting. A plausible but incorrect transcription can be more harmful than an explicit unreadable marker. This project aims to produce editable text while showing which spans require human verification.

## Pipeline and review app

1. Load the image, apply EXIF orientation, and create a light contrast-enhanced copy.
2. Detect and read text regions with PaddleOCR. Keep original pixel boxes and recognition scores.
3. With OCR available, read each original crop with the selected Qwen-VL or Gemini reader. Without OCR, read the page once to control latency. Ask for literal characters and explicit `[[illegible]]` markers.
4. When readers disagree, ask the selected vision reader to resolve only what the image supports. Keep both raw readings.
5. Assign an evidence score from agreement, OCR score, and image sharpness. Flag disagreements, substantial corrections, low-quality crops, and unreadable spans.
6. Compare the assembled text with a whole-page vision reading to flag possible omitted lines.
7. Split each region into word records with approximate image boxes, aligned reader candidates, agreement, legibility, eligibility, risk, and high-impact tags for numbers, units, negations, medical terms, and names.
8. Return the real normalized image, word crops, a risk-sorted review queue, editable text, and a page review flag through the API. Human corrections retain the original model evidence and an audit history.
9. Serve a local review workspace at `/` for photo upload, source-linked text, image zoom and overlays, candidate inspection, word actions, page verification, exports, and a side-by-side whole-page baseline. The interface never invents accuracy metrics for an unlabelled upload.

The current score is heuristic. Its acceptance threshold must be tuned on development data; it is not a calibrated correctness probability.

## Baseline and evaluation

The fixed baseline is a single whole-page reading by the selected vision model using the same input image. `benchmarks/evaluate.py` reports baseline and pipeline CER/WER on a held-out `test` split. Optional line boxes enable automatic coverage and error-flag recall. Split by writer or document and freeze the threshold before testing. Record difficult-category results manually or extend the manifest with category tags.

## What is new in the comparison

With PaddleOCR available, the comparison isolates the value of explicit localization, independent OCR and vision readings, constrained correction, and abstention over a single whole-page model. Without PaddleOCR, the pipeline shares the single page reading with the baseline and adds review and source-linking; it should not be presented as an independent accuracy gain. A follow-up ablation can disable correction or the second reader while holding the test images fixed. No accuracy improvement is claimed until annotated hard samples are evaluated.

## Current boundaries

Crossed-out classification is model-assisted and can miss strokes that OCR never localizes. The page audit flags major omissions but does not always locate them. Word boxes are estimated inside detected line boxes, so the frontend uses them as crop/highlight guides rather than character-perfect segmentation. If line alignment is uncertain, source links point to the full page. Mixed-script text is preserved as returned by the readers, but no mixed-script accuracy claim has been tested. Table reconstruction and diagram parsing are not implemented. On this Windows host, an application control policy blocks PaddlePaddle's native DLL. The app therefore uses one whole-page vision request by default and flags all text for review. Local Qwen2.5-VL 3B through Ollama completed a generated-image smoke test and a saved 1529×860 handwriting upload. That upload produced page-scoped text with all 35 recognized words requiring review; this verifies operation, not transcription accuracy. Failed uploads can continue through manual review.
