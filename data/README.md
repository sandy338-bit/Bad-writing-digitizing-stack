# Annotation rules

Collect genuine difficult handwriting from multiple writers. Obtain permission before using private pages, and remove personal details from any shared examples.

For each page, record its exact visible text in reading order. Preserve the original spelling, numbers, punctuation, and script. Use `[[illegible]]` only where a human annotator also cannot read the strokes. Represent fully crossed-out text as `[crossed out]`; transcribe a replacement separately. Place margin notes at their visual reading point or make a consistent page-level ordering rule before evaluation.

The benchmark manifest is JSONL. Required fields are `image`, `text`, and `split`; use `split` values `development` or `test`. Add a `writer_id` so pages by one writer stay in one split. Optional `regions` entries contain a pixel `box` and exact `text`. The benchmark reads only `split=test` rows.

Do not change test transcriptions or the acceptance threshold after inspecting test predictions. Two independent annotators should resolve ambiguous ground truth before judging where possible.
