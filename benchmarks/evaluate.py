"""Evaluate a writer-disjoint JSONL manifest; optionally run model ablations."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from benchmarks.metrics import region_metrics, text_metrics
from pipeline.images import load_image
from pipeline.schemas import BaselineResult
from pipeline.service import Digitizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/latest.json"))
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--ablation", action="store_true", help="Also run individual-reader comparisons; incurs additional calls")
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    writers = {}
    for row in all_rows:
        if row.get("writer_id"):
            writers.setdefault(row["writer_id"], set()).add(row.get("split"))
    if any(len(splits) > 1 for splits in writers.values()):
        raise SystemExit("Writer leakage: the same writer appears in multiple splits")
    rows = [row for row in all_rows if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"Manifest has no split={args.split} rows")
    engine = Digitizer()
    references, baselines, predictions, details, regions = [], [], [], [], []
    ensemble = engine.settings.vision_reader == "ensemble"
    ablations = ({"gemini_only": [], "qwen_only": [], "trocr_only": []} if ensemble else
                 {"without_correction": [], "ocr_only": []})
    for row in rows:
        path = (args.manifest.parent / row["image"]).resolve()
        image = load_image(path.read_bytes())
        if ensemble:
            result = engine.transcribe(image, path.name)
            page_reader = next((c for c in result.page_candidates if c.source == "qwen_page"), None)
            page_reader = page_reader or next((c for c in result.page_candidates if c.source == "gemini_page"), None)
            baseline = (BaselineResult(filename=path.name, text=page_reader.text, model=page_reader.source)
                        if page_reader else engine.baseline(image, path.name))
        else:
            baseline = engine.baseline(image, path.name)
            result = engine.transcribe(image, path.name, baseline.text)
        references.append(row["text"])
        baselines.append(baseline.text)
        predictions.append(result.text)
        regions.append((row.get("regions", []), result))
        details.append({"image": row["image"], "category": row.get("category", "unspecified"),
                        "truth": row["text"], "baseline": baseline.model_dump(), "pipeline": result.model_dump()})
        if args.ablation:
            if ensemble:
                for reader_name in ("gemini", "qwen", "trocr"):
                    try:
                        single = Digitizer(replace(engine.settings, vision_reader=reader_name))
                        ablations[reader_name + "_only"].append(single.transcribe(image, path.name).text)
                    except Exception as exc:
                        raise SystemExit(f"{reader_name} ablation failed on {row['image']}: {exc}") from exc
            else:
                without_correction = Digitizer(engine.settings, engine.ocr, engine.vision, correction=False)
                ablations["without_correction"].append(without_correction.transcribe(image, path.name, baseline.text).text)
                if engine.ocr is None:
                    raise SystemExit("OCR-only ablation requires a working independent OCR reader")
                ocr_only = Digitizer(replace(engine.settings, gemini_api_key="", qwen_api_key=""), ocr=engine.ocr)
                ablations["ocr_only"].append(ocr_only.transcribe(image, path.name).text)
    categories = {}
    for category in {row["category"] for row in details}:
        subset = [row for row in details if row["category"] == category]
        categories[category] = {
            "pages": len(subset),
            "baseline": text_metrics([r["truth"] for r in subset], [r["baseline"]["text"] for r in subset]),
            "pipeline": text_metrics([r["truth"] for r in subset], [r["pipeline"]["text"] for r in subset]),
        }
    report = {"pages": len(rows), "split": args.split, "model": engine.capabilities()["vision_model"],
              "accept_threshold": engine.settings.accept_threshold,
              "baseline": text_metrics(references, baselines), "pipeline": text_metrics(references, predictions),
              "region_metrics": region_metrics(regions), "categories": categories, "details": details}
    report["meets_90_percent_word_accuracy"] = report["pipeline"]["word_accuracy"] >= .9
    if args.ablation:
        report["ablations"] = {name: text_metrics(references, values) for name, values in ablations.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
