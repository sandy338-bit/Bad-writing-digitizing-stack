"""One-to-one region matching; unmatched writing stays visible in evaluation."""
from jiwer import cer, wer


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def region_metrics(records):
    labels_total = matched = missing = accepted = accepted_wrong = errors = flagged_errors = 0
    missing_page_flagged = flagged = 0
    ranked = []
    for labels, result in records:
        labels_total += len(labels)
        pairs = []
        for index, label in enumerate(labels):
            for prediction, region in enumerate(result.regions):
                b = region.box
                overlap = iou(label["box"], [b.x1, b.y1, b.x2, b.y2])
                if overlap >= 0.3:
                    pairs.append((overlap, index, prediction))
        used_labels, used_predictions = set(), set()
        for _, index, prediction in sorted(pairs, reverse=True):
            if index in used_labels or prediction in used_predictions:
                continue
            used_labels.add(index)
            used_predictions.add(prediction)
            matched += 1
            region = result.regions[prediction]
            predicted = "[crossed out]" if region.status == "crossed_out" else region.text
            wrong = " ".join(predicted.split()) != " ".join(labels[index]["text"].split())
            automatic = region.status == "accepted"
            accepted += automatic
            accepted_wrong += automatic and wrong
            errors += wrong
            flagged += not automatic
            flagged_errors += wrong and not automatic
            ranked.append((region.confidence, wrong))
        omitted = len(labels) - len(used_labels)
        missing += omitted
        missing_page_flagged += omitted if result.page_status == "review" else 0
    curve = []
    wrong_sum = 0
    for count, (_, wrong) in enumerate(sorted(ranked, key=lambda x: x[0], reverse=True), 1):
        wrong_sum += wrong
        curve.append({"coverage": count / labels_total, "error_rate": wrong_sum / count})
    return {
        "ground_truth_regions": labels_total, "matched": matched, "missing": missing,
        "detection_coverage": matched / labels_total if labels_total else None,
        "automatic_coverage": accepted / labels_total if labels_total else None,
        "accepted_error_rate": accepted_wrong / accepted if accepted else None,
        "matched_error_flag_recall": flagged_errors / errors if errors else None,
        "matched_flag_precision": flagged_errors / flagged if flagged else None,
        "missing_text_page_flag_recall": missing_page_flagged / missing if missing else None,
        "risk_coverage_curve": curve,
        "note": "One-to-one IoU >= 0.3 matching; region metrics need ground-truth boxes",
    }


def text_metrics(references, predictions):
    character_error = cer(references, predictions)
    word_error = wer(references, predictions)
    return {"cer": character_error, "wer": word_error,
            "character_accuracy": max(0.0, 1.0 - character_error),
            "word_accuracy": max(0.0, 1.0 - word_error)}
