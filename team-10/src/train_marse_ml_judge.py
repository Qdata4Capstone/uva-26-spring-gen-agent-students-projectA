import argparse
import glob
import json
import os
import random

from ml_violation_detector import train_detector, rows_from_jsonl, rows_from_marse_log


def _split_stratified(rows, val_ratio=0.2, seed=17):
    positives = [row for row in rows if int(row.get("label", 0)) == 1]
    negatives = [row for row in rows if int(row.get("label", 0)) == 0]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_pos_val = max(1, int(len(positives) * val_ratio)) if positives else 0
    n_neg_val = max(1, int(len(negatives) * val_ratio)) if negatives else 0

    # val = positives[:n_pos_val]
    val = positives[:n_pos_val] + negatives[:n_neg_val]
    train = positives[n_pos_val:] + negatives[n_neg_val:]

    if not train or not val:
        return rows, []

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _metrics(detector, rows):
    if not rows:
        return {}

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for row in rows:
        y = int(row["label"])
        predVal = int(detector.predict_violation(row["attack"], row["response"]))

        if y == 1 and predVal == 1:
            tp += 1
        elif y == 0 and predVal == 1:
            fp += 1
        elif y == 0 and predVal == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)


    return {"n": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def _load_rows(args):
    rows = []
    matchedPaths = []
    logsWithRows = 0
    logsWithoutRows = 0

    if args.input_jsonl:
        rows.extend(rows_from_jsonl(args.input_jsonl))

    for pattern in args.log_glob:
        for path in glob.glob(pattern, recursive=True):
            matchedPaths.append(path)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            extracted = rows_from_marse_log(payload)
            if extracted:
                logsWithRows += 1
                rows.extend(extracted)
            else:
                logsWithoutRows += 1


    return {"rows": rows, "matched_paths": matchedPaths, "logs_with_turn_rows": logsWithRows, "logs_without_turn_rows": logsWithoutRows}


def main():
    parser = argparse.ArgumentParser(description="train marse lightweight ml violation detector")
    parser.add_argument("--input-jsonl", default="", help="jsonl with attack,response,label")
    parser.add_argument("--log-glob", action="append", default=[], help="glob for marse campaign logs (repeatable)")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2", help="sentence transformer model name")

    # parser.add_argument("--threshold", type=float, default=0.5, help="violation threshold in [0,1]")
    # parser.add_argument("--threshold", type=float, default=0.7, help="violation threshold in [0,1]")
    parser.add_argument("--threshold", type=float, default=0.6, help="violation threshold in [0,1]")

    parser.add_argument("--val-ratio", type=float, default=0.2, help="validation ratio")
    parser.add_argument("--seed", type=int, default=17, help="random seed")
    parser.add_argument("--output-artifact", default="experiments/models/marse_ml_detector.joblib", help="where to write trained artifact")

    args = parser.parse_args()
    loadInfo = _load_rows(args)
    rows = loadInfo["rows"]

    print("log ingestion:", f"matched_files={len(loadInfo['matched_paths'])}", f"logs_with_turn_rows={loadInfo['logs_with_turn_rows']}", f"logs_without_turn_rows={loadInfo['logs_without_turn_rows']}")  # keep this short and easy

    if not rows:
        matched = len(loadInfo["matched_paths"])
        withTurns = loadInfo["logs_with_turn_rows"]
        noTurns = loadInfo["logs_without_turn_rows"]
        raise RuntimeError("No rows found. " f"Matched files: {matched}, logs with turn rows: {withTurns}, logs without turn rows: {noTurns}. " "Try globs like: " "'src/experiments/standard/*_campaign_log.json' or " "'src/experiments/cross/*_campaign_log.json' or " "'src/experiments/**/*_campaign_log.json'.")

    trainRows, valRows = _split_stratified(rows, val_ratio=args.val_ratio, seed=args.seed)
    print(f"loaded rows: total={len(rows)} train={len(trainRows)} val={len(valRows)}")

    detector = train_detector(trainRows, encoder_model_name=args.encoder_model, threshold=args.threshold, random_state=args.seed)

    os.makedirs(os.path.dirname(args.output_artifact), exist_ok=True)  # make dir if needed, avoid weird fail
    detector.save(args.output_artifact)


    print(f"saved detector artifact to: {args.output_artifact}")

    trainMetrics = _metrics(detector, trainRows)
    valMetrics = _metrics(detector, valRows)

    print("train metrics:", json.dumps(trainMetrics, indent=2))
    print("val metrics:", json.dumps(valMetrics, indent=2))

    sidecarPath = args.output_artifact + ".metrics.json"
    with open(sidecarPath, "w", encoding="utf-8") as handle:
        json.dump({"encoder_model": args.encoder_model, "threshold": args.threshold, "seed": args.seed, "train_metrics": trainMetrics, "val_metrics": valMetrics, "rows_total": len(rows)}, handle, indent=2)


    print(f"saved metrics to: {sidecarPath}")


if __name__ == "__main__":
    main()
