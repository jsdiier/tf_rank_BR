#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re


FILE_RE = re.compile(
    r"rolling_test_ckpt_(\d{8})_from_(\d{8})_to_(\d{8})_(\d{12,14})$"
)
METRIC_RE = re.compile(
    r"test_(buy|cat|click|ext)\s+auc:([\d.]+)\s+gauc:([\d.]+)\s+"
    r"uauc:([\d.]+).*?size:(\d+).*?pos:\s*(\d+)"
)


def main():
    parser = argparse.ArgumentParser(description="Summarize rolling-test logs into a TSV file")
    parser.add_argument("--log-dir", default="log")
    parser.add_argument("--output", default="model/rolling_metrics.tsv")
    args = parser.parse_args()

    latest = {}
    for path in glob.glob(os.path.join(args.log_dir, "rolling_test_ckpt_*")):
        match = FILE_RE.search(os.path.basename(path))
        if not match:
            continue
        ckpt_day, test_start_day, test_end_day, timestamp = match.groups()
        key = (ckpt_day, test_start_day, test_end_day)
        if key not in latest or timestamp > latest[key][0]:
            latest[key] = (timestamp, path)

    rows = []
    for (ckpt_day, test_start_day, test_end_day), (_, path) in sorted(latest.items()):
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
        for task, auc, gauc, uauc, size, pos in METRIC_RE.findall(text):
            rows.append({
                "checkpoint_day": ckpt_day,
                "test_start_day": test_start_day,
                "test_end_day": test_end_day,
                "task": task,
                "auc": auc,
                "gauc": gauc,
                "uauc": uauc,
                "size": size,
                "pos": pos,
                "log_path": path,
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fields = ["checkpoint_day", "test_start_day", "test_end_day", "task", "auc", "gauc", "uauc", "size", "pos", "log_path"]
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d metric rows to %s" % (len(rows), args.output))


if __name__ == "__main__":
    main()
