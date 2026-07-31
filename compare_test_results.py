#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比多个实验分支/目录下 test.py 跑出来的日志，输出关键指标对比表。

用法（在任意一个分支目录下执行都行，各分支目录是兄弟目录）：
    python3 compare_test_results.py br_nearby_rank_base br_nearby_rank_dev tf_train_base_new_try

默认去脚本所在目录的上一级里找 <实验名>/log/test_log_* ，同一实验有多份日志时取最新修改时间的一份。
"""
import argparse
import glob
import os
import re
import sys

TASKS = ["buy", "cat", "click", "ext"]

PATTERNS = {
    "all_num": re.compile(r"all_num:(\d+)"),
    "low_score_rate": re.compile(r"low score rate:([\d.]+)"),
    "using_time": re.compile(r"using_time_training:\s*([\d.]+)"),
}
for t in TASKS:
    PATTERNS[t + "_pos"] = re.compile(r"pos_{}:(\d+)".format(t))
    PATTERNS[t + "_rate"] = re.compile(r"{}_rate:([\d.]+)".format(t))
    PATTERNS[t + "_res"] = re.compile(r"res_{}:([\d.]+)".format(t))
    PATTERNS[t + "_test"] = re.compile(
        r"test_{} auc:([\d.]+) gauc:([\d.]+) uauc:([\d.]+) size:(\d+) loss:([\d.]+), pos: (\d+)".format(t)
    )
    PATTERNS[t + "_online"] = re.compile(
        r"online_{} auc:([\d.]+) gauc:([\d.]+) uauc:([\d.]+)".format(t)
    )


def find_latest_log(exp_dir):
    candidates = glob.glob(os.path.join(exp_dir, "log", "test_log_*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def parse_log(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    result = {}
    m = PATTERNS["all_num"].search(content)
    result["all_num"] = int(m.group(1)) if m else None
    m = PATTERNS["low_score_rate"].search(content)
    result["low_score_rate"] = float(m.group(1)) if m else None
    m = PATTERNS["using_time"].search(content)
    result["using_time_sec"] = float(m.group(1)) if m else None

    for t in TASKS:
        task_result = {}
        m = PATTERNS[t + "_pos"].search(content)
        task_result["pos"] = int(m.group(1)) if m else None
        m = PATTERNS[t + "_rate"].search(content)
        task_result["pos_rate"] = float(m.group(1)) if m else None
        m = PATTERNS[t + "_res"].search(content)
        task_result["avg_pred"] = float(m.group(1)) if m else None
        m = PATTERNS[t + "_test"].search(content)
        if m:
            task_result["auc"] = float(m.group(1))
            task_result["gauc"] = float(m.group(2))
            task_result["uauc"] = float(m.group(3))
            task_result["size"] = int(m.group(4))
            task_result["loss"] = float(m.group(5))
        m = PATTERNS[t + "_online"].search(content)
        if m:
            task_result["online_auc"] = float(m.group(1))
            task_result["online_gauc"] = float(m.group(2))
            task_result["online_uauc"] = float(m.group(3))
        result[t] = task_result

    return result


def format_table(rows, headers):
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt_row(vals):
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="对比多个实验目录的 test.py 输出日志")
    parser.add_argument("experiments", nargs="+",
                        help="实验目录名（跟本脚本所在目录同级），如 br_nearby_rank_base br_nearby_rank_dev")
    parser.add_argument("--base_dir", default=None,
                        help="实验目录的上级目录，默认取本脚本所在目录的上一级")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir or os.path.dirname(script_dir)

    all_results = {}
    for exp in args.experiments:
        exp_dir = os.path.join(base_dir, exp)
        log_path = find_latest_log(exp_dir)
        if log_path is None:
            print("[WARN] {}: 找不到 log/test_log_* 文件，跳过".format(exp))
            continue
        print("[INFO] {}: 使用日志 {}".format(exp, log_path))
        all_results[exp] = parse_log(log_path)

    if not all_results:
        print("没有任何实验解析到结果")
        sys.exit(1)

    overview_headers = ["experiment", "all_num", "low_score_rate", "using_time_sec"]
    overview_rows = []
    for exp, r in all_results.items():
        overview_rows.append([exp, r.get("all_num"), r.get("low_score_rate"), r.get("using_time_sec")])
    print("\n=== 总览 ===")
    print(format_table(overview_rows, overview_headers))

    for t in TASKS:
        headers = ["experiment", "auc", "gauc", "uauc", "loss", "pos", "pos_rate", "avg_pred",
                    "online_auc", "online_gauc", "online_uauc"]
        rows = []
        for exp, r in all_results.items():
            tr = r.get(t, {})
            rows.append([
                exp,
                tr.get("auc"), tr.get("gauc"), tr.get("uauc"), tr.get("loss"),
                tr.get("pos"), tr.get("pos_rate"), tr.get("avg_pred"),
                tr.get("online_auc"), tr.get("online_gauc"), tr.get("online_uauc"),
            ])
        print("\n=== 任务: {} ===".format(t))
        print(format_table(rows, headers))


if __name__ == "__main__":
    main()
