#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment 1 本地评测脚本（学生自评用）

用法（在仓库根目录，与 solution.py / model.py 同级）：

    python evaluate.py                          # 在 test_public.csv 上评测
    python evaluate.py --student-id 2026001 --name 张三
    python evaluate.py --rows 5000              # 快速试跑前 5000 行

流程：禁用库扫描 -> 加载你的 Solution -> 10 线程并发逐样本推理 ->
计算 MAE / MSE / RMSE / 预测耗时 -> 打印并写入 result.json。

提交作业时：把 solution.py、model.py 和 result.json 一起上交。
自评分数仅供参考，最终成绩以教师在保留数据上的复测为准。
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EVALUATOR_VERSION = "2026.1.0"
ASSIGNMENT = "ML2026-Assignment-1"
DEFAULT_TEST = "test_public.csv"
CODE_FILES = ["solution.py", "model.py"]

FORBIDDEN_LIBS = [
    "sklearn", "scikit-learn", "tensorflow", "torch", "pytorch",
    "keras", "xgboost", "lightgbm", "catboost", "statsmodels",
]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_forbidden():
    hits = []
    for name in CODE_FILES:
        text = Path(name).read_text(encoding="utf-8", errors="ignore").lower()
        hits += ["{}: {}".format(name, lib) for lib in FORBIDDEN_LIBS if lib in text]
    return hits


def load_solution():
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import solution
    except Exception as e:
        raise RuntimeError("导入 solution.py 失败: {}".format(e))
    try:
        return solution.Solution()
    except Exception as e:
        raise RuntimeError("初始化 Solution 失败: {}".format(e))


def run_forward(sol, samples):
    n = len(samples)
    results = [None] * n
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(sol.forward, s): i for i, s in enumerate(samples)}
        try:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        except Exception as e:
            raise RuntimeError("forward 在第 {} 个样本上失败: {}".format(futures[fut], e))
    wall = time.perf_counter() - t0

    preds = np.empty(n, dtype=float)
    for i, r in enumerate(results):
        try:
            preds[i] = float(r["prediction"])
        except Exception:
            raise RuntimeError(
                "第 {} 个样本的返回值无法解析，应返回 {{'prediction': float}}，"
                "实际得到: {!r}".format(i, r))
    if not np.all(np.isfinite(preds)):
        bad = int(np.sum(~np.isfinite(preds)))
        raise RuntimeError("预测结果中包含 {} 个 NaN/Inf".format(bad))
    return preds, wall


def main():
    ap = argparse.ArgumentParser(description="Assignment 1 本地评测")
    ap.add_argument("--test", default=DEFAULT_TEST, help="测试数据文件")
    ap.add_argument("--rows", type=int, default=0, help="只评测前 N 行（0=全部）")
    ap.add_argument("--out", default="result.json", help="结果输出文件")
    ap.add_argument("--student-id", default="", help="学号（可选，仅写入结果文件）")
    ap.add_argument("--name", default="", help="姓名（可选）")
    args = ap.parse_args()

    for name in CODE_FILES + [args.test]:
        if not Path(name).exists():
            sys.exit("缺少文件: {}（请在仓库根目录运行）".format(name))

    hits = scan_forbidden()
    if hits:
        sys.exit("[禁止] 检测到禁用库引用:\n  " + "\n  ".join(hits))

    sol = load_solution()

    df = pd.read_csv(args.test)
    if args.rows:
        df = df.head(args.rows)
    if "age" not in df.columns:
        sys.exit("测试数据缺少 age 列")
    samples = df.drop(columns=["age"]).to_dict("records")
    y_true = df["age"].to_numpy(dtype=float)

    print("Evaluating {} rows from {} ...".format(len(df), args.test))
    preds, wall = run_forward(sol, samples)

    mae = float(np.mean(np.abs(preds - y_true)))
    mse = float(np.mean((preds - y_true) ** 2))
    rmse = float(np.sqrt(mse))

    print("=" * 50)
    print("MAE:  {:.6f}".format(mae))
    print("MSE:  {:.6f}".format(mse))
    print("RMSE: {:.6f}".format(rmse))
    print("Time: {:.3f}s".format(wall))
    print("=" * 50)

    result = {
        "assignment": ASSIGNMENT,
        "evaluator_version": EVALUATOR_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student": {"id": args.student_id, "name": args.name},
        "test_file": {"name": Path(args.test).name, "md5": md5_file(args.test),
                      "rows": int(len(df))},
        "metrics": {"mae": round(mae, 6), "mse": round(mse, 6), "rmse": round(rmse, 6)},
        "prediction_time_s": round(wall, 3),
        "code_md5": {name: md5_file(name) for name in CODE_FILES},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 {}（提交作业时与代码一起上交）".format(args.out))


if __name__ == "__main__":
    main()
