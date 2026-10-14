#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assignment 1 本地评测脚本（学生自评用）

用法（在仓库根目录，与 solution.py / model.py 同级）：

    python evaluate.py                          # 在 test_public.csv 上评测
    python evaluate.py --student-id 2026001 --name 张三
    python evaluate.py --rows 5000              # 快速试跑前 5000 行

流程：禁用库扫描 -> 在隔离的子进程中加载你的 Solution ->
10 线程并发逐样本推理 -> 计算 MAE / MSE / RMSE / 预测耗时 ->
打印并写入 result.json。

你的代码在子进程中运行，输入不含 id 和 age。该脚本是本地自评工具，
不构成操作系统安全沙箱；教师另用容器核验器执行提交。

提交五个文件：solution.py、model.py、analysis.py、report.md 和 result.json。
analysis.py 与 report.md 用于过程评价，不参与模型推理。
自评分数仅供参考，最终成绩以教师在保留数据上的复测为准。
"""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EVALUATOR_VERSION = "2026.2.0"
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
        tree = ast.parse(Path(name).read_text(encoding="utf-8-sig"), filename=name)
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                      [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            for module in modules:
                if module.split('.')[0] in FORBIDDEN_LIBS:
                    hits.append('{}:{}: {}'.format(name, node.lineno, module))
    return hits


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------- 子进程：加载学生 Solution 并并发推理 ----------------

def _forward_all(samples):
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import solution
    except Exception as e:
        return None, 0.0, "导入 solution.py 失败: {}".format(e)
    try:
        sol = solution.Solution()
    except Exception as e:
        return None, 0.0, "初始化 Solution 失败: {}".format(e)

    # Initialization includes training; report it separately from prediction time.
    n = len(samples)
    results = [None] * n
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(sol.forward, s): i for i, s in enumerate(samples)}
        try:
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        except Exception as e:
            return None, time.perf_counter() - t0, \
                "forward 在第 {} 个样本上失败: {}".format(futures[fut], e)
    wall = time.perf_counter() - t0

    preds = [None] * n
    for i, r in enumerate(results):
        try:
            preds[i] = float(r["prediction"])
        except Exception:
            return None, wall, \
                "第 {} 个样本的返回值无法解析，应返回 {{'prediction': float}}，实际得到: {!r}".format(i, r)
    if not np.all(np.isfinite(np.asarray(preds, dtype=float))):
        return None, wall, "预测结果中包含 NaN/Inf"
    return preds, wall, None


def _worker_main(workdir):
    samples = []
    with open(os.path.join(workdir, "samples.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    start = time.perf_counter()
    preds, wall, error = _forward_all(samples)
    total = time.perf_counter() - start
    out = {"predictions": preds,
           "info": {"prediction_time_s": round(wall, 3),
                    "training_and_setup_time_s": round(max(0, total-wall), 3),
                    "total_worker_time_s": round(total, 3), "error": error}}
    with open(os.path.join(workdir, "worker_out.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


# ---------------- 主进程：数据、指标与结果文件 ----------------

def _run_worker(samples, timeout):
    """把无 ID、无标签样本写入临时目录，供本地自评子进程使用。"""
    workdir = tempfile.mkdtemp(prefix="ml2026_eval_")
    try:
        with open(os.path.join(workdir, "samples.jsonl"), "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({k: _jsonable(v) for k, v in s.items()},
                                   ensure_ascii=False) + "\n")
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", workdir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="ignore",
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("推理超时（>{}s），请检查 forward 是否存在死循环或过慢".format(timeout))
        out_file = os.path.join(workdir, "worker_out.json")
        if not os.path.exists(out_file):
            tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
            raise RuntimeError("推理子进程异常退出（code={}）：\n{}".format(
                proc.returncode, tail))
        with open(out_file, encoding="utf-8") as f:
            out = json.load(f)
        stdout_tail = (proc.stdout or "").strip()[-1500:]
        return out, stdout_tail
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Assignment 1 本地评测")
    ap.add_argument("--test", default=DEFAULT_TEST, help="测试数据文件")
    ap.add_argument("--rows", type=int, default=0, help="只评测前 N 行（0=全部）")
    ap.add_argument("--out", default="result.json", help="结果输出文件")
    ap.add_argument("--student-id", default="", help="学号（可选，仅写入结果文件）")
    ap.add_argument("--name", default="", help="姓名（可选）")
    ap.add_argument("--timeout", type=int, default=3600, help="推理阶段超时秒数")
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.rows < 0 or args.timeout <= 0:
        ap.error('--rows 必须非负，--timeout 必须为正')

    if args._worker:
        _worker_main(args._worker)
        return

    for name in CODE_FILES + [args.test]:
        if not Path(name).exists():
            sys.exit("缺少文件: {}（请在仓库根目录运行）".format(name))

    hits = scan_forbidden()
    if hits:
        sys.exit("[禁止] 检测到禁用库引用:\n  " + "\n  ".join(hits))

    df = pd.read_csv(args.test)
    if args.rows:
        df = df.head(args.rows)
    if "age" not in df.columns:
        sys.exit("测试数据缺少 age 列")
    samples = [{k: _jsonable(v) for k, v in s.items()}
               for s in df.drop(columns=["age", "id"], errors='ignore').to_dict("records")]
    y_true = df["age"].to_numpy(dtype=float)

    print("Evaluating {} rows from {} ...".format(len(df), args.test))
    out, stdout_tail = _run_worker(samples, args.timeout)

    error = out["info"].get("error")
    if error:
        sys.exit("[评测失败] {}".format(error))
    preds = out["predictions"]
    if preds is None or len(preds) != len(samples):
        sys.exit("[评测失败] 预测数量与样本数不一致")
    if stdout_tail:
        print("---- solution 输出 ----\n{}\n----------------------".format(stdout_tail))

    wall = out["info"]["prediction_time_s"]
    preds = np.asarray(preds, dtype=float)
    if not np.all(np.isfinite(preds)):
        sys.exit('[评测失败] 预测结果包含 NaN/Inf')
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
        "prediction_time_s": wall,
        "training_and_setup_time_s": out['info'].get('training_and_setup_time_s'),
        "total_worker_time_s": out['info'].get('total_worker_time_s'),
        "code_md5": {name: md5_file(name) for name in CODE_FILES},
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in CODE_FILES},
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 {}（提交作业时与代码一起上交）".format(args.out))


if __name__ == "__main__":
    main()
