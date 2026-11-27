"""Parallel, resumable batch execution of simulation jobs.

A job is a dict {"method", "params", "scen", "seed", "tag"}.  Results are
appended to a JSON-lines file; jobs whose key already appears in the file are
skipped, so an interrupted batch can simply be re-launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")


def job_key(job: dict) -> str:
    blob = json.dumps({k: job.get(k) for k in ("method", "params", "scen", "seed", "tag")},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def _run(job: dict) -> dict:
    from wsncc.runner import run_one
    try:
        out = run_one(job["method"], job.get("params") or {}, job.get("scen") or {},
                      seed=int(job["seed"]), timeseries=bool(job.get("timeseries", False)),
                      rl_train_scen=job.get("rl_train_scen"))
        out["status"] = "ok"
    except Exception as exc:  # keep the batch alive; record the failure
        out = {"method": job["method"], "seed": job["seed"], "status": "error",
               "error": repr(exc), "trace": traceback.format_exc()}
    out["key"] = job_key(job)
    out["tag"] = job.get("tag", "")
    out["scen_overrides"] = json.dumps(job.get("scen") or {}, sort_keys=True)
    return out


def done_keys(path: Path) -> set[str]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    keys.add(rec["key"])
            except json.JSONDecodeError:
                pass
    return keys


def run_batch(jobs: list[dict], out_path: Path, workers: int = 2, time_budget: float | None = None,
              verbose: bool = True) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = done_keys(out_path)
    todo = [j for j in jobs if job_key(j) not in have]
    if verbose:
        print(f"[batch] {len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)
    t0 = time.time()
    n = 0
    # long jobs first improves load balance
    todo.sort(key=lambda j: -(j.get("scen") or {}).get("n", 100) * (j.get("scen") or {}).get("T", 300))
    with Pool(workers) as pool, out_path.open("a") as fh:
        for res in pool.imap_unordered(_run, todo):
            fh.write(json.dumps(res, default=float) + "\n")
            fh.flush()
            n += 1
            if verbose and (n % 20 == 0 or n == len(todo)):
                print(f"[batch] {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
            if time_budget and time.time() - t0 > time_budget:
                pool.terminate()
                break
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", help="JSON file with a list of jobs")
    ap.add_argument("out", help="JSON-lines output file")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--budget", type=float, default=None, help="stop after this many seconds")
    a = ap.parse_args()
    jobs = json.loads(Path(a.jobs).read_text())
    run_batch(jobs, Path(a.out), a.workers, a.budget)


if __name__ == "__main__":
    main()
