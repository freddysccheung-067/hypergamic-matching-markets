"""Run a file of jobs, at most W at a time, resumably.

    python scripts/run_queue.py --jobs D:/hm_runs/jobs.jsonl --workers 22 \
        --log-dir D:/hm_runs/logs --linger 14400

``jobs.jsonl`` holds one JSON object per line:

    {"id": "e09-s100", "cwd": "D:/hm_runs/wt_abc1234", "cmd": ["python", ...],
     "done": "D:/.../seed100.json", "priority": 1}

A job is complete when its ``done`` file exists; complete jobs are skipped, so a
killed queue restarts where it stopped. The file is re-read every poll, so jobs
appended while the queue runs are picked up. Lower ``priority`` runs first, which
is how tiers are ordered. A job that exits non-zero is recorded in
``<log-dir>/<id>.failed`` and is not retried until that file is removed: a
failure is a fact to look at, not a reason to loop.

Each job runs with ``OMP_NUM_THREADS=1`` and the BLAS thread variables at 1, so W
jobs use W cores and no job's reductions depend on how many others are running.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "PYTHONUTF8": "1",
}


def load(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    jobs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            jobs.append(json.loads(line))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--linger", type=float, default=0.0,
                        help="seconds to wait for new jobs once the file is drained")
    args = parser.parse_args()

    jobs_path = pathlib.Path(args.jobs)
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **SINGLE_THREAD}
    running: dict[str, tuple[subprocess.Popen, object, float]] = {}
    idle_since: float | None = None

    while True:
        for job_id, (proc, handle, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del running[job_id]
            took = time.time() - started
            if code != 0:
                (log_dir / f"{job_id}.failed").write_text(f"exit {code}\n", encoding="utf-8")
            print(f"{time.strftime('%H:%M:%S')} {'done' if code == 0 else 'FAILED'} "
                  f"{job_id} ({took / 60:.1f} min, exit {code})", flush=True)

        jobs = load(jobs_path)
        pending = [j for j in jobs
                   if j["id"] not in running
                   and not pathlib.Path(j["done"]).exists()
                   and not (log_dir / f"{j['id']}.failed").exists()]
        pending.sort(key=lambda j: (j.get("priority", 0), jobs.index(j)))
        while pending and len(running) < args.workers:
            job = pending.pop(0)
            handle = open(log_dir / f"{job['id']}.log", "a", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], cwd=job["cwd"], env=env,
                                    stdout=handle, stderr=subprocess.STDOUT)
            running[job["id"]] = (proc, handle, time.time())
            print(f"{time.strftime('%H:%M:%S')} start {job['id']} "
                  f"({len(running)} running, {len(pending)} queued)", flush=True)

        if not running and not pending:
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= args.linger:
                print(f"{time.strftime('%H:%M:%S')} queue drained", flush=True)
                return 0
        else:
            idle_since = None
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
