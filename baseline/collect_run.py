#!/usr/bin/env python3
"""Automate baseline docker runs and capture coordinator logs."""

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd, capture=True):
    print(f"[baseline-collect] running: {' '.join(cmd)}", flush=True)
    if capture:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        if capture:
            sys.stderr.write(result.stdout or "")
            sys.stderr.write(result.stderr or "")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.stdout if capture else ""


def _next_run_number(log_path: Path) -> int:
    pattern = re.compile(r"^===== RUN (\d+)")
    if not log_path.exists():
        return 1
    last = 0
    for line in log_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            last = max(last, int(match.group(1)))
    return last + 1


def _collect_single_run(project_dir: Path, log_path: Path, run_number: int):
    timestamp = _dt.datetime.now().isoformat(sep=" ", timespec="seconds")
    wait_output = ""
    coordinator_logs = ""

    try:
        _run(["docker", "compose", "up", "-d", "--build"], project_dir, capture=False)
        wait_output = _run(["docker", "compose", "wait", "coordinator"], project_dir)
        coordinator_logs = _run(
            ["docker", "compose", "logs", "--no-color", "--no-log-prefix", "coordinator"],
            project_dir,
        )
    finally:
        try:
            _run(["docker", "compose", "down"], project_dir, capture=False)
        except subprocess.CalledProcessError:
            print("[baseline-collect] warning: docker compose down failed", file=sys.stderr)

    header = f"===== RUN {run_number} ({timestamp}) =====\n"

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(wait_output)
        if wait_output and not wait_output.endswith("\n"):
            fh.write("\n")
        fh.write(coordinator_logs)
        if coordinator_logs and not coordinator_logs.endswith("\n"):
            fh.write("\n")
        fh.write("\n")

    print(f"[baseline-collect] wrote results for run {run_number} to {log_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Collect baseline coordinator logs")
    parser.add_argument(
        "--project-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory containing docker-compose.yaml (default: script directory)",
    )
    parser.add_argument(
        "--output",
        default="run_log.txt",
        type=Path,
        help="Relative path for aggregated log file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs to execute sequentially (default: 1)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    project_dir = args.project_dir.resolve()
    if not (project_dir / "docker-compose.yaml").exists():
        raise SystemExit(f"No docker-compose.yaml in {project_dir}")

    log_path = (project_dir / args.output).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    next_run = _next_run_number(log_path)

    for offset in range(args.runs):
        _collect_single_run(project_dir, log_path, next_run + offset)


if __name__ == "__main__":
    main()

