#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_replication.py"
sys.path.insert(0, str(ROOT))
from src.replication.config import load_config


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal replication locally with checkpointed workers"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "analysis.json"
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(
    stage: str,
    config: Path,
    device: str,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(RUNNER),
        stage,
        "--config",
        str(config),
        "--device",
        device,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
    ]


def worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(threads)
    return environment


def display_command(values: list[str]) -> str:
    return subprocess.list2cmdline(values)


def run_stage(
    stage: str,
    config: Path,
    device: str,
    log_dir: Path,
    environment: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    values = command(stage, config, device)
    if dry_run:
        print(display_command(values))
        return
    stdout_path = log_dir / f"{stage}.stdout.log"
    stderr_path = log_dir / f"{stage}.stderr.log"
    print(f"starting {stage}; logs: {stdout_path.name}, {stderr_path.name}")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            values,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{stage} failed; see {stderr_path}")
    print(f"completed {stage}")


def run_parallel_stage(
    stage: str,
    config: Path,
    device: str,
    workers: int,
    log_dir: Path,
    environment: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    commands = [
        command(
            stage,
            config,
            device,
            shard_index=index,
            shard_count=workers,
        )
        for index in range(workers)
    ]
    if dry_run:
        for values in commands:
            print(display_command(values))
        return

    processes = []
    print(f"starting {stage} with {workers} workers")
    try:
        for index, values in enumerate(commands):
            stdout_path = log_dir / f"{stage}-shard-{index:02d}.stdout.log"
            stderr_path = log_dir / f"{stage}-shard-{index:02d}.stderr.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                values,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            processes.append((index, process, stdout, stderr, stderr_path))

        pending = {index for index, *_ in processes}
        while pending:
            for index, process, _, _, stderr_path in processes:
                if index not in pending:
                    continue
                returncode = process.poll()
                if returncode is None:
                    continue
                pending.remove(index)
                if returncode:
                    for other_index, other, *_ in processes:
                        if other_index in pending:
                            other.terminate()
                    raise RuntimeError(
                        f"{stage} shard {index} failed; see {stderr_path}"
                    )
                print(f"completed {stage} shard {index + 1}/{workers}")
            if pending:
                time.sleep(2)
    except KeyboardInterrupt:
        for _, process, *_ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, process, stdout, stderr, _ in processes:
            if process.poll() is None:
                process.wait()
            stdout.close()
            stderr.close()
    print(f"completed {stage}")


def main() -> None:
    args = arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    logical_cpus = os.cpu_count() or 1
    threads = args.threads_per_worker or max(1, logical_cpus // args.workers)
    if threads < 1:
        raise ValueError("--threads-per-worker must be at least 1")

    config = args.config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    log_dir = load_config(config).output_root / "logs" / "main"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    environment = worker_environment(threads)

    print(f"workers={args.workers} threads_per_worker={threads} device={args.device}")
    for stage in ("validate", "plan", "depths"):
        run_stage(
            stage,
            config,
            args.device,
            log_dir,
            environment,
            dry_run=args.dry_run,
        )
    for stage in ("brain", "universality"):
        run_parallel_stage(
            stage,
            config,
            args.device,
            args.workers,
            log_dir,
            environment,
            dry_run=args.dry_run,
        )
    run_stage(
        "summarize",
        config,
        args.device,
        log_dir,
        environment,
        dry_run=args.dry_run,
    )
    run_parallel_stage(
        "rsa",
        config,
        args.device,
        args.workers,
        log_dir,
        environment,
        dry_run=args.dry_run,
    )
    run_stage(
        "figures",
        config,
        args.device,
        log_dir,
        environment,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
