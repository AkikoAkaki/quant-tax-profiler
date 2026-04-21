#!/usr/bin/env python3
"""One-shot Phase 2 runner: repeated benchmarks + canonical analysis/report."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def to_wsl_path(path: Path) -> str:
    """Convert a Windows path into a /mnt/<drive>/... WSL path."""
    resolved = path.resolve()
    path_str = resolved.as_posix()
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        return f"/mnt/{drive}{path_str[2:]}"
    return path_str


def run_checked(cmd: list[str], cwd: Path | None = None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def build_wsl_python_command(repo_root: Path, python_args: list[str]) -> list[str]:
    repo_wsl = to_wsl_path(repo_root)
    quoted_args = " ".join(shlex.quote(arg) for arg in python_args)
    shell_cmd = (
        f"cd {shlex.quote(repo_wsl)} && "
        "source venv/bin/activate && "
        f"python {quoted_args}"
    )
    return ["wsl", "bash", "-lc", shell_cmd]


def run_benchmark_in_wsl(
    repo_root: Path,
    model: str,
    quantization: str,
    prompt_len: int,
    max_new_tokens: int,
    output_dir: Path,
    run_id: str,
):
    metadata_path = output_dir / "benchmark_metadata.json"
    python_args = [
        "scripts/run_benchmark.py",
        "--model", model,
        "--quantization", quantization,
        "--prompt-len", str(prompt_len),
        "--max-new-tokens", str(max_new_tokens),
        "--output-dir", to_wsl_path(output_dir),
        "--run-id", run_id,
        "--metadata-path", to_wsl_path(metadata_path),
    ]
    run_checked(build_wsl_python_command(repo_root, python_args))


def run_analysis_in_wsl(repo_root: Path, data_dir: Path, output_dir: Path, report_path: Path):
    python_args = [
        "scripts/run_analysis.py",
        "--data-dir", to_wsl_path(data_dir),
        "--output-dir", to_wsl_path(output_dir),
        "--report-path", to_wsl_path(report_path),
    ]
    run_checked(build_wsl_python_command(repo_root, python_args))


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved manifest: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run the full Phase 2 workflow in WSL2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-root", default="data/phase2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--report-path", default="report/analysis_report.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("phase2_%Y%m%d_%H%M%S")
    run_root = (repo_root / args.data_root / timestamp).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    report_path = (repo_root / args.report_path).resolve()

    manifest = {
        "phase": "phase2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "runs": [],
    }

    for quantization in ("fp16", "int4"):
        for run_index in range(1, args.repeats + 1):
            run_id = f"{quantization}_run{run_index:02d}"
            run_dir = run_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest["runs"].append({
                "run_id": run_id,
                "quantization": quantization,
                "data_dir": str(run_dir),
            })
            print(f"\n=== Running {run_id} ===")
            run_benchmark_in_wsl(
                repo_root=repo_root,
                model=args.model,
                quantization=quantization,
                prompt_len=args.prompt_len,
                max_new_tokens=args.max_new_tokens,
                output_dir=run_dir,
                run_id=run_id,
            )

    print("\n=== Generating canonical analysis artifacts ===")
    run_analysis_in_wsl(
        repo_root=repo_root,
        data_dir=run_root,
        output_dir=output_dir,
        report_path=report_path,
    )

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_root / "phase2_manifest.json", manifest)

    print("\n=== Phase 2 complete ===")
    print(f"Raw runs: {run_root}")
    print(f"Charts:   {output_dir}")
    print(f"Report:   {report_path}")


if __name__ == "__main__":
    main()
