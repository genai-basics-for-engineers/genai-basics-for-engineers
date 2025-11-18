#!/usr/bin/env python3
"""Chapter7 の主要デモを順番に実行し、ログを保存するスクリプト"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "7-3_react_search.py",
    "7-4_langgraph_search.py",
    "7-5_research_agent.py",
    "7-6_rag_agent.py",
]


def run(script: str) -> None:
    print(f"\n=== Running {script} ===\n")
    script_path = ROOT / script
    out_path = OUTPUT_DIR / f"{script.replace('.py', '')}-out.txt"
    with out_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            ["python", str(script_path)],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()
        if proc.returncode != 0:
            raise SystemExit(f"{script} failed with exit code {proc.returncode}")


def main() -> None:
    for script in SCRIPTS:
        run(script)


if __name__ == "__main__":
    main()
