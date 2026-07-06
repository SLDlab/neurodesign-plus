from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from validation.helpers.regenerate_v2_notebooks import NOTEBOOKS, REPO_ROOT


def execute_notebook(source: Path, output_root: Path) -> dict[str, object]:
    """Execute one tutorial notebook and save the executed copy."""
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    target = output_root.resolve() / source.relative_to(REPO_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(nbformat.writes(notebook), encoding="utf-8")
    return {
        "path": str(source.relative_to(REPO_ROOT)),
        "output_path": str(target.relative_to(REPO_ROOT)),
        "cells": len(notebook.cells),
    }


def main() -> None:
    """Execute the maintained validation notebook set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
    os.environ.setdefault("IPYTHONDIR", str(REPO_ROOT / ".tmp_ipython"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    executed = [execute_notebook(path, args.output_dir) for path in NOTEBOOKS]
    (args.output_dir / "notebook_results.json").write_text(
        json.dumps(executed, indent=2), encoding="utf-8"
    )
    for item in executed:
        print(item["path"])


if __name__ == "__main__":
    main()
