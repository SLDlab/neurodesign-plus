from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from validation.manifest import REPO_ROOT, WORKFLOWS


def _artifact_root() -> Path:
    return Path(
        os.environ.get(
            "NEURODESIGN_VALIDATION_ROOT",
            REPO_ROOT / "validation" / "_artifacts" / "run_all",
        )
    )


def _workflow_env(artifact_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mpl"))
    env.setdefault("IPYTHONDIR", str(REPO_ROOT / ".tmp_ipython"))
    env.setdefault("NEURODESIGN_DOCS_OFFLINE", "1")
    env["NEURODESIGN_VALIDATION_CASE10_OUTPUT"] = str(artifact_dir / "case10_output")
    env["NEURODESIGN_VALIDATION_MANUSCRIPT_OUTPUT"] = str(
        artifact_dir / "manuscript_output"
    )
    env["NEURODESIGN_VALIDATION_DETERMINISM_OUTPUT"] = str(
        artifact_dir / "determinism_output"
    )
    env["NEURODESIGN_VALIDATION_SVG_OUTPUT"] = str(artifact_dir / "svg_output")
    return env


def main() -> int:
    """Run each validation workflow and persist the collected logs and summary."""
    root = _artifact_root()
    root.mkdir(parents=True, exist_ok=True)
    results = []

    for workflow in WORKFLOWS:
        artifact_dir = root / workflow["artifact_subdir"]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        command = [
            part.format(artifact_dir=artifact_dir.as_posix())
            for part in workflow["command"]
        ]
        start = time.time()
        completed = subprocess.run(
            [sys.executable, *command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=_workflow_env(artifact_dir),
        )
        runtime = time.time() - start
        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        result = {
            "name": workflow["name"],
            "kind": workflow["kind"],
            "command": shlex.join([sys.executable, *command]),
            "status": "passed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "runtime_seconds": round(runtime, 3),
            "artifact_dir": str(artifact_dir),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "assertion_summary": workflow["assertion_summary"],
            "ci": workflow["ci"],
        }
        if completed.returncode != 0:
            result["failure_message"] = (
                completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "workflow failed"
            )
            results.append(result)
            break
        results.append(result)

    summary_path = root / "validation_results.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    inventory_path = root / "validation_inventory.json"
    inventory_path.write_text(json.dumps(WORKFLOWS, indent=2), encoding="utf-8")
    return (
        0
        if all(item["status"] == "passed" for item in results)
        and len(results) == len(WORKFLOWS)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
