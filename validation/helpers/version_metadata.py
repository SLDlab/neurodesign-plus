from __future__ import annotations

import subprocess
from pathlib import Path


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _read_generated_version(repo_root: Path) -> str | None:
    version_file = repo_root / "neurodesign" / "_version.py"
    if not version_file.exists():
        return None

    namespace: dict[str, object] = {}
    try:
        exec(version_file.read_text(encoding="utf-8"), namespace)
    except Exception:
        return None

    version = namespace.get("__version__")
    return version if isinstance(version, str) and version else None


def capture_version_metadata(repo_root: Path) -> dict[str, object]:
    """Capture git- and package-derived version metadata for validation artifacts."""
    git_commit_hash = _git_output(repo_root, ["rev-parse", "HEAD"])
    git_commit_short = _git_output(repo_root, ["rev-parse", "--short", "HEAD"])
    git_describe = _git_output(
        repo_root, ["describe", "--tags", "--long", "--dirty", "--always"]
    )
    generated_version = _read_generated_version(repo_root)

    if generated_version:
        package_version = generated_version
        package_version_source = "generated_version_file"
    elif git_describe:
        package_version = git_describe
        package_version_source = "git_describe"
    else:
        package_version = git_commit_hash
        package_version_source = "git_commit_hash"

    return {
        "git_commit_hash": git_commit_hash,
        "git_commit_short": git_commit_short,
        "package_version": package_version,
        "package_version_source": package_version_source,
        "package_version_git_describe": git_describe,
        "package_version_generated_file": generated_version,
        "package_version_generated_file_matches_package_version": (
            generated_version == package_version
            if generated_version and package_version
            else None
        ),
    }
