"""Shared paths, configuration, and deterministic metadata helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "local"
RESULTS = ROOT / "results"
CONFIG_PATH = ROOT / "config.json"


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config():
    config = read_json(CONFIG_PATH)
    config_id = sha256(CONFIG_PATH)
    return config, config_id


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def repository_files() -> list[Path]:
    """Return the files selected for the GitHub package."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file()
            and LOCAL not in path.parents
            and ".git" not in path.parts
            and "__pycache__" not in path.parts
        ]
    return [ROOT / relative for relative in result.stdout.split("\0") if relative]
