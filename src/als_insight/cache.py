"""Minimal content-addressed caching for pipeline steps.

Each step writes a ``_step.json`` next to its outputs that records a fingerprint of
its configuration and of its input files. A step is skipped when the fingerprint is
unchanged and all declared outputs still exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STEP_FILE = "_step.json"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(params: str, inputs: Iterable[Path] = ()) -> str:
    digest = hashlib.sha256(params.encode())
    for path in sorted(inputs):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def is_fresh(step_dir: Path, fp: str, outputs: Iterable[str]) -> bool:
    record_path = step_dir / STEP_FILE
    if not record_path.exists():
        return False
    try:
        record = json.loads(record_path.read_text())
    except json.JSONDecodeError:
        return False
    return record.get("fingerprint") == fp and all((step_dir / o).exists() for o in outputs)


def mark_done(step_dir: Path, fp: str, outputs: Iterable[str]) -> None:
    record = {
        "fingerprint": fp,
        "outputs": sorted(outputs),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (step_dir / STEP_FILE).write_text(json.dumps(record, indent=2))
