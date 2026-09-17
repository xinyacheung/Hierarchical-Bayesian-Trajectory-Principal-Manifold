#!/usr/bin/env python3
"""Write a deterministic file-size and SHA-256 manifest for the package."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "PACKAGE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == OUTPUT
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "schema_version": "hb-tpm-handoff-package-1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count_excluding_manifest": len(files),
        "files": files,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUTPUT} with {len(files)} file hashes.")


if __name__ == "__main__":
    main()
