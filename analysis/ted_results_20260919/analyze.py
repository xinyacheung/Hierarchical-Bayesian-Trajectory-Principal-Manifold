#!/usr/bin/env python3
"""Public CLI entry point for the complete TED results analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from checkpoint_reader import load_checkpoint_state_without_torch
import summaries

# Use the nested-state checkpoint reader for prior diagnostics.
summaries.load_checkpoint_state_without_torch = load_checkpoint_state_without_torch

import run_analysis
from analyze_ted_results import sha256_file


def main() -> None:
    args = run_analysis.parse_args()
    run_analysis.main()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {entry["path"] for entry in manifest["analysis_scripts"]}
    for name in ["analyze.py", "checkpoint_reader.py"]:
        path = Path(__file__).resolve().parent / name
        if str(path) not in existing:
            manifest["analysis_scripts"].append({"path": str(path), "sha256": sha256_file(path)})
    manifest["analysis_scripts"] = sorted(manifest["analysis_scripts"], key=lambda item: item["path"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
