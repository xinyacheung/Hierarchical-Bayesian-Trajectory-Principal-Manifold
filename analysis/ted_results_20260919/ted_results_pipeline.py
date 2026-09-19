#!/usr/bin/env python3
"""Stable CLI entry point for the complete TED results analysis."""

from __future__ import annotations

import json
from pathlib import Path
import re

from checkpoint_reader import load_checkpoint_state_without_torch
import summaries

# Inject the safe nested checkpoint reader and the standard regex module used by
# the diagnostics. Keeping these bindings here makes this file the single public
# entry point while the analysis remains split into testable modules.
summaries.load_checkpoint_state_without_torch = load_checkpoint_state_without_torch
summaries.re = re

import run_analysis
from analyze_ted_results import sha256_file


def main() -> None:
    args = run_analysis.parse_args()
    run_analysis.main()
    manifest_path = args.output_dir.resolve() / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {entry["path"] for entry in manifest["analysis_scripts"]}
    for name in ["ted_results_pipeline.py", "checkpoint_reader.py"]:
        path = Path(__file__).resolve().parent / name
        if str(path) not in existing:
            manifest["analysis_scripts"].append({"path": str(path), "sha256": sha256_file(path)})
    manifest["analysis_scripts"] = sorted(manifest["analysis_scripts"], key=lambda item: item["path"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
