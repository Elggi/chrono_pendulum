#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from stage2_feature_map import DEFAULT_FEATURES
from stage2_sindy import run_stage2, save_stage2_summary


def _parse_features(raw: str) -> list[str]:
    items = [s.strip() for s in raw.split(",")]
    return [s for s in items if s]


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 greybox residual-torque SINDy entrypoint")
    ap.add_argument("--csv", type=Path, nargs="+", required=True, help="finalized trajectory CSVs (multi trajectory allowed)")
    ap.add_argument(
        "--model-parameter-json",
        type=Path,
        default=Path("host/model_parameter.json"),
        help="canonical model registry json to read/update",
    )
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=1.0e-4, help="sparsification threshold")
    ap.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="comma-separated candidate features (runtime-compatible names)",
    )
    args = ap.parse_args()

    features = _parse_features(args.features)
    result = run_stage2(
        csv_paths=list(args.csv),
        model_parameter_json=args.model_parameter_json,
        outdir=args.outdir,
        features=features,
        threshold=float(args.threshold),
    )
    summary_path = save_stage2_summary(result, args.outdir)
    (args.outdir / "stage2_result.json").write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Stage2 summary: {summary_path}")
    print(f"[DONE] Updated model parameter json: {args.model_parameter_json}")


if __name__ == "__main__":
    main()

