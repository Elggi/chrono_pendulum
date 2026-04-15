#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import offline_id_pem_sindy_ppo as bench


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage3 PPO standalone entrypoint")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--train-ratio", type=float, default=0.75)
    ap.add_argument("--fit-lcom", action="store_true")
    ap.add_argument("--ppo-steps", type=int, default=12000)
    ap.add_argument("--ppo-seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--w-theta", type=float, default=5.0)
    ap.add_argument("--w-omega", type=float, default=2.5)
    ap.add_argument("--w-alpha", type=float, default=0.7)
    args = ap.parse_args()

    bench.ensure_dir(args.outdir)
    run = bench.load_run(args.csv)
    cfg = bench.infer_physical_config(args.meta)
    run_train, run_val = bench.split_run(run, train_ratio=args.train_ratio)
    meta = bench._load_json(args.meta)
    cfg_meta = meta.get("config", {})
    init_params = {
        "K_u": float(cfg_meta.get("K_i_init", cfg_meta.get("K_u_init", 1e-5))),
        "b_eq": float(cfg_meta.get("b_eq_init", 0.02)),
        "tau_eq": float(cfg_meta.get("tau_eq_init", 0.01)),
        "l_com": float(cfg_meta.get("l_com_init", 0.1425)),
    }
    weights = {"theta": args.w_theta, "omega": args.w_omega, "alpha": args.w_alpha}
    lo = np.array([1e-9, 0.0, 0.0, 0.03] if args.fit_lcom else [1e-9, 0.0, 0.0], dtype=float)
    hi = np.array([1.0, 10.0, 10.0, 0.45] if args.fit_lcom else [1.0, 10.0, 10.0], dtype=float)
    stage1 = bench.fit_stage1_pem(
        run_train=run_train,
        run_val=run_val,
        cfg=cfg,
        init_params=init_params,
        bounds=(lo, hi),
        weights=weights,
        fit_lcom=args.fit_lcom,
        outdir=args.outdir,
    )
    stage3 = bench.fit_stage3_ppo(
        runs=[run],
        cfg=cfg,
        nominal_params=stage1.params,
        residual_fn=None,
        weights=weights,
        bounds={"K_u": (1e-9, 1.0), "b_eq": (0.0, 10.0), "tau_eq": (0.0, 10.0), "l_com": (0.03, 0.45)},
        seed_list=list(args.ppo_seeds),
        total_timesteps=args.ppo_steps,
        outdir=args.outdir,
    )
    (args.outdir / "stage1_result.json").write_text(json.dumps(asdict(stage1), indent=2), encoding="utf-8")
    (args.outdir / "stage3_result.json").write_text(json.dumps(asdict(stage3), indent=2), encoding="utf-8")
    (args.outdir / "config_used.json").write_text(
        json.dumps({"stage": 3, "csv": str(args.csv), "meta": str(args.meta), "weights": weights}, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Stage3 PPO artifacts: {args.outdir}")


if __name__ == "__main__":
    main()
