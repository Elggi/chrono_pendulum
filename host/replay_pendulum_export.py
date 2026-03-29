#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from chrono_core.pendulum_csv_schema import PENDULUM_CSV_HEADER
from chrono_core.pendulum_rl_env import simulate_replay


def main():
    ap = argparse.ArgumentParser(description="Replay a logged pendulum trajectory and export CSV using runtime schema")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--param_json", required=True)
    ap.add_argument("--delay_sec", type=float, default=None)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    p = json.loads(Path(args.param_json).read_text())
    delay_sec = float(args.delay_sec) if args.delay_sec is not None else float(df.get("delay_sec_est", pd.Series([0.12])).iloc[0])
    sim = simulate_replay(df, p, delay_sec)

    out = []
    for i, r in df.iterrows():
        row = {k: "" for k in PENDULUM_CSV_HEADER}
        row.update({k: r[k] for k in df.columns if k in row})
        row["cmd_u_delayed"] = float(sim["cmd_u_delayed"][i])
        row["theta"] = float(sim["theta"][i])
        row["omega"] = float(sim["omega"][i])
        row["alpha"] = float(sim["alpha"][i])
        row["tau_motor"] = float(sim["tau_motor"][i])
        row["tau_res"] = float(sim["tau_res"][i])
        row["i_pred"] = float(sim["i_pred"][i])
        row["v_applied"] = float(sim["v_applied"][i])
        row["delay_sec_est"] = delay_sec
        row["delay_ms"] = delay_sec * 1000.0
        out.append(row)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=PENDULUM_CSV_HEADER)
        wr.writeheader()
        wr.writerows(out)


if __name__ == "__main__":
    main()
