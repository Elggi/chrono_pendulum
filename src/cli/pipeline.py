"""CLI for running digital twin pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..identification.nominal_free_decay.train import NominalTrainConfig, train_nominal_model
from ..identification.actuator_a2.train import ActuatorTrainConfig, train_actuator_a2
from ..identification.regression.fit_actuator_regression import RegressionConfig, fit_linear_actuator
from ..identification.sparse.fit_sindy import SparseConfig, fit_sparse_residual
from ..calibration_rl.train_ppo import RLTrainConfig, train_rl_calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Chrono pendulum digital twin pipelines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    n = sub.add_parser("train-nominal")
    n.add_argument("--data", type=Path, required=True)

    a2 = sub.add_parser("train-actuator-a2")
    a2.add_argument("--data", type=Path, required=True)

    reg = sub.add_parser("fit-regression")
    reg.add_argument("--data", type=Path, required=True)

    sp = sub.add_parser("fit-sparse")
    sp.add_argument("--data", type=Path, required=True)

    rl = sub.add_parser("train-rl-calibrator")
    rl.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "train-nominal":
        train_nominal_model(NominalTrainConfig(data_csv=args.data))
    elif args.cmd == "train-actuator-a2":
        train_actuator_a2(ActuatorTrainConfig(data_csv=args.data))
    elif args.cmd == "fit-regression":
        fit_linear_actuator(RegressionConfig(data_csv=args.data))
    elif args.cmd == "fit-sparse":
        fit_sparse_residual(SparseConfig(data_csv=args.data))
    elif args.cmd == "train-rl-calibrator":
        train_rl_calibrator(RLTrainConfig(data_csv=args.data))


if __name__ == "__main__":
    main()
