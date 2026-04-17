#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper.

Core Stage2 known-parameter parsing and residual-target construction
is centralized in ``stage2_settings.py``.
"""

from stage2_settings import KnownParams, ResidualTarget, known_params_from_model_json, compute_residual_target
from stage2_dataset import Stage2Trajectory


def build_residual_target(traj: Stage2Trajectory, p: KnownParams, target_mode: str = "greybox") -> ResidualTarget:
    return compute_residual_target(
        theta=traj.theta,
        omega=traj.omega,
        alpha=traj.alpha,
        motor_input=traj.motor_input_a,
        p=p,
        target_mode=target_mode,
    )
