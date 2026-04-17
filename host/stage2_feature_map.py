#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper.

Core Stage2 feature logic is centralized in ``stage2_settings.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stage2_settings import DEFAULT_FEATURES, build_feature_matrix as _build_feature_matrix


@dataclass
class FeatureMatrix:
    names: list[str]
    phi: np.ndarray


def build_feature_matrix(
    theta: np.ndarray,
    omega: np.ndarray,
    motor_input: np.ndarray,
    feature_names: list[str],
    *,
    eps: float,
) -> FeatureMatrix:
    names, phi = _build_feature_matrix(theta, omega, motor_input, feature_names, eps=eps)
    return FeatureMatrix(names=names, phi=phi)
