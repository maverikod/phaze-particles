#!/usr/bin/env python3
"""
Mathematical utilities for particle modeling.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import math
from typing import List, Tuple


def calculate_rms_radius(density: List[float], positions: List[float]) -> float:
    """
    Calculate root mean square radius from density distribution.

    Args:
        density: Density values
        positions: Position values

    Returns:
        RMS radius
    """
    if len(density) != len(positions):
        raise ValueError("Density and positions must have same length")

    numerator = sum(d * r**2 for d, r in zip(density, positions))
    denominator = sum(density)

    if denominator == 0:
        return 0.0

    return math.sqrt(numerator / denominator)


def normalize_charge_distribution(
    density: List[float], target_charge: float = 1.0
) -> List[float]:
    """
    Normalize charge distribution to target charge.

    Args:
        density: Input density values
        target_charge: Target total charge

    Returns:
        Normalized density values
    """
    total_charge = sum(density)
    if total_charge == 0:
        return density

    normalization_factor = target_charge / total_charge
    return [d * normalization_factor for d in density]


def calculate_energy_balance(
    e2: float, e4: float, e6: float = 0.0
) -> Tuple[float, float, float]:
    """
    Calculate energy balance percentages.

    Args:
        e2: E2 energy term
        e4: E4 energy term
        e6: E6 energy term (optional)

    Returns:
        Tuple of (E2%, E4%, E6%) percentages
    """
    total_energy = e2 + e4 + e6
    if total_energy == 0:
        return (0.0, 0.0, 0.0)

    e2_percent = (e2 / total_energy) * 100.0
    e4_percent = (e4 / total_energy) * 100.0
    e6_percent = (e6 / total_energy) * 100.0

    return (e2_percent, e4_percent, e6_percent)


def check_virial_condition(
    e2_percent: float, e4_percent: float, tolerance: float = 5.0
) -> bool:
    """
    Check if virial condition (50-50 balance) is satisfied.

    Args:
        e2_percent: E2 energy percentage
        e4_percent: E4 energy percentage
        tolerance: Tolerance for balance check

    Returns:
        True if virial condition is satisfied
    """
    return abs(e2_percent - 50.0) <= tolerance and abs(e4_percent - 50.0) <= tolerance
