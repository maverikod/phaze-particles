#!/usr/bin/env python3
"""
Mathematical foundations for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Any


class PhysicalConstants:
    """Physical constants for proton model."""

    # Experimental values
    PROTON_CHARGE = 1.0  # Exact value
    BARYON_NUMBER = 1.0  # Exact value
    PROTON_MASS_MEV = 938.272  # MeV
    PROTON_MASS_ERROR = 0.006  # MeV
    CHARGE_RADIUS_FM = 0.841  # fm
    CHARGE_RADIUS_ERROR = 0.019  # fm
    MAGNETIC_MOMENT_MU_N = 2.793  # μN
    MAGNETIC_MOMENT_ERROR = 0.001  # μN

    # Physical constants
    HBAR_C = 197.3269804  # MeV·fm
    ALPHA_EM = 1.0 / 137.035999139  # Fine structure constant

    # Scale factors
    ENERGY_SCALE = 1.0  # MeV
    LENGTH_SCALE = 1.0  # fm


class SkyrmeConstants:
    """Skyrme model constants."""

    def __init__(self, c2: float = 1.0, c4: float = 1.0, c6: float = 1.0):
        """
        Initialize Skyrme model constants.

        Args:
            c2: Constant for Tr(L_i L_i) term
            c4: Constant for Tr([L_i, L_j]^2) term
            c6: Constant for stabilizing b_0^2 term
        """
        self.c2 = c2
        self.c4 = c4
        self.c6 = c6

    def validate(self) -> bool:
        """
        Check if constants are valid.

        Returns:
            True if constants are valid
        """
        return all(c > 0 for c in [self.c2, self.c4, self.c6])


class PauliMatrices:
    """Pauli matrices for SU(2) operations."""

    # Pauli matrices
    SIGMA_1 = np.array([[0, 1], [1, 0]], dtype=complex)
    SIGMA_2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    SIGMA_3 = np.array([[1, 0], [0, -1]], dtype=complex)

    @classmethod
    def get_sigma(cls, i: int) -> Any:
        """
        Get Pauli matrix by index.

        Args:
            i: Matrix index (1, 2, 3)

        Returns:
            Pauli matrix
        """
        if i == 1:
            return cls.SIGMA_1.copy()
        elif i == 2:
            return cls.SIGMA_2.copy()
        elif i == 3:
            return cls.SIGMA_3.copy()
        else:
            raise ValueError(f"Invalid Pauli matrix index: {i}")

    @classmethod
    def get_all_sigmas(cls) -> List[Any]:
        """
        Get all Pauli matrices.

        Returns:
            List of all Pauli matrices
        """
        return [cls.SIGMA_1.copy(), cls.SIGMA_2.copy(), cls.SIGMA_3.copy()]


class TensorOperations:
    """Tensor operations for proton model."""

    @staticmethod
    def epsilon_tensor() -> Any:
        """
        Antisymmetric tensor εⁱʲᵏ.

        Returns:
            3x3x3 array with antisymmetric tensor
        """
        epsilon = np.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon

    @staticmethod
    def trace_product(matrices: List[np.ndarray]) -> complex:
        """
        Calculate trace of matrix product.

        Args:
            matrices: List of matrices to multiply

        Returns:
            Trace of product
        """
        if not matrices:
            return 0.0

        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.dot(result, matrix)

        return complex(np.trace(result))

    @staticmethod
    def commutator(A: np.ndarray, B: np.ndarray) -> Any:
        """
        Calculate commutator [A, B] = AB - BA.

        Args:
            A, B: Matrices for commutator calculation

        Returns:
            Commutator [A, B]
        """
        return np.array(np.dot(A, B) - np.dot(B, A))


class CoordinateSystem:
    """Coordinate system for toroidal structures."""

    def __init__(self, grid_size: int, box_size: float):
        """
        Initialize coordinate system.

        Args:
            grid_size: Grid size (N x N x N)
            box_size: Box size in fm
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size

        # Create coordinate grids
        x = np.linspace(-box_size / 2, box_size / 2, grid_size)
        y = np.linspace(-box_size / 2, box_size / 2, grid_size)
        z = np.linspace(-box_size / 2, box_size / 2, grid_size)

        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing="ij")
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get coordinate grids.

        Returns:
            Tuple (X, Y, Z) of coordinate grids
        """
        return self.X, self.Y, self.Z

    def get_radial_coordinate(self) -> Any:
        """
        Get radial coordinate.

        Returns:
            Radial coordinate r = sqrt(x² + y² + z²)
        """
        return np.array(self.R)

    def get_volume_element(self) -> float:
        """
        Get volume element.

        Returns:
            dx³ - volume element
        """
        return self.dx**3


class NumericalUtils:
    """Utilities for numerical computations."""

    @staticmethod
    def gradient_3d(
        field: np.ndarray, dx: float
    ) -> Tuple[Any, Any, Any]:
        """
        Calculate 3D field gradient.

        Args:
            field: 3D scalar field
            dx: Grid step

        Returns:
            Tuple (∂f/∂x, ∂f/∂y, ∂f/∂z)
        """
        grad_x = np.gradient(field, dx, axis=0)
        grad_y = np.gradient(field, dx, axis=1)
        grad_z = np.gradient(field, dx, axis=2)

        return grad_x, grad_y, grad_z

    @staticmethod
    def divergence_3d(
        field_x: np.ndarray, field_y: np.ndarray,
        field_z: np.ndarray, dx: float
    ) -> Any:
        """
        Calculate 3D vector field divergence.

        Args:
            field_x, field_y, field_z: Vector field components
            dx: Grid step

        Returns:
            Divergence ∇·F
        """
        div_x = np.gradient(field_x, dx, axis=0)
        div_y = np.gradient(field_y, dx, axis=1)
        div_z = np.gradient(field_z, dx, axis=2)

        return div_x + div_y + div_z

    @staticmethod
    def integrate_3d(field: np.ndarray, dx: float) -> float:
        """
        Integrate 3D field over volume.

        Args:
            field: 3D field to integrate
            dx: Grid step

        Returns:
            Integration result
        """
        return np.sum(field) * dx**3


class ValidationUtils:
    """Utilities for result validation."""

    @staticmethod
    def check_su2_matrix(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix is SU(2) element.

        Args:
            U: Matrix to check
            tolerance: Allowed tolerance

        Returns:
            True if matrix ∈ SU(2)
        """
        # Check unitarity: U†U = I
        unitary_check = np.allclose(
            np.dot(U.conj().T, U), np.eye(2), atol=tolerance
        )

        # Check determinant: det(U) = 1
        det_check = abs(np.linalg.det(U) - 1.0) < tolerance

        return unitary_check and det_check

    @staticmethod
    def check_physical_bounds(
        value: float, expected: float, tolerance: float
    ) -> bool:
        """
        Check physical bounds.

        Args:
            value: Calculated value
            expected: Expected value
            tolerance: Allowed deviation

        Returns:
            True if value is within bounds
        """
        return abs(value - expected) <= tolerance


# Main class for mathematical foundations
class MathematicalFoundations:
    """Main class combining all mathematical components."""

    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Initialize mathematical foundations.

        Args:
            grid_size: Grid size
            box_size: Box size in fm
        """
        self.constants = PhysicalConstants()
        self.skyrme = SkyrmeConstants()
        self.pauli = PauliMatrices()
        self.tensor = TensorOperations()
        self.coords = CoordinateSystem(grid_size, box_size)
        self.numerical = NumericalUtils()
        self.validation = ValidationUtils()

    def validate_setup(self) -> bool:
        """
        Check setup correctness.

        Returns:
            True if setup is correct
        """
        return self.skyrme.validate()

    def get_physical_constants(self) -> Dict[str, float]:
        """
        Get physical constants.

        Returns:
            Dictionary with physical constants
        """
        return {
            "proton_charge": self.constants.PROTON_CHARGE,
            "baryon_number": self.constants.BARYON_NUMBER,
            "proton_mass_mev": self.constants.PROTON_MASS_MEV,
            "charge_radius_fm": self.constants.CHARGE_RADIUS_FM,
            "magnetic_moment_mu_n": self.constants.MAGNETIC_MOMENT_MU_N,
            "hbar_c": self.constants.HBAR_C,
            "alpha_em": self.constants.ALPHA_EM,
        }

    def get_skyrme_constants(self) -> Dict[str, float]:
        """
        Get Skyrme model constants.

        Returns:
            Dictionary with Skyrme constants
        """
        return {
            "c2": self.skyrme.c2,
            "c4": self.skyrme.c4,
            "c6": self.skyrme.c6
        }
