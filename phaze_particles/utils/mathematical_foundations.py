#!/usr/bin/env python3
"""
Mathematical foundations for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from .cuda import get_cuda_manager, is_cuda_available


class ArrayBackend:
    """CUDA-aware array backend for mathematical operations."""

    def __init__(self) -> None:
        """Initialize array backend."""
        self.cuda_manager = get_cuda_manager()
        self._use_cuda = is_cuda_available()

        # Try to import CuPy if CUDA is available
        self._cp = None
        if self._use_cuda:
            try:
                import cupy as cp

                self._cp = cp
            except ImportError:
                self._use_cuda = False

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._use_cuda and self._cp is not None

    def get_array_module(self) -> Any:
        """
        Get appropriate array module (CuPy or NumPy).

        Returns:
            CuPy module if CUDA available, NumPy otherwise
        """
        if self.is_cuda_available:
            return self._cp
        return np

    def array(self, data: Any, dtype: Any = None) -> Any:
        """
        Create array using appropriate backend.

        Args:
            data: Array data
            dtype: Data type

        Returns:
            Array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.array(data, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        """
        Create zeros array using appropriate backend.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            Zeros array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        """
        Create ones array using appropriate backend.

        Args:
            shape: Array shape
            dtype: Data type

        Returns:
            Ones array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.ones(shape, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int, dtype: Any = None) -> Any:
        """
        Create linspace array using appropriate backend.

        Args:
            start: Start value
            stop: Stop value
            num: Number of points
            dtype: Data type

        Returns:
            Linspace array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.linspace(start, stop, num, dtype=dtype)

    def meshgrid(self, *arrays: Any, indexing: str = "ij") -> Any:
        """
        Create meshgrid using appropriate backend.

        Args:
            *arrays: Input arrays
            indexing: Indexing mode

        Returns:
            Meshgrid arrays in appropriate backend
        """
        xp = self.get_array_module()
        return xp.meshgrid(*arrays, indexing=indexing)

    def sqrt(self, x: Any) -> Any:
        """
        Compute square root using appropriate backend.

        Args:
            x: Input array

        Returns:
            Square root array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.sqrt(x)

    def dot(self, a: Any, b: Any) -> Any:
        """
        Compute dot product using appropriate backend.

        Args:
            a, b: Input arrays

        Returns:
            Dot product in appropriate backend
        """
        xp = self.get_array_module()
        return xp.dot(a, b)

    def trace(self, a: Any) -> Any:
        """
        Compute trace using appropriate backend.

        Args:
            a: Input array

        Returns:
            Trace in appropriate backend
        """
        xp = self.get_array_module()
        return xp.trace(a)

    def sum(self, a: Any, axis: Any = None) -> Any:
        """
        Compute sum using appropriate backend.

        Args:
            a: Input array
            axis: Axis to sum over

        Returns:
            Sum in appropriate backend
        """
        xp = self.get_array_module()
        return xp.sum(a, axis=axis)

    def eye(self, n: int, dtype: Any = None) -> Any:
        """
        Create identity matrix using appropriate backend.

        Args:
            n: Matrix size
            dtype: Data type

        Returns:
            Identity matrix in appropriate backend
        """
        xp = self.get_array_module()
        return xp.eye(n, dtype=dtype)

    def det(self, a: Any) -> Any:
        """
        Compute determinant using appropriate backend.

        Args:
            a: Input matrix

        Returns:
            Determinant in appropriate backend
        """
        xp = self.get_array_module()
        return xp.linalg.det(a)

    def to_numpy(self, array: Any) -> Any:
        """
        Convert array to NumPy if needed.

        Args:
            array: Input array

        Returns:
            NumPy array
        """
        if self.is_cuda_available and hasattr(array, "get"):
            # CuPy array - convert to NumPy
            return array.get()
        return array

    def get_backend_name(self) -> str:
        """
        Get current backend name.

        Returns:
            Backend name ('cuda' or 'cpu')
        """
        return "cuda" if self.is_cuda_available else "cpu"
    
    def zeros_like(self, array: Any) -> Any:
        """
        Create zeros array with same shape as input.
        
        Args:
            array: Input array
            
        Returns:
            Zeros array with same shape
        """
        xp = self.get_array_module()
        return xp.zeros_like(array)
    
    def exp(self, x: Any) -> Any:
        """
        Compute exponential using appropriate backend.
        
        Args:
            x: Input array
            
        Returns:
            Exponential array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.exp(x)
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """
        Compute where condition using appropriate backend.
        
        Args:
            condition: Boolean condition
            x: Value if condition is True
            y: Value if condition is False
            
        Returns:
            Where result in appropriate backend
        """
        xp = self.get_array_module()
        return xp.where(condition, x, y)
    
    def abs(self, x: Any) -> Any:
        """
        Compute absolute value using appropriate backend.
        
        Args:
            x: Input array
            
        Returns:
            Absolute value array in appropriate backend
        """
        xp = self.get_array_module()
        return xp.abs(x)


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

    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize Pauli matrices.

        Args:
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.backend = backend or ArrayBackend()

        # Create Pauli matrices using appropriate backend
        self.SIGMA_1 = self.backend.array([[0, 1], [1, 0]], dtype=complex)
        self.SIGMA_2 = self.backend.array([[0, -1j], [1j, 0]], dtype=complex)
        self.SIGMA_3 = self.backend.array([[1, 0], [0, -1]], dtype=complex)

    def get_sigma(self, i: int) -> Any:
        """
        Get Pauli matrix by index.

        Args:
            i: Matrix index (1, 2, 3)

        Returns:
            Pauli matrix
        """
        if i == 1:
            return self.SIGMA_1
        elif i == 2:
            return self.SIGMA_2
        elif i == 3:
            return self.SIGMA_3
        else:
            raise ValueError(f"Invalid Pauli matrix index: {i}")

    def get_all_sigmas(self) -> List[Any]:
        """
        Get all Pauli matrices.

        Returns:
            List of all Pauli matrices
        """
        return [self.SIGMA_1, self.SIGMA_2, self.SIGMA_3]


class TensorOperations:
    """Tensor operations for proton model."""

    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize tensor operations.

        Args:
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.backend = backend or ArrayBackend()

    def epsilon_tensor(self) -> Any:
        """
        Antisymmetric tensor εⁱʲᵏ.

        Returns:
            3x3x3 array with antisymmetric tensor
        """
        epsilon = self.backend.zeros((3, 3, 3))
        epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
        epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
        return epsilon

    def trace_product(self, matrices: List[Any]) -> complex:
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
            result = self.backend.dot(result, matrix)

        return complex(self.backend.trace(result))

    def commutator(self, A: Any, B: Any) -> Any:
        """
        Calculate commutator [A, B] = AB - BA.

        Args:
            A, B: Matrices for commutator calculation

        Returns:
            Commutator [A, B]
        """
        return self.backend.dot(A, B) - self.backend.dot(B, A)


class CoordinateSystem:
    """Coordinate system for toroidal structures."""

    def __init__(
        self, grid_size: int, box_size: float, backend: Optional[ArrayBackend] = None
    ):
        """
        Initialize coordinate system.

        Args:
            grid_size: Grid size (N x N x N)
            box_size: Box size in fm
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.backend = backend or ArrayBackend()

        # Create coordinate grids
        x = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)
        y = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)
        z = self.backend.linspace(-box_size / 2, box_size / 2, grid_size)

        self.X, self.Y, self.Z = self.backend.meshgrid(x, y, z, indexing="ij")
        self.R = self.backend.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    def get_coordinates(self) -> Tuple[Any, Any, Any]:
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
        return self.R

    def get_volume_element(self) -> float:
        """
        Get volume element.

        Returns:
            dx³ - volume element
        """
        return self.dx**3


class NumericalUtils:
    """Utilities for numerical computations."""

    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize numerical utilities.

        Args:
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.backend = backend or ArrayBackend()

    def gradient_3d(self, field: Any, dx: float) -> Tuple[Any, Any, Any]:
        """
        Calculate 3D field gradient.

        Args:
            field: 3D scalar field
            dx: Grid step

        Returns:
            Tuple (∂f/∂x, ∂f/∂y, ∂f/∂z)
        """
        xp = self.backend.get_array_module()
        grad_x = xp.gradient(field, dx, axis=0)
        grad_y = xp.gradient(field, dx, axis=1)
        grad_z = xp.gradient(field, dx, axis=2)

        return grad_x, grad_y, grad_z

    def divergence_3d(self, field_x: Any, field_y: Any, field_z: Any, dx: float) -> Any:
        """
        Calculate 3D vector field divergence.

        Args:
            field_x, field_y, field_z: Vector field components
            dx: Grid step

        Returns:
            Divergence ∇·F
        """
        xp = self.backend.get_array_module()
        div_x = xp.gradient(field_x, dx, axis=0)
        div_y = xp.gradient(field_y, dx, axis=1)
        div_z = xp.gradient(field_z, dx, axis=2)

        return div_x + div_y + div_z

    def integrate_3d(self, field: Any, dx: float) -> float:
        """
        Integrate 3D field over volume.

        Args:
            field: 3D field to integrate
            dx: Grid step

        Returns:
            Integration result
        """
        return float(self.backend.sum(field)) * dx**3


class ValidationUtils:
    """Utilities for result validation."""

    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize validation utilities.

        Args:
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.backend = backend or ArrayBackend()

    def check_su2_matrix(self, U: Any, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix is SU(2) element.

        Args:
            U: Matrix to check
            tolerance: Allowed tolerance

        Returns:
            True if matrix ∈ SU(2)
        """
        xp = self.backend.get_array_module()

        # Check unitarity: U†U = I
        unitary_check = xp.allclose(
            self.backend.dot(U.conj().T, U),
            self.backend.eye(2),
            atol=tolerance
        )

        # Check determinant: det(U) = 1
        det_check = abs(self.backend.det(U) - 1.0) < tolerance

        return bool(unitary_check and det_check)

    def check_physical_bounds(
        self, value: float, expected: float, tolerance: float
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

    def __init__(
        self, grid_size: int = 64, box_size: float = 4.0, use_cuda: bool = True
    ):
        """
        Initialize mathematical foundations.

        Args:
            grid_size: Grid size
            box_size: Box size in fm
            use_cuda: Whether to use CUDA if available
        """
        self.constants = PhysicalConstants()
        self.skyrme = SkyrmeConstants()

        # Initialize CUDA-aware backend
        self.backend = ArrayBackend()
        if not use_cuda:
            # Force CPU mode
            self.backend._use_cuda = False
            self.backend._cp = None

        # Initialize components with CUDA-aware backend
        self.pauli = PauliMatrices(self.backend)
        self.tensor = TensorOperations(self.backend)
        self.coords = CoordinateSystem(grid_size, box_size, self.backend)
        self.numerical = NumericalUtils(self.backend)
        self.validation = ValidationUtils(self.backend)

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
            "c6": self.skyrme.c6,
        }

    def get_cuda_status(self) -> str:
        """
        Get CUDA status information.

        Returns:
            CUDA status string
        """
        return self.backend.cuda_manager.get_status_string()

    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            True if CUDA is available
        """
        return self.backend.is_cuda_available

    def get_backend_info(self) -> Dict[str, str]:
        """
        Get backend information.

        Returns:
            Dictionary with backend information
        """
        return {
            "backend": self.backend.get_backend_name(),
            "cuda_status": self.get_cuda_status(),
            "cuda_available": str(self.is_cuda_available()),
        }
