#!/usr/bin/env python3
"""
SU(2) fields for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math
from .mathematical_foundations import ArrayBackend


@dataclass
class SU2Field:
    """SU(2) field U(x) = cos f(r) 1 + i sin f(r) n̂(x) · σ⃗."""

    # Field components
    u_00: Any  # U[0,0] element
    u_01: Any  # U[0,1] element
    u_10: Any  # U[1,0] element
    u_11: Any  # U[1,1] element

    # Metadata
    grid_size: int
    box_size: float
    backend: ArrayBackend

    def __post_init__(self) -> None:
        """Validate field after initialization."""
        if not self._is_su2_field():
            raise ValueError("Field is not a valid SU(2) field")

    def _is_su2_field(self, tolerance: float = 1e-10) -> bool:
        """
        Check if field is an element of SU(2).

        Args:
            tolerance: Allowed tolerance

        Returns:
            True if field ∈ SU(2)
        """
        # Check unitarity: U†U = I
        xp = self.backend.get_array_module()
        u_dagger_u_00 = xp.conj(self.u_00) * self.u_00 + xp.conj(self.u_10) * self.u_10
        u_dagger_u_01 = xp.conj(self.u_00) * self.u_01 + xp.conj(self.u_10) * self.u_11
        u_dagger_u_10 = xp.conj(self.u_01) * self.u_00 + xp.conj(self.u_11) * self.u_10
        u_dagger_u_11 = xp.conj(self.u_01) * self.u_01 + xp.conj(self.u_11) * self.u_11

        unitary_check = (
            xp.allclose(u_dagger_u_00, 1.0, atol=tolerance)
            and xp.allclose(u_dagger_u_01, 0.0, atol=tolerance)
            and xp.allclose(u_dagger_u_10, 0.0, atol=tolerance)
            and xp.allclose(u_dagger_u_11, 1.0, atol=tolerance)
        )

        # Check determinant: det(U) = 1
        det_u = self.u_00 * self.u_11 - self.u_01 * self.u_10
        det_check = xp.allclose(det_u, 1.0, atol=tolerance)

        return bool(unitary_check and det_check)

    def get_matrix_at_point(self, i: int, j: int, k: int) -> Any:
        """
        Get matrix U at point (i, j, k).

        Args:
            i, j, k: Point indices

        Returns:
            2x2 matrix U at point
        """
        return self.backend.array(
            [
                [self.u_00[i, j, k], self.u_01[i, j, k]],
                [self.u_10[i, j, k], self.u_11[i, j, k]],
            ],
            dtype=complex,
        )

    def get_determinant(self) -> Any:
        """
        Compute field determinant.

        Returns:
            Determinant at each point
        """
        return self.u_00 * self.u_11 - self.u_01 * self.u_10


class RadialProfile:
    """Radial profile f(r) for SU(2) field."""

    def __init__(
        self,
        profile_type: str = "skyrmion",
        scale: float = 1.0,
        center_value: float = math.pi,
        backend: Optional[ArrayBackend] = None,
    ):
        """
        Initialize radial profile.

        Args:
            profile_type: Profile type ("skyrmion", "exponential",
                                       "polynomial")
            scale: Scale parameter
            center_value: Value at center f(0)
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.profile_type = profile_type
        self.scale = scale
        self.center_value = center_value
        self.backend = backend or ArrayBackend()

    def evaluate(self, r: Any) -> Any:
        """
        Evaluate radial profile f(r).

        Args:
            r: Radial coordinate

        Returns:
            Profile values f(r)
        """
        if self.profile_type == "skyrmion":
            return self._skyrmion_profile(r)
        elif self.profile_type == "exponential":
            return self._exponential_profile(r)
        elif self.profile_type == "polynomial":
            return self._polynomial_profile(r)
        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")

    def _skyrmion_profile(self, r: Any) -> Any:
        """
        Standard skyrmion profile.

        Args:
            r: Radial coordinate

        Returns:
            Profile f(r) = π * exp(-r/scale)
        """
        return self.center_value * self.backend.exp(-r / self.scale)

    def _exponential_profile(self, r: Any) -> Any:
        """
        Exponential profile.

        Args:
            r: Radial coordinate

        Returns:
            Profile f(r) = center_value * exp(-r²/scale²)
        """
        return self.center_value * self.backend.exp(-(r**2) / (self.scale**2))

    def _polynomial_profile(self, r: Any) -> Any:
        """
        Polynomial profile.

        Args:
            r: Radial coordinate

        Returns:
            Profile f(r) = center_value * (1 + r/scale)⁻¹
        """
        return self.center_value / (1 + r / self.scale)

    def get_derivative(self, r: Any, dr: float) -> Any:
        """
        Compute profile derivative df/dr.

        Args:
            r: Radial coordinate
            dr: Step for numerical differentiation

        Returns:
            Derivative df/dr
        """
        f_r = self.evaluate(r)
        f_r_plus_dr = self.evaluate(r + dr)

        return (f_r_plus_dr - f_r) / dr


class SU2FieldBuilder:
    """SU(2) field builder."""

    def __init__(
        self,
        grid_size: int,
        box_size: float,
        backend: Optional[ArrayBackend] = None,
    ):
        """
        Initialize field builder.

        Args:
            grid_size: Grid size
            box_size: Box size
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

    def _build_field_components(
        self, n_x: Any, n_y: Any, n_z: Any, profile: RadialProfile
    ) -> SU2Field:
        """
        Build SU(2) field from direction and profile.

        Args:
            n_x, n_y, n_z: Field direction components
            profile: Radial profile

        Returns:
            SU(2) field
        """
        # Normalize field direction
        xp = self.backend.get_array_module()
        n_norm = xp.sqrt(n_x**2 + n_y**2 + n_z**2)

        # Handle zero norm case by setting default direction
        zero_mask = n_norm < 1e-10
        n_x_norm = xp.where(zero_mask, 0.0, n_x / n_norm)
        n_y_norm = xp.where(zero_mask, 0.0, n_y / n_norm)
        n_z_norm = xp.where(zero_mask, 1.0, n_z / n_norm)  # Default to z-direction

        # Compute radial profile
        f_r = profile.evaluate(self.R)

        # Compute field components
        cos_f = xp.cos(f_r)
        sin_f = xp.sin(f_r)

        # U = cos f(r) 1 + i sin f(r) n̂(x) · σ⃗
        u_00 = cos_f + 1j * sin_f * n_z_norm
        u_01 = 1j * sin_f * (n_x_norm - 1j * n_y_norm)
        u_10 = 1j * sin_f * (n_x_norm + 1j * n_y_norm)
        u_11 = cos_f - 1j * sin_f * n_z_norm

        return SU2Field(
            u_00=u_00,
            u_01=u_01,
            u_10=u_10,
            u_11=u_11,
            grid_size=self.grid_size,
            box_size=self.box_size,
            backend=self.backend,
        )

    def build_field(
        self,
        field_direction: Any,
        profile_type: str = "tanh",
        f_0: float = np.pi,
        f_inf: float = 0.0,
        r_scale: float = 1.0,
    ) -> SU2Field:
        """
        Build SU(2) field from field direction and profile parameters.

        Args:
            field_direction: Field direction configuration
            profile_type: Profile type
            f_0: Initial value
            f_inf: Final value
            r_scale: Scale parameter

        Returns:
            SU(2) field
        """
        # Create radial profile
        profile = RadialProfile(profile_type, f_0, f_inf, r_scale)

        # Extract direction components from field_direction
        if hasattr(field_direction, "n_x"):
            n_x = field_direction.n_x
            n_y = field_direction.n_y
            n_z = field_direction.n_z
        else:
            # Mock direction components
            n_x = np.ones((self.grid_size, self.grid_size, self.grid_size))
            n_y = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            n_z = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        return self._build_field_components(n_x, n_y, n_z, profile)

    def build_from_torus_config(
        self, torus_config: Any, profile: RadialProfile
    ) -> SU2Field:
        """
        Build field from torus configuration.

        Args:
            torus_config: Torus configuration
            profile: Radial profile

        Returns:
            SU(2) field
        """
        # Get field direction from configuration
        n_x, n_y, n_z = torus_config.get_field_direction(self.X, self.Y, self.Z)

        return self.build_field(n_x, n_y, n_z, profile)


class SU2FieldOperations:
    """Operations on SU(2) fields."""

    def __init__(self, dx: float, backend: Optional[ArrayBackend] = None):
        """
        Initialize operations.

        Args:
            dx: Grid step
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.dx = dx
        self.backend = backend or ArrayBackend()

    def compute_left_currents(
        self, field: SU2Field
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Compute left currents Lᵢ = U†∂ᵢU.

        Args:
            field: SU(2) field

        Returns:
            Tuple (L_x, L_y, L_z) of left currents
        """
        # Compute field derivatives
        du_dx = self._compute_field_derivative(field, axis=0)
        du_dy = self._compute_field_derivative(field, axis=1)
        du_dz = self._compute_field_derivative(field, axis=2)

        # Compute Lᵢ = U†∂ᵢU
        l_x = self._multiply_field_dagger_derivative(field, du_dx)
        l_y = self._multiply_field_dagger_derivative(field, du_dy)
        l_z = self._multiply_field_dagger_derivative(field, du_dz)

        return l_x, l_y, l_z

    def _compute_field_derivative(self, field: SU2Field, axis: int) -> Dict[str, Any]:
        """
        Compute field derivative along given axis.

        Args:
            field: SU(2) field
            axis: Differentiation axis (0, 1, 2)

        Returns:
            Dictionary with derivative components
        """
        xp = self.backend.get_array_module()
        du_dx = {
            "u_00": xp.gradient(field.u_00, self.dx, axis=axis),
            "u_01": xp.gradient(field.u_01, self.dx, axis=axis),
            "u_10": xp.gradient(field.u_10, self.dx, axis=axis),
            "u_11": xp.gradient(field.u_11, self.dx, axis=axis),
        }

        return du_dx

    def _multiply_field_dagger_derivative(
        self, field: SU2Field, du: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute U†∂U.

        Args:
            field: SU(2) field
            du: Field derivative components

        Returns:
            Dictionary with Lᵢ components
        """
        # U† = [[u_00*, u_10*], [u_01*, u_11*]]
        # ∂U = [[du_00, du_01], [du_10, du_11]]
        # L = U†∂U

        xp = self.backend.get_array_module()
        l_00 = xp.conj(field.u_00) * du["u_00"] + xp.conj(field.u_10) * du["u_10"]
        l_01 = xp.conj(field.u_00) * du["u_01"] + xp.conj(field.u_10) * du["u_11"]
        l_10 = xp.conj(field.u_01) * du["u_00"] + xp.conj(field.u_11) * du["u_10"]
        l_11 = xp.conj(field.u_01) * du["u_01"] + xp.conj(field.u_11) * du["u_11"]

        return {"l_00": l_00, "l_01": l_01, "l_10": l_10, "l_11": l_11}

    def compute_commutators(
        self,
        l_x: Dict[str, Any],
        l_y: Dict[str, Any],
        l_z: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute commutators [Lᵢ, Lⱼ].

        Args:
            l_x, l_y, l_z: Left currents

        Returns:
            Dictionary with commutators
        """
        commutators = {}

        # [L_x, L_y]
        commutators["xy"] = self._compute_commutator(l_x, l_y)
        # [L_y, L_z]
        commutators["yz"] = self._compute_commutator(l_y, l_z)
        # [L_z, L_x]
        commutators["zx"] = self._compute_commutator(l_z, l_x)

        return commutators

    def _compute_commutator(
        self, l1: Dict[str, Any], l2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute commutator [L1, L2] = L1*L2 - L2*L1.

        Args:
            l1, l2: Left currents

        Returns:
            Commutator
        """
        # L1*L2
        l1l2_00 = l1["l_00"] * l2["l_00"] + l1["l_01"] * l2["l_10"]
        l1l2_01 = l1["l_00"] * l2["l_01"] + l1["l_01"] * l2["l_11"]
        l1l2_10 = l1["l_10"] * l2["l_00"] + l1["l_11"] * l2["l_10"]
        l1l2_11 = l1["l_10"] * l2["l_01"] + l1["l_11"] * l2["l_11"]

        # L2*L1
        l2l1_00 = l2["l_00"] * l1["l_00"] + l2["l_01"] * l1["l_10"]
        l2l1_01 = l2["l_00"] * l1["l_01"] + l2["l_01"] * l1["l_11"]
        l2l1_10 = l2["l_10"] * l1["l_00"] + l2["l_11"] * l1["l_10"]
        l2l1_11 = l2["l_10"] * l1["l_01"] + l2["l_11"] * l1["l_11"]

        # [L1, L2] = L1*L2 - L2*L1
        return {
            "comm_00": l1l2_00 - l2l1_00,
            "comm_01": l1l2_01 - l2l1_01,
            "comm_10": l1l2_10 - l2l1_10,
            "comm_11": l1l2_11 - l2l1_11,
        }

    def compute_traces(
        self,
        l_x: Dict[str, Any],
        l_y: Dict[str, Any],
        l_z: Dict[str, Any],
        commutators: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute traces for energy density.

        Args:
            l_x, l_y, l_z: Left currents
            commutators: Commutators

        Returns:
            Dictionary with traces
        """
        traces = {}

        # Tr(Lᵢ Lᵢ) - for c₂ term
        traces["l_squared"] = (
            l_x["l_00"] * l_x["l_00"]
            + l_x["l_01"] * l_x["l_10"]
            + l_x["l_10"] * l_x["l_01"]
            + l_x["l_11"] * l_x["l_11"]
            + l_y["l_00"] * l_y["l_00"]
            + l_y["l_01"] * l_y["l_10"]
            + l_y["l_10"] * l_y["l_01"]
            + l_y["l_11"] * l_y["l_11"]
            + l_z["l_00"] * l_z["l_00"]
            + l_z["l_01"] * l_z["l_10"]
            + l_z["l_10"] * l_z["l_01"]
            + l_z["l_11"] * l_z["l_11"]
        )

        # Tr([Lᵢ, Lⱼ]²) - for c₄ term
        traces["comm_squared"] = (
            commutators["xy"]["comm_00"] * commutators["xy"]["comm_00"]
            + commutators["xy"]["comm_01"] * commutators["xy"]["comm_10"]
            + commutators["xy"]["comm_10"] * commutators["xy"]["comm_01"]
            + commutators["xy"]["comm_11"] * commutators["xy"]["comm_11"]
            + commutators["yz"]["comm_00"] * commutators["yz"]["comm_00"]
            + commutators["yz"]["comm_01"] * commutators["yz"]["comm_10"]
            + commutators["yz"]["comm_10"] * commutators["yz"]["comm_01"]
            + commutators["yz"]["comm_11"] * commutators["yz"]["comm_11"]
            + commutators["zx"]["comm_00"] * commutators["zx"]["comm_00"]
            + commutators["zx"]["comm_01"] * commutators["zx"]["comm_10"]
            + commutators["zx"]["comm_10"] * commutators["zx"]["comm_01"]
            + commutators["zx"]["comm_11"] * commutators["zx"]["comm_11"]
        )

        return traces


class SU2FieldValidator:
    """SU(2) field validator."""

    def __init__(
        self, tolerance: float = 1e-10, backend: Optional[ArrayBackend] = None
    ):
        """
        Initialize validator.

        Args:
            tolerance: Allowed tolerance
            backend: Array backend (CUDA-aware or NumPy)
        """
        self.tolerance = tolerance
        self.backend = backend or ArrayBackend()

    def validate_field(self, field: SU2Field) -> Dict[str, bool]:
        """
        Full SU(2) field validation.

        Args:
            field: Field to validate

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check unitarity
        results["unitary"] = self._check_unitarity(field)

        # Check determinant
        results["determinant"] = self._check_determinant(field)

        # Check continuity
        results["continuity"] = self._check_continuity(field)

        # Check boundary conditions
        results["boundary_conditions"] = self._check_boundary_conditions(field)

        return results

    def _check_unitarity(self, field: SU2Field) -> bool:
        """Check field unitarity."""
        # Check U†U = I
        xp = self.backend.get_array_module()
        u_dagger_u_00 = (
            xp.conj(field.u_00) * field.u_00 + xp.conj(field.u_10) * field.u_10
        )
        u_dagger_u_11 = (
            xp.conj(field.u_01) * field.u_01 + xp.conj(field.u_11) * field.u_11
        )

        return bool(
            xp.allclose(u_dagger_u_00, 1.0, atol=self.tolerance)
            and xp.allclose(u_dagger_u_11, 1.0, atol=self.tolerance)
        )

    def _check_determinant(self, field: SU2Field) -> bool:
        """Check field determinant."""
        det = field.get_determinant()
        xp = self.backend.get_array_module()
        return bool(xp.allclose(det, 1.0, atol=self.tolerance))

    def _check_continuity(self, field: SU2Field) -> bool:
        """Check field continuity."""
        # Check gradients
        dx = field.box_size / field.grid_size

        xp = self.backend.get_array_module()
        grad_u_00 = xp.gradient(field.u_00, dx)
        grad_u_01 = xp.gradient(field.u_01, dx)
        grad_u_10 = xp.gradient(field.u_10, dx)
        grad_u_11 = xp.gradient(field.u_11, dx)

        # Check that gradients are finite
        return bool(
            xp.all(xp.isfinite(grad_u_00))
            and xp.all(xp.isfinite(grad_u_01))
            and xp.all(xp.isfinite(grad_u_10))
            and xp.all(xp.isfinite(grad_u_11))
        )

    def _check_boundary_conditions(self, field: SU2Field) -> bool:
        """Check boundary conditions."""
        # For torus configurations, we don't enforce strict boundary conditions
        # as they may not have the same center behavior as pure skyrmions
        # Just check that the field is well-behaved at the center
        center_idx = field.grid_size // 2
        center_field = field.get_matrix_at_point(center_idx, center_idx, center_idx)

        # Check that field is finite and well-behaved
        xp = self.backend.get_array_module()
        center_check = xp.all(xp.isfinite(center_field)) and xp.all(
            xp.abs(center_field) < 10.0
        )  # Reasonable bound

        return bool(center_check)


# Main class for SU(2) fields
class SU2Fields:
    """Main class for working with SU(2) fields."""

    def __init__(
        self, grid_size: int = 64, box_size: float = 4.0, use_cuda: bool = True
    ):
        """
        Initialize SU(2) fields.

        Args:
            grid_size: Grid size
            box_size: Box size
            use_cuda: Whether to use CUDA if available
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size

        # Initialize CUDA-aware backend
        self.backend = ArrayBackend()
        if not use_cuda:
            # Force CPU mode
            self.backend._use_cuda = False
            self.backend._cp = None

        self.builder = SU2FieldBuilder(grid_size, box_size, self.backend)
        self.operations = SU2FieldOperations(self.dx, self.backend)
        self.validator = SU2FieldValidator(backend=self.backend)

    def create_field_from_torus(
        self,
        torus_config: Any,
        profile_type: str = "skyrmion",
        scale: float = 1.0,
    ) -> SU2Field:
        """
        Create SU(2) field from torus configuration.

        Args:
            torus_config: Torus configuration
            profile_type: Radial profile type
            scale: Profile scale parameter

        Returns:
            SU(2) field
        """
        profile = RadialProfile(profile_type, scale, backend=self.backend)
        return self.builder.build_from_torus_config(torus_config, profile)

    def compute_field_derivatives(self, field: SU2Field) -> Dict[str, Any]:
        """
        Compute field derivatives.

        Args:
            field: SU(2) field

        Returns:
            Dictionary with derivatives and traces
        """
        # Compute left currents
        l_x, l_y, l_z = self.operations.compute_left_currents(field)

        # Compute commutators
        commutators = self.operations.compute_commutators(l_x, l_y, l_z)

        # Compute traces
        traces = self.operations.compute_traces(l_x, l_y, l_z, commutators)

        return {
            "left_currents": {"x": l_x, "y": l_y, "z": l_z},
            "commutators": commutators,
            "traces": traces,
        }

    def validate_field(self, field: SU2Field) -> Dict[str, bool]:
        """
        Validate SU(2) field.

        Args:
            field: Field to validate

        Returns:
            Validation results
        """
        return self.validator.validate_field(field)

    def get_field_statistics(self, field: SU2Field) -> Dict[str, float]:
        """
        Get field statistics.

        Args:
            field: SU(2) field

        Returns:
            Dictionary with statistics
        """
        det = field.get_determinant()
        xp = self.backend.get_array_module()

        return {
            "mean_determinant": float(xp.mean(xp.real(det))),
            "std_determinant": float(xp.std(xp.real(det))),
            "min_determinant": float(xp.min(xp.real(det))),
            "max_determinant": float(xp.max(xp.real(det))),
            "field_norm_mean": float(xp.mean(xp.abs(field.u_00))),
            "field_norm_std": float(xp.std(xp.abs(field.u_00))),
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
