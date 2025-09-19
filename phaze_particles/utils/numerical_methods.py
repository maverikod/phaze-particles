#!/usr/bin/env python3
"""
Numerical methods for proton model optimization.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import time
import logging
from typing import Tuple, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

# from .cuda import get_cuda_manager, is_cuda_available
from .mathematical_foundations import ArrayBackend
from .progress import ProgressBar


class RelaxationMethod(Enum):
    """Relaxation methods."""

    GRADIENT_DESCENT = "gradient_descent"
    LBFGS = "lbfgs"
    ADAM = "adam"


@dataclass
class RelaxationConfig:
    """Relaxation configuration."""

    def __init__(
        self,
        method: Union[RelaxationMethod, str] = RelaxationMethod.GRADIENT_DESCENT,
        max_iterations: int = 1000,
        convergence_tol: float = 1e-6,
        step_size: float = 0.01,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize relaxation configuration.

        Args:
            method: Relaxation method (enum or string)
            max_iterations: Maximum iterations
            convergence_tol: Convergence tolerance
            step_size: Step size
            momentum: Momentum parameter
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            epsilon: Adam epsilon parameter
        """
        if isinstance(method, str):
            method = RelaxationMethod(method)
        self.method = method
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.step_size = step_size
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


@dataclass
class ConstraintConfig:
    """Constraint configuration."""

    lambda_B: float = 1000.0
    lambda_Q: float = 1000.0
    lambda_virial: float = 1000.0
    tolerance_B: float = 0.02
    tolerance_Q: float = 1e-6
    tolerance_virial: float = 0.01


class SU2Projection:
    """Projection onto SU(2) group with CUDA support."""

    def __init__(self, backend: Optional[ArrayBackend] = None):
        """
        Initialize SU(2) projection.

        Args:
            backend: Array backend for CUDA support
        """
        self.backend = backend or ArrayBackend()
        self.logger = logging.getLogger(__name__)

    def project_to_su2(self, U: np.ndarray) -> np.ndarray:
        """
        Project matrix onto SU(2) group.

        Args:
            U: Matrix to project

        Returns:
            Matrix in SU(2)
        """
        xp = self.backend.get_array_module()

        # Convert to appropriate backend array if needed
        if hasattr(U, 'get') and self.backend.is_cuda_available:
            # Already CuPy array
            U_backend = U
        elif self.backend.is_cuda_available:
            # Convert NumPy to CuPy
            U_backend = xp.asarray(U)
        else:
            # Use NumPy
            U_backend = np.asarray(U)

        # QR decomposition
        Q, R = xp.linalg.qr(U_backend)

        # Correct determinant
        det_Q = xp.linalg.det(Q)
        Q = Q / (det_Q ** (1 / 2))

        # Check unitarity
        if not xp.allclose(xp.dot(Q.conj().T, Q), xp.eye(2), atol=1e-10):
            # Re-project
            Q = (Q + Q.conj().T) / 2
            Q = Q / xp.sqrt(xp.trace(xp.dot(Q.conj().T, Q)) / 2)

        return Q

    def validate_su2(self, U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix belongs to SU(2).

        Args:
            U: Matrix to check
            tolerance: Allowed tolerance

        Returns:
            True if matrix ∈ SU(2)
        """
        xp = self.backend.get_array_module()

        # Convert to appropriate backend array if needed
        if hasattr(U, 'get') and self.backend.is_cuda_available:
            # Already CuPy array
            U_backend = U
        elif self.backend.is_cuda_available:
            # Convert NumPy to CuPy
            U_backend = xp.asarray(U)
        else:
            # Use NumPy
            U_backend = np.asarray(U)

        # Check unitarity
        unitary_check = xp.allclose(xp.dot(U_backend.conj().T, U_backend), xp.eye(2), atol=tolerance)

        # Check determinant
        det_check = abs(xp.linalg.det(U_backend) - 1.0) < tolerance

        return bool(unitary_check and det_check)

    @staticmethod
    def project_to_su2_static(U: np.ndarray) -> np.ndarray:
        """
        Static method for backward compatibility.

        Args:
            U: Matrix to project

        Returns:
            Matrix in SU(2)
        """
        projection = SU2Projection()
        return projection.project_to_su2(U)

    @staticmethod
    def validate_su2_static(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Static method for backward compatibility.

        Args:
            U: Matrix to check
            tolerance: Allowed tolerance

        Returns:
            True if matrix ∈ SU(2)
        """
        projection = SU2Projection()
        return projection.validate_su2(U, tolerance)


class GradientDescent:
    """Gradient descent with SU(2) projection and CUDA support."""

    def __init__(
        self, config: RelaxationConfig, backend: Optional[ArrayBackend] = None
    ):
        """
        Initialize gradient descent.

        Args:
            config: Relaxation configuration
            backend: Array backend for CUDA support
        """
        self.config = config
        self.step_size = config.step_size
        self.momentum = config.momentum
        self.velocity = None
        self.backend = backend or ArrayBackend()
        self.su2_projection = SU2Projection(self.backend)
        self.logger = logging.getLogger(__name__)

    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One gradient descent step.

        Args:
            U: Current field
            gradient: Functional gradient

        Returns:
            Updated field
        """
        xp = self.backend.get_array_module()

        # Convert to appropriate backend arrays if needed
        if hasattr(U, 'get') and self.backend.is_cuda_available:
            U_backend = U
        elif self.backend.is_cuda_available:
            U_backend = xp.asarray(U)
        else:
            U_backend = np.asarray(U)

        if hasattr(gradient, 'get') and self.backend.is_cuda_available:
            gradient_backend = gradient
        elif self.backend.is_cuda_available:
            gradient_backend = xp.asarray(gradient)
        else:
            gradient_backend = np.asarray(gradient)

        if self.velocity is None:
            self.velocity = xp.zeros_like(gradient_backend)

        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity + self.step_size * gradient_backend

        # Update field
        U_new = U_backend - self.velocity

        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = self.su2_projection.project_to_su2(
                            U_new[i, j, k]
                        )
        else:  # Single matrix
            U_new = self.su2_projection.project_to_su2(U_new)

        return U_new

    def reset(self):
        """Reset optimizer state."""
        self.velocity = None


class LBFGSOptimizer:
    """L-BFGS optimizer with CUDA support."""

    def __init__(
        self, config: RelaxationConfig, backend: Optional[ArrayBackend] = None
    ):
        """
        Initialize L-BFGS.

        Args:
            config: Relaxation configuration
            backend: Array backend for CUDA support
        """
        self.config = config
        self.backend = backend or ArrayBackend()
        self.su2_projection = SU2Projection(self.backend)
        self.memory_size = 10
        self.s_history = []
        self.y_history = []
        self.rho_history = []
        self.logger = logging.getLogger(__name__)

    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One L-BFGS step.

        Args:
            U: Current field
            gradient: Functional gradient

        Returns:
            Updated field
        """
        xp = self.backend.get_array_module()

        # Simple implementation (full L-BFGS needed in reality)
        U_new = U - self.config.step_size * gradient

        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = self.su2_projection.project_to_su2(
                            U_new[i, j, k]
                        )
        else:  # Single matrix
            U_new = self.su2_projection.project_to_su2(U_new)

        return U_new

    def reset(self):
        """Reset optimizer state."""
        self.s_history = []
        self.y_history = []
        self.rho_history = []


class AdamOptimizer:
    """Adam optimizer with CUDA support."""

    def __init__(
        self, config: RelaxationConfig, backend: Optional[ArrayBackend] = None
    ):
        """
        Initialize Adam.

        Args:
            config: Relaxation configuration
            backend: Array backend for CUDA support
        """
        self.config = config
        self.backend = backend or ArrayBackend()
        self.su2_projection = SU2Projection(self.backend)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.logger = logging.getLogger(__name__)

    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One Adam step.

        Args:
            U: Current field
            gradient: Functional gradient

        Returns:
            Updated field
        """
        xp = self.backend.get_array_module()

        # Convert to appropriate backend arrays if needed
        if hasattr(U, 'get') and self.backend.is_cuda_available:
            U_backend = U
        elif self.backend.is_cuda_available:
            U_backend = xp.asarray(U)
        else:
            U_backend = np.asarray(U)

        if hasattr(gradient, 'get') and self.backend.is_cuda_available:
            gradient_backend = gradient
        elif self.backend.is_cuda_available:
            gradient_backend = xp.asarray(gradient)
        else:
            gradient_backend = np.asarray(gradient)

        if self.m is None:
            self.m = xp.zeros_like(gradient_backend)
            self.v = xp.zeros_like(gradient_backend)

        self.t += 1

        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient_backend
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient_backend**2)

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update field
        U_new = U_backend - self.config.step_size * m_hat / (xp.sqrt(v_hat) + self.epsilon)

        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = self.su2_projection.project_to_su2(
                            U_new[i, j, k]
                        )
        else:  # Single matrix
            U_new = self.su2_projection.project_to_su2(U_new)

        return U_new

    def reset(self):
        """Reset optimizer state."""
        self.m = None
        self.v = None
        self.t = 0


class ConstraintController:
    """Constraint controller."""

    def __init__(self, config: ConstraintConfig):
        """
        Initialize constraint controller.

        Args:
            config: Constraint configuration
        """
        self.config = config
        self.lambda_B = config.lambda_B
        self.lambda_Q = config.lambda_Q
        self.lambda_virial = config.lambda_virial

    def compute_constraint_penalty(
        self,
        U: np.ndarray,
        baryon_number: float,
        electric_charge: float,
        energy_balance: float,
    ) -> float:
        """
        Compute constraint violation penalty.

        Args:
            U: SU(2) field
            baryon_number: Baryon number
            electric_charge: Electric charge
            energy_balance: Energy balance

        Returns:
            Constraint penalty
        """
        penalty = 0.0

        # Baryon number penalty
        penalty += self.lambda_B * (baryon_number - 1.0) ** 2

        # Electric charge penalty
        penalty += self.lambda_Q * (electric_charge - 1.0) ** 2

        # Virial condition penalty
        penalty += self.lambda_virial * (energy_balance - 0.5) ** 2

        return penalty

    def check_constraints(
        self, baryon_number: float, electric_charge: float, energy_balance: float
    ) -> Dict[str, bool]:
        """
        Check constraint satisfaction.

        Args:
            baryon_number: Baryon number
            electric_charge: Electric charge
            energy_balance: Energy balance

        Returns:
            Dictionary with constraint check results
        """
        return {
            "baryon_number": abs(baryon_number - 1.0) <= self.config.tolerance_B,
            "electric_charge": abs(electric_charge - 1.0) <= self.config.tolerance_Q,
            "energy_balance": abs(energy_balance - 0.5) <= self.config.tolerance_virial,
        }


class RelaxationSolver:
    """Main relaxation solver with CUDA support and progress tracking."""

    def __init__(
        self,
        config: RelaxationConfig,
        constraint_config: ConstraintConfig,
        backend: Optional[ArrayBackend] = None,
    ):
        """
        Initialize solver.

        Args:
            config: Relaxation configuration
            constraint_config: Constraint configuration
            backend: Array backend for CUDA support
        """
        self.config = config
        self.constraint_controller = ConstraintController(constraint_config)
        self.backend = backend or ArrayBackend()
        self.logger = logging.getLogger(__name__)

        # Choose optimizer
        if config.method == RelaxationMethod.GRADIENT_DESCENT:
            self.optimizer = GradientDescent(config, self.backend)
        elif config.method == RelaxationMethod.LBFGS:
            self.optimizer = LBFGSOptimizer(config, self.backend)
        elif config.method == RelaxationMethod.ADAM:
            self.optimizer = AdamOptimizer(config, self.backend)
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")

        # Initialize progress tracker
        self.progress_tracker = ProgressBar(
            total=config.max_iterations, description="Relaxation optimization"
        )

    def solve(
        self,
        U_init: np.ndarray,
        energy_function: Callable,
        gradient_function: Callable,
        constraint_functions: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """
        Solve relaxation problem.

        Args:
            U_init: Initial field
            energy_function: Energy computation function
            gradient_function: Gradient computation function
            constraint_functions: Constraint computation functions

        Returns:
            Dictionary with results
        """
        U = U_init.copy()
        energy_history = []
        constraint_history = []

        start_time = time.time()

        # Log CUDA status
        cuda_status = "CUDA" if self.backend.is_cuda_available else "CPU"
        self.logger.info(f"Starting relaxation optimization using {cuda_status}")

        try:
            for iteration in range(self.config.max_iterations):
                # Update progress
                self.progress_tracker.set_progress(iteration)

                # Compute energy
                energy = energy_function(U)
                energy_history.append(energy)

                # Compute constraints
                constraints = {}
                for name, func in constraint_functions.items():
                    constraints[name] = func(U)
                constraint_history.append(constraints)

                # Check convergence
                if iteration > 0:
                    energy_change = abs(energy - energy_history[-2])
                    if energy_change < self.config.convergence_tol:
                        self.logger.info(f"Converged after {iteration} iterations")
                        break

                # Compute gradient
                gradient = gradient_function(U)

                # Optimization step
                U = self.optimizer.step(U, gradient)

                # Check constraints
                constraint_check = self.constraint_controller.check_constraints(
                    constraints.get("baryon_number", 0),
                    constraints.get("electric_charge", 0),
                    constraints.get("energy_balance", 0),
                )

                # Log progress
                if iteration % 100 == 0:
                    self.logger.info(
                        f"Iteration {iteration}: Energy = {energy:.6f}, "
                        f"Constraints = {constraint_check}"
                    )

        finally:
            self.progress_tracker.finish()

        end_time = time.time()

        return {
            "solution": U,
            "energy_history": energy_history,
            "constraint_history": constraint_history,
            "iterations": iteration + 1,
            "converged": iteration < self.config.max_iterations - 1,
            "execution_time": end_time - start_time,
            "final_energy": energy_history[-1] if energy_history else 0.0,
            "final_constraints": constraint_history[-1] if constraint_history else {},
        }

    def reset(self):
        """Reset solver state."""
        self.optimizer.reset()


class NumericalMethods:
    """Main numerical methods class with CUDA support."""

    def __init__(
        self,
        grid_size: int = 64,
        box_size: float = 4.0,
        backend: Optional[ArrayBackend] = None,
    ):
        """
        Initialize numerical methods.

        Args:
            grid_size: Grid size
            box_size: Box size in fm
            backend: Array backend for CUDA support
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        self.backend = backend or ArrayBackend()
        self.logger = logging.getLogger(__name__)

        # Create coordinate grids
        xp = self.backend.get_array_module()
        x = xp.linspace(-box_size / 2, box_size / 2, grid_size)
        y = xp.linspace(-box_size / 2, box_size / 2, grid_size)
        z = xp.linspace(-box_size / 2, box_size / 2, grid_size)
        self.X, self.Y, self.Z = xp.meshgrid(x, y, z, indexing="ij")
        self.R = xp.sqrt(self.X**2 + self.Y**2 + self.Z**2)

        # Log CUDA status
        cuda_status = "CUDA" if self.backend.is_cuda_available else "CPU"
        self.logger.info(f"NumericalMethods initialized using {cuda_status}")

    def compute_gradient(
        self, field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute field gradient.

        Args:
            field: Scalar field

        Returns:
            Tuple (grad_x, grad_y, grad_z)
        """
        xp = self.backend.get_array_module()
        grad_x = xp.gradient(field, self.dx, axis=0)
        grad_y = xp.gradient(field, self.dx, axis=1)
        grad_z = xp.gradient(field, self.dx, axis=2)

        return grad_x, grad_y, grad_z

    def compute_divergence(
        self, field_x: np.ndarray, field_y: np.ndarray, field_z: np.ndarray
    ) -> np.ndarray:
        """
        Compute vector field divergence.

        Args:
            field_x, field_y, field_z: Vector field components

        Returns:
            Divergence
        """
        xp = self.backend.get_array_module()
        div_x = xp.gradient(field_x, self.dx, axis=0)
        div_y = xp.gradient(field_y, self.dx, axis=1)
        div_z = xp.gradient(field_z, self.dx, axis=2)

        return div_x + div_y + div_z

    def integrate_3d(self, field: np.ndarray) -> float:
        """
        Integrate 3D field over volume.

        Args:
            field: 3D field to integrate

        Returns:
            Integration result
        """
        xp = self.backend.get_array_module()
        return float(xp.sum(field) * self.dx**3)

    def create_initial_field(self, profile_type: str = "tanh") -> np.ndarray:
        """
        Create initial field.

        Args:
            profile_type: Profile type

        Returns:
            Initial SU(2) field
        """
        xp = self.backend.get_array_module()

        if profile_type == "tanh":
            f = xp.pi * (1 - xp.tanh(self.R))
        elif profile_type == "exp":
            f = xp.pi * xp.exp(-self.R)
        elif profile_type == "gaussian":
            f = xp.pi * xp.exp(-self.R**2)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

        # Create SU(2) field
        U = xp.zeros(
            (self.grid_size, self.grid_size, self.grid_size, 2, 2), dtype=complex
        )

        # Fill field
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    f_val = f[i, j, k]
                    U[i, j, k] = xp.array(
                        [
                            [xp.cos(f_val), 1j * xp.sin(f_val)],
                            [1j * xp.sin(f_val), xp.cos(f_val)],
                        ]
                    )

        return U

    def validate_solution(self, U: np.ndarray) -> Dict[str, bool]:
        """
        Validate solution.

        Args:
            U: SU(2) field

        Returns:
            Dictionary with validation results
        """
        xp = self.backend.get_array_module()
        su2_projection = SU2Projection(self.backend)

        results = {}

        # Check SU(2) for each point
        su2_valid = True
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if not su2_projection.validate_su2(U[i, j, k]):
                        su2_valid = False
                        break
                if not su2_valid:
                    break
            if not su2_valid:
                break

        results["su2_valid"] = su2_valid

        # Check boundary conditions
        center_field = U[self.grid_size // 2, self.grid_size // 2, self.grid_size // 2]
        boundary_field = U[0, 0, 0]

        results["boundary_conditions"] = (
            abs(xp.linalg.det(center_field) + 1) < 1e-6
            and abs(xp.linalg.det(boundary_field) - 1) < 1e-6
        )

        return results
