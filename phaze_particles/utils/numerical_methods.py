#!/usr/bin/env python3
"""
Numerical methods for proton model optimization.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RelaxationMethod(Enum):
    """Relaxation methods."""
    GRADIENT_DESCENT = "gradient_descent"
    LBFGS = "lbfgs"
    ADAM = "adam"


@dataclass
class RelaxationConfig:
    """Relaxation configuration."""
    method: RelaxationMethod
    max_iterations: int = 1000
    convergence_tol: float = 1e-6
    step_size: float = 0.01
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


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
    """Projection onto SU(2) group."""
    
    @staticmethod
    def project_to_su2(U: np.ndarray) -> np.ndarray:
        """
        Project matrix onto SU(2) group.
        
        Args:
            U: Matrix to project
            
        Returns:
            Matrix in SU(2)
        """
        # QR decomposition
        Q, R = np.linalg.qr(U)
        
        # Correct determinant
        det_Q = np.linalg.det(Q)
        Q = Q / (det_Q**(1/2))
        
        # Check unitarity
        if not np.allclose(np.dot(Q.conj().T, Q), np.eye(2), atol=1e-10):
            # Re-project
            Q = (Q + Q.conj().T) / 2
            Q = Q / np.sqrt(np.trace(np.dot(Q.conj().T, Q)) / 2)
        
        return Q
    
    @staticmethod
    def validate_su2(U: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if matrix belongs to SU(2).
        
        Args:
            U: Matrix to check
            tolerance: Allowed tolerance
            
        Returns:
            True if matrix ∈ SU(2)
        """
        # Check unitarity
        unitary_check = np.allclose(np.dot(U.conj().T, U), np.eye(2), atol=tolerance)
        
        # Check determinant
        det_check = abs(np.linalg.det(U) - 1.0) < tolerance
        
        return unitary_check and det_check


class GradientDescent:
    """Gradient descent with SU(2) projection."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Initialize gradient descent.
        
        Args:
            config: Relaxation configuration
        """
        self.config = config
        self.step_size = config.step_size
        self.momentum = config.momentum
        self.velocity = None
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One gradient descent step.
        
        Args:
            U: Current field
            gradient: Functional gradient
            
        Returns:
            Updated field
        """
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient)
        
        # Update velocity with momentum
        self.velocity = self.momentum * self.velocity + self.step_size * gradient
        
        # Update field
        U_new = U - self.velocity
        
        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = SU2Projection.project_to_su2(U_new[i, j, k])
        else:  # Single matrix
            U_new = SU2Projection.project_to_su2(U_new)
        
        return U_new
    
    def reset(self):
        """Reset optimizer state."""
        self.velocity = None


class LBFGSOptimizer:
    """L-BFGS optimizer."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Initialize L-BFGS.
        
        Args:
            config: Relaxation configuration
        """
        self.config = config
        self.memory_size = 10
        self.s_history = []
        self.y_history = []
        self.rho_history = []
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One L-BFGS step.
        
        Args:
            U: Current field
            gradient: Functional gradient
            
        Returns:
            Updated field
        """
        # Simple implementation (full L-BFGS needed in reality)
        U_new = U - self.config.step_size * gradient
        
        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = SU2Projection.project_to_su2(U_new[i, j, k])
        else:  # Single matrix
            U_new = SU2Projection.project_to_su2(U_new)
        
        return U_new
    
    def reset(self):
        """Reset optimizer state."""
        self.s_history = []
        self.y_history = []
        self.rho_history = []


class AdamOptimizer:
    """Adam optimizer."""
    
    def __init__(self, config: RelaxationConfig):
        """
        Initialize Adam.
        
        Args:
            config: Relaxation configuration
        """
        self.config = config
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, U: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        One Adam step.
        
        Args:
            U: Current field
            gradient: Functional gradient
            
        Returns:
            Updated field
        """
        if self.m is None:
            self.m = np.zeros_like(gradient)
            self.v = np.zeros_like(gradient)
        
        self.t += 1
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update field
        U_new = U - self.config.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Project onto SU(2) for each point
        if U_new.ndim == 5:  # 3D field
            for i in range(U_new.shape[0]):
                for j in range(U_new.shape[1]):
                    for k in range(U_new.shape[2]):
                        U_new[i, j, k] = SU2Projection.project_to_su2(U_new[i, j, k])
        else:  # Single matrix
            U_new = SU2Projection.project_to_su2(U_new)
        
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
    
    def compute_constraint_penalty(self, U: np.ndarray, 
                                 baryon_number: float, 
                                 electric_charge: float,
                                 energy_balance: float) -> float:
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
        penalty += self.lambda_B * (baryon_number - 1.0)**2
        
        # Electric charge penalty
        penalty += self.lambda_Q * (electric_charge - 1.0)**2
        
        # Virial condition penalty
        penalty += self.lambda_virial * (energy_balance - 0.5)**2
        
        return penalty
    
    def check_constraints(self, baryon_number: float, 
                         electric_charge: float,
                         energy_balance: float) -> Dict[str, bool]:
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
            'baryon_number': abs(baryon_number - 1.0) <= self.config.tolerance_B,
            'electric_charge': abs(electric_charge - 1.0) <= self.config.tolerance_Q,
            'energy_balance': abs(energy_balance - 0.5) <= self.config.tolerance_virial
        }


class RelaxationSolver:
    """Main relaxation solver."""
    
    def __init__(self, config: RelaxationConfig, constraint_config: ConstraintConfig):
        """
        Initialize solver.
        
        Args:
            config: Relaxation configuration
            constraint_config: Constraint configuration
        """
        self.config = config
        self.constraint_controller = ConstraintController(constraint_config)
        
        # Choose optimizer
        if config.method == RelaxationMethod.GRADIENT_DESCENT:
            self.optimizer = GradientDescent(config)
        elif config.method == RelaxationMethod.LBFGS:
            self.optimizer = LBFGSOptimizer(config)
        elif config.method == RelaxationMethod.ADAM:
            self.optimizer = AdamOptimizer(config)
        else:
            raise ValueError(f"Unknown optimization method: {config.method}")
    
    def solve(self, U_init: np.ndarray, 
              energy_function: Callable,
              gradient_function: Callable,
              constraint_functions: Dict[str, Callable]) -> Dict[str, Any]:
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
        
        for iteration in range(self.config.max_iterations):
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
                    break
            
            # Compute gradient
            gradient = gradient_function(U)
            
            # Optimization step
            U = self.optimizer.step(U, gradient)
            
            # Check constraints
            constraint_check = self.constraint_controller.check_constraints(
                constraints.get('baryon_number', 0),
                constraints.get('electric_charge', 0),
                constraints.get('energy_balance', 0)
            )
            
            # Log progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Energy = {energy:.6f}, "
                      f"Constraints = {constraint_check}")
        
        end_time = time.time()
        
        return {
            'solution': U,
            'energy_history': energy_history,
            'constraint_history': constraint_history,
            'iterations': iteration + 1,
            'converged': iteration < self.config.max_iterations - 1,
            'execution_time': end_time - start_time,
            'final_energy': energy_history[-1] if energy_history else 0.0,
            'final_constraints': constraint_history[-1] if constraint_history else {}
        }
    
    def reset(self):
        """Reset solver state."""
        self.optimizer.reset()


class NumericalMethods:
    """Main numerical methods class."""
    
    def __init__(self, grid_size: int = 64, box_size: float = 4.0):
        """
        Initialize numerical methods.
        
        Args:
            grid_size: Grid size
            box_size: Box size in fm
        """
        self.grid_size = grid_size
        self.box_size = box_size
        self.dx = box_size / grid_size
        
        # Create coordinate grids
        x = np.linspace(-box_size/2, box_size/2, grid_size)
        y = np.linspace(-box_size/2, box_size/2, grid_size)
        z = np.linspace(-box_size/2, box_size/2, grid_size)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
    
    def compute_gradient(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute field gradient.
        
        Args:
            field: Scalar field
            
        Returns:
            Tuple (grad_x, grad_y, grad_z)
        """
        grad_x = np.gradient(field, self.dx, axis=0)
        grad_y = np.gradient(field, self.dx, axis=1)
        grad_z = np.gradient(field, self.dx, axis=2)
        
        return grad_x, grad_y, grad_z
    
    def compute_divergence(self, field_x: np.ndarray, field_y: np.ndarray, 
                          field_z: np.ndarray) -> np.ndarray:
        """
        Compute vector field divergence.
        
        Args:
            field_x, field_y, field_z: Vector field components
            
        Returns:
            Divergence
        """
        div_x = np.gradient(field_x, self.dx, axis=0)
        div_y = np.gradient(field_y, self.dx, axis=1)
        div_z = np.gradient(field_z, self.dx, axis=2)
        
        return div_x + div_y + div_z
    
    def integrate_3d(self, field: np.ndarray) -> float:
        """
        Integrate 3D field over volume.
        
        Args:
            field: 3D field to integrate
            
        Returns:
            Integration result
        """
        return np.sum(field) * self.dx**3
    
    def create_initial_field(self, profile_type: str = "tanh") -> np.ndarray:
        """
        Create initial field.
        
        Args:
            profile_type: Profile type
            
        Returns:
            Initial SU(2) field
        """
        if profile_type == "tanh":
            f = np.pi * (1 - np.tanh(self.R))
        elif profile_type == "exp":
            f = np.pi * np.exp(-self.R)
        elif profile_type == "gaussian":
            f = np.pi * np.exp(-self.R**2)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")
        
        # Create SU(2) field
        U = np.zeros((self.grid_size, self.grid_size, self.grid_size, 2, 2), dtype=complex)
        
        # Fill field
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    f_val = f[i, j, k]
                    U[i, j, k] = np.array([
                        [np.cos(f_val), 1j * np.sin(f_val)],
                        [1j * np.sin(f_val), np.cos(f_val)]
                    ])
        
        return U
    
    def validate_solution(self, U: np.ndarray) -> Dict[str, bool]:
        """
        Validate solution.
        
        Args:
            U: SU(2) field
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Check SU(2) for each point
        su2_valid = True
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    if not SU2Projection.validate_su2(U[i, j, k]):
                        su2_valid = False
                        break
                if not su2_valid:
                    break
            if not su2_valid:
                break
        
        results['su2_valid'] = su2_valid
        
        # Check boundary conditions
        center_field = U[self.grid_size//2, self.grid_size//2, self.grid_size//2]
        boundary_field = U[0, 0, 0]
        
        results['boundary_conditions'] = (
            abs(np.linalg.det(center_field) + 1) < 1e-6 and
            abs(np.linalg.det(boundary_field) - 1) < 1e-6
        )
        
        return results
