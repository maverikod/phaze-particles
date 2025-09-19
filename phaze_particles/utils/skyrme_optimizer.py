#!/usr/bin/env python3
"""
Skyrme constants optimizer for achieving virial balance and energy balance.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class OptimizationTargets:
    """Target values for optimization."""
    
    # Energy balance targets
    target_e2_ratio: float = 0.5  # E₂/E_total = 50%
    target_e4_ratio: float = 0.5  # E₄/E_total = 50%
    target_e6_ratio: float = 0.0  # E₆/E_total = 0% (minimal)
    
    # Virial condition target
    target_virial_residual: float = 0.05  # |virial_residual| < 5%
    
    # Tolerances
    energy_balance_tolerance: float = 0.05  # ±5% for energy ratios
    virial_tolerance: float = 0.01  # ±1% for virial residual


@dataclass
class OptimizationResult:
    """Result of constants optimization."""
    
    # Optimized constants
    c2: float
    c4: float
    c6: float
    
    # Achieved values
    e2_ratio: float
    e4_ratio: float
    e6_ratio: float
    virial_residual: float
    
    # Optimization metrics
    iterations: int
    converged: bool
    final_error: float
    
    # Energy components
    E2: float
    E4: float
    E6: float
    E_total: float


class SkyrmeConstantsOptimizer:
    """
    Optimizer for Skyrme constants to achieve virial balance and energy balance.
    """
    
    def __init__(
        self,
        targets: Optional[OptimizationTargets] = None,
        max_iterations: int = 100,
        learning_rate: float = 0.1,
        convergence_tolerance: float = 1e-4
    ):
        """
        Initialize optimizer.
        
        Args:
            targets: Optimization targets
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            convergence_tolerance: Convergence tolerance
        """
        self.targets = targets or OptimizationTargets()
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.convergence_tolerance = convergence_tolerance
        
    def optimize_constants(
        self,
        energy_calculator,
        su2_field,
        initial_c2: float = 1.0,
        initial_c4: float = 1.0,
        initial_c6: float = 1.0,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize Skyrme constants for virial balance.
        
        Args:
            energy_calculator: Energy density calculator
            su2_field: SU(2) field
            initial_c2, initial_c4, initial_c6: Initial constants
            verbose: Verbose output
            
        Returns:
            Optimization result
        """
        # Initialize constants
        c2, c4, c6 = initial_c2, initial_c4, initial_c6
        
        if verbose:
            print("Starting Skyrme constants optimization...")
            print(f"Targets: E₂/E₄ = {self.targets.target_e2_ratio:.1%}/{self.targets.target_e4_ratio:.1%}")
            print(f"Virial residual target: < {self.targets.target_virial_residual:.1%}")
            print("-" * 60)
        
        for iteration in range(self.max_iterations):
            # Calculate energy with current constants
            energy_calculator.c2 = c2
            energy_calculator.c4 = c4
            energy_calculator.c6 = c6
            
            # Recalculate energy density
            energy_density = energy_calculator.calculate_energy_density(su2_field)
            
            # Get energy components and ratios
            components = energy_density.get_energy_components()
            balance = energy_density.get_energy_balance()
            virial_residual = energy_density.get_virial_residual()
            
            E2, E4, E6 = components["E2"], components["E4"], components["E6"]
            E_total = components["E_total"]
            
            e2_ratio = balance["E2_ratio"]
            e4_ratio = balance["E4_ratio"]
            e6_ratio = balance["E6_ratio"]
            
            # Calculate errors
            e2_error = e2_ratio - self.targets.target_e2_ratio
            e4_error = e4_ratio - self.targets.target_e4_ratio
            virial_error = abs(virial_residual) - self.targets.target_virial_residual
            
            # Total error
            total_error = abs(e2_error) + abs(e4_error) + max(0, virial_error)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration:3d}: "
                      f"E₂/E₄ = {e2_ratio:.3f}/{e4_ratio:.3f}, "
                      f"Virial = {virial_residual:.4f}, "
                      f"Error = {total_error:.4f}")
            
            # Check convergence
            if total_error < self.convergence_tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
            # Gradient-based updates
            # Adjust c2 to increase E2 ratio
            if e2_error < 0:  # E2 too small
                c2 *= (1 + self.learning_rate * abs(e2_error))
            else:  # E2 too large
                c2 *= (1 - self.learning_rate * abs(e2_error))
            
            # Adjust c4 to decrease E4 ratio
            if e4_error > 0:  # E4 too large
                c4 *= (1 - self.learning_rate * abs(e4_error))
            else:  # E4 too small
                c4 *= (1 + self.learning_rate * abs(e4_error))
            
            # Adjust c6 to minimize virial residual
            if virial_error > 0:  # Virial residual too large
                # Increase c6 to balance the virial condition
                c6 *= (1 + self.learning_rate * virial_error)
            
            # Ensure positive constants
            c2 = max(c2, 0.01)
            c4 = max(c4, 0.01)
            c6 = max(c6, 0.01)
        
        # Final calculation
        energy_calculator.c2 = c2
        energy_calculator.c4 = c4
        energy_calculator.c6 = c6
        
        final_energy_density = energy_calculator.calculate_energy_density(su2_field)
        
        final_components = final_energy_density.get_energy_components()
        final_balance = final_energy_density.get_energy_balance()
        final_virial_residual = final_energy_density.get_virial_residual()
        
        converged = total_error < self.convergence_tolerance
        
        if verbose:
            print("-" * 60)
            print("Optimization completed:")
            print(f"Final constants: c₂ = {c2:.6f}, c₄ = {c4:.6f}, c₆ = {c6:.6f}")
            print(f"Final ratios: E₂/E₄ = {final_balance['E2_ratio']:.3f}/{final_balance['E4_ratio']:.3f}")
            print(f"Final virial residual: {final_virial_residual:.6f}")
            print(f"Converged: {converged}")
        
        return OptimizationResult(
            c2=c2,
            c4=c4,
            c6=c6,
            e2_ratio=final_balance["E2_ratio"],
            e4_ratio=final_balance["E4_ratio"],
            e6_ratio=final_balance["E6_ratio"],
            virial_residual=final_virial_residual,
            iterations=iteration + 1,
            converged=converged,
            final_error=total_error,
            E2=final_components["E2"],
            E4=final_components["E4"],
            E6=final_components["E6"],
            E_total=final_components["E_total"]
        )
    
    def optimize_for_virial_balance(
        self,
        energy_calculator,
        su2_field,
        initial_c2: float = 1.0,
        initial_c4: float = 1.0,
        initial_c6: float = 1.0,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize specifically for virial balance: -E₂ + E₄ + 3E₆ = 0.
        
        Args:
            energy_calculator: Energy density calculator
            su2_field: SU(2) field
            initial_c2, initial_c4, initial_c6: Initial constants
            verbose: Verbose output
            
        Returns:
            Optimization result
        """
        # Set targets for virial balance
        virial_targets = OptimizationTargets(
            target_e2_ratio=0.5,
            target_e4_ratio=0.5,
            target_e6_ratio=0.0,
            target_virial_residual=0.01,  # Very tight virial tolerance
            energy_balance_tolerance=0.1,  # Relaxed energy balance
            virial_tolerance=0.005
        )
        
        # Create optimizer with virial-focused targets
        virial_optimizer = SkyrmeConstantsOptimizer(
            targets=virial_targets,
            max_iterations=self.max_iterations,
            learning_rate=self.learning_rate * 0.5,  # Slower learning for stability
            convergence_tolerance=self.convergence_tolerance
        )
        
        return virial_optimizer.optimize_constants(
            energy_calculator, su2_field, initial_c2, initial_c4, initial_c6, verbose
        )
    
    def get_optimization_report(self, result: OptimizationResult) -> str:
        """
        Generate optimization report.
        
        Args:
            result: Optimization result
            
        Returns:
            Report string
        """
        report = f"""
SKYRME CONSTANTS OPTIMIZATION REPORT
====================================

Optimized Constants:
  c₂ = {result.c2:.6f}
  c₄ = {result.c4:.6f}
  c₆ = {result.c6:.6f}

Energy Components:
  E₂ = {result.E2:.6f} MeV
  E₄ = {result.E4:.6f} MeV
  E₆ = {result.E6:.6f} MeV
  E_total = {result.E_total:.6f} MeV

Energy Balance:
  E₂/E_total = {result.e2_ratio:.3f} (target: {self.targets.target_e2_ratio:.3f})
  E₄/E_total = {result.e4_ratio:.3f} (target: {self.targets.target_e4_ratio:.3f})
  E₆/E_total = {result.e6_ratio:.3f} (target: {self.targets.target_e6_ratio:.3f})

Virial Analysis:
  Virial Residual = {result.virial_residual:.6f} (target: < {self.targets.target_virial_residual:.3f})
  Virial Condition: {'✓ PASS' if abs(result.virial_residual) < self.targets.target_virial_residual else '✗ FAIL'}

Optimization Status:
  Iterations: {result.iterations}/{self.max_iterations}
  Converged: {'✓ YES' if result.converged else '✗ NO'}
  Final Error: {result.final_error:.6f}

Quality Assessment:
"""
        
        # Quality assessment
        e2_quality = "✓ EXCELLENT" if abs(result.e2_ratio - self.targets.target_e2_ratio) < 0.05 else \
                    "✓ GOOD" if abs(result.e2_ratio - self.targets.target_e2_ratio) < 0.1 else \
                    "⚠ FAIR" if abs(result.e2_ratio - self.targets.target_e2_ratio) < 0.2 else "✗ POOR"
        
        e4_quality = "✓ EXCELLENT" if abs(result.e4_ratio - self.targets.target_e4_ratio) < 0.05 else \
                    "✓ GOOD" if abs(result.e4_ratio - self.targets.target_e4_ratio) < 0.1 else \
                    "⚠ FAIR" if abs(result.e4_ratio - self.targets.target_e4_ratio) < 0.2 else "✗ POOR"
        
        virial_quality = "✓ EXCELLENT" if abs(result.virial_residual) < 0.01 else \
                        "✓ GOOD" if abs(result.virial_residual) < 0.05 else \
                        "⚠ FAIR" if abs(result.virial_residual) < 0.1 else "✗ POOR"
        
        report += f"""
  E₂ Balance: {e2_quality}
  E₄ Balance: {e4_quality}
  Virial: {virial_quality}
"""
        
        return report


class AdaptiveOptimizer:
    """
    Adaptive optimizer that adjusts learning rate and strategy based on progress.
    """
    
    def __init__(self, base_optimizer: SkyrmeConstantsOptimizer):
        """
        Initialize adaptive optimizer.
        
        Args:
            base_optimizer: Base optimizer to enhance
        """
        self.base_optimizer = base_optimizer
        self.learning_rate_history = []
        self.error_history = []
    
    def optimize_with_adaptation(
        self,
        energy_calculator,
        su2_field,
        initial_c2: float = 1.0,
        initial_c4: float = 1.0,
        initial_c6: float = 1.0,
        verbose: bool = False
    ) -> OptimizationResult:
        """
        Optimize with adaptive learning rate.
        
        Args:
            energy_calculator: Energy density calculator
            su2_field: SU(2) field
            initial_c2, initial_c4, initial_c6: Initial constants
            verbose: Verbose output
            
        Returns:
            Optimization result
        """
        # Start with base optimization
        result = self.base_optimizer.optimize_constants(
            energy_calculator, su2_field, initial_c2, initial_c4, initial_c6, verbose
        )
        
        # If not converged, try with different strategies
        if not result.converged:
            if verbose:
                print("Base optimization did not converge, trying adaptive strategies...")
            
            # Strategy 1: Reduce learning rate
            self.base_optimizer.learning_rate *= 0.5
            result = self.base_optimizer.optimize_constants(
                energy_calculator, su2_field, result.c2, result.c4, result.c6, verbose
            )
            
            # Strategy 2: Focus on virial balance
            if not result.converged:
                if verbose:
                    print("Trying virial-focused optimization...")
                result = self.base_optimizer.optimize_for_virial_balance(
                    energy_calculator, su2_field, result.c2, result.c4, result.c6, verbose
                )
        
        return result
