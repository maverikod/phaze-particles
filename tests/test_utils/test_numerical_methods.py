#!/usr/bin/env python3
"""
Tests for numerical methods module.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from phaze_particles.utils.numerical_methods import (
    RelaxationMethod,
    RelaxationConfig,
    ConstraintConfig,
    SU2Projection,
    GradientDescent,
    LBFGSOptimizer,
    AdamOptimizer,
    ConstraintController,
    RelaxationSolver,
    NumericalMethods,
)


class TestSU2Projection(unittest.TestCase):
    """Test SU(2) projection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance = 1e-10

    def test_project_to_su2_unitary_matrix(self):
        """Test projection of unitary matrix."""
        # Create a unitary matrix
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        projected = SU2Projection.project_to_su2(U)
        
        # Check unitarity
        self.assertTrue(np.allclose(
            np.dot(projected.conj().T, projected), 
            np.eye(2), 
            atol=self.tolerance
        ))
        
        # Check determinant
        self.assertTrue(abs(np.linalg.det(projected) - 1.0) < self.tolerance)

    def test_project_to_su2_random_matrix(self):
        """Test projection of random matrix."""
        # Create random matrix
        np.random.seed(42)
        U = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        projected = SU2Projection.project_to_su2(U)
        
        # Check unitarity
        self.assertTrue(np.allclose(
            np.dot(projected.conj().T, projected), 
            np.eye(2), 
            atol=self.tolerance
        ))
        
        # Check determinant
        self.assertTrue(abs(np.linalg.det(projected) - 1.0) < self.tolerance)

    def test_validate_su2_valid_matrix(self):
        """Test validation of valid SU(2) matrix."""
        # Create valid SU(2) matrix
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        self.assertTrue(SU2Projection.validate_su2(U))

    def test_validate_su2_invalid_matrix(self):
        """Test validation of invalid matrix."""
        # Create invalid matrix
        U = np.array([[2, 0], [0, 1]], dtype=complex)
        self.assertFalse(SU2Projection.validate_su2(U))


class TestGradientDescent(unittest.TestCase):
    """Test gradient descent optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(
            method=RelaxationMethod.GRADIENT_DESCENT,
            step_size=0.01,
            momentum=0.9
        )
        self.optimizer = GradientDescent(self.config)

    def test_step_basic(self):
        """Test basic gradient descent step."""
        # Use non-identity matrix
        U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
        gradient = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=complex)
        
        U_new = self.optimizer.step(U, gradient)
        
        # Check that field is updated
        self.assertFalse(np.allclose(U, U_new))
        
        # Check SU(2) property
        self.assertTrue(SU2Projection.validate_su2(U_new))

    def test_step_with_momentum(self):
        """Test gradient descent with momentum."""
        U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
        gradient = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=complex)
        
        # First step
        U1 = self.optimizer.step(U, gradient)
        
        # Second step with same gradient
        U2 = self.optimizer.step(U1, gradient)
        
        # Should be different due to momentum
        self.assertFalse(np.allclose(U1, U2))

    def test_reset(self):
        """Test optimizer reset."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)
        
        # Take a step
        self.optimizer.step(U, gradient)
        
        # Reset
        self.optimizer.reset()
        
        # Velocity should be None
        self.assertIsNone(self.optimizer.velocity)


class TestLBFGSOptimizer(unittest.TestCase):
    """Test L-BFGS optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(
            method=RelaxationMethod.LBFGS,
            step_size=0.01
        )
        self.optimizer = LBFGSOptimizer(self.config)

    def test_step_basic(self):
        """Test basic L-BFGS step."""
        U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
        gradient = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=complex)
        
        U_new = self.optimizer.step(U, gradient)
        
        # Check that field is updated
        self.assertFalse(np.allclose(U, U_new))
        
        # Check SU(2) property
        self.assertTrue(SU2Projection.validate_su2(U_new))

    def test_reset(self):
        """Test optimizer reset."""
        self.optimizer.reset()
        
        # History should be empty
        self.assertEqual(len(self.optimizer.s_history), 0)
        self.assertEqual(len(self.optimizer.y_history), 0)
        self.assertEqual(len(self.optimizer.rho_history), 0)


class TestAdamOptimizer(unittest.TestCase):
    """Test Adam optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = RelaxationConfig(
            method=RelaxationMethod.ADAM,
            step_size=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        self.optimizer = AdamOptimizer(self.config)

    def test_step_basic(self):
        """Test basic Adam step."""
        U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
        gradient = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=complex)
        
        U_new = self.optimizer.step(U, gradient)
        
        # Check that field is updated
        self.assertFalse(np.allclose(U, U_new))
        
        # Check SU(2) property
        self.assertTrue(SU2Projection.validate_su2(U_new))

    def test_step_adaptive(self):
        """Test adaptive learning rate."""
        U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
        gradient = np.array([[0.1, 0.05], [0.05, 0.1]], dtype=complex)
        
        # First step
        U1 = self.optimizer.step(U, gradient)
        
        # Second step with same gradient
        U2 = self.optimizer.step(U1, gradient)
        
        # Should be different due to adaptive learning
        self.assertFalse(np.allclose(U1, U2))

    def test_reset(self):
        """Test optimizer reset."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        gradient = np.array([[0.1, 0], [0, 0.1]], dtype=complex)
        
        # Take a step
        self.optimizer.step(U, gradient)
        
        # Reset
        self.optimizer.reset()
        
        # State should be reset
        self.assertIsNone(self.optimizer.m)
        self.assertIsNone(self.optimizer.v)
        self.assertEqual(self.optimizer.t, 0)


class TestConstraintController(unittest.TestCase):
    """Test constraint controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ConstraintConfig(
            lambda_B=1000.0,
            lambda_Q=1000.0,
            lambda_virial=1000.0,
            tolerance_B=0.02,
            tolerance_Q=1e-6,
            tolerance_virial=0.01
        )
        self.controller = ConstraintController(self.config)

    def test_compute_constraint_penalty_satisfied(self):
        """Test penalty computation when constraints are satisfied."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        penalty = self.controller.compute_constraint_penalty(
            U, baryon_number=1.0, electric_charge=1.0, energy_balance=0.5
        )
        
        # Should be zero when constraints are satisfied
        self.assertEqual(penalty, 0.0)

    def test_compute_constraint_penalty_violated(self):
        """Test penalty computation when constraints are violated."""
        U = np.array([[1, 0], [0, 1]], dtype=complex)
        penalty = self.controller.compute_constraint_penalty(
            U, baryon_number=1.1, electric_charge=1.1, energy_balance=0.6
        )
        
        # Should be positive when constraints are violated
        self.assertGreater(penalty, 0.0)

    def test_check_constraints_satisfied(self):
        """Test constraint checking when satisfied."""
        result = self.controller.check_constraints(
            baryon_number=1.0, electric_charge=1.0, energy_balance=0.5
        )
        
        # All constraints should be satisfied
        self.assertTrue(result['baryon_number'])
        self.assertTrue(result['electric_charge'])
        self.assertTrue(result['energy_balance'])

    def test_check_constraints_violated(self):
        """Test constraint checking when violated."""
        result = self.controller.check_constraints(
            baryon_number=1.1, electric_charge=1.1, energy_balance=0.6
        )
        
        # All constraints should be violated
        self.assertFalse(result['baryon_number'])
        self.assertFalse(result['electric_charge'])
        self.assertFalse(result['energy_balance'])


class TestRelaxationSolver(unittest.TestCase):
    """Test relaxation solver."""

    def setUp(self):
        """Set up test fixtures."""
        self.relaxation_config = RelaxationConfig(
            method=RelaxationMethod.GRADIENT_DESCENT,
            max_iterations=10,
            convergence_tol=1e-6
        )
        self.constraint_config = ConstraintConfig()
        self.solver = RelaxationSolver(self.relaxation_config, self.constraint_config)

    def test_init_gradient_descent(self):
        """Test initialization with gradient descent."""
        self.assertIsInstance(self.solver.optimizer, GradientDescent)

    def test_init_lbfgs(self):
        """Test initialization with L-BFGS."""
        config = RelaxationConfig(method=RelaxationMethod.LBFGS)
        solver = RelaxationSolver(config, self.constraint_config)
        self.assertIsInstance(solver.optimizer, LBFGSOptimizer)

    def test_init_adam(self):
        """Test initialization with Adam."""
        config = RelaxationConfig(method=RelaxationMethod.ADAM)
        solver = RelaxationSolver(config, self.constraint_config)
        self.assertIsInstance(solver.optimizer, AdamOptimizer)

    def test_init_invalid_method(self):
        """Test initialization with invalid method."""
        config = RelaxationConfig(method="invalid")
        with self.assertRaises(ValueError):
            RelaxationSolver(config, self.constraint_config)

    @patch('builtins.print')
    def test_solve_convergence(self, mock_print):
        """Test solver convergence."""
        # Mock functions
        U_init = np.array([[1, 0], [0, 1]], dtype=complex)
        
        def energy_function(U):
            return 1.0
        
        def gradient_function(U):
            return np.array([[0.01, 0], [0, 0.01]], dtype=complex)
        
        def baryon_function(U):
            return 1.0
        
        def charge_function(U):
            return 1.0
        
        def energy_balance_function(U):
            return 0.5
        
        constraint_functions = {
            'baryon_number': baryon_function,
            'electric_charge': charge_function,
            'energy_balance': energy_balance_function
        }
        
        result = self.solver.solve(
            U_init, energy_function, gradient_function, constraint_functions
        )
        
        # Check result structure
        self.assertIn('solution', result)
        self.assertIn('energy_history', result)
        self.assertIn('constraint_history', result)
        self.assertIn('iterations', result)
        self.assertIn('converged', result)
        self.assertIn('execution_time', result)
        self.assertIn('final_energy', result)
        self.assertIn('final_constraints', result)

    def test_reset(self):
        """Test solver reset."""
        self.solver.reset()
        # Should not raise any exceptions


class TestNumericalMethods(unittest.TestCase):
    """Test numerical methods class."""

    def setUp(self):
        """Set up test fixtures."""
        self.numerical = NumericalMethods(grid_size=16, box_size=2.0)

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.numerical.grid_size, 16)
        self.assertEqual(self.numerical.box_size, 2.0)
        self.assertEqual(self.numerical.dx, 2.0 / 16)

    def test_compute_gradient(self):
        """Test gradient computation."""
        # Create test field
        field = self.numerical.X**2 + self.numerical.Y**2 + self.numerical.Z**2
        
        grad_x, grad_y, grad_z = self.numerical.compute_gradient(field)
        
        # Check shapes
        self.assertEqual(grad_x.shape, field.shape)
        self.assertEqual(grad_y.shape, field.shape)
        self.assertEqual(grad_z.shape, field.shape)

    def test_compute_divergence(self):
        """Test divergence computation."""
        # Create test vector field
        field_x = self.numerical.X
        field_y = self.numerical.Y
        field_z = self.numerical.Z
        
        divergence = self.numerical.compute_divergence(field_x, field_y, field_z)
        
        # Check shape
        self.assertEqual(divergence.shape, field_x.shape)

    def test_integrate_3d(self):
        """Test 3D integration."""
        # Create test field
        field = np.ones((16, 16, 16))
        
        result = self.numerical.integrate_3d(field)
        
        # Should be volume
        expected = 16**3 * (2.0/16)**3
        self.assertAlmostEqual(result, expected, places=10)

    def test_create_initial_field_tanh(self):
        """Test initial field creation with tanh profile."""
        U = self.numerical.create_initial_field("tanh")
        
        # Check shape
        self.assertEqual(U.shape, (16, 16, 16, 2, 2))
        
        # Check SU(2) property for a few points
        for i in [0, 8, 15]:
            for j in [0, 8, 15]:
                for k in [0, 8, 15]:
                    self.assertTrue(SU2Projection.validate_su2(U[i, j, k]))

    def test_create_initial_field_exp(self):
        """Test initial field creation with exp profile."""
        U = self.numerical.create_initial_field("exp")
        
        # Check shape
        self.assertEqual(U.shape, (16, 16, 16, 2, 2))
        
        # Check SU(2) property for a few points
        for i in [0, 8, 15]:
            for j in [0, 8, 15]:
                for k in [0, 8, 15]:
                    self.assertTrue(SU2Projection.validate_su2(U[i, j, k]))

    def test_create_initial_field_gaussian(self):
        """Test initial field creation with gaussian profile."""
        U = self.numerical.create_initial_field("gaussian")
        
        # Check shape
        self.assertEqual(U.shape, (16, 16, 16, 2, 2))
        
        # Check SU(2) property for a few points
        for i in [0, 8, 15]:
            for j in [0, 8, 15]:
                for k in [0, 8, 15]:
                    self.assertTrue(SU2Projection.validate_su2(U[i, j, k]))

    def test_create_initial_field_invalid(self):
        """Test initial field creation with invalid profile."""
        with self.assertRaises(ValueError):
            self.numerical.create_initial_field("invalid")

    def test_validate_solution(self):
        """Test solution validation."""
        # Create valid SU(2) field
        U = self.numerical.create_initial_field("tanh")
        
        results = self.numerical.validate_solution(U)
        
        # Check result structure
        self.assertIn('su2_valid', results)
        self.assertIn('boundary_conditions', results)
        
        # Should be valid
        self.assertTrue(results['su2_valid'])


if __name__ == '__main__':
    unittest.main()
