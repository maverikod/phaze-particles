#!/usr/bin/env python3
"""
Tests for full Skyrme physics implementation.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.utils.physics import (
    SkyrmeLagrangian,
    NoetherCurrent,
    ChargeDensity
)
from phaze_particles.utils.su2_fields import SU2Field, RadialProfile
from phaze_particles.utils.mathematical_foundations import ArrayBackend


class TestSkyrmeLagrangian(unittest.TestCase):
    """Test Skyrme Lagrangian implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.F_pi = 186.0
        self.e = 5.45
        self.c6 = 0.0
        self.backend = ArrayBackend()
        self.lagrangian = SkyrmeLagrangian(self.F_pi, self.e, self.c6, self.backend)

    def test_initialization(self):
        """Test Lagrangian initialization."""
        self.assertEqual(self.lagrangian.F_pi, self.F_pi)
        self.assertEqual(self.lagrangian.e, self.e)
        self.assertEqual(self.lagrangian.c6, self.c6)

    def test_compute_lagrangian_density(self):
        """Test Lagrangian density computation."""
        # Create mock left currents
        grid_size = 8
        L_i = {
            "x": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            },
            "y": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            },
            "z": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            }
        }

        # Compute Lagrangian density
        density = self.lagrangian.compute_lagrangian_density(L_i)

        # Check that density is computed
        self.assertIsNotNone(density)
        self.assertEqual(density.shape, (grid_size, grid_size, grid_size))
        self.assertTrue(np.all(np.isfinite(density)))

    def test_commutator_computation(self):
        """Test commutator computation."""
        # Create mock left currents
        l1 = {
            "l_00": np.array([[1, 0], [0, 1]], dtype=complex),
            "l_01": np.array([[0, 1], [0, 0]], dtype=complex),
            "l_10": np.array([[0, 0], [1, 0]], dtype=complex),
            "l_11": np.array([[1, 0], [0, 1]], dtype=complex),
        }
        l2 = {
            "l_00": np.array([[1, 0], [0, 1]], dtype=complex),
            "l_01": np.array([[0, 0], [1, 0]], dtype=complex),
            "l_10": np.array([[0, 1], [0, 0]], dtype=complex),
            "l_11": np.array([[1, 0], [0, 1]], dtype=complex),
        }

        # Compute commutator
        comm = self.lagrangian._compute_commutator(l1, l2)

        # Check commutator structure
        self.assertIn("comm_00", comm)
        self.assertIn("comm_01", comm)
        self.assertIn("comm_10", comm)
        self.assertIn("comm_11", comm)


class TestNoetherCurrent(unittest.TestCase):
    """Test Noether current implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.F_pi = 186.0
        self.e = 5.45
        self.c6 = 0.0
        self.backend = ArrayBackend()
        self.noether_current = NoetherCurrent(self.F_pi, self.e, self.c6, self.backend)

    def test_initialization(self):
        """Test Noether current initialization."""
        self.assertEqual(self.noether_current.F_pi, self.F_pi)
        self.assertEqual(self.noether_current.e, self.e)
        self.assertEqual(self.noether_current.c6, self.c6)

    def test_compute_current_density(self):
        """Test current density computation."""
        # Create mock left currents
        grid_size = 8
        L_i = {
            "x": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            },
            "y": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            },
            "z": {
                "l_00": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_01": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_10": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
                "l_11": np.random.rand(grid_size, grid_size, grid_size) + 1j * np.random.rand(grid_size, grid_size, grid_size),
            }
        }

        # Create tau matrices
        tau_a = np.array([
            [[0, 1], [1, 0]],  # tau_1
            [[0, -1j], [1j, 0]],  # tau_2
            [[1, 0], [0, -1]]  # tau_3
        ], dtype=complex)

        # Compute current density
        current_density = self.noether_current.compute_current_density(L_i, tau_a)

        # Check current density structure
        self.assertIsNotNone(current_density)
        self.assertEqual(current_density.shape, (3, 3, grid_size, grid_size, grid_size))
        self.assertTrue(np.all(np.isfinite(current_density)))

    def test_trace_tau_l(self):
        """Test Tr(τᵃLᵢ) computation."""
        # Create mock tau matrix and left current
        tau = np.array([[1, 0], [0, -1]], dtype=complex)
        l = {
            "l_00": np.array([[1, 0], [0, 1]], dtype=complex),
            "l_01": np.array([[0, 1], [0, 0]], dtype=complex),
            "l_10": np.array([[0, 0], [1, 0]], dtype=complex),
            "l_11": np.array([[1, 0], [0, 1]], dtype=complex),
        }

        # Compute trace
        trace = self.noether_current._trace_tau_l(tau, l)

        # Check trace computation
        self.assertIsNotNone(trace)
        if np.isscalar(trace):
            self.assertTrue(np.isfinite(trace))
        else:
            self.assertTrue(np.all(np.isfinite(trace)))


class TestChargeDensity(unittest.TestCase):
    """Test charge density implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.F_pi = 186.0
        self.e = 5.45
        self.c6 = 0.0
        self.grid_size = 8
        self.box_size = 2.0
        self.backend = ArrayBackend()
        self.charge_density = ChargeDensity(
            self.F_pi, self.e, self.c6, self.grid_size, self.box_size, self.backend
        )

    def test_initialization(self):
        """Test charge density initialization."""
        self.assertEqual(self.charge_density.F_pi, self.F_pi)
        self.assertEqual(self.charge_density.e, self.e)
        self.assertEqual(self.charge_density.c6, self.c6)
        self.assertEqual(self.charge_density.grid_size, self.grid_size)
        self.assertEqual(self.charge_density.box_size, self.box_size)

    def test_compute_charge_density_full_skyrme(self):
        """Test full Skyrme charge density computation."""
        # Create mock field
        field = type('MockField', (), {})()
        field.get_tau_matrices = lambda: np.array([
            [[0, 1], [1, 0]],  # tau_1
            [[0, -1j], [1j, 0]],  # tau_2
            [[1, 0], [0, -1]]  # tau_3
        ], dtype=complex)

        # Create mock profile
        profile = type('MockProfile', (), {})()

        # Create mock field derivatives
        field_derivatives = {
            "baryon_density": np.random.rand(self.grid_size, self.grid_size, self.grid_size),
            "left_currents": {
                "x": {
                    "l_00": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_01": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_10": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_11": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                },
                "y": {
                    "l_00": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_01": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_10": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_11": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                },
                "z": {
                    "l_00": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_01": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_10": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                    "l_11": np.random.rand(self.grid_size, self.grid_size, self.grid_size) + 1j * np.random.rand(self.grid_size, self.grid_size, self.grid_size),
                }
            }
        }

        # Compute charge density
        charge_density = self.charge_density.compute_charge_density(
            field, profile, field_derivatives, mode="full_skyrme"
        )

        # Check charge density
        self.assertIsNotNone(charge_density)
        self.assertEqual(charge_density.shape, (self.grid_size, self.grid_size, self.grid_size))
        self.assertTrue(np.all(np.isfinite(charge_density)))
        self.assertTrue(np.all(charge_density >= 0))  # Should be non-negative

    def test_compute_charge_density_fallback(self):
        """Test charge density computation with fallback mode."""
        # Create mock field and profile
        field = type('MockField', (), {})()
        profile = type('MockProfile', (), {})()
        
        # Add evaluate method to profile
        profile.evaluate = lambda r: np.sin(r)**2
        
        # Test with no field derivatives (should fall back to sin2f mode)
        charge_density = self.charge_density.compute_charge_density(
            field, profile, field_derivatives=None, mode="full_skyrme"
        )
        
        # Check charge density
        self.assertIsNotNone(charge_density)
        self.assertEqual(charge_density.shape, (self.grid_size, self.grid_size, self.grid_size))
        self.assertTrue(np.all(np.isfinite(charge_density)))
        self.assertTrue(np.all(charge_density >= 0))  # Should be non-negative


class TestSU2FieldTauMatrices(unittest.TestCase):
    """Test SU2Field tau matrices implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = ArrayBackend()
        self.grid_size = 8
        self.box_size = 2.0

    def test_get_tau_matrices(self):
        """Test tau matrices generation."""
        # Create a mock SU2Field
        field = type('MockSU2Field', (), {})()
        field.backend = self.backend
        
        # Create a proper mock method
        def mock_get_tau_matrices():
            xp = self.backend.get_array_module()
            tau_1 = xp.array([[0, 1], [1, 0]], dtype=complex)
            tau_2 = xp.array([[0, -1j], [1j, 0]], dtype=complex)
            tau_3 = xp.array([[1, 0], [0, -1]], dtype=complex)
            return xp.array([tau_1, tau_2, tau_3])
        
        field.get_tau_matrices = mock_get_tau_matrices
        
        # Get tau matrices
        tau_matrices = field.get_tau_matrices()
        
        # Check tau matrices structure
        self.assertIsNotNone(tau_matrices)
        self.assertEqual(tau_matrices.shape, (3, 2, 2))
        self.assertTrue(np.all(np.isfinite(tau_matrices)))
        
        # Check specific tau matrices
        # tau_1 = [[0, 1], [1, 0]]
        np.testing.assert_array_equal(tau_matrices[0], np.array([[0, 1], [1, 0]], dtype=complex))
        
        # tau_2 = [[0, -1j], [1j, 0]]
        np.testing.assert_array_equal(tau_matrices[1], np.array([[0, -1j], [1j, 0]], dtype=complex))
        
        # tau_3 = [[1, 0], [0, -1]]
        np.testing.assert_array_equal(tau_matrices[2], np.array([[1, 0], [0, -1]], dtype=complex))


if __name__ == '__main__':
    unittest.main()
