#!/usr/bin/env python3
"""
Integration tests for ProtonModel with full Skyrme physics.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
from phaze_particles.utils.physics import (
    SkyrmeLagrangian,
    NoetherCurrent,
    ChargeDensity
)


class TestProtonModelIntegration(unittest.TestCase):
    """Test ProtonModel integration with full Skyrme physics."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            grid_size=16,
            box_size=2.0,
            torus_config="120deg",
            F_pi=186.0,
            e=5.45,
            c6=0.0,
            max_iterations=10,  # Small number for testing
            convergence_tol=1e-3,
            validation_enabled=False  # Disable validation for faster testing
        )

    def test_model_initialization(self):
        """Test ProtonModel initialization with new physics components."""
        model = ProtonModel(self.config)
        
        # Check that all components are initialized
        self.assertIsNotNone(model.skyrme_lagrangian)
        self.assertIsNotNone(model.noether_current)
        self.assertIsNotNone(model.charge_density)
        
        # Check that components have correct parameters
        self.assertEqual(model.skyrme_lagrangian.F_pi, self.config.F_pi)
        self.assertEqual(model.skyrme_lagrangian.e, self.config.e)
        self.assertEqual(model.skyrme_lagrangian.c6, self.config.c6)
        
        self.assertEqual(model.noether_current.F_pi, self.config.F_pi)
        self.assertEqual(model.noether_current.e, self.config.e)
        self.assertEqual(model.noether_current.c6, self.config.c6)
        
        self.assertEqual(model.charge_density.F_pi, self.config.F_pi)
        self.assertEqual(model.charge_density.e, self.config.e)
        self.assertEqual(model.charge_density.c6, self.config.c6)

    def test_skyrme_lagrangian_component(self):
        """Test SkyrmeLagrangian component initialization."""
        model = ProtonModel(self.config)
        
        # Check that SkyrmeLagrangian is properly initialized
        self.assertIsInstance(model.skyrme_lagrangian, SkyrmeLagrangian)
        self.assertEqual(model.skyrme_lagrangian.F_pi, 186.0)
        self.assertEqual(model.skyrme_lagrangian.e, 5.45)
        self.assertEqual(model.skyrme_lagrangian.c6, 0.0)

    def test_noether_current_component(self):
        """Test NoetherCurrent component initialization."""
        model = ProtonModel(self.config)
        
        # Check that NoetherCurrent is properly initialized
        self.assertIsInstance(model.noether_current, NoetherCurrent)
        self.assertEqual(model.noether_current.F_pi, 186.0)
        self.assertEqual(model.noether_current.e, 5.45)
        self.assertEqual(model.noether_current.c6, 0.0)

    def test_charge_density_component(self):
        """Test ChargeDensity component initialization."""
        model = ProtonModel(self.config)
        
        # Check that ChargeDensity is properly initialized
        self.assertIsInstance(model.charge_density, ChargeDensity)
        self.assertEqual(model.charge_density.F_pi, 186.0)
        self.assertEqual(model.charge_density.e, 5.45)
        self.assertEqual(model.charge_density.c6, 0.0)
        self.assertEqual(model.charge_density.grid_size, 16)
        self.assertEqual(model.charge_density.box_size, 2.0)

    def test_model_config_with_physical_constants(self):
        """Test ModelConfig with physical constants."""
        # Test that ModelConfig accepts new physical constants
        config = ModelConfig(
            grid_size=32,
            box_size=4.0,
            F_pi=200.0,  # Different value
            e=6.0,       # Different value
            c6=1.0       # Non-zero value
        )
        
        self.assertEqual(config.F_pi, 200.0)
        self.assertEqual(config.e, 6.0)
        self.assertEqual(config.c6, 1.0)
        
        # Test that model can be initialized with these values
        model = ProtonModel(config)
        self.assertEqual(model.skyrme_lagrangian.F_pi, 200.0)
        self.assertEqual(model.skyrme_lagrangian.e, 6.0)
        self.assertEqual(model.skyrme_lagrangian.c6, 1.0)

    def test_model_creation_geometry(self):
        """Test model geometry creation."""
        model = ProtonModel(self.config)
        
        # Test geometry creation
        success = model.create_geometry()
        self.assertTrue(success)
        self.assertEqual(model.status.value, "geometry_created")

    def test_model_build_fields(self):
        """Test model field building."""
        model = ProtonModel(self.config)
        
        # Create geometry first
        model.create_geometry()
        
        # Test field building
        success = model.build_fields()
        self.assertTrue(success)
        self.assertEqual(model.status.value, "fields_built")
        
        # Check that SU2 field is created
        self.assertIsNotNone(model.su2_field)

    def test_model_calculate_energy(self):
        """Test model energy calculation."""
        model = ProtonModel(self.config)
        
        # Create geometry and fields
        model.create_geometry()
        model.build_fields()
        
        # Test energy calculation
        success = model.calculate_energy()
        self.assertTrue(success)
        self.assertEqual(model.status.value, "energy_calculated")
        
        # Check that energy is calculated
        self.assertIsNotNone(model.energy_density)
        self.assertIsNotNone(model.field_derivatives)

    def test_model_calculate_physics(self):
        """Test model physics calculation with new charge density calculator."""
        model = ProtonModel(self.config)
        
        # Create geometry, fields, and energy
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        
        # Test physics calculation
        success = model.calculate_physics()
        self.assertTrue(success)
        self.assertEqual(model.status.value, "physics_calculated")
        
        # Check that physical quantities are calculated
        self.assertIsNotNone(model.physical_quantities)

    def test_model_run_complete(self):
        """Test complete model run."""
        model = ProtonModel(self.config)
        
        # Run complete model
        results = model.run()
        
        # Check that results are generated
        self.assertIsNotNone(results)
        
        # Check that physical quantities have reasonable values
        self.assertGreater(results.electric_charge, 0)
        # Baryon number might be complex, so check real part
        baryon_real = np.real(results.baryon_number)
        self.assertGreaterEqual(baryon_real, -1e-4)  # Allow small negative due to numerical precision
        self.assertGreater(results.charge_radius, 0)
        self.assertGreater(results.proton_mass, 0)

    def test_model_with_different_constants(self):
        """Test model with different physical constants."""
        # Test with different constants
        config = ModelConfig(
            grid_size=16,
            box_size=2.0,
            torus_config="120deg",
            F_pi=200.0,  # Different F_pi
            e=6.0,       # Different e
            c6=0.5,      # Non-zero c6
            max_iterations=5,
            convergence_tol=1e-2,
            validation_enabled=False
        )
        
        model = ProtonModel(config)
        
        # Check that components use the new constants
        self.assertEqual(model.skyrme_lagrangian.F_pi, 200.0)
        self.assertEqual(model.skyrme_lagrangian.e, 6.0)
        self.assertEqual(model.skyrme_lagrangian.c6, 0.5)
        
        # Run model
        results = model.run()
        self.assertIsNotNone(results)


if __name__ == '__main__':
    unittest.main()
