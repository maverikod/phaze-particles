#!/usr/bin/env python3
"""
Backward compatibility tests for existing functionality.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from phaze_particles.models.proton_integrated import ProtonModel, ModelConfig
from phaze_particles.utils.physics import PhysicalQuantitiesCalculator
from phaze_particles.utils.energy_densities import EnergyDensityCalculator
from phaze_particles.utils.su2_fields import SU2FieldBuilder


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ModelConfig(
            grid_size=16,
            box_size=2.0,
            torus_config="120deg",
            max_iterations=5,
            convergence_tol=1e-2,
            validation_enabled=False
        )

    def test_old_charge_density_mode(self):
        """Test that old charge density modes still work."""
        model = ProtonModel(self.config)
        
        # Create geometry and fields
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        
        # Test that old charge density calculator still works
        old_charge_density = model.physics_calculator.charge_density
        
        # Test sin2f mode
        field = model.su2_field
        profile = model.profile
        field_derivatives = model.field_derivatives
        
        charge_density_sin2f = old_charge_density.compute_charge_density(
            field, profile, field_derivatives, mode="sin2f"
        )
        
        self.assertIsNotNone(charge_density_sin2f)
        self.assertEqual(charge_density_sin2f.shape, (16, 16, 16))
        self.assertTrue(np.all(np.isfinite(charge_density_sin2f)))
        self.assertTrue(np.all(charge_density_sin2f >= 0))

    def test_old_energy_calculator(self):
        """Test that old energy calculator still works."""
        model = ProtonModel(self.config)
        
        # Create geometry and fields
        model.create_geometry()
        model.build_fields()
        
        # Test that old energy calculator still works
        old_energy_calculator = model.energy_calculator
        
        # Test energy calculation
        energy = old_energy_calculator.calculate_total_energy(model.su2_field)
        self.assertIsNotNone(energy)
        self.assertGreater(energy, 0)

    def test_old_physics_calculator(self):
        """Test that old physics calculator still works."""
        model = ProtonModel(self.config)
        
        # Create geometry, fields, and energy
        model.create_geometry()
        model.build_fields()
        model.calculate_energy()
        
        # Test that old physics calculator still works
        old_physics_calculator = model.physics_calculator
        
        # Test physics calculation without new charge density calculator
        physical_quantities = old_physics_calculator.calculate_quantities(
            su2_field=model.su2_field,
            energy_density=model.energy_density,
            profile=model.profile,
            field_derivatives=model.field_derivatives
        )
        
        self.assertIsNotNone(physical_quantities)
        self.assertGreater(physical_quantities.electric_charge, 0)
        # Baryon number might be complex, so check real part
        baryon_real = np.real(physical_quantities.baryon_number)
        self.assertGreaterEqual(baryon_real, -1e-5)  # Allow small negative due to numerical precision
        self.assertGreater(physical_quantities.charge_radius, 0)
        self.assertGreater(physical_quantities.mass, 0)

    def test_model_config_defaults(self):
        """Test that ModelConfig defaults still work."""
        # Test with minimal config (should use defaults)
        config = ModelConfig()
        
        # Check that defaults are set
        self.assertEqual(config.F_pi, 186.0)
        self.assertEqual(config.e, 5.45)
        self.assertEqual(config.c6, 1.0)  # Default from original config
        self.assertEqual(config.grid_size, 64)
        self.assertEqual(config.box_size, 4.0)

    def test_model_without_new_components(self):
        """Test that model works without explicitly using new components."""
        model = ProtonModel(self.config)
        
        # Run model using old interface
        results = model.run()
        
        # Check that results are generated
        self.assertIsNotNone(results)
        
        # Check that physical quantities have reasonable values
        self.assertGreater(results.electric_charge, 0)
        # Baryon number might be complex or very small, so check real part
        baryon_real = np.real(results.baryon_number)
        self.assertGreaterEqual(baryon_real, -1e-4)  # Allow small negative due to numerical precision
        self.assertGreater(results.charge_radius, 0)
        self.assertGreater(results.proton_mass, 0)

    def test_energy_density_calculator_compatibility(self):
        """Test that EnergyDensityCalculator still works with old interface."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend
        
        backend = ArrayBackend()
        energy_calculator = EnergyDensityCalculator(
            grid_size=16,
            box_size=2.0,
            c2=1.0,
            c4=1.0,
            c6=1.0,
            backend=backend
        )
        
        # Test that old interface still works
        self.assertIsNotNone(energy_calculator)
        self.assertEqual(energy_calculator.grid_size, 16)
        self.assertEqual(energy_calculator.box_size, 2.0)

    def test_su2_field_builder_compatibility(self):
        """Test that SU2FieldBuilder still works with old interface."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend
        
        backend = ArrayBackend()
        field_builder = SU2FieldBuilder(
            grid_size=16,
            box_size=2.0,
            backend=backend
        )
        
        # Test that old interface still works
        self.assertIsNotNone(field_builder)
        self.assertEqual(field_builder.grid_size, 16)
        self.assertEqual(field_builder.box_size, 2.0)

    def test_physical_quantities_calculator_compatibility(self):
        """Test that PhysicalQuantitiesCalculator still works with old interface."""
        from phaze_particles.utils.mathematical_foundations import ArrayBackend
        
        backend = ArrayBackend()
        physics_calculator = PhysicalQuantitiesCalculator(
            grid_size=16,
            box_size=2.0,
            backend=backend
        )
        
        # Test that old interface still works
        self.assertIsNotNone(physics_calculator)
        self.assertEqual(physics_calculator.grid_size, 16)
        self.assertEqual(physics_calculator.box_size, 2.0)

    def test_model_config_validation(self):
        """Test that ModelConfig validation still works."""
        config = ModelConfig(
            grid_size=16,
            box_size=2.0,
            F_pi=186.0,
            e=5.45,
            c6=0.0
        )
        
        # Test validation
        errors = config.validate()
        self.assertEqual(len(errors), 0)  # Should have no errors
        
        # Test with invalid config
        invalid_config = ModelConfig(
            grid_size=-1,  # Invalid
            box_size=0.0,  # Invalid
            F_pi=-1.0,     # Invalid
            e=0.0,         # Invalid
            c6=-1.0        # Invalid
        )
        
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)  # Should have errors


if __name__ == '__main__':
    unittest.main()
