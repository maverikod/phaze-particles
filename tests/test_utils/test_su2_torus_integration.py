#!/usr/bin/env python3
"""
Integration tests for SU(2) fields with torus geometries.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
import math
from typing import Dict, Any

from phaze_particles.utils.su2_fields import SU2Fields, RadialProfile
from phaze_particles.utils.torus_geometries import (
    Torus120Degrees,
    TorusClover,
    TorusCartesian,
    TorusGeometryManager,
    TorusConfiguration,
)


class TestSU2TorusIntegration(unittest.TestCase):
    """Integration tests for SU(2) fields with torus geometries."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.grid_size = 16
        self.box_size = 4.0
        self.su2_fields = SU2Fields(self.grid_size, self.box_size, use_cuda=False)

    def test_120deg_configuration_integration(self) -> None:
        """Test integration with 120° torus configuration."""
        # Create 120° torus configuration
        torus_120 = Torus120Degrees(radius=1.0, backend=self.su2_fields.backend)
        
        # Create SU(2) field from torus configuration
        field = self.su2_fields.create_field_from_torus(
            torus_120, "skyrmion", 1.0
        )
        
        # Validate field
        validation_results = self.su2_fields.validate_field(field)
        
        # Check results
        test_field = self.su2_fields.builder.build_field(
            (np.zeros((self.grid_size, self.grid_size, self.grid_size)),
             np.zeros((self.grid_size, self.grid_size, self.grid_size)),
             np.ones((self.grid_size, self.grid_size, self.grid_size))),
            RadialProfile("skyrmion", 1.0, backend=self.su2_fields.backend)
        )
        self.assertIsInstance(field, type(test_field))
        
        # Field should be valid
        self.assertTrue(validation_results["unitary"])
        self.assertTrue(validation_results["determinant"])
        self.assertTrue(validation_results["continuity"])

    def test_clover_configuration_integration(self) -> None:
        """Test integration with clover torus configuration."""
        # Create clover torus configuration
        torus_clover = TorusClover(radius=1.0, backend=self.su2_fields.backend)
        
        # Create SU(2) field from torus configuration
        field = self.su2_fields.create_field_from_torus(
            torus_clover, "exponential", 1.5
        )
        
        # Validate field
        validation_results = self.su2_fields.validate_field(field)
        
        # Field should be valid
        self.assertTrue(validation_results["unitary"])
        self.assertTrue(validation_results["determinant"])
        self.assertTrue(validation_results["continuity"])

    def test_cartesian_configuration_integration(self) -> None:
        """Test integration with cartesian torus configuration."""
        # Create cartesian torus configuration
        torus_cartesian = TorusCartesian(radius=1.0, backend=self.su2_fields.backend)
        
        # Create SU(2) field from torus configuration
        field = self.su2_fields.create_field_from_torus(
            torus_cartesian, "polynomial", 2.0
        )
        
        # Validate field
        validation_results = self.su2_fields.validate_field(field)
        
        # Field should be valid
        self.assertTrue(validation_results["unitary"])
        self.assertTrue(validation_results["determinant"])
        self.assertTrue(validation_results["continuity"])

    def test_torus_geometry_manager_integration(self) -> None:
        """Test integration with torus geometry manager."""
        # Create torus geometry manager
        manager = TorusGeometryManager(backend=self.su2_fields.backend)
        
        # Test all configurations
        configs = [
            TorusConfiguration.CONFIG_120_DEG,
            TorusConfiguration.CONFIG_CLOVER,
            TorusConfiguration.CONFIG_CARTESIAN
        ]
        
        for config_type in configs:
            with self.subTest(config=config_type.value):
                # Get torus configuration
                torus_config = manager.create_configuration(config_type, radius=1.0)
                
                # Create SU(2) field
                field = self.su2_fields.create_field_from_torus(
                    torus_config, "skyrmion", 1.0
                )
                
                # Validate field
                validation_results = self.su2_fields.validate_field(field)
                
                # Field should be valid
                self.assertTrue(validation_results["unitary"])
                self.assertTrue(validation_results["determinant"])
                self.assertTrue(validation_results["continuity"])

    def test_different_profile_types_integration(self) -> None:
        """Test different profile types with torus configurations."""
        # Create torus configuration
        torus_120 = Torus120Degrees(radius=1.0, backend=self.su2_fields.backend)
        
        # Test different profile types
        profile_types = ["skyrmion", "exponential", "polynomial"]
        
        for profile_type in profile_types:
            with self.subTest(profile=profile_type):
                # Create SU(2) field
                field = self.su2_fields.create_field_from_torus(
                    torus_120, profile_type, 1.0
                )
                
                # Validate field
                validation_results = self.su2_fields.validate_field(field)
                
                # Field should be valid
                self.assertTrue(validation_results["unitary"])
                self.assertTrue(validation_results["determinant"])
                self.assertTrue(validation_results["continuity"])

    def test_field_derivatives_computation(self) -> None:
        """Test field derivatives computation with torus configurations."""
        # Create torus configuration
        torus_120 = Torus120Degrees(radius=1.0, backend=self.su2_fields.backend)
        
        # Create SU(2) field
        field = self.su2_fields.create_field_from_torus(
            torus_120, "skyrmion", 1.0
        )
        
        # Compute derivatives
        derivatives = self.su2_fields.compute_field_derivatives(field)
        
        # Check structure
        self.assertIn("left_currents", derivatives)
        self.assertIn("commutators", derivatives)
        self.assertIn("traces", derivatives)
        
        # Check left currents
        left_currents = derivatives["left_currents"]
        self.assertIn("x", left_currents)
        self.assertIn("y", left_currents)
        self.assertIn("z", left_currents)
        
        # Check commutators
        commutators = derivatives["commutators"]
        self.assertIn("xy", commutators)
        self.assertIn("yz", commutators)
        self.assertIn("zx", commutators)
        
        # Check traces
        traces = derivatives["traces"]
        self.assertIn("l_squared", traces)
        self.assertIn("comm_squared", traces)

    def test_field_statistics_computation(self) -> None:
        """Test field statistics computation."""
        # Create torus configuration
        torus_120 = Torus120Degrees(radius=1.0, backend=self.su2_fields.backend)
        
        # Create SU(2) field
        field = self.su2_fields.create_field_from_torus(
            torus_120, "skyrmion", 1.0
        )
        
        # Get statistics
        stats = self.su2_fields.get_field_statistics(field)
        
        # Check statistics structure
        expected_keys = [
            "mean_determinant",
            "std_determinant", 
            "min_determinant",
            "max_determinant",
            "field_norm_mean",
            "field_norm_std"
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], float)
        
        # Check determinant statistics (should be close to 1 for SU(2) field)
        self.assertAlmostEqual(stats["mean_determinant"], 1.0, places=5)
        self.assertLess(stats["std_determinant"], 0.1)  # Should be small
        self.assertAlmostEqual(stats["min_determinant"], 1.0, places=5)
        self.assertAlmostEqual(stats["max_determinant"], 1.0, places=5)

    def test_cuda_backend_integration(self) -> None:
        """Test CUDA backend integration."""
        # Test with CUDA if available
        su2_fields_cuda = SU2Fields(self.grid_size, self.box_size, use_cuda=True)
        
        # Check CUDA status
        cuda_status = su2_fields_cuda.get_cuda_status()
        self.assertIsInstance(cuda_status, str)
        
        # Check backend info
        backend_info = su2_fields_cuda.get_backend_info()
        self.assertIn("backend", backend_info)
        self.assertIn("cuda_status", backend_info)
        self.assertIn("cuda_available", backend_info)
        
        # Create field with CUDA backend
        torus_120 = Torus120Degrees(radius=1.0, backend=su2_fields_cuda.backend)
        field = su2_fields_cuda.create_field_from_torus(
            torus_120, "skyrmion", 1.0
        )
        
        # Validate field
        validation_results = su2_fields_cuda.validate_field(field)
        
        # Field should be valid regardless of backend
        self.assertTrue(validation_results["unitary"])
        self.assertTrue(validation_results["determinant"])
        self.assertTrue(validation_results["continuity"])

    def test_performance_comparison(self) -> None:
        """Test performance comparison between CPU and CUDA backends."""
        import time
        
        # Create torus configuration
        torus_120 = Torus120Degrees(radius=1.0, backend=self.su2_fields.backend)
        
        # Test CPU performance
        start_time = time.time()
        field_cpu = self.su2_fields.create_field_from_torus(
            torus_120, "skyrmion", 1.0
        )
        derivatives_cpu = self.su2_fields.compute_field_derivatives(field_cpu)
        cpu_time = time.time() - start_time
        
        # Test CUDA performance if available
        if self.su2_fields.is_cuda_available():
            su2_fields_cuda = SU2Fields(self.grid_size, self.box_size, use_cuda=True)
            torus_120_cuda = Torus120Degrees(radius=1.0, backend=su2_fields_cuda.backend)
            
            start_time = time.time()
            field_cuda = su2_fields_cuda.create_field_from_torus(
                torus_120_cuda, "skyrmion", 1.0
            )
            derivatives_cuda = su2_fields_cuda.compute_field_derivatives(field_cuda)
            cuda_time = time.time() - start_time
            
            # CUDA should be faster for large grids
            print(f"CPU time: {cpu_time:.4f}s, CUDA time: {cuda_time:.4f}s")
            
            # Both should produce valid results
            validation_cpu = self.su2_fields.validate_field(field_cpu)
            validation_cuda = su2_fields_cuda.validate_field(field_cuda)
            
            self.assertTrue(validation_cpu["unitary"])
            self.assertTrue(validation_cuda["unitary"])
        else:
            print("CUDA not available, skipping performance comparison")


if __name__ == "__main__":
    unittest.main()
