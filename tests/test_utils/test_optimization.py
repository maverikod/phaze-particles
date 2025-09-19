#!/usr/bin/env python3
"""
Tests for optimization utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from phaze_particles.utils.optimization import (
    OptimizationLevel,
    OptimizationConfig,
    PerformanceMetrics,
    MemoryOptimizer,
    AlgorithmOptimizer,
    AdaptiveParameterOptimizer,
    PerformanceProfiler,
    OptimizationSuite,
    OptimizedProtonModel,
)


class TestOptimizationLevel:
    """Test optimization level enum."""

    def test_optimization_levels(self):
        """Test optimization level values."""
        assert OptimizationLevel.NONE.value == "none"
        assert OptimizationLevel.BASIC.value == "basic"
        assert OptimizationLevel.ADVANCED.value == "advanced"
        assert OptimizationLevel.MAXIMUM.value == "maximum"


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = OptimizationConfig()
        assert config.use_cuda is True
        assert config.optimization_level == OptimizationLevel.ADVANCED
        assert config.memory_optimization is True
        assert config.algorithm_optimization is True
        assert config.adaptive_parameters is True
        assert config.profiling_enabled is True
        assert config.cache_enabled is True
        assert config.parallel_processing is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            use_cuda=False,
            optimization_level=OptimizationLevel.BASIC,
            memory_optimization=False,
            algorithm_optimization=False,
            adaptive_parameters=False,
            profiling_enabled=False,
            cache_enabled=False,
            parallel_processing=False,
        )
        assert config.use_cuda is False
        assert config.optimization_level == OptimizationLevel.BASIC
        assert config.memory_optimization is False
        assert config.algorithm_optimization is False
        assert config.adaptive_parameters is False
        assert config.profiling_enabled is False
        assert config.cache_enabled is False
        assert config.parallel_processing is False


class TestPerformanceMetrics:
    """Test performance metrics."""

    def test_performance_metrics(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            execution_time=1.5,
            memory_usage=100.0,
            gpu_utilization=50.0,
            cpu_utilization=75.0,
            iterations=100,
            convergence_rate=0.01,
            throughput=1000.0,
        )
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage == 100.0
        assert metrics.gpu_utilization == 50.0
        assert metrics.cpu_utilization == 75.0
        assert metrics.iterations == 100
        assert metrics.convergence_rate == 0.01
        assert metrics.throughput == 1000.0


class TestMemoryOptimizer:
    """Test memory optimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cuda_manager = Mock()
        self.cuda_manager.is_available = False
        self.optimizer = MemoryOptimizer(self.cuda_manager)

    def test_initialization(self):
        """Test memory optimizer initialization."""
        assert self.optimizer.cuda_manager == self.cuda_manager
        assert self.optimizer.memory_pool == {}
        assert self.optimizer.cache == {}

    @patch("psutil.Process")
    def test_get_memory_usage_cpu_only(self, mock_process):
        """Test memory usage with CPU only."""
        mock_process.return_value.memory_info.return_value.rss = (
            100 * 1024 * 1024
        )  # 100 MB

        memory_info = self.optimizer.get_memory_usage()

        assert "cpu_memory_mb" in memory_info
        assert memory_info["cpu_memory_mb"] == 100.0
        assert memory_info["gpu_memory_mb"] == 0
        assert memory_info["gpu_memory_total_mb"] == 0

    def test_optimize_memory_layout(self):
        """Test memory layout optimization."""
        # Create test arrays
        array1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        array2 = np.array([[5, 6], [7, 8]], dtype=np.float32)

        arrays = [array1, array2]
        optimized = self.optimizer.optimize_memory_layout(arrays)

        # Check that float64 was converted to float32
        assert optimized[0].dtype == np.float32
        assert optimized[1].dtype == np.float32

        # Check that arrays are contiguous
        assert optimized[0].flags["C_CONTIGUOUS"]
        assert optimized[1].flags["C_CONTIGUOUS"]

    def test_cache_operations(self):
        """Test cache operations."""
        # Test caching
        self.optimizer.cache_result("test_key", "test_value")
        assert self.optimizer.get_cached_result("test_key") == "test_value"

        # Test non-existent key
        assert self.optimizer.get_cached_result("non_existent") is None


class TestAlgorithmOptimizer:
    """Test algorithm optimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cuda_manager = Mock()
        self.cuda_manager.is_available = False
        self.optimizer = AlgorithmOptimizer(self.cuda_manager)

    def test_initialization(self):
        """Test algorithm optimizer initialization."""
        assert self.optimizer.cuda_manager == self.cuda_manager
        assert self.optimizer.optimized_kernels == {}

    def test_gradient_calculation_cpu(self):
        """Test gradient calculation on CPU."""
        # Create test field
        field = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        dx = 0.1

        grad_x, grad_y, grad_z = self.optimizer.optimize_gradient_calculation(field, dx)

        assert grad_x.shape == field.shape
        assert grad_y.shape == field.shape
        assert grad_z.shape == field.shape

    def test_integration_cpu(self):
        """Test integration on CPU."""
        field = np.ones((10, 10, 10), dtype=np.float32)
        dx = 0.1

        result = self.optimizer.optimize_integration(field, dx)

        expected = 10 * 10 * 10 * dx**3
        assert abs(result - expected) < 1e-6

    def test_matrix_operations_cpu(self):
        """Test matrix operations on CPU."""
        matrices = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        ]

        result = self.optimizer.optimize_matrix_operations(matrices)

        expected = np.dot(matrices[0], matrices[1])
        np.testing.assert_array_almost_equal(result, expected)


class TestAdaptiveParameterOptimizer:
    """Test adaptive parameter optimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = AdaptiveParameterOptimizer()

    def test_initialization(self):
        """Test parameter optimizer initialization."""
        assert self.optimizer.parameter_history == {}
        assert self.optimizer.performance_history == {}

    def test_step_size_optimization(self):
        """Test step size optimization."""
        # Fast convergence - should increase step
        new_step = self.optimizer.optimize_step_size(0.01, 0.2, 10)
        assert new_step > 0.01

        # Slow convergence - should decrease step
        new_step = self.optimizer.optimize_step_size(0.01, 0.005, 10)
        assert new_step < 0.01

        # Normal convergence - should stay same
        new_step = self.optimizer.optimize_step_size(0.01, 0.05, 10)
        assert new_step == 0.01

    def test_tolerance_optimization(self):
        """Test tolerance optimization."""
        # Very small error - should increase tolerance
        new_tolerance = self.optimizer.optimize_tolerance(
            1e-6, 1e-8
        )  # error < tolerance * 0.1
        assert new_tolerance > 1e-6

        # Large error - should decrease tolerance
        new_tolerance = self.optimizer.optimize_tolerance(
            1e-6, 1e-4
        )  # error > tolerance * 10
        assert new_tolerance < 1e-6

        # Normal error - should stay same
        new_tolerance = self.optimizer.optimize_tolerance(
            1e-6, 5e-6
        )  # tolerance * 0.1 < error < tolerance * 10
        assert new_tolerance == 1e-6

    def test_constraints_optimization(self):
        """Test constraints optimization."""
        constraints = {"constraint1": 1.0, "constraint2": 2.0}
        violations = {
            "constraint1": 0.2,
            "constraint2": 0.005,
        }  # constraint2 violation < 0.01

        new_constraints = self.optimizer.optimize_constraints(constraints, violations)

        # Large violation - should increase weight
        assert new_constraints["constraint1"] > constraints["constraint1"]

        # Small violation - should decrease weight
        assert new_constraints["constraint2"] < constraints["constraint2"]


class TestPerformanceProfiler:
    """Test performance profiler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()

    def test_initialization(self):
        """Test profiler initialization."""
        assert self.profiler.profiles == {}
        assert self.profiler.current_profile is None

    @patch("psutil.cpu_percent")
    @patch("psutil.Process")
    def test_profiling(self, mock_process, mock_cpu):
        """Test profiling functionality."""
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_cpu.return_value = 50.0

        # Start profiling
        self.profiler.start_profile("test_profile")
        assert self.profiler.current_profile is not None
        assert self.profiler.current_profile["name"] == "test_profile"

        # Simulate some work
        time.sleep(0.01)

        # End profiling
        result = self.profiler.end_profile("test_profile")

        assert result["name"] == "test_profile"
        assert result["execution_time"] > 0
        assert "memory_delta" in result
        assert "cpu_usage" in result
        assert "timestamp" in result

        # Check that profile was saved
        assert "test_profile" in self.profiler.profiles

    def test_profile_summary(self):
        """Test profile summary."""
        # Add some mock profiles
        self.profiler.profiles = {
            "profile1": {"execution_time": 1.0, "memory_delta": 10.0},
            "profile2": {"execution_time": 2.0, "memory_delta": 20.0},
        }

        summary = self.profiler.get_profile_summary()

        assert summary["total_execution_time"] == 3.0
        assert summary["total_memory_delta"] == 30.0
        assert summary["profile_count"] == 2
        assert "profiles" in summary


class TestOptimizationSuite:
    """Test optimization suite."""

    def setup_method(self):
        """Set up test fixtures."""
        self.suite = OptimizationSuite()

    def test_initialization(self):
        """Test optimization suite initialization."""
        assert len(self.suite.optimization_levels) == 4
        assert OptimizationLevel.NONE in self.suite.optimization_levels
        assert OptimizationLevel.BASIC in self.suite.optimization_levels
        assert OptimizationLevel.ADVANCED in self.suite.optimization_levels
        assert OptimizationLevel.MAXIMUM in self.suite.optimization_levels

    def test_optimization_levels_config(self):
        """Test optimization level configurations."""
        # Test NONE level
        none_config = self.suite.optimization_levels[OptimizationLevel.NONE]
        assert none_config.use_cuda is False
        assert none_config.memory_optimization is False
        assert none_config.algorithm_optimization is False

        # Test MAXIMUM level
        max_config = self.suite.optimization_levels[OptimizationLevel.MAXIMUM]
        assert max_config.use_cuda is True
        assert max_config.memory_optimization is True
        assert max_config.algorithm_optimization is True

    @patch("phaze_particles.utils.optimization.OptimizedProtonModel")
    def test_optimization_comparison(self, mock_model_class):
        """Test optimization comparison."""
        # Mock the optimized model
        mock_model = Mock()
        mock_model.run_optimized.return_value = {
            "performance_metrics": PerformanceMetrics(
                execution_time=1.0,
                memory_usage=100.0,
                gpu_utilization=50.0,
                cpu_utilization=75.0,
                iterations=100,
                convergence_rate=0.01,
                throughput=1000.0,
            ),
            "device_info": {"available": False},
        }
        mock_model.get_optimization_report.return_value = "Test report"
        mock_model_class.return_value = mock_model

        # Create mock config
        config = Mock()
        config.grid_size = 64
        config.box_size = 4.0
        config.max_iterations = 1000

        # Run comparison
        results = self.suite.run_optimization_comparison(config)

        # Check results
        assert len(results) == 4
        for level in ["none", "basic", "advanced", "maximum"]:
            assert level in results
            assert "execution_time" in results[level]
            assert "performance_metrics" in results[level]
            assert "device_info" in results[level]
            assert "optimization_report" in results[level]

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        # Create mock comparison results
        comparison_results = {
            "none": {
                "performance_metrics": PerformanceMetrics(
                    execution_time=2.0,
                    memory_usage=200.0,
                    gpu_utilization=0.0,
                    cpu_utilization=100.0,
                    iterations=100,
                    convergence_rate=0.01,
                    throughput=500.0,
                )
            },
            "maximum": {
                "performance_metrics": PerformanceMetrics(
                    execution_time=1.0,
                    memory_usage=100.0,
                    gpu_utilization=50.0,
                    cpu_utilization=75.0,
                    iterations=100,
                    convergence_rate=0.01,
                    throughput=1000.0,
                )
            },
        }

        report = self.suite.generate_comparison_report(comparison_results)

        assert "OPTIMIZATION COMPARISON REPORT" in report
        assert "COMPARISON TABLE:" in report
        assert "ANALYSIS:" in report
        assert "Best execution time: maximum" in report
        assert "Best throughput: maximum" in report


class TestOptimizedProtonModel:
    """Test optimized proton model."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.config.grid_size = 64
        self.config.box_size = 4.0
        self.config.max_iterations = 1000

        self.optimization_config = OptimizationConfig()

        # Mock the CUDA manager
        with patch("phaze_particles.utils.optimization.get_cuda_manager") as mock_cuda:
            mock_cuda.return_value.get_detailed_status.return_value = {
                "available": False
            }
            self.model = OptimizedProtonModel(self.config, self.optimization_config)

    def test_initialization(self):
        """Test optimized model initialization."""
        assert self.model.config == self.config
        assert self.model.optimization_config == self.optimization_config
        assert self.model.performance_metrics is None

    def test_optimize_configuration_cpu(self):
        """Test configuration optimization for CPU."""
        optimized_config = self.model.optimize_configuration()

        # Should optimize for CPU
        assert optimized_config.grid_size <= 64
        assert optimized_config.max_iterations <= 1000

    @patch("phaze_particles.utils.optimization.get_cuda_manager")
    def test_optimize_configuration_gpu(self, mock_cuda):
        """Test configuration optimization for GPU."""
        # Mock GPU availability
        mock_cuda.return_value.get_detailed_status.return_value = {"available": True}

        model = OptimizedProtonModel(self.config, self.optimization_config)
        optimized_config = model.optimize_configuration()

        # Should optimize for GPU
        assert optimized_config.grid_size <= 128
        assert optimized_config.max_iterations <= 2000

    def test_run_optimized(self):
        """Test running optimized model."""
        results = self.model.run_optimized()

        assert "results" in results
        assert "performance_metrics" in results
        assert "profile_summary" in results
        assert "device_info" in results
        assert "optimization_config" in results

        # Check performance metrics
        metrics = results["performance_metrics"]
        assert metrics.execution_time > 0
        assert metrics.memory_usage >= 0
        assert metrics.iterations > 0
        assert metrics.throughput > 0

    def test_get_optimization_report(self):
        """Test optimization report generation."""
        # Run optimization first
        self.model.run_optimized()

        report = self.model.get_optimization_report()

        assert "PROTON MODEL OPTIMIZATION REPORT" in report
        assert "DEVICE INFORMATION:" in report
        assert "PERFORMANCE METRICS:" in report
        assert "OPTIMIZATION CONFIGURATION:" in report


if __name__ == "__main__":
    pytest.main([__file__])
