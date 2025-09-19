#!/usr/bin/env python3
"""
Comprehensive test suite for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import time
import tempfile
import shutil
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Import all test modules
from tests.test_cli.test_base import TestBaseCommand
from tests.test_cli.test_proton_command import TestProtonCommand
from tests.test_models.test_proton_integrated import (
    TestProtonModel,
    TestModelConfig,
    TestModelResults,
    TestProtonModelFactory,
)
from tests.test_utils.test_mathematical_foundations import (
    TestPhysicalConstants,
    TestSkyrmeConstants,
    TestPauliMatrices,
    TestTensorOperations,
    TestCoordinateSystem,
    TestNumericalUtils,
    TestValidationUtils,
    TestMathematicalFoundations,
)
from tests.test_utils.test_torus_geometries import (
    TestTorusGeometries,
    TestTorusParameters,
    TestTorus120Degrees,
    TestTorusClover,
    TestTorusCartesian,
    TestTorusGeometries,
)
from tests.test_utils.test_su2_fields import (
    TestSU2Field,
    TestRadialProfile,
    TestSU2FieldBuilder,
    TestSU2FieldOperations,
    TestSU2FieldValidator,
    TestSU2Fields,
    TestSU2FieldsIntegration,
)
from tests.test_utils.test_energy_densities import (
    TestEnergyDensity,
    TestBaryonDensity,
    TestEnergyDensityCalculator,
    TestEnergyAnalyzer,
    TestEnergyOptimizer,
    TestEnergyDensities,
)
from tests.test_utils.test_physics import (
    TestPhysicalParameter,
    TestAnalysisResult,
    TestPhysicsAnalyzer,
    TestPhysicalQuantities,
    TestChargeDensity,
    TestBaryonNumberCalculator,
    TestMagneticMomentCalculator,
    TestMassCalculator,
    TestPhysicalQuantitiesCalculator,
)
from tests.test_utils.test_numerical_methods import (
    TestSU2Projection,
    TestGradientDescent,
    TestLBFGSOptimizer,
    TestAdamOptimizer,
    TestConstraintController,
    TestRelaxationSolver,
    TestNumericalMethods,
    TestCUDAIntegration,
)
from tests.test_utils.test_validation import (
    TestValidationStatus,
    TestExperimentalData,
    TestCalculatedData,
    TestValidationResult,
    TestParameterValidator,
    TestModelQualityAssessor,
    TestValidationReportGenerator,
    TestValidationSystem,
    TestValidationFunctions,
)
from tests.test_utils.test_cuda import (
    TestCUDADevice,
    TestCUDAMemoryManager,
    TestCUDAOperations,
    TestCUDAManager,
    TestGetCUDAManager,
    TestCUDAIntegration,
)
from tests.test_utils.test_progress import (
    TestProgressBar,
    TestTimeEstimator,
    TestPerformanceMonitor,
    TestProgressCallback,
    TestCreateProgressBar,
    TestCreatePerformanceMonitor,
    TestProgressIntegration,
)


@dataclass
class TestConfig:
    """Test configuration."""

    grid_size: int = 32
    box_size: float = 2.0
    max_iterations: int = 100
    tolerance: float = 1e-6
    performance_threshold: float = 1.0  # seconds
    memory_threshold: float = 100.0  # MB


class TestSuite:
    """Main test suite class."""

    def __init__(self):
        """Initialize test suite."""
        self.test_config = TestConfig()
        self.results = {}
        self.temp_dir = None

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests.

        Returns:
            Test results
        """
        self.setUp()

        try:
            # Create test suite
            test_suite = unittest.TestSuite()

            # Add all test classes
            test_classes = [
                # CLI tests
                TestBaseCommand,
                TestProtonCommand,
                # Model tests
                TestProtonModel,
                TestModelConfig,
                TestModelResults,
                TestProtonModelFactory,
                # Mathematical foundations tests
                TestPhysicalConstants,
                TestSkyrmeConstants,
                TestPauliMatrices,
                TestTensorOperations,
                TestCoordinateSystem,
                TestNumericalUtils,
                TestValidationUtils,
                TestMathematicalFoundations,
                # Torus geometries tests
                TestTorusGeometries,
                TestTorusParameters,
                TestTorus120Degrees,
                TestTorusClover,
                TestTorusCartesian,
                TestTorusGeometries,
                # SU(2) fields tests
                TestSU2Field,
                TestRadialProfile,
                TestSU2FieldBuilder,
                TestSU2FieldOperations,
                TestSU2FieldValidator,
                TestSU2Fields,
                TestSU2FieldsIntegration,
                # Energy densities tests
                TestEnergyDensity,
                TestBaryonDensity,
                TestEnergyDensityCalculator,
                TestEnergyAnalyzer,
                TestEnergyOptimizer,
                TestEnergyDensities,
                # Physics tests
                TestPhysicalParameter,
                TestAnalysisResult,
                TestPhysicsAnalyzer,
                TestPhysicalQuantities,
                TestChargeDensity,
                TestBaryonNumberCalculator,
                TestMagneticMomentCalculator,
                TestMassCalculator,
                TestPhysicalQuantitiesCalculator,
                # Numerical methods tests
                TestSU2Projection,
                TestGradientDescent,
                TestLBFGSOptimizer,
                TestAdamOptimizer,
                TestConstraintController,
                TestRelaxationSolver,
                TestNumericalMethods,
                TestCUDAIntegration,
                # Validation tests
                TestValidationStatus,
                TestExperimentalData,
                TestCalculatedData,
                TestValidationResult,
                TestParameterValidator,
                TestModelQualityAssessor,
                TestValidationReportGenerator,
                TestValidationSystem,
                TestValidationFunctions,
                # CUDA tests
                TestCUDADevice,
                TestCUDAMemoryManager,
                TestCUDAOperations,
                TestCUDAManager,
                TestGetCUDAManager,
                TestCUDAIntegration,
                # Progress tests
                TestProgressBar,
                TestTimeEstimator,
                TestPerformanceMonitor,
                TestProgressCallback,
                TestCreateProgressBar,
                TestCreatePerformanceMonitor,
                TestProgressIntegration,
            ]

            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                test_suite.addTests(tests)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(test_suite)

            # Save results
            self.results = {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (
                    (result.testsRun - len(result.failures) - len(result.errors))
                    / result.testsRun
                    if result.testsRun > 0
                    else 0
                ),
                "failure_details": result.failures,
                "error_details": result.errors,
                "execution_time": time.time()
                - getattr(self, "start_time", time.time()),
            }

            return self.results

        finally:
            self.tearDown()

    def run_unit_tests(self) -> Dict[str, Any]:
        """
        Run unit tests only.

        Returns:
            Unit test results
        """
        self.setUp()

        try:
            # Create test suite for unit tests only
            test_suite = unittest.TestSuite()

            # Add unit test classes
            unit_test_classes = [
                # Mathematical foundations tests
                TestPhysicalConstants,
                TestSkyrmeConstants,
                TestPauliMatrices,
                TestTensorOperations,
                TestCoordinateSystem,
                TestNumericalUtils,
                TestValidationUtils,
                TestMathematicalFoundations,
                # Torus geometries tests
                TestTorusGeometries,
                TestTorusParameters,
                TestTorus120Degrees,
                TestTorusClover,
                TestTorusCartesian,
                TestTorusGeometries,
                # SU(2) fields tests
                TestSU2Field,
                TestRadialProfile,
                TestSU2FieldBuilder,
                TestSU2FieldOperations,
                TestSU2FieldValidator,
                TestSU2Fields,
                TestSU2FieldsIntegration,
                # Energy densities tests
                TestEnergyDensity,
                TestBaryonDensity,
                TestEnergyDensityCalculator,
                TestEnergyAnalyzer,
                TestEnergyOptimizer,
                TestEnergyDensities,
                # Physics tests
                TestPhysicalParameter,
                TestAnalysisResult,
                TestPhysicsAnalyzer,
                TestPhysicalQuantities,
                TestChargeDensity,
                TestBaryonNumberCalculator,
                TestMagneticMomentCalculator,
                TestMassCalculator,
                TestPhysicalQuantitiesCalculator,
                # Numerical methods tests
                TestSU2Projection,
                TestGradientDescent,
                TestLBFGSOptimizer,
                TestAdamOptimizer,
                TestConstraintController,
                TestRelaxationSolver,
                TestNumericalMethods,
                TestCUDAIntegration,
                # Validation tests
                TestValidationStatus,
                TestExperimentalData,
                TestCalculatedData,
                TestValidationResult,
                TestParameterValidator,
                TestModelQualityAssessor,
                TestValidationReportGenerator,
                TestValidationSystem,
                TestValidationFunctions,
                # CUDA tests
                TestCUDADevice,
                TestCUDAMemoryManager,
                TestCUDAOperations,
                TestCUDAManager,
                TestGetCUDAManager,
                TestCUDAIntegration,
                # Progress tests
                TestProgressBar,
                TestTimeEstimator,
                TestPerformanceMonitor,
                TestProgressCallback,
                TestCreateProgressBar,
                TestCreatePerformanceMonitor,
                TestProgressIntegration,
            ]

            for test_class in unit_test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                test_suite.addTests(tests)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(test_suite)

            # Save results
            unit_results = {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (
                    (result.testsRun - len(result.failures) - len(result.errors))
                    / result.testsRun
                    if result.testsRun > 0
                    else 0
                ),
                "failure_details": result.failures,
                "error_details": result.errors,
                "execution_time": time.time()
                - getattr(self, "start_time", time.time()),
            }

            return unit_results

        finally:
            self.tearDown()

    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run integration tests only.

        Returns:
            Integration test results
        """
        self.setUp()

        try:
            # Create test suite for integration tests only
            test_suite = unittest.TestSuite()

            # Add integration test classes
            integration_test_classes = [
                # Model tests
                TestProtonModel,
                TestModelConfig,
                TestModelResults,
                TestProtonModelFactory,
                # CLI tests
                TestBaseCommand,
                TestProtonCommand,
            ]

            for test_class in integration_test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                test_suite.addTests(tests)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(test_suite)

            # Save results
            integration_results = {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (
                    (result.testsRun - len(result.failures) - len(result.errors))
                    / result.testsRun
                    if result.testsRun > 0
                    else 0
                ),
                "failure_details": result.failures,
                "error_details": result.errors,
                "execution_time": time.time()
                - getattr(self, "start_time", time.time()),
            }

            return integration_results

        finally:
            self.tearDown()

    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run performance tests.

        Returns:
            Performance test results
        """
        self.setUp()

        try:
            # Performance tests are integrated into other test classes
            # This method runs a subset of tests with performance monitoring

            performance_results = {
                "tests_run": 0,
                "failures": 0,
                "errors": 0,
                "success_rate": 0.0,
                "performance_metrics": {},
                "execution_time": 0.0,
            }

            # Run performance-critical tests
            start_time = time.time()

            # Test model execution performance
            try:
                from phaze_particles.models.proton_integrated import (
                    ProtonModel,
                    ModelConfig,
                )

                config = ModelConfig(
                    grid_size=32,
                    box_size=2.0,
                    max_iterations=100,
                    validation_enabled=False,
                )

                model = ProtonModel(config)
                model_start = time.time()
                results = model.run()
                model_time = time.time() - model_start

                performance_results["performance_metrics"][
                    "model_execution_time"
                ] = model_time
                performance_results["tests_run"] += 1

                if results.status.value == "optimized":
                    performance_results["success_rate"] = 1.0
                else:
                    performance_results["failures"] += 1

            except Exception as e:
                performance_results["errors"] += 1
                performance_results["error_details"] = [str(e)]

            performance_results["execution_time"] = time.time() - start_time

            return performance_results

        finally:
            self.tearDown()

    def run_stress_tests(self) -> Dict[str, Any]:
        """
        Run stress tests.

        Returns:
            Stress test results
        """
        self.setUp()

        try:
            stress_results = {
                "tests_run": 0,
                "failures": 0,
                "errors": 0,
                "success_rate": 0.0,
                "stress_metrics": {},
                "execution_time": 0.0,
            }

            start_time = time.time()

            # Test with extreme parameters
            try:
                from phaze_particles.models.proton_integrated import (
                    ProtonModel,
                    ModelConfig,
                )

                # Test with very small grid
                config = ModelConfig(
                    grid_size=8,
                    box_size=1.0,
                    max_iterations=10,
                    validation_enabled=False,
                )

                model = ProtonModel(config)
                results = model.run()

                stress_results["tests_run"] += 1
                if results.status.value == "optimized":
                    stress_results["success_rate"] = 1.0
                else:
                    stress_results["failures"] += 1

            except Exception as e:
                stress_results["errors"] += 1
                stress_results["error_details"] = [str(e)]

            # Test with large iterations
            try:
                config = ModelConfig(
                    grid_size=32,
                    box_size=2.0,
                    max_iterations=1000,
                    validation_enabled=False,
                )

                model = ProtonModel(config)
                results = model.run()

                stress_results["tests_run"] += 1
                if results.status.value == "optimized":
                    stress_results["success_rate"] = 1.0
                else:
                    stress_results["failures"] += 1

            except Exception as e:
                stress_results["errors"] += 1
                if "error_details" not in stress_results:
                    stress_results["error_details"] = []
                stress_results["error_details"].append(str(e))

            stress_results["execution_time"] = time.time() - start_time

            return stress_results

        finally:
            self.tearDown()

    def generate_report(self) -> str:
        """
        Generate test report.

        Returns:
            Text report
        """
        if not self.results:
            return "No test results available. Run tests first."

        report = []
        report.append("=" * 80)
        report.append("PROTON MODEL TESTING REPORT")
        report.append("=" * 80)
        report.append(f"Total tests run: {self.results['tests_run']}")
        report.append(
            f"Successful: {self.results['tests_run'] - self.results['failures'] - self.results['errors']}"
        )
        report.append(f"Failed: {self.results['failures']}")
        report.append(f"Errors: {self.results['errors']}")
        report.append(f"Success rate: {self.results['success_rate']:.2%}")
        report.append(
            f"Execution time: {self.results.get('execution_time', 0):.2f} seconds"
        )
        report.append("")

        if self.results["failures"]:
            report.append("FAILED TESTS:")
            report.append("-" * 40)
            for test, traceback in self.results["failure_details"]:
                report.append(f"  - {test}: {traceback}")
            report.append("")

        if self.results["errors"]:
            report.append("ERRORS:")
            report.append("-" * 40)
            for test, traceback in self.results["error_details"]:
                report.append(f"  - {test}: {traceback}")
            report.append("")

        # Add performance metrics if available
        if "performance_metrics" in self.results:
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 40)
            for metric, value in self.results["performance_metrics"].items():
                report.append(f"  - {metric}: {value}")
            report.append("")

        # Add stress test metrics if available
        if "stress_metrics" in self.results:
            report.append("STRESS TEST METRICS:")
            report.append("-" * 40)
            for metric, value in self.results["stress_metrics"].items():
                report.append(f"  - {metric}: {value}")
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self, filepath: str):
        """
        Save test results to file.

        Args:
            filepath: Path to save results
        """
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def load_results(self, filepath: str):
        """
        Load test results from file.

        Args:
            filepath: Path to load results from
        """
        with open(filepath, "r") as f:
            self.results = json.load(f)


def run_tests():
    """Run all tests."""
    test_suite = TestSuite()
    results = test_suite.run_all_tests()
    report = test_suite.generate_report()

    print(report)

    return results


def run_unit_tests():
    """Run unit tests only."""
    test_suite = TestSuite()
    results = test_suite.run_unit_tests()
    report = test_suite.generate_report()

    print(report)

    return results


def run_integration_tests():
    """Run integration tests only."""
    test_suite = TestSuite()
    results = test_suite.run_integration_tests()
    report = test_suite.generate_report()

    print(report)

    return results


def run_performance_tests():
    """Run performance tests only."""
    test_suite = TestSuite()
    results = test_suite.run_performance_tests()
    report = test_suite.generate_report()

    print(report)

    return results


def run_stress_tests():
    """Run stress tests only."""
    test_suite = TestSuite()
    results = test_suite.run_stress_tests()
    report = test_suite.generate_report()

    print(report)

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "unit":
            run_unit_tests()
        elif test_type == "integration":
            run_integration_tests()
        elif test_type == "performance":
            run_performance_tests()
        elif test_type == "stress":
            run_stress_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Available types: unit, integration, performance, stress")
            sys.exit(1)
    else:
        run_tests()
