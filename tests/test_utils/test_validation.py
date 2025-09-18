#!/usr/bin/env python3
"""
Tests for validation system.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
from unittest.mock import patch

from phaze_particles.utils.validation import (
    ValidationStatus,
    ExperimentalData,
    CalculatedData,
    ValidationResult,
    ParameterValidator,
    ModelQualityAssessor,
    ValidationReportGenerator,
    ValidationSystem,
    create_validation_system,
    validate_proton_model_results,
)


class TestValidationStatus(unittest.TestCase):
    """Test ValidationStatus enum."""

    def test_validation_status_values(self):
        """Test validation status enum values."""
        self.assertEqual(ValidationStatus.EXCELLENT.value, "excellent")
        self.assertEqual(ValidationStatus.GOOD.value, "good")
        self.assertEqual(ValidationStatus.FAIR.value, "fair")
        self.assertEqual(ValidationStatus.POOR.value, "poor")
        self.assertEqual(ValidationStatus.FAILED.value, "failed")


class TestExperimentalData(unittest.TestCase):
    """Test ExperimentalData dataclass."""

    def test_default_values(self):
        """Test default experimental data values."""
        data = ExperimentalData()

        self.assertEqual(data.proton_mass, 938.272)
        self.assertEqual(data.proton_mass_error, 0.006)
        self.assertEqual(data.charge_radius, 0.841)
        self.assertEqual(data.charge_radius_error, 0.019)
        self.assertEqual(data.magnetic_moment, 2.793)
        self.assertEqual(data.magnetic_moment_error, 0.001)
        self.assertEqual(data.electric_charge, 1.0)
        self.assertEqual(data.baryon_number, 1.0)

    def test_custom_values(self):
        """Test custom experimental data values."""
        data = ExperimentalData(
            proton_mass=939.0,
            proton_mass_error=0.01,
            charge_radius=0.85,
            charge_radius_error=0.02,
        )

        self.assertEqual(data.proton_mass, 939.0)
        self.assertEqual(data.proton_mass_error, 0.01)
        self.assertEqual(data.charge_radius, 0.85)
        self.assertEqual(data.charge_radius_error, 0.02)


class TestCalculatedData(unittest.TestCase):
    """Test CalculatedData dataclass."""

    def test_calculated_data_creation(self):
        """Test calculated data creation."""
        data = CalculatedData(
            proton_mass=938.5,
            charge_radius=0.84,
            magnetic_moment=2.8,
            electric_charge=1.0,
            baryon_number=1.0,
            energy_balance=0.5,
            total_energy=1000.0,
            execution_time=10.5,
        )

        self.assertEqual(data.proton_mass, 938.5)
        self.assertEqual(data.charge_radius, 0.84)
        self.assertEqual(data.magnetic_moment, 2.8)
        self.assertEqual(data.electric_charge, 1.0)
        self.assertEqual(data.baryon_number, 1.0)
        self.assertEqual(data.energy_balance, 0.5)
        self.assertEqual(data.total_energy, 1000.0)
        self.assertEqual(data.execution_time, 10.5)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            parameter_name="proton_mass",
            calculated_value=938.5,
            experimental_value=938.272,
            experimental_error=0.006,
            deviation=0.228,
            deviation_percent=0.024,
            within_tolerance=False,
            status=ValidationStatus.GOOD,
        )

        self.assertEqual(result.parameter_name, "proton_mass")
        self.assertEqual(result.calculated_value, 938.5)
        self.assertEqual(result.experimental_value, 938.272)
        self.assertEqual(result.experimental_error, 0.006)
        self.assertEqual(result.deviation, 0.228)
        self.assertEqual(result.deviation_percent, 0.024)
        self.assertFalse(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.GOOD)


class TestParameterValidator(unittest.TestCase):
    """Test ParameterValidator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.experimental_data = ExperimentalData()
        self.validator = ParameterValidator(self.experimental_data)

    def test_validate_mass_excellent(self):
        """Test mass validation with excellent result."""
        result = self.validator.validate_mass(938.272)

        self.assertEqual(result.parameter_name, "proton_mass")
        self.assertEqual(result.calculated_value, 938.272)
        self.assertEqual(result.experimental_value, 938.272)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)

    def test_validate_mass_good(self):
        """Test mass validation with good result."""
        result = self.validator.validate_mass(938.280)

        self.assertEqual(result.parameter_name, "proton_mass")
        self.assertFalse(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.GOOD)

    def test_validate_mass_failed(self):
        """Test mass validation with failed result."""
        result = self.validator.validate_mass(1000.0)

        self.assertEqual(result.parameter_name, "proton_mass")
        self.assertFalse(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.FAILED)

    def test_validate_radius_excellent(self):
        """Test radius validation with excellent result."""
        result = self.validator.validate_radius(0.841)

        self.assertEqual(result.parameter_name, "charge_radius")
        self.assertEqual(result.calculated_value, 0.841)
        self.assertEqual(result.experimental_value, 0.841)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)

    def test_validate_magnetic_moment_excellent(self):
        """Test magnetic moment validation with excellent result."""
        result = self.validator.validate_magnetic_moment(2.793)

        self.assertEqual(result.parameter_name, "magnetic_moment")
        self.assertEqual(result.calculated_value, 2.793)
        self.assertEqual(result.experimental_value, 2.793)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)

    def test_validate_charge_excellent(self):
        """Test electric charge validation with excellent result."""
        result = self.validator.validate_charge(1.0)

        self.assertEqual(result.parameter_name, "electric_charge")
        self.assertEqual(result.calculated_value, 1.0)
        self.assertEqual(result.experimental_value, 1.0)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)

    def test_validate_baryon_number_excellent(self):
        """Test baryon number validation with excellent result."""
        result = self.validator.validate_baryon_number(1.0)

        self.assertEqual(result.parameter_name, "baryon_number")
        self.assertEqual(result.calculated_value, 1.0)
        self.assertEqual(result.experimental_value, 1.0)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)

    def test_validate_energy_balance_excellent(self):
        """Test energy balance validation with excellent result."""
        result = self.validator.validate_energy_balance(0.5)

        self.assertEqual(result.parameter_name, "energy_balance")
        self.assertEqual(result.calculated_value, 0.5)
        self.assertEqual(result.experimental_value, 0.5)
        self.assertEqual(result.deviation, 0.0)
        self.assertEqual(result.deviation_percent, 0.0)
        self.assertTrue(result.within_tolerance)
        self.assertEqual(result.status, ValidationStatus.EXCELLENT)


class TestModelQualityAssessor(unittest.TestCase):
    """Test ModelQualityAssessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.assessor = ModelQualityAssessor()

    def test_assess_quality_excellent(self):
        """Test quality assessment with excellent results."""
        results = [
            ValidationResult(
                "proton_mass",
                938.272,
                938.272,
                0.006,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "charge_radius",
                0.841,
                0.841,
                0.019,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "magnetic_moment",
                2.793,
                2.793,
                0.001,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "electric_charge",
                1.0,
                1.0,
                1e-6,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "baryon_number",
                1.0,
                1.0,
                0.02,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "energy_balance",
                0.5,
                0.5,
                0.01,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
        ]

        assessment = self.assessor.assess_quality(results)

        self.assertEqual(assessment["overall_status"], ValidationStatus.EXCELLENT)
        self.assertEqual(assessment["weighted_score"], 1.0)
        self.assertEqual(assessment["total_parameters"], 6)
        self.assertEqual(assessment["passed_parameters"], 6)
        self.assertEqual(assessment["status_counts"][ValidationStatus.EXCELLENT], 6)

    def test_assess_quality_mixed(self):
        """Test quality assessment with mixed results."""
        results = [
            ValidationResult(
                "proton_mass",
                938.5,
                938.272,
                0.006,
                0.228,
                0.024,
                False,
                ValidationStatus.GOOD,
            ),
            ValidationResult(
                "charge_radius",
                0.85,
                0.841,
                0.019,
                0.009,
                1.07,
                False,
                ValidationStatus.GOOD,
            ),
            ValidationResult(
                "magnetic_moment",
                2.8,
                2.793,
                0.001,
                0.007,
                0.25,
                False,
                ValidationStatus.GOOD,
            ),
            ValidationResult(
                "electric_charge",
                1.0,
                1.0,
                1e-6,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "baryon_number",
                1.0,
                1.0,
                0.02,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
            ValidationResult(
                "energy_balance",
                0.5,
                0.5,
                0.01,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
        ]

        assessment = self.assessor.assess_quality(results)

        self.assertEqual(assessment["overall_status"], ValidationStatus.GOOD)
        self.assertGreater(assessment["weighted_score"], 0.7)
        self.assertLess(assessment["weighted_score"], 0.9)
        self.assertEqual(assessment["total_parameters"], 6)
        self.assertEqual(assessment["passed_parameters"], 3)
        self.assertEqual(assessment["status_counts"][ValidationStatus.EXCELLENT], 3)
        self.assertEqual(assessment["status_counts"][ValidationStatus.GOOD], 3)


class TestValidationReportGenerator(unittest.TestCase):
    """Test ValidationReportGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = ValidationReportGenerator()

    def test_generate_text_report(self):
        """Test text report generation."""
        results = [
            ValidationResult(
                "proton_mass",
                938.5,
                938.272,
                0.006,
                0.228,
                0.024,
                False,
                ValidationStatus.GOOD,
            ),
            ValidationResult(
                "electric_charge",
                1.0,
                1.0,
                1e-6,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
        ]

        quality_assessment = {
            "overall_status": ValidationStatus.GOOD,
            "weighted_score": 0.8,
            "status_counts": {ValidationStatus.EXCELLENT: 1, ValidationStatus.GOOD: 1},
            "total_parameters": 2,
            "passed_parameters": 1,
        }

        report = self.generator.generate_text_report(results, quality_assessment)

        self.assertIn("PROTON MODEL VALIDATION REPORT", report)
        self.assertIn("GOOD", report)
        self.assertIn("proton_mass", report)
        self.assertIn("electric_charge", report)
        self.assertIn("0.02%", report)

    def test_generate_json_report(self):
        """Test JSON report generation."""
        results = [
            ValidationResult(
                "proton_mass",
                938.5,
                938.272,
                0.006,
                0.228,
                0.024,
                False,
                ValidationStatus.GOOD,
            ),
        ]

        quality_assessment = {
            "overall_status": ValidationStatus.GOOD,
            "weighted_score": 0.8,
            "status_counts": {ValidationStatus.GOOD: 1},
            "total_parameters": 1,
            "passed_parameters": 0,
        }

        report = self.generator.generate_json_report(results, quality_assessment)

        # Parse JSON to verify structure
        import json

        data = json.loads(report)

        self.assertEqual(data["overall_status"], "good")
        self.assertEqual(data["weighted_score"], 0.8)
        self.assertEqual(data["total_parameters"], 1)
        self.assertEqual(data["passed_parameters"], 0)
        self.assertEqual(len(data["validation_results"]), 1)
        self.assertEqual(data["validation_results"][0]["parameter_name"], "proton_mass")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_plots(self, mock_close, mock_savefig):
        """Test plot generation."""
        results = [
            ValidationResult(
                "proton_mass",
                938.5,
                938.272,
                0.006,
                0.228,
                0.024,
                False,
                ValidationStatus.GOOD,
            ),
            ValidationResult(
                "electric_charge",
                1.0,
                1.0,
                1e-6,
                0.0,
                0.0,
                True,
                ValidationStatus.EXCELLENT,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            self.generator.generate_plots(results, temp_dir)

            # Verify plots were saved
            self.assertEqual(mock_savefig.call_count, 2)
            mock_close.assert_called()


class TestValidationSystem(unittest.TestCase):
    """Test ValidationSystem class."""

    def setUp(self):
        """Set up test fixtures."""
        self.experimental_data = ExperimentalData()
        self.validation_system = ValidationSystem(self.experimental_data)

    def test_validate_model(self):
        """Test model validation."""
        calculated_data = CalculatedData(
            proton_mass=938.5,
            charge_radius=0.85,
            magnetic_moment=2.8,
            electric_charge=1.0,
            baryon_number=1.0,
            energy_balance=0.5,
            total_energy=1000.0,
            execution_time=10.5,
        )

        results = self.validation_system.validate_model(calculated_data)

        self.assertIn("validation_results", results)
        self.assertIn("quality_assessment", results)
        self.assertIn("text_report", results)
        self.assertIn("json_report", results)
        self.assertIn("overall_status", results)
        self.assertIn("weighted_score", results)

        self.assertEqual(len(results["validation_results"]), 6)
        self.assertIsInstance(results["overall_status"], ValidationStatus)
        self.assertIsInstance(results["weighted_score"], float)

    @patch("os.makedirs")
    def test_save_reports(self, mock_makedirs):
        """Test saving reports."""
        validation_results = {
            "text_report": "Test report",
            "json_report": '{"test": "data"}',
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            self.validation_system.save_reports(validation_results, temp_dir)

            # Verify directory was created
            mock_makedirs.assert_called_with(temp_dir, exist_ok=True)


class TestValidationFunctions(unittest.TestCase):
    """Test validation utility functions."""

    def test_create_validation_system(self):
        """Test validation system creation."""
        system = create_validation_system()

        self.assertIsInstance(system, ValidationSystem)
        self.assertIsInstance(system.experimental_data, ExperimentalData)

    def test_validate_proton_model_results(self):
        """Test proton model results validation."""
        results = {
            "mass": 938.5,
            "radius": 0.85,
            "magnetic_moment": 2.8,
            "electric_charge": 1.0,
            "baryon_number": 1.0,
            "energy_balance": {"E2_percentage": 50.0},
            "total_energy": 1000.0,
            "execution_time": 10.5,
        }

        validation_results = validate_proton_model_results(results)

        self.assertIn("validation_results", validation_results)
        self.assertIn("quality_assessment", validation_results)
        self.assertIn("text_report", validation_results)
        self.assertIn("json_report", validation_results)
        self.assertIn("overall_status", validation_results)
        self.assertIn("weighted_score", validation_results)


if __name__ == "__main__":
    unittest.main()
