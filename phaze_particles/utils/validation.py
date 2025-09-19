#!/usr/bin/env python3
"""
Proton model validation system.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os


class ValidationStatus(Enum):
    """Validation status enumeration."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class ExperimentalData:
    """Experimental data for proton model validation."""

    proton_mass: float = 938.272
    proton_mass_error: float = 0.006
    charge_radius: float = 0.841
    charge_radius_error: float = 0.019
    magnetic_moment: float = 2.793
    magnetic_moment_error: float = 0.001
    electric_charge: float = 1.0
    baryon_number: float = 1.0


@dataclass
class CalculatedData:
    """Calculated data from proton model."""

    proton_mass: float
    charge_radius: float
    magnetic_moment: float
    electric_charge: float
    baryon_number: float
    energy_balance: float
    total_energy: float
    execution_time: float


@dataclass
class ValidationResult:
    """Result of parameter validation."""

    parameter_name: str
    calculated_value: float
    experimental_value: float
    experimental_error: float
    deviation: float
    deviation_percent: float
    within_tolerance: bool
    status: ValidationStatus


class ParameterValidator:
    """Validator for physical parameters."""

    def __init__(self, experimental_data: ExperimentalData):
        """
        Initialize validator.

        Args:
            experimental_data: Experimental data for comparison
        """
        self.experimental_data = experimental_data

    def validate_mass(self, calculated_mass: float) -> ValidationResult:
        """
        Validate proton mass.

        Args:
            calculated_mass: Calculated mass value

        Returns:
            Validation result
        """
        exp_mass = self.experimental_data.proton_mass
        exp_error = self.experimental_data.proton_mass_error

        deviation = abs(calculated_mass - exp_mass)
        deviation_percent = (deviation / exp_mass) * 100
        within_tolerance = deviation <= exp_error

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="proton_mass",
            calculated_value=calculated_mass,
            experimental_value=exp_mass,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )

    def validate_radius(self, calculated_radius: float) -> ValidationResult:
        """
        Validate charge radius.

        Args:
            calculated_radius: Calculated radius value

        Returns:
            Validation result
        """
        exp_radius = self.experimental_data.charge_radius
        exp_error = self.experimental_data.charge_radius_error

        deviation = abs(calculated_radius - exp_radius)
        deviation_percent = (deviation / exp_radius) * 100
        within_tolerance = deviation <= exp_error

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="charge_radius",
            calculated_value=calculated_radius,
            experimental_value=exp_radius,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )

    def validate_magnetic_moment(self, calculated_moment: float) -> ValidationResult:
        """
        Validate magnetic moment.

        Args:
            calculated_moment: Calculated magnetic moment value

        Returns:
            Validation result
        """
        exp_moment = self.experimental_data.magnetic_moment
        exp_error = self.experimental_data.magnetic_moment_error

        deviation = abs(calculated_moment - exp_moment)
        deviation_percent = (deviation / exp_moment) * 100
        within_tolerance = deviation <= exp_error

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 2 * exp_error:
            status = ValidationStatus.GOOD
        elif deviation <= 5 * exp_error:
            status = ValidationStatus.FAIR
        elif deviation <= 10 * exp_error:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="magnetic_moment",
            calculated_value=calculated_moment,
            experimental_value=exp_moment,
            experimental_error=exp_error,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )

    def validate_charge(self, calculated_charge: float) -> ValidationResult:
        """
        Validate electric charge.

        Args:
            calculated_charge: Calculated electric charge value

        Returns:
            Validation result
        """
        exp_charge = self.experimental_data.electric_charge
        tolerance = 1e-6  # Exact value

        deviation = abs(calculated_charge - exp_charge)
        deviation_percent = deviation * 100
        within_tolerance = deviation <= tolerance

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 1e-4:
            status = ValidationStatus.GOOD
        elif deviation <= 1e-3:
            status = ValidationStatus.FAIR
        elif deviation <= 1e-2:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="electric_charge",
            calculated_value=calculated_charge,
            experimental_value=exp_charge,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )

    def validate_baryon_number(self, calculated_baryon: float) -> ValidationResult:
        """
        Validate baryon number.

        Args:
            calculated_baryon: Calculated baryon number value

        Returns:
            Validation result
        """
        exp_baryon = self.experimental_data.baryon_number
        tolerance = 0.02  # Tolerance for baryon number

        deviation = abs(calculated_baryon - exp_baryon)
        deviation_percent = deviation * 100
        within_tolerance = deviation <= tolerance

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 0.05:
            status = ValidationStatus.GOOD
        elif deviation <= 0.1:
            status = ValidationStatus.FAIR
        elif deviation <= 0.2:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="baryon_number",
            calculated_value=calculated_baryon,
            experimental_value=exp_baryon,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )

    def validate_energy_balance(self, energy_balance: float) -> ValidationResult:
        """
        Validate energy balance.

        Args:
            energy_balance: Energy balance E₂/E₄ ratio

        Returns:
            Validation result
        """
        target_balance = 0.5
        tolerance = 0.01

        deviation = abs(energy_balance - target_balance)
        deviation_percent = deviation * 200  # In percentage of 0.5
        within_tolerance = deviation <= tolerance

        # Determine status
        if within_tolerance:
            status = ValidationStatus.EXCELLENT
        elif deviation <= 0.02:
            status = ValidationStatus.GOOD
        elif deviation <= 0.05:
            status = ValidationStatus.FAIR
        elif deviation <= 0.1:
            status = ValidationStatus.POOR
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            parameter_name="energy_balance",
            calculated_value=energy_balance,
            experimental_value=target_balance,
            experimental_error=tolerance,
            deviation=deviation,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            status=status,
        )


class ModelQualityAssessor:
    """Model quality assessor."""

    def __init__(self) -> None:
        """Initialize quality assessor."""
        self.weights = {
            "proton_mass": 0.25,
            "charge_radius": 0.25,
            "magnetic_moment": 0.20,
            "electric_charge": 0.15,
            "baryon_number": 0.10,
            "energy_balance": 0.05,
        }

    def assess_quality(
        self, validation_results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Assess model quality.

        Args:
            validation_results: List of validation results

        Returns:
            Quality assessment dictionary
        """
        # Count statuses
        status_counts = {}
        for status in ValidationStatus:
            status_counts[status] = 0

        for result in validation_results:
            status_counts[result.status] += 1

        # Calculate weighted score
        weighted_score = 0.0
        total_weight = 0.0

        for result in validation_results:
            weight = self.weights.get(result.parameter_name, 0.0)
            total_weight += weight

            # Convert status to numerical score
            if result.status == ValidationStatus.EXCELLENT:
                score = 1.0
            elif result.status == ValidationStatus.GOOD:
                score = 0.8
            elif result.status == ValidationStatus.FAIR:
                score = 0.6
            elif result.status == ValidationStatus.POOR:
                score = 0.4
            else:  # FAILED
                score = 0.0

            weighted_score += weight * score

        if total_weight > 0:
            weighted_score /= total_weight

        # Determine overall status
        if weighted_score >= 0.9:
            overall_status = ValidationStatus.EXCELLENT
        elif weighted_score >= 0.7:
            overall_status = ValidationStatus.GOOD
        elif weighted_score >= 0.5:
            overall_status = ValidationStatus.FAIR
        elif weighted_score >= 0.3:
            overall_status = ValidationStatus.POOR
        else:
            overall_status = ValidationStatus.FAILED

        return {
            "overall_status": overall_status,
            "weighted_score": weighted_score,
            "status_counts": status_counts,
            "total_parameters": len(validation_results),
            "passed_parameters": sum(
                1 for r in validation_results if r.within_tolerance
            ),
        }


class ValidationReportGenerator:
    """Validation report generator."""

    def __init__(self) -> None:
        """Initialize report generator."""
        self.timestamp = datetime.now()

    def generate_text_report(
        self,
        validation_results: List[ValidationResult],
        quality_assessment: Dict[str, Any],
    ) -> str:
        """
        Generate text report.

        Args:
            validation_results: List of validation results
            quality_assessment: Quality assessment results

        Returns:
            Text report string
        """
        report = []
        report.append("=" * 80)
        report.append("PROTON MODEL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Date and time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall assessment
        report.append("OVERALL ASSESSMENT:")
        report.append("-" * 40)
        report.append(f"Status: {quality_assessment['overall_status'].value.upper()}")
        report.append(f"Weighted score: {quality_assessment['weighted_score']:.3f}")
        passed = quality_assessment["passed_parameters"]
        total = quality_assessment["total_parameters"]
        report.append(f"Passed parameters: {passed}/{total}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)

        for result in validation_results:
            report.append(f"Parameter: {result.parameter_name}")
            report.append(f"  Calculated value: {result.calculated_value:.6f}")
            exp_val = result.experimental_value
            exp_err = result.experimental_error
            report.append(f"  Experimental value: {exp_val:.6f} ± {exp_err:.6f}")
            report.append(
                f"  Deviation: {result.deviation:.6f} "
                f"({result.deviation_percent:.2f}%)"
            )
            report.append(
                f"  Within tolerance: " f"{'YES' if result.within_tolerance else 'NO'}"
            )
            report.append(f"  Status: {result.status.value.upper()}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)

        failed_params = [
            r for r in validation_results if r.status == ValidationStatus.FAILED
        ]
        poor_params = [
            r for r in validation_results if r.status == ValidationStatus.POOR
        ]

        if failed_params:
            report.append("CRITICAL ISSUES:")
            for param in failed_params:
                report.append(
                    f"  - {param.parameter_name}: " f"requires immediate correction"
                )

        if poor_params:
            report.append("ISSUES REQUIRING ATTENTION:")
            for param in poor_params:
                report.append(f"  - {param.parameter_name}: improvement recommended")

        if not failed_params and not poor_params:
            report.append("Model meets all validation requirements.")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def generate_json_report(
        self,
        validation_results: List[ValidationResult],
        quality_assessment: Dict[str, Any],
    ) -> str:
        """
        Generate JSON report.

        Args:
            validation_results: List of validation results
            quality_assessment: Quality assessment results

        Returns:
            JSON report string
        """
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(value):
            try:
                if hasattr(value, 'tolist'):  # numpy array
                    return value.tolist()
                elif hasattr(value, 'get'):  # CuPy array
                    return value.get().tolist()
                elif isinstance(value, (complex, np.complex128, np.complex64)):  # complex number
                    return {"real": float(value.real), "imag": float(value.imag)}
                elif hasattr(value, 'dtype') and 'complex' in str(value.dtype):  # numpy complex array
                    return value.tolist()
                elif hasattr(value, 'dtype') and hasattr(value, 'get') and 'complex' in str(value.dtype):  # CuPy complex array
                    return value.get().tolist()
                else:
                    return value
            except Exception as e:
                # Fallback: try to convert to string representation
                print(f"Warning: Failed to convert value {type(value)} to JSON: {e}")
                return str(value)

        report_data = {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": quality_assessment["overall_status"].value,
            "weighted_score": convert_for_json(quality_assessment["weighted_score"]),
            "status_counts": {
                k.value: v for k, v in quality_assessment["status_counts"].items()
            },
            "total_parameters": quality_assessment["total_parameters"],
            "passed_parameters": quality_assessment["passed_parameters"],
            "validation_results": [],
        }

        for result in validation_results:
            try:
                report_data["validation_results"].append(
                    {
                        "parameter_name": result.parameter_name,
                        "calculated_value": convert_for_json(result.calculated_value),
                        "experimental_value": convert_for_json(result.experimental_value),
                        "experimental_error": convert_for_json(result.experimental_error),
                        "deviation": convert_for_json(result.deviation),
                        "deviation_percent": convert_for_json(result.deviation_percent),
                        "within_tolerance": bool(result.within_tolerance),
                        "status": result.status.value,
                    }
                )
            except Exception as e:
                print(f"Error processing validation result for {result.parameter_name}: {e}")
                print(f"  calculated_value type: {type(result.calculated_value)}")
                print(f"  experimental_value type: {type(result.experimental_value)}")
                print(f"  deviation type: {type(result.deviation)}")
                # Add a simplified version
                report_data["validation_results"].append(
                    {
                        "parameter_name": result.parameter_name,
                        "calculated_value": str(result.calculated_value),
                        "experimental_value": str(result.experimental_value),
                        "experimental_error": str(result.experimental_error),
                        "deviation": str(result.deviation),
                        "deviation_percent": str(result.deviation_percent),
                        "within_tolerance": bool(result.within_tolerance),
                        "status": result.status.value,
                    }
                )

        # Debug: check for complex numbers in report_data
        def check_for_complex(obj, path=""):
            if isinstance(obj, (complex, np.complex128, np.complex64)):
                print(f"Found complex number at {path}: {obj}")
                return True
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if check_for_complex(v, f"{path}.{k}"):
                        return True
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    if check_for_complex(v, f"{path}[{i}]"):
                        return True
            return False
        
        if check_for_complex(report_data, "report_data"):
            print("Complex numbers found in report_data, converting...")
            # Recursively convert all complex numbers
            def convert_all_complex(obj):
                if isinstance(obj, (complex, np.complex128, np.complex64)):
                    return {"real": float(obj.real), "imag": float(obj.imag)}
                elif isinstance(obj, dict):
                    return {k: convert_all_complex(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_all_complex(v) for v in obj]
                else:
                    return obj
            
            report_data = convert_all_complex(report_data)
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)

    def generate_plots(
        self,
        validation_results: List[ValidationResult],
        output_dir: str = "plots",
    ) -> None:
        """
        Generate validation plots.

        Args:
            validation_results: List of validation results
            output_dir: Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)

        # Deviation plot
        fig, ax = plt.subplots(figsize=(12, 8))

        param_names = [r.parameter_name for r in validation_results]
        # Convert CuPy arrays to NumPy for matplotlib
        deviations = []
        for r in validation_results:
            if hasattr(r.deviation_percent, 'get'):
                deviations.append(float(r.deviation_percent.get()))
            else:
                deviations.append(float(r.deviation_percent))
        colors = ["green" if r.within_tolerance else "red" for r in validation_results]

        bars = ax.bar(param_names, deviations, color=colors, alpha=0.7)
        ax.set_ylabel("Deviation (%)")
        ax.set_title("Deviations of Calculated Parameters from Experimental Values")
        ax.tick_params(axis="x", rotation=45)

        # Add values on bars
        for bar, deviation in zip(bars, deviations):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{deviation:.2f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/validation_deviations.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Comparison plot
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(param_names))
        width = 0.35

        # Convert CuPy arrays to NumPy for matplotlib
        calculated_values = []
        experimental_values = []
        experimental_errors = []
        
        for r in validation_results:
            # Handle calculated_value
            if hasattr(r.calculated_value, 'get'):
                calculated_values.append(float(r.calculated_value.get()))
            else:
                calculated_values.append(float(r.calculated_value))
            
            # Handle experimental_value
            if hasattr(r.experimental_value, 'get'):
                experimental_values.append(float(r.experimental_value.get()))
            else:
                experimental_values.append(float(r.experimental_value))
            
            # Handle experimental_error
            if hasattr(r.experimental_error, 'get'):
                experimental_errors.append(float(r.experimental_error.get()))
            else:
                experimental_errors.append(float(r.experimental_error))

        ax.bar(
            x - width / 2,
            calculated_values,
            width,
            label="Calculated",
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            experimental_values,
            width,
            label="Experimental",
            alpha=0.7,
        )

        # Add error bars
        ax.errorbar(
            x + width / 2,
            experimental_values,
            yerr=experimental_errors,
            fmt="none",
            color="black",
            capsize=5,
        )

        ax.set_ylabel("Parameter Values")
        ax.set_title("Comparison of Calculated and Experimental Values")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/validation_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


class ValidationSystem:
    """Main validation system."""

    def __init__(self, experimental_data: ExperimentalData):
        """
        Initialize validation system.

        Args:
            experimental_data: Experimental data for comparison
        """
        self.experimental_data = experimental_data
        self.validator = ParameterValidator(experimental_data)
        self.quality_assessor = ModelQualityAssessor()
        self.report_generator = ValidationReportGenerator()

    def validate_model(self, calculated_data: CalculatedData) -> Dict[str, Any]:
        """
        Validate model.

        Args:
            calculated_data: Calculated data from model

        Returns:
            Validation results dictionary
        """
        # Validate all parameters
        validation_results = [
            self.validator.validate_mass(calculated_data.proton_mass),
            self.validator.validate_radius(calculated_data.charge_radius),
            self.validator.validate_magnetic_moment(calculated_data.magnetic_moment),
            self.validator.validate_charge(calculated_data.electric_charge),
            self.validator.validate_baryon_number(calculated_data.baryon_number),
            self.validator.validate_energy_balance(calculated_data.energy_balance),
        ]

        # Assess quality
        quality_assessment = self.quality_assessor.assess_quality(validation_results)

        # Generate reports
        text_report = self.report_generator.generate_text_report(
            validation_results, quality_assessment
        )
        json_report = self.report_generator.generate_json_report(
            validation_results, quality_assessment
        )

        # Generate plots
        self.report_generator.generate_plots(validation_results)

        return {
            "validation_results": validation_results,
            "quality_assessment": quality_assessment,
            "text_report": text_report,
            "json_report": json_report,
            "overall_status": quality_assessment["overall_status"],
            "weighted_score": quality_assessment["weighted_score"],
        }

    def save_reports(
        self,
        validation_results: Dict[str, Any],
        output_dir: str = "validation_reports",
    ) -> None:
        """
        Save validation reports.

        Args:
            validation_results: Validation results
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save text report
        with open(
            f"{output_dir}/validation_report_{timestamp}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(validation_results["text_report"])

        # Save JSON report
        with open(
            f"{output_dir}/validation_report_{timestamp}.json",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(validation_results["json_report"])

        print(f"Reports saved in directory: {output_dir}")


def create_validation_system() -> ValidationSystem:
    """
    Create validation system with default experimental data.

    Returns:
        Initialized validation system
    """
    experimental_data = ExperimentalData()
    return ValidationSystem(experimental_data)


def validate_proton_model_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate proton model results.

    Args:
        results: Model calculation results

    Returns:
        Validation results
    """
    # Extract calculated data from results
    calculated_data = CalculatedData(
        proton_mass=results.get("mass", 0.0),
        charge_radius=results.get("radius", 0.0),
        magnetic_moment=results.get("magnetic_moment", 0.0),
        electric_charge=results.get("electric_charge", 0.0),
        baryon_number=results.get("baryon_number", 0.0),
        energy_balance=(
            results.get("energy_balance", {}).get("E2_percentage", 50.0) / 100.0
        ),
        total_energy=results.get("total_energy", 0.0),
        execution_time=results.get("execution_time", 0.0),
    )

    # Create validation system and validate
    validation_system = create_validation_system()
    validation_results = validation_system.validate_model(calculated_data)

    return validation_results
