#!/usr/bin/env python3
"""
Physical analysis and validation utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class PhysicalParameter:
    """Physical parameter with experimental value and tolerance."""
    
    name: str
    calculated_value: float
    experimental_value: float
    tolerance: float
    unit: str
    description: str


@dataclass
class AnalysisResult:
    """Result of physical analysis."""
    
    parameter: PhysicalParameter
    deviation_percent: float
    within_tolerance: bool
    quality_rating: str  # excellent/good/fair/poor


class PhysicsAnalyzer:
    """
    Analyzer for physical parameters and experimental comparison.
    
    Performs post-execution analysis of calculated physical parameters
    against experimental data and provides quality assessment.
    """
    
    # Experimental values for proton model
    EXPERIMENTAL_VALUES = {
        "electric_charge": {
            "value": 1.0,
            "tolerance": 0.0,  # Exact value
            "unit": "e",
            "description": "Electric charge"
        },
        "baryon_number": {
            "value": 1.0,
            "tolerance": 0.0,  # Exact value
            "unit": "",
            "description": "Baryon number"
        },
        "mass": {
            "value": 938.272,
            "tolerance": 0.006,
            "unit": "MeV",
            "description": "Proton mass"
        },
        "radius": {
            "value": 0.841,
            "tolerance": 0.019,
            "unit": "fm",
            "description": "Charge radius"
        },
        "magnetic_moment": {
            "value": 2.793,
            "tolerance": 0.001,
            "unit": "μN",
            "description": "Magnetic moment"
        },
        "energy_balance_e2": {
            "value": 50.0,
            "tolerance": 5.0,
            "unit": "%",
            "description": "E2 energy component"
        },
        "energy_balance_e4": {
            "value": 50.0,
            "tolerance": 5.0,
            "unit": "%",
            "description": "E4 energy component"
        }
    }
    
    def __init__(self):
        """Initialize physics analyzer."""
        self.results: List[AnalysisResult] = []
    
    def analyze_results(self, calculated_values: Dict[str, float]) -> List[AnalysisResult]:
        """
        Analyze calculated values against experimental data.
        
        Args:
            calculated_values: Dictionary of calculated physical parameters
            
        Returns:
            List of analysis results
        """
        self.results = []
        
        for param_name, exp_data in self.EXPERIMENTAL_VALUES.items():
            if param_name in calculated_values:
                param = PhysicalParameter(
                    name=param_name,
                    calculated_value=calculated_values[param_name],
                    experimental_value=exp_data["value"],
                    tolerance=exp_data["tolerance"],
                    unit=exp_data["unit"],
                    description=exp_data["description"]
                )
                
                result = self._analyze_parameter(param)
                self.results.append(result)
        
        return self.results
    
    def _analyze_parameter(self, param: PhysicalParameter) -> AnalysisResult:
        """
        Analyze a single physical parameter.
        
        Args:
            param: Physical parameter to analyze
            
        Returns:
            Analysis result
        """
        # Calculate percentage deviation
        if param.experimental_value != 0:
            deviation_percent = abs(
                (param.calculated_value - param.experimental_value) / 
                param.experimental_value * 100
            )
        else:
            deviation_percent = abs(param.calculated_value - param.experimental_value)
        
        # Check if within tolerance
        within_tolerance = deviation_percent <= param.tolerance
        
        # Determine quality rating
        quality_rating = self._determine_quality_rating(deviation_percent, param.tolerance)
        
        return AnalysisResult(
            parameter=param,
            deviation_percent=deviation_percent,
            within_tolerance=within_tolerance,
            quality_rating=quality_rating
        )
    
    def _determine_quality_rating(self, deviation_percent: float, tolerance: float) -> str:
        """
        Determine quality rating based on deviation.
        
        Args:
            deviation_percent: Percentage deviation from experimental value
            tolerance: Experimental tolerance
            
        Returns:
            Quality rating string
        """
        if deviation_percent <= tolerance * 0.1:
            return "excellent"
        elif deviation_percent <= tolerance * 0.5:
            return "good"
        elif deviation_percent <= tolerance:
            return "fair"
        else:
            return "poor"
    
    def get_overall_quality(self) -> str:
        """
        Get overall model quality assessment.
        
        Returns:
            Overall quality rating
        """
        if not self.results:
            return "unknown"
        
        quality_scores = {
            "excellent": 4,
            "good": 3,
            "fair": 2,
            "poor": 1
        }
        
        total_score = sum(quality_scores[result.quality_rating] for result in self.results)
        average_score = total_score / len(self.results)
        
        if average_score >= 3.5:
            return "excellent"
        elif average_score >= 2.5:
            return "good"
        elif average_score >= 1.5:
            return "fair"
        else:
            return "poor"
    
    def get_validation_status(self) -> str:
        """
        Get overall validation status.
        
        Returns:
            "pass" or "fail"
        """
        if not self.results:
            return "fail"
        
        # Check if all parameters are within tolerance
        all_within_tolerance = all(result.within_tolerance for result in self.results)
        
        return "pass" if all_within_tolerance else "fail"
    
    def generate_comparison_table(self) -> str:
        """
        Generate comparison table as string.
        
        Returns:
            Formatted comparison table
        """
        if not self.results:
            return "No analysis results available."
        
        table = "\n" + "="*80 + "\n"
        table += "PHYSICAL PARAMETER ANALYSIS\n"
        table += "="*80 + "\n"
        table += f"{'Parameter':<20} {'Calculated':<12} {'Experimental':<12} {'Deviation':<10} {'Status':<8} {'Quality':<10}\n"
        table += "-"*80 + "\n"
        
        for result in self.results:
            param = result.parameter
            status = "✓ PASS" if result.within_tolerance else "✗ FAIL"
            
            table += f"{param.description:<20} "
            table += f"{param.calculated_value:<12.3f} "
            table += f"{param.experimental_value:<12.3f} "
            table += f"{result.deviation_percent:<10.2f}% "
            table += f"{status:<8} "
            table += f"{result.quality_rating:<10}\n"
        
        table += "-"*80 + "\n"
        table += f"Overall Quality: {self.get_overall_quality().upper()}\n"
        table += f"Validation Status: {self.get_validation_status().upper()}\n"
        table += "="*80 + "\n"
        
        return table
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for model improvement.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.results:
            return ["No analysis results available for recommendations."]
        
        # Check for poor quality parameters
        poor_params = [r for r in self.results if r.quality_rating == "poor"]
        if poor_params:
            recommendations.append(
                f"Improve accuracy for {len(poor_params)} parameters with poor quality: "
                f"{', '.join(p.parameter.description for p in poor_params)}"
            )
        
        # Check for failed validations
        failed_params = [r for r in self.results if not r.within_tolerance]
        if failed_params:
            recommendations.append(
                f"Parameters outside experimental tolerance: "
                f"{', '.join(p.parameter.description for p in failed_params)}"
            )
        
        # General recommendations based on overall quality
        overall_quality = self.get_overall_quality()
        if overall_quality == "poor":
            recommendations.append("Consider fundamental model improvements or parameter tuning")
        elif overall_quality == "fair":
            recommendations.append("Fine-tune model parameters for better accuracy")
        elif overall_quality == "good":
            recommendations.append("Model shows good agreement with experimental data")
        else:
            recommendations.append("Excellent model performance - consider publication")
        
        return recommendations
