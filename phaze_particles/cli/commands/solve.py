"""
Universal solver command for Skyrme field equations.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from phaze_particles.cli.base import BaseCommand
from phaze_particles.utils.universal_solver import solve_skyrme_field, SolverInput


class SolveCommand(BaseCommand):
    """Universal solver command for Skyrme field equations."""
    
    def __init__(self):
        super().__init__("solve", "Universal solver for Skyrme field equations")
        self.subcommands = []
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--grid-size", type=int, default=32,
            help="Grid size (default: 32)"
        )
        parser.add_argument(
            "--box-size", type=float, default=4.0,
            help="Box size in fm (default: 4.0)"
        )
        parser.add_argument(
            "--config-type", type=str, default="120deg",
            choices=["120deg", "clover", "cartesian"],
            help="Configuration type (default: 120deg)"
        )
        parser.add_argument(
            "--c2", type=float, default=1.0,
            help="Skyrme constant c2 (default: 1.0)"
        )
        parser.add_argument(
            "--c4", type=float, default=1.0,
            help="Skyrme constant c4 (default: 1.0)"
        )
        parser.add_argument(
            "--c6", type=float, default=1.0,
            help="Skyrme constant c6 (default: 1.0)"
        )
        parser.add_argument(
            "--F-pi", type=float, default=186.0,
            help="Pion decay constant in MeV (default: 186.0)"
        )
        parser.add_argument(
            "--e", type=float, default=5.45,
            help="Dimensionless Skyrme constant (default: 5.45)"
        )
        parser.add_argument(
            "--target-mass", type=float,
            help="Target mass in MeV (e.g., 938.272 for proton)"
        )
        parser.add_argument(
            "--target-radius", type=float,
            help="Target radius in fm (e.g., 0.841 for proton)"
        )
        parser.add_argument(
            "--target-magnetic-moment", type=float,
            help="Target magnetic moment in μN (e.g., 2.793 for proton)"
        )
        parser.add_argument(
            "--target-bands", type=int,
            help="Target number of energy bands (e.g., 11)"
        )
        parser.add_argument(
            "--optimization-strategy", type=str, default="auto",
            choices=["auto", "energy_balance", "physical_params", "quantization"],
            help="Optimization strategy (default: auto)"
        )
        parser.add_argument(
            "--output", type=str,
            help="Output directory for results"
        )
        parser.add_argument(
            "--verbose", action="store_true",
            help="Verbose output"
        )
    
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the solve command."""
        try:
            print("UNIVERSAL SKYRME FIELD SOLVER")
            print("=" * 50)
            
            # Create solver input
            input_params = SolverInput(
                grid_size=args.grid_size,
                box_size=args.box_size,
                config_type=args.config_type,
                c2=args.c2,
                c4=args.c4,
                c6=args.c6,
                F_pi=args.F_pi,
                e=args.e,
                target_mass=args.target_mass,
                target_radius=args.target_radius,
                target_magnetic_moment=args.target_magnetic_moment,
                target_bands=args.target_bands,
                optimization_strategy=args.optimization_strategy
            )
            
            if args.verbose:
                print(f"Input parameters:")
                print(f"  Grid size: {input_params.grid_size}")
                print(f"  Box size: {input_params.box_size} fm")
                print(f"  Config type: {input_params.config_type}")
                print(f"  Constants: c2={input_params.c2}, c4={input_params.c4}, c6={input_params.c6}")
                print(f"  Physical: F_π={input_params.F_pi} MeV, e={input_params.e}")
                if input_params.target_mass:
                    print(f"  Target mass: {input_params.target_mass} MeV")
                if input_params.target_radius:
                    print(f"  Target radius: {input_params.target_radius} fm")
                if input_params.target_magnetic_moment:
                    print(f"  Target magnetic moment: {input_params.target_magnetic_moment} μN")
                if input_params.target_bands:
                    print(f"  Target bands: {input_params.target_bands}")
                print(f"  Strategy: {input_params.optimization_strategy}")
                print()
            
            # Solve
            print("Solving Skyrme field equations...")
            result = solve_skyrme_field(
                grid_size=input_params.grid_size,
                box_size=input_params.box_size,
                config_type=input_params.config_type,
                c2=input_params.c2,
                c4=input_params.c4,
                c6=input_params.c6,
                F_pi=input_params.F_pi,
                e=input_params.e,
                target_mass=input_params.target_mass,
                target_radius=input_params.target_radius,
                target_magnetic_moment=input_params.target_magnetic_moment,
                target_bands=input_params.target_bands,
                optimization_strategy=input_params.optimization_strategy
            )
            
            if not result.success:
                print(f"❌ Solver failed: {result.convergence_info}")
                return 1
            
            # Display results
            print("✅ Solver completed successfully!")
            print()
            print("PHYSICAL PARAMETERS:")
            print(f"  Mass: {result.physical_parameters['mass']:.1f} MeV")
            print(f"  Charge radius: {result.physical_parameters['charge_radius']:.3f} fm")
            print(f"  Magnetic moment: {result.physical_parameters['magnetic_moment']:.3f} μN")
            print(f"  Electric charge: {result.physical_parameters['electric_charge']:.3f}")
            print(f"  Baryon number: {result.physical_parameters['baryon_number']:.3f}")
            
            print()
            print("ENERGY ANALYSIS:")
            print(f"  E₂: {result.energy_analysis['e2']:.1f} MeV")
            print(f"  E₄: {result.energy_analysis['e4']:.1f} MeV")
            print(f"  E₆: {result.energy_analysis['e6']:.1f} MeV")
            print(f"  Total: {result.energy_analysis['total']:.1f} MeV")
            print(f"  Ratios: E₂/E₄ = {result.energy_analysis['e2_ratio']:.2f}/{result.energy_analysis['e4_ratio']:.2f}")
            
            print()
            print("MODE ANALYSIS:")
            print(f"  Total modes: {result.mode_analysis['total_modes']}")
            print(f"  Energy bands: {result.mode_analysis['energy_bands']}")
            print(f"  Core radius: {result.mode_analysis['core_radius']:.3f} fm")
            print(f"  Quantization parameter: {result.mode_analysis['quantization_parameter']:.3f}")
            
            print()
            print("TOPOLOGICAL ANALYSIS:")
            print(f"  Geometric radius: {result.topological_analysis['geometric_radius']:.3f} fm")
            print(f"  Phase radius: {result.topological_analysis['phase_radius']:.3f} fm")
            print(f"  Effective radius: {result.topological_analysis['effective_radius']:.3f} fm")
            print(f"  Topological charge: {result.topological_analysis['topological_charge']:.3f}")
            print(f"  Phase transitions: {result.topological_analysis['phase_transitions']}")
            
            print()
            print("INTERFERENCE ANALYSIS:")
            print(f"  Fluctuation energy: {result.interference_analysis['fluctuation_energy']:.1f} MeV")
            print(f"  Background field strength: {result.interference_analysis['background_field_strength']:.3f}")
            print(f"  Constructive regions: {result.interference_analysis['constructive_regions']}")
            print(f"  Destructive regions: {result.interference_analysis['destructive_regions']}")
            print(f"  Interference strength: {result.interference_analysis['interference_strength']:.3f}")
            print(f"  Fluctuation amplitude: {result.interference_analysis['fluctuation_amplitude']:.3f}")
            
            print()
            print("OPTIMIZED CONSTANTS:")
            print(f"  c₂: {result.optimized_constants['c2']:.6f}")
            print(f"  c₄: {result.optimized_constants['c4']:.6f}")
            print(f"  c₆: {result.optimized_constants['c6']:.6f}")
            print(f"  F_π: {result.optimized_constants['F_pi']:.1f} MeV")
            print(f"  e: {result.optimized_constants['e']:.3f}")
            
            print()
            print("PERFORMANCE:")
            print(f"  Execution time: {result.execution_time:.1f} seconds")
            print(f"  Iterations: {result.iterations}")
            print(f"  Strategy used: {result.convergence_info['strategy_used']}")
            
            # Save results if output directory specified
            if args.output:
                self._save_results(result, args.output, input_params)
                print(f"\nResults saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error executing solve command: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _save_results(self, result, output_dir: str, input_params: SolverInput) -> None:
        """Save results to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_data = {
            "input_parameters": {
                "grid_size": input_params.grid_size,
                "box_size": input_params.box_size,
                "config_type": input_params.config_type,
                "c2": input_params.c2,
                "c4": input_params.c4,
                "c6": input_params.c6,
                "F_pi": input_params.F_pi,
                "e": input_params.e,
                "target_mass": input_params.target_mass,
                "target_radius": input_params.target_radius,
                "target_magnetic_moment": input_params.target_magnetic_moment,
                "target_bands": input_params.target_bands,
                "optimization_strategy": input_params.optimization_strategy
            },
            "results": {
                "success": result.success,
                "physical_parameters": result.physical_parameters,
                "energy_analysis": result.energy_analysis,
                "mode_analysis": result.mode_analysis,
                "topological_analysis": result.topological_analysis,
                "interference_analysis": result.interference_analysis,
                "optimized_constants": result.optimized_constants,
                "convergence_info": result.convergence_info,
                "execution_time": result.execution_time,
                "iterations": result.iterations
            }
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                item = obj.item()
                if isinstance(item, complex):
                    return float(item.real)
                return item
            elif isinstance(obj, (np.complex128, np.complex64, complex)):
                return float(obj.real)  # Take real part for JSON serialization
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        results_data = convert_numpy(results_data)
        
        # Save JSON file
        json_file = output_path / "solver_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save summary as text
        summary_file = output_path / "solver_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("UNIVERSAL SKYRME FIELD SOLVER RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PHYSICAL PARAMETERS:\n")
            f.write(f"  Mass: {result.physical_parameters['mass']:.1f} MeV\n")
            f.write(f"  Charge radius: {result.physical_parameters['charge_radius']:.3f} fm\n")
            f.write(f"  Magnetic moment: {result.physical_parameters['magnetic_moment']:.3f} μN\n")
            f.write(f"  Electric charge: {result.physical_parameters['electric_charge']:.3f}\n")
            f.write(f"  Baryon number: {result.physical_parameters['baryon_number']:.3f}\n\n")
            
            f.write("ENERGY ANALYSIS:\n")
            f.write(f"  E₂: {result.energy_analysis['e2']:.1f} MeV\n")
            f.write(f"  E₄: {result.energy_analysis['e4']:.1f} MeV\n")
            f.write(f"  E₆: {result.energy_analysis['e6']:.1f} MeV\n")
            f.write(f"  Total: {result.energy_analysis['total']:.1f} MeV\n")
            f.write(f"  Ratios: E₂/E₄ = {result.energy_analysis['e2_ratio']:.2f}/{result.energy_analysis['e4_ratio']:.2f}\n\n")
            
            f.write("OPTIMIZED CONSTANTS:\n")
            f.write(f"  c₂: {result.optimized_constants['c2']:.6f}\n")
            f.write(f"  c₄: {result.optimized_constants['c4']:.6f}\n")
            f.write(f"  c₆: {result.optimized_constants['c6']:.6f}\n")
            f.write(f"  F_π: {result.optimized_constants['F_pi']:.1f} MeV\n")
            f.write(f"  e: {result.optimized_constants['e']:.3f}\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"  Execution time: {result.execution_time:.1f} seconds\n")
            f.write(f"  Iterations: {result.iterations}\n")
            f.write(f"  Strategy used: {result.convergence_info['strategy_used']}\n")
    
    def get_subcommands(self) -> list:
        """Get available subcommands."""
        return []
    
    def get_help(self) -> str:
        """Get help text."""
        return """
Universal solver for Skyrme field equations.

This command can solve for protons, neutrons, or any topological defect
using various optimization strategies.

Examples:
  # Basic proton calculation
  phaze-particles solve --grid-size 32 --box-size 4.0

  # Optimize for target radius
  phaze-particles solve --target-radius 0.841 --optimization-strategy physical_params

  # Optimize for target bands
  phaze-particles solve --target-bands 11 --optimization-strategy quantization

  # Full optimization with all targets
  phaze-particles solve --target-mass 938.272 --target-radius 0.841 --target-magnetic-moment 2.793
        """
