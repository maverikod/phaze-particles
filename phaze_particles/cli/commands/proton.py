#!/usr/bin/env python3
"""
Proton modeling commands for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import sys
import os
from datetime import datetime
from typing import List

from ..base import BaseCommand
from ...models.proton_integrated import (
    ProtonModel,
    ModelConfig,
    ModelStatus,
)
from ...utils.cuda import get_cuda_status
from ...utils.progress import create_performance_monitor
from ...utils.io import save_results


class ProtonCommand(BaseCommand):
    """
    Proton modeling command with subcommands for different model types.
    """

    def __init__(self) -> None:
        """Initialize proton command."""
        super().__init__(
            name="proton",
            description="Proton modeling using topological soliton approaches",
        )
        self.subcommands = {
            "static": ProtonStaticCommand(),
            # Future subcommands will be added here
            # 'dynamic': ProtonDynamicCommand(),
        }

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add proton command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        subparsers = parser.add_subparsers(
            dest="subcommand",
            help="Available proton model types",
            metavar="MODEL_TYPE",
        )

        # Add subcommand parsers
        for subcmd_name, subcmd_instance in self.subcommands.items():
            subcmd_parser = subparsers.add_parser(
                subcmd_name,
                help=subcmd_instance.description,
                description=subcmd_instance.description,
            )
            subcmd_instance.add_arguments(subcmd_parser)

    def get_subcommands(self) -> List[str]:
        """
        Get list of available subcommands.

        Returns:
            List of subcommand names
        """
        return list(self.subcommands.keys())

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        help_text = f"{self.description}\n\n"
        help_text += "Available subcommands:\n"
        for subcmd_name, subcmd_instance in self.subcommands.items():
            help_text += f"  {subcmd_name}: {subcmd_instance.description}\n"
        return help_text

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute proton command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        if not args.subcommand:
            print(
                "Error: No subcommand specified. " "Use --help for available options."
            )
            return 1

        if args.subcommand not in self.subcommands:
            print(f"Error: Unknown subcommand '{args.subcommand}'")
            return 1

        # Execute the subcommand
        subcmd_instance = self.subcommands[args.subcommand]
        return subcmd_instance.execute(args)


class ProtonStaticCommand(BaseCommand):
    """
    Static proton model command based on three torus configurations.
    """

    def __init__(self) -> None:
        """Initialize static proton command."""
        super().__init__(
            name="proton static",
            description=(
                "Static proton model with three torus configurations "
                "(120°, clover, cartesian)"
            ),
        )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add static proton command arguments.

        Args:
            parser: Argument parser to add arguments to
        """
        parser.add_argument(
            "--config", type=str, help="Configuration file path (YAML format)"
        )

        parser.add_argument(
            "--output",
            type=str,
            default="proton_static_results",
            help=("Output directory for results " "(default: proton_static_results)"),
        )

        parser.add_argument(
            "--grid-size",
            type=int,
            default=64,
            help="Grid size for numerical calculations (default: 64)",
        )

        parser.add_argument(
            "--box-size",
            type=float,
            default=4.0,
            help="Box size in femtometers (default: 4.0)",
        )

        parser.add_argument(
            "--config-type",
            type=str,
            choices=["120deg", "clover", "cartesian", "all"],
            default="all",
            help="Torus configuration type to run (default: all)",
        )

        parser.add_argument(
            "--verbose", action="store_true", help="Enable verbose output"
        )

        parser.add_argument(
            "--save-data", action="store_true", help="Save numerical data to files"
        )

        parser.add_argument(
            "--generate-plots", action="store_true", help="Generate visualization plots"
        )

    def get_subcommands(self) -> List[str]:
        """
        Get list of available subcommands.

        Returns:
            List of subcommand names (empty for leaf commands)
        """
        return []

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        return (
            f"{self.description}\n\n"
            f"This command implements the static proton model "
            f"with three torus configurations."
        )

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate grid size
        grid_size = self.get_config("grid_size", 64)
        if not isinstance(grid_size, int) or grid_size < 16:
            return False

        # Validate box size
        box_size = self.get_config("box_size", 4.0)
        if not isinstance(box_size, (int, float)) or box_size <= 0:
            return False

        # Validate config type
        config_type = self.get_config("config_type", "all")
        if config_type not in ["120deg", "clover", "cartesian", "all"]:
            return False

        return True

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute static proton model.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        try:
            # Load configuration if provided
            if args.config:
                self.load_config(args.config)
                if not self.validate_config():
                    print("Error: Invalid configuration file", file=sys.stderr)
                    return 1

            # Use config values or command line arguments
            grid_size = self.get_config("grid_size", args.grid_size)
            box_size = self.get_config("box_size", args.box_size)
            config_type = self.get_config("config_type", args.config_type)
            output_dir = self.get_config("output", args.output)
            verbose = self.get_config("verbose", args.verbose)
            save_data = self.get_config("save_data", args.save_data)
            generate_plots = self.get_config("generate_plots", args.generate_plots)

            # Display CUDA status
            print("Running integrated proton model...")
            print(get_cuda_status())
            print(f"Configuration: {config_type}")
            print(f"Grid size: {grid_size}")
            print(f"Box size: {box_size} fm")
            print(f"Output directory: {output_dir}")

            if verbose:
                print("Verbose mode enabled")

            if save_data:
                print("Data saving enabled")

            if generate_plots:
                print("Plot generation enabled")

            if args.config:
                print(f"Using configuration file: {args.config}")

            # Create model configuration
            model_config = ModelConfig(
                grid_size=grid_size,
                box_size=box_size,
                torus_config=config_type,
                validation_enabled=True,
                save_reports=True,
                output_dir=output_dir,
            )

            # Create and run model
            with create_performance_monitor("Proton Static Model") as monitor:
                # Create model
                model = ProtonModel(model_config)

                # Run model with progress tracking
                print("Starting proton model calculation...")
                results = model.run()

                # Add performance metrics
                monitor.add_metric("grid_size", grid_size)
                monitor.add_metric("box_size", box_size)
                monitor.add_metric("config_type", config_type)
                monitor.add_metric("iterations", results.iterations)
                monitor.add_metric("execution_time", results.execution_time)

            # Check if model was successful
            if results.status == ModelStatus.FAILED:
                print(
                    f"Model execution failed: {results.error_message}",
                    file=sys.stderr,
                )
                return 1

            # Print results
            print("\n=== PROTON MODEL RESULTS ===")
            print(f"Status: {results.status.value}")
            print(f"Execution time: {results.execution_time:.2f} seconds")
            print(f"Iterations: {results.iterations}")
            print(f"Converged: {results.converged}")
            print()
            print("Physical Parameters:")
            print(f"  Proton mass: {results.proton_mass:.3f} MeV")
            print(f"  Charge radius: {results.charge_radius:.3f} fm")
            print(f"  Magnetic moment: {results.magnetic_moment:.3f} μN")
            print(f"  Electric charge: {results.electric_charge:.3f} e")
            print(f"  Baryon number: {results.baryon_number:.3f}")
            print(f"  Energy balance: {results.energy_balance:.3f}")
            print(f"  Total energy: {results.total_energy:.3f} MeV")

            # Print validation results
            if results.validation_status:
                print(f"\nValidation Status: {results.validation_status}")
                if results.validation_score is not None:
                    print(f"Validation Score: {results.validation_score:.3f}")

            # Save results to CSV
            if save_data:
                self._save_results_to_csv(
                    results, output_dir, config_type, grid_size, box_size
                )

            # Save model results
            results_path = os.path.join(output_dir, "model_results.json")
            os.makedirs(output_dir, exist_ok=True)
            results.save_to_file(results_path)
            print(f"\nResults saved to: {results_path}")

            print("\nStatic proton model execution completed successfully.")
            return 0

        except Exception as e:
            print(f"Error executing static proton model: {e}", file=sys.stderr)
            return 1

    def _save_results_to_csv(
        self,
        results,
        output_dir: str,
        config_type: str,
        grid_size: int,
        box_size: float,
    ) -> None:
        """
        Save results to CSV file.

        Args:
            results: Model results
            output_dir: Output directory
            config_type: Configuration type
            grid_size: Grid size
            box_size: Box size
        """
        try:
            # Create results directory
            results_dir = os.path.join(output_dir, "proton", "static")
            os.makedirs(results_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
            short_desc = f"grid{grid_size}-box{box_size}-{config_type}"
            filename = f"-{short_desc}-{timestamp}.csv"
            csv_path = os.path.join(results_dir, filename)

            # Prepare data for CSV
            csv_data = {
                "timestamp": datetime.now().isoformat(),
                "grid_size": grid_size,
                "box_size": box_size,
                "config_type": config_type,
                "status": results.status.value,
                "execution_time": results.execution_time,
                "iterations": results.iterations,
                "converged": results.converged,
                "proton_mass": results.proton_mass,
                "charge_radius": results.charge_radius,
                "magnetic_moment": results.magnetic_moment,
                "electric_charge": results.electric_charge,
                "baryon_number": results.baryon_number,
                "energy_balance": results.energy_balance,
                "total_energy": results.total_energy,
                "validation_status": results.validation_status or "N/A",
                "validation_score": results.validation_score or "N/A",
            }

            # Save CSV
            save_results(csv_data, csv_path)
            print(f"CSV results saved to: {csv_path}")

        except Exception as e:
            print(f"Warning: Failed to save CSV results: {e}", file=sys.stderr)
