#!/usr/bin/env python3
"""
Proton modeling commands for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import sys
from typing import Optional

from ..base import BaseCommand


class ProtonCommand(BaseCommand):
    """
    Proton modeling command with subcommands for different model types.
    """

    def __init__(self):
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
            dest="subcommand", help="Available proton model types", metavar="MODEL_TYPE"
        )

        # Add subcommand parsers
        for subcmd_name, subcmd_instance in self.subcommands.items():
            subcmd_parser = subparsers.add_parser(
                subcmd_name,
                help=subcmd_instance.description,
                description=subcmd_instance.description,
            )
            subcmd_instance.add_arguments(subcmd_parser)

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute proton command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        if not args.subcommand:
            print("Error: No subcommand specified. Use --help for available options.")
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

    def __init__(self):
        """Initialize static proton command."""
        super().__init__(
            name="proton static",
            description="Static proton model with three torus configurations (120°, clover, cartesian)",
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
            help="Output directory for results (default: proton_static_results)",
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

    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute static proton model.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code
        """
        try:
            print("Hello from proton_model (prototype).")
            print(f"Running static proton model with configuration: {args.config_type}")
            print(f"Grid size: {args.grid_size}")
            print(f"Box size: {args.box_size} fm")
            print(f"Output directory: {args.output}")

            if args.verbose:
                print("Verbose mode enabled")

            if args.save_data:
                print("Data saving enabled")

            if args.generate_plots:
                print("Plot generation enabled")

            if args.config:
                print(f"Using configuration file: {args.config}")

            # TODO: Implement actual proton model calculations
            # This is where the original proton_model.py logic will be integrated

            print("Static proton model execution completed successfully.")
            return 0

        except Exception as e:
            print(f"Error executing static proton model: {e}", file=sys.stderr)
            return 1
