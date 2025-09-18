#!/usr/bin/env python3
"""
Proton modeling commands for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import sys
from typing import List, Optional

from ..base import BaseCommand


class ProtonCommand(BaseCommand):
    """
    Proton modeling command with subcommands for different model types.
    """
    
    def __init__(self):
        """Initialize proton command."""
        super().__init__(
            name="proton",
            description="Proton modeling using topological soliton approaches"
        )
        self.subcommands = {
            'static': ProtonStaticCommand(),
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
            dest='subcommand',
            help='Available proton model types',
            metavar='MODEL_TYPE'
        )
        
        # Add subcommand parsers
        for subcmd_name, subcmd_instance in self.subcommands.items():
            subcmd_parser = subparsers.add_parser(
                subcmd_name,
                help=subcmd_instance.description,
                description=subcmd_instance.description
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
            description="Static proton model with three torus configurations (120°, clover, cartesian)"
        )
    
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add static proton command arguments.
        
        Args:
            parser: Argument parser to add arguments to
        """
        parser.add_argument(
            '--config',
            type=str,
            help='Configuration file path (YAML format)'
        )
        
        parser.add_argument(
            '--output',
            type=str,
            default='proton_static_results',
            help='Output directory for results (default: proton_static_results)'
        )
        
        parser.add_argument(
            '--grid-size',
            type=int,
            default=64,
            help='Grid size for numerical calculations (default: 64)'
        )
        
        parser.add_argument(
            '--box-size',
            type=float,
            default=4.0,
            help='Box size in femtometers (default: 4.0)'
        )
        
        parser.add_argument(
            '--config-type',
            type=str,
            choices=['120deg', 'clover', 'cartesian', 'all'],
            default='all',
            help='Torus configuration type to run (default: all)'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--save-data',
            action='store_true',
            help='Save numerical data to files'
        )
        
        parser.add_argument(
            '--generate-plots',
            action='store_true',
            help='Generate visualization plots'
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
        return f"{self.description}\n\nThis command implements the static proton model with three torus configurations."

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

            print("Hello from proton_model (prototype).")
            print(f"Running static proton model with configuration: {config_type}")
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

            # TODO: Implement actual proton model calculations
            # This is where the original proton_model.py logic will be integrated

            print("Static proton model execution completed successfully.")
            return 0

        except Exception as e:
            print(f"Error executing static proton model: {e}", file=sys.stderr)
            return 1
