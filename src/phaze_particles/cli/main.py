#!/usr/bin/env python3
"""
Main CLI entry point for Phaze-Particles.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import sys
from typing import Dict, Type

from .base import BaseCommand
from .commands.proton import ProtonCommand


class PhazeParticlesCLI:
    """
    Main CLI application for Phaze-Particles.

    Manages command registration, argument parsing, and command execution.
    """

    def __init__(self):
        """Initialize CLI application."""
        self.commands: Dict[str, Type[BaseCommand]] = {}
        self._register_commands()

    def _register_commands(self) -> None:
        """Register all available commands."""
        self.commands = {
            "proton": ProtonCommand,
            # Future commands will be added here
            # 'neutron': NeutronCommand,
        }

    def create_main_parser(self) -> argparse.ArgumentParser:
        """
        Create main argument parser.

        Returns:
            Configured main argument parser
        """
        parser = argparse.ArgumentParser(
            prog="phaze-particles",
            description="Elementary Particle Modeling Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  phaze-particles proton static --config config.yaml
  phaze-particles proton static --help
  phaze-particles --help

For more information, visit: https://github.com/vasilyvz/phaze-particles
            """,
        )

        parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

        subparsers = parser.add_subparsers(
            dest="command", help="Available commands", metavar="COMMAND"
        )

        # Add command subparsers
        for cmd_name, cmd_class in self.commands.items():
            cmd_instance = cmd_class()
            cmd_parser = subparsers.add_parser(
                cmd_name,
                help=cmd_instance.description,
                description=cmd_instance.description,
            )
            cmd_instance.add_arguments(cmd_parser)

        return parser

    def run(self, args: list = None) -> int:
        """
        Run the CLI application.

        Args:
            args: Command line arguments (if None, uses sys.argv[1:])

        Returns:
            Exit code
        """
        if args is None:
            args = sys.argv[1:]

        try:
            parser = self.create_main_parser()
            parsed_args = parser.parse_args(args)

            if not parsed_args.command:
                parser.print_help()
                return 1

            # Get command class and execute
            cmd_class = self.commands[parsed_args.command]
            cmd_instance = cmd_class()

            # Execute command directly with parsed args
            return cmd_instance.execute(parsed_args)

        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1


def main() -> int:
    """
    Main entry point for the CLI application.

    Returns:
        Exit code
    """
    cli = PhazeParticlesCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
