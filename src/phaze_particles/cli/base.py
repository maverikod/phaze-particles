#!/usr/bin/env python3
"""
Base command class for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseCommand(ABC):
    """
    Base class for all CLI commands.

    Provides common functionality for command execution, help system,
    and graceful shutdown handling.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize base command.

        Args:
            name: Command name
            description: Command description
        """
        self.name = name
        self.description = description
        self._shutdown_requested = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self._shutdown_requested = True
        sys.exit(0)

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """
        Add command-specific arguments to parser.

        Args:
            parser: Argument parser to add arguments to
        """
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """
        Execute the command.

        Args:
            args: Parsed command arguments

        Returns:
            Exit code (0 for success, non-zero for error)
        """
        pass

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create argument parser for this command.

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(
            prog=f"phaze-particles {self.name}",
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self.add_arguments(parser)
        return parser

    def run(self, args: Optional[list] = None) -> int:
        """
        Run the command with given arguments.

        Args:
            args: Command line arguments (if None, uses sys.argv)

        Returns:
            Exit code
        """
        try:
            parser = self.create_parser()
            parsed_args = parser.parse_args(args)
            return self.execute(parsed_args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
