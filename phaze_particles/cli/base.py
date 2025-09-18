#!/usr/bin/env python3
"""
Base command class for Phaze-Particles CLI.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import argparse
import json
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        self._config: Dict[str, Any] = {}
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

    def get_subcommands(self) -> List[str]:
        """
        Get list of available subcommands.

        Returns:
            List of subcommand names
        """
        return []

    def get_help(self) -> str:
        """
        Get detailed help text for the command.

        Returns:
            Help text string
        """
        return self.description

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_file.suffix.lower() != ".json":
            raise ValueError(f"Configuration file must be JSON format: {config_path}")

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value

    def validate_config(self) -> bool:
        """
        Validate current configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

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
