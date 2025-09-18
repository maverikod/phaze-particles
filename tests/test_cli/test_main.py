#!/usr/bin/env python3
"""
Tests for main CLI functionality.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from phaze_particles.cli.main import PhazeParticlesCLI


def test_cli_initialization():
    """Test CLI initialization."""
    cli = PhazeParticlesCLI()
    assert "proton" in cli.commands
    assert len(cli.commands) >= 1


def test_help_command():
    """Test help command execution."""
    cli = PhazeParticlesCLI()

    with patch("sys.argv", ["phaze-particles", "--help"]):
        result = cli.run(["--help"])
        # Help should return 0 (success) or 1 (no command specified)
        assert result in [0, 1]


def test_version_command():
    """Test version command execution."""
    cli = PhazeParticlesCLI()

    with patch("sys.argv", ["phaze-particles", "--version"]):
        result = cli.run(["--version"])
        # Version should return 0 (success)
        assert result == 0


def test_proton_command_help():
    """Test proton command help."""
    cli = PhazeParticlesCLI()

    result = cli.run(["proton", "--help"])
    # Help should return 0 (success)
    assert result == 0


def test_proton_static_command():
    """Test proton static command execution."""
    cli = PhazeParticlesCLI()

    result = cli.run(["proton", "static"])
    # Command should execute successfully
    assert result == 0


if __name__ == "__main__":
    # Run tests if executed directly
    test_cli_initialization()
    test_help_command()
    test_version_command()
    test_proton_command_help()
    test_proton_static_command()
    print("All CLI tests passed!")
