#!/usr/bin/env python3
"""
Application settings and configuration management.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


class Settings:
    """
    Application settings and configuration management.

    Handles loading and validation of configuration parameters
    for the Phaze-Particles application.
    """

    def __init__(self):
        """Initialize settings with default values."""
        self._settings: Dict[str, Any] = {
            # Default physical constants
            "proton_mass": 938.272,  # MeV
            "electron_charge": 1.0,  # e
            "proton_radius": 0.84,  # fm
            "proton_magnetic_moment": 2.793,  # μN
            # Default numerical parameters
            "default_grid_size": 64,
            "default_box_size": 4.0,  # fm
            "convergence_tolerance": 1e-6,
            "max_iterations": 1000,
            # Default output settings
            "output_format": "yaml",
            "save_plots": True,
            "save_data": True,
            "verbose": False,
            # Default paths
            "config_dir": Path.home() / ".phaze-particles",
            "output_dir": Path.cwd() / "output",
            "reports_dir": Path.cwd() / "docs" / "reports",
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get setting value by key.

        Args:
            key: Setting key
            default: Default value if key not found

        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set setting value.

        Args:
            key: Setting key
            value: Setting value
        """
        self._settings[key] = value

    def load_from_file(self, config_path: str) -> None:
        """
        Load settings from configuration file.

        Args:
            config_path: Path to configuration file
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # TODO: Implement YAML/JSON configuration loading
        # For now, just validate that the file exists
        print(f"Loading configuration from: {config_path}")

    def validate(self) -> bool:
        """
        Validate current settings.

        Returns:
            True if settings are valid, False otherwise
        """
        # Validate physical constants
        if self._settings["proton_mass"] <= 0:
            return False

        if self._settings["proton_radius"] <= 0:
            return False

        # Validate numerical parameters
        if self._settings["default_grid_size"] < 16:
            return False

        if self._settings["default_box_size"] <= 0:
            return False

        return True

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        for dir_key in ["config_dir", "output_dir", "reports_dir"]:
            dir_path = self._settings[dir_key]
            if isinstance(dir_path, Path):
                dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
