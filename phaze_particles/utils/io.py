#!/usr/bin/env python3
"""
Input/output utilities for particle modeling.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_results(
    results: Dict[str, Any], output_path: str, format: str = "json"
) -> None:
    """
    Save model results to file.

    Args:
        results: Results dictionary to save
        output_path: Output file path
        format: Output format ('json' or 'yaml')
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    elif format.lower() == "yaml":
        # TODO: Implement YAML saving when yaml package is available
        raise NotImplementedError("YAML format not yet implemented")
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Configuration file path

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_file.suffix.lower() == ".json":
        with open(config_file, "r") as f:
            return json.load(f)
    else:
        # TODO: Implement YAML loading when yaml package is available
        raise NotImplementedError(
            f"Configuration format not supported: {config_file.suffix}"
        )


def ensure_output_directory(output_dir: str) -> Path:
    """
    Ensure output directory exists.

    Args:
        output_dir: Output directory path

    Returns:
        Path object for output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def generate_report_filename(
    model_name: str, config_type: str, timestamp: Optional[str] = None
) -> str:
    """
    Generate standardized report filename.

    Args:
        model_name: Name of the model
        config_type: Configuration type
        timestamp: Optional timestamp string

    Returns:
        Generated filename
    """
    if timestamp is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    return f"{timestamp}_{model_name}_{config_type}_report.md"
