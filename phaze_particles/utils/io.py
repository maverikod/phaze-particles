#!/usr/bin/env python3
"""
Input/output utilities for particle modeling.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


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


class CSVWriter:
    """
    CSV file writer with UTF-8 BOM support for Excel compatibility.
    """
    
    def __init__(self, file_path: Union[str, Path], encoding: str = "utf-8-sig"):
        """
        Initialize CSV writer.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding (default: utf-8-sig for Excel compatibility)
        """
        self.file_path = Path(file_path)
        self.encoding = encoding
        self._file = None
        self._writer = None
        self._headers_written = False
    
    def __enter__(self):
        """Context manager entry."""
        self._file = open(self.file_path, 'w', newline='', encoding=self.encoding)
        self._writer = csv.writer(self._file)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()
    
    def write_header(self, headers: List[str]) -> None:
        """
        Write CSV header row.
        
        Args:
            headers: List of column headers
        """
        if self._writer and not self._headers_written:
            self._writer.writerow(headers)
            self._headers_written = True
    
    def write_row(self, row: List[Any]) -> None:
        """
        Write a single row to CSV.
        
        Args:
            row: List of values to write
        """
        if self._writer:
            # Convert None values to empty strings
            processed_row = ["" if value is None else str(value) for value in row]
            self._writer.writerow(processed_row)
    
    def write_rows(self, rows: List[List[Any]]) -> None:
        """
        Write multiple rows to CSV.
        
        Args:
            rows: List of rows to write
        """
        for row in rows:
            self.write_row(row)


class JSONWriter:
    """
    JSON file writer with pretty formatting support.
    """
    
    def __init__(self, file_path: Union[str, Path], pretty: bool = True):
        """
        Initialize JSON writer.
        
        Args:
            file_path: Path to JSON file
            pretty: Whether to use pretty formatting
        """
        self.file_path = Path(file_path)
        self.pretty = pretty
    
    def write_data(self, data: Any) -> None:
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
        """
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        processed_data = self._process_data_for_json(data)
        
        with open(self.file_path, 'w', encoding='utf-8') as f:
            if self.pretty:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(processed_data, f, ensure_ascii=False)
    
    def _process_data_for_json(self, data: Any) -> Any:
        """Process data for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: self._process_data_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._process_data_for_json(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        else:
            return data


class ResultsManager:
    """
    Manager for organizing and saving model results.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize results manager.
        
        Args:
            base_dir: Base directory for results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_results_directory(self, command: str, subcommand: str) -> Path:
        """
        Create results directory for command/subcommand.
        
        Args:
            command: Main command name
            subcommand: Subcommand name
            
        Returns:
            Path to created directory
        """
        results_dir = self.base_dir / command / subcommand
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def generate_filename(self, short_desc: str, timestamp: Optional[str] = None) -> str:
        """
        Generate standardized filename.
        
        Args:
            short_desc: Short description of parameters
            timestamp: Optional timestamp
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
        
        return f"-{short_desc}-{timestamp}.csv"
    
    def save_results_csv(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save results to CSV file.
        
        Args:
            data: Results data
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with CSVWriter(file_path) as writer:
            # Write headers
            headers = list(data.keys())
            writer.write_header(headers)
            
            # Write data row
            row = [data.get(header, "") for header in headers]
            writer.write_row(row)
    
    def save_results_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save results to JSON file.
        
        Args:
            data: Results data
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = JSONWriter(file_path, pretty=True)
        writer.write_data(data)
    
    def list_results(self, command: Optional[str] = None, subcommand: Optional[str] = None) -> List[Path]:
        """
        List result files.
        
        Args:
            command: Optional command filter
            subcommand: Optional subcommand filter
            
        Returns:
            List of result file paths
        """
        if command and subcommand:
            search_dir = self.base_dir / command / subcommand
        elif command:
            search_dir = self.base_dir / command
        else:
            search_dir = self.base_dir
        
        if not search_dir.exists():
            return []
        
        result_files = []
        for file_path in search_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.csv', '.json']:
                result_files.append(file_path)
        
        return sorted(result_files)
    
    def cleanup_old_results(self, days: int = 30) -> int:
        """
        Clean up old result files.
        
        Args:
            days: Number of days to keep files
            
        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in self.list_results():
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        return deleted_count


class ConfigLoader:
    """
    Configuration file loader with validation and defaults.
    """
    
    def __init__(self):
        """Initialize config loader."""
        self._defaults: Dict[str, Any] = {}
    
    def load_config_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {file_path.suffix}")
        
        return self._merge_with_defaults(config)
    
    def load_config_with_defaults(self, file_path: Union[str, Path], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration with defaults.
        
        Args:
            file_path: Path to configuration file
            defaults: Default values
            
        Returns:
            Merged configuration dictionary
        """
        self._defaults = defaults
        return self.load_config_from_file(file_path)
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Validation schema
            
        Returns:
            True if valid, False otherwise
        """
        # Simple validation - check required keys
        required_keys = schema.get('required', [])
        for key in required_keys:
            if key not in config:
                return False
        
        return True
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration with defaults."""
        if not self._defaults:
            return config
        
        return self.merge_configs(self._defaults, config)


class FileManager:
    """
    File and directory management utilities.
    """
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize file manager.
        
        Args:
            base_dir: Base directory for operations
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def file_exists(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        return Path(file_path).exists()
    
    def directory_exists(self, dir_path: Union[str, Path]) -> bool:
        """
        Check if directory exists.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            True if directory exists, False otherwise
        """
        return Path(dir_path).is_dir()
    
    def create_directory(self, dir_path: Union[str, Path]) -> Path:
        """
        Create directory.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            Path to created directory
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def create_nested_directory(self, dir_path: Union[str, Path]) -> Path:
        """
        Create nested directory structure.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            Path to created directory
        """
        return self.create_directory(dir_path)
    
    def get_file_size(self, file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    def list_files(self, dir_path: Union[str, Path], pattern: str = "*") -> List[Path]:
        """
        List files in directory.
        
        Args:
            dir_path: Directory path
            pattern: File pattern to match
            
        Returns:
            List of file paths
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []
        
        return list(dir_path.glob(pattern))
    
    def list_directories(self, dir_path: Union[str, Path]) -> List[Path]:
        """
        List directories in directory.
        
        Args:
            dir_path: Directory path
            
        Returns:
            List of directory paths
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []
        
        return [item for item in dir_path.iterdir() if item.is_dir()]
    
    def backup_file(self, file_path: Union[str, Path], backup_suffix: str = ".bak") -> Path:
        """
        Create backup of file.
        
        Args:
            file_path: Path to file
            backup_suffix: Backup file suffix
            
        Returns:
            Path to backup file
        """
        file_path = Path(file_path)
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def cleanup_directory(self, dir_path: Union[str, Path], keep_files: Optional[List[str]] = None) -> int:
        """
        Clean up directory.
        
        Args:
            dir_path: Directory path
            keep_files: List of files to keep
            
        Returns:
            Number of files deleted
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return 0
        
        keep_files = keep_files or []
        deleted_count = 0
        
        for item in dir_path.iterdir():
            if item.name not in keep_files:
                if item.is_file():
                    item.unlink()
                    deleted_count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    deleted_count += 1
        
        return deleted_count


class DataExporter:
    """
    Data export utilities for various formats.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataframe(self, df: pd.DataFrame, filename: str, format: str = "csv") -> Path:
        """
        Export DataFrame to file.
        
        Args:
            df: DataFrame to export
            filename: Output filename
            format: Export format ('csv' or 'json')
            
        Returns:
            Path to exported file
        """
        file_path = self.output_dir / filename
        
        if format.lower() == "csv":
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif format.lower() == "json":
            df.to_json(file_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return file_path
    
    def export_numpy_array(self, array: np.ndarray, filename: str, format: str = "npy") -> Path:
        """
        Export NumPy array to file.
        
        Args:
            array: Array to export
            filename: Output filename
            format: Export format ('npy' or 'csv')
            
        Returns:
            Path to exported file
        """
        file_path = self.output_dir / filename
        
        if format.lower() == "npy":
            np.save(file_path, array)
        elif format.lower() == "csv":
            np.savetxt(file_path, array, delimiter=',')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return file_path
    
    def export_metadata(self, metadata: Dict[str, Any], filename: str) -> Path:
        """
        Export metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        file_path = self.output_dir / filename
        writer = JSONWriter(file_path, pretty=True)
        writer.write_data(metadata)
        return file_path
    
    def export_plot(self, plot_data: Any, filename: str, format: str = "png") -> Path:
        """
        Export plot to file.
        
        Args:
            plot_data: Plot object or data
            filename: Output filename
            format: Export format
            
        Returns:
            Path to exported file
        """
        file_path = self.output_dir / filename
        
        # This is a placeholder - actual implementation would depend on plotting library
        # For now, just save plot data as JSON
        if hasattr(plot_data, 'savefig'):
            plot_data.savefig(file_path, format=format)
        else:
            # Save as JSON if not a plot object
            writer = JSONWriter(file_path.with_suffix('.json'), pretty=True)
            writer.write_data(plot_data)
        
        return file_path


class DataImporter:
    """
    Data import utilities for various formats.
    """
    
    def __init__(self, input_dir: Union[str, Path]):
        """
        Initialize data importer.
        
        Args:
            input_dir: Input directory
        """
        self.input_dir = Path(input_dir)
    
    def import_dataframe(self, filename: str, format: str = "csv") -> pd.DataFrame:
        """
        Import DataFrame from file.
        
        Args:
            filename: Input filename
            format: Import format ('csv' or 'json')
            
        Returns:
            Imported DataFrame
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if format.lower() == "csv":
            return pd.read_csv(file_path, encoding='utf-8-sig')
        elif format.lower() == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def import_numpy_array(self, filename: str, format: str = "npy") -> np.ndarray:
        """
        Import NumPy array from file.
        
        Args:
            filename: Input filename
            format: Import format ('npy' or 'csv')
            
        Returns:
            Imported array
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if format.lower() == "npy":
            return np.load(file_path)
        elif format.lower() == "csv":
            return np.loadtxt(file_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported import format: {format}")
    
    def import_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Import metadata from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            Metadata dictionary
        """
        file_path = self.input_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def import_invalid_file(self, filename: str) -> None:
        """
        Test import of invalid file (should raise exception).
        
        Args:
            filename: Invalid filename
            
        Raises:
            Exception: Always raises an exception for testing
        """
        file_path = self.input_dir / filename
        raise ValueError(f"Invalid file format: {file_path}")
