#!/usr/bin/env python3
"""
Unit tests for IO utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
import os
import json
import csv
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from phaze_particles.utils.io import (
    CSVWriter,
    JSONWriter,
    ResultsManager,
    ConfigLoader,
    FileManager,
    DataExporter,
    DataImporter
)


class TestCSVWriter(unittest.TestCase):
    """Test CSVWriter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.csv_file = os.path.join(self.temp_dir, "test.csv")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_csv_writer_initialization(self):
        """Test CSV writer initialization."""
        writer = CSVWriter(self.csv_file)
        self.assertEqual(writer.filepath, self.csv_file)
        self.assertIsNone(writer.file_handle)

    def test_csv_writer_context_manager(self):
        """Test CSV writer as context manager."""
        with CSVWriter(self.csv_file) as writer:
            self.assertIsNotNone(writer.file_handle)
            self.assertTrue(os.path.exists(self.csv_file))

    def test_csv_writer_write_header(self):
        """Test writing CSV header."""
        headers = ["column1", "column2", "column3"]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("column1", content)
            self.assertIn("column2", content)
            self.assertIn("column3", content)

    def test_csv_writer_write_row(self):
        """Test writing CSV row."""
        headers = ["column1", "column2", "column3"]
        row_data = ["value1", "value2", "value3"]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
            writer.write_row(row_data)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("value1", content)
            self.assertIn("value2", content)
            self.assertIn("value3", content)

    def test_csv_writer_write_multiple_rows(self):
        """Test writing multiple CSV rows."""
        headers = ["column1", "column2", "column3"]
        rows_data = [
            ["value1", "value2", "value3"],
            ["value4", "value5", "value6"],
            ["value7", "value8", "value9"]
        ]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
            for row in rows_data:
                writer.write_row(row)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            for row in rows_data:
                for value in row:
                    self.assertIn(value, content)

    def test_csv_writer_numeric_data(self):
        """Test writing numeric data."""
        headers = ["int_value", "float_value", "scientific_value"]
        row_data = [42, 3.14159, 1.23e-4]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
            writer.write_row(row_data)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("42", content)
            self.assertIn("3.14159", content)
            self.assertIn("1.23e-04", content)

    def test_csv_writer_unicode_data(self):
        """Test writing unicode data."""
        headers = ["unicode_column"]
        row_data = ["тест", "αβγ", "🚀"]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
            writer.write_row(row_data)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("тест", content)
            self.assertIn("αβγ", content)
            self.assertIn("🚀", content)

    def test_csv_writer_missing_values(self):
        """Test writing missing values."""
        headers = ["column1", "column2", "column3"]
        row_data = ["value1", None, "value3"]
        
        with CSVWriter(self.csv_file) as writer:
            writer.write_header(headers)
            writer.write_row(row_data)
        
        # Check file content
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("value1", content)
            self.assertIn("value3", content)
            # Check that None is handled properly
            lines = content.strip().split('\n')
            self.assertEqual(len(lines), 2)  # Header + 1 data row


class TestJSONWriter(unittest.TestCase):
    """Test JSONWriter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_file = os.path.join(self.temp_dir, "test.json")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_json_writer_initialization(self):
        """Test JSON writer initialization."""
        writer = JSONWriter(self.json_file)
        self.assertEqual(writer.filepath, self.json_file)

    def test_json_writer_write_data(self):
        """Test writing JSON data."""
        test_data = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14159,
            "list_value": [1, 2, 3],
            "nested_dict": {"key": "value"}
        }
        
        writer = JSONWriter(self.json_file)
        writer.write_data(test_data)
        
        # Check file content
        with open(self.json_file, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data, test_data)

    def test_json_writer_write_numpy_data(self):
        """Test writing numpy data."""
        test_data = {
            "numpy_array": np.array([1, 2, 3, 4, 5]),
            "numpy_float": np.float64(3.14159),
            "numpy_int": np.int32(42)
        }
        
        writer = JSONWriter(self.json_file)
        writer.write_data(test_data)
        
        # Check file content
        with open(self.json_file, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data["numpy_array"], [1, 2, 3, 4, 5])
            self.assertEqual(loaded_data["numpy_float"], 3.14159)
            self.assertEqual(loaded_data["numpy_int"], 42)

    def test_json_writer_pretty_formatting(self):
        """Test JSON pretty formatting."""
        test_data = {"key1": "value1", "key2": "value2"}
        
        writer = JSONWriter(self.json_file, pretty=True)
        writer.write_data(test_data)
        
        # Check file content
        with open(self.json_file, 'r') as f:
            content = f.read()
            # Pretty formatting should include newlines and indentation
            self.assertIn('\n', content)
            self.assertIn('  ', content)  # Indentation


class TestResultsManager(unittest.TestCase):
    """Test ResultsManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_manager = ResultsManager(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_results_manager_initialization(self):
        """Test results manager initialization."""
        self.assertEqual(self.results_manager.base_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_create_results_directory(self):
        """Test creating results directory."""
        command = "proton"
        subcommand = "static"
        
        results_dir = self.results_manager.create_results_directory(command, subcommand)
        
        expected_path = os.path.join(self.temp_dir, command, subcommand)
        self.assertEqual(results_dir, expected_path)
        self.assertTrue(os.path.exists(results_dir))

    def test_generate_filename(self):
        """Test filename generation."""
        command = "proton"
        subcommand = "static"
        short_desc = "grid64-box4.0-all"
        
        filename = self.results_manager.generate_filename(
            command, subcommand, short_desc
        )
        
        # Check filename format
        self.assertIn("grid64-box4.0-all", filename)
        self.assertIn(".csv", filename)
        self.assertIn("T", filename)  # ISO timestamp format

    def test_save_results_csv(self):
        """Test saving results to CSV."""
        command = "proton"
        subcommand = "static"
        short_desc = "test"
        
        results_data = {
            "proton_mass": 938.272,
            "charge_radius": 0.841,
            "magnetic_moment": 2.793,
            "electric_charge": 1.0,
            "baryon_number": 1.0
        }
        
        filepath = self.results_manager.save_results_csv(
            command, subcommand, short_desc, results_data
        )
        
        self.assertTrue(os.path.exists(filepath))
        
        # Check CSV content
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            self.assertIn("proton_mass", content)
            self.assertIn("938.272", content)

    def test_save_results_json(self):
        """Test saving results to JSON."""
        command = "proton"
        subcommand = "static"
        short_desc = "test"
        
        results_data = {
            "proton_mass": 938.272,
            "charge_radius": 0.841,
            "magnetic_moment": 2.793,
            "electric_charge": 1.0,
            "baryon_number": 1.0
        }
        
        filepath = self.results_manager.save_results_json(
            command, subcommand, short_desc, results_data
        )
        
        self.assertTrue(os.path.exists(filepath))
        
        # Check JSON content
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data, results_data)

    def test_list_results(self):
        """Test listing results."""
        # Create some test results
        self.results_manager.save_results_csv(
            "proton", "static", "test1", {"mass": 938.272}
        )
        self.results_manager.save_results_csv(
            "proton", "static", "test2", {"mass": 938.272}
        )
        
        results = self.results_manager.list_results("proton", "static")
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all("test" in result for result in results))

    def test_cleanup_old_results(self):
        """Test cleaning up old results."""
        # Create some test results
        self.results_manager.save_results_csv(
            "proton", "static", "test1", {"mass": 938.272}
        )
        
        # Cleanup old results (older than 0 days)
        self.results_manager.cleanup_old_results("proton", "static", days=0)
        
        # Results should be cleaned up
        results = self.results_manager.list_results("proton", "static")
        self.assertEqual(len(results), 0)


class TestConfigLoader(unittest.TestCase):
    """Test ConfigLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_config_loader_initialization(self):
        """Test config loader initialization."""
        loader = ConfigLoader()
        self.assertIsNotNone(loader)

    def test_load_config_from_file(self):
        """Test loading config from file."""
        test_config = {
            "grid_size": 64,
            "box_size": 4.0,
            "max_iterations": 1000,
            "validation_enabled": True
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        loader = ConfigLoader()
        loaded_config = loader.load_config(self.config_file)
        
        self.assertEqual(loaded_config, test_config)

    def test_load_config_with_defaults(self):
        """Test loading config with defaults."""
        test_config = {
            "grid_size": 64,
            "box_size": 4.0
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)
        
        defaults = {
            "max_iterations": 1000,
            "validation_enabled": True,
            "tolerance": 1e-6
        }
        
        loader = ConfigLoader()
        loaded_config = loader.load_config(self.config_file, defaults=defaults)
        
        self.assertEqual(loaded_config["grid_size"], 64)
        self.assertEqual(loaded_config["box_size"], 4.0)
        self.assertEqual(loaded_config["max_iterations"], 1000)
        self.assertEqual(loaded_config["validation_enabled"], True)
        self.assertEqual(loaded_config["tolerance"], 1e-6)

    def test_load_config_file_not_found(self):
        """Test loading config from non-existent file."""
        loader = ConfigLoader()
        
        with self.assertRaises(FileNotFoundError):
            loader.load_config("nonexistent_config.json")

    def test_load_config_invalid_json(self):
        """Test loading config from invalid JSON file."""
        with open(self.config_file, 'w') as f:
            f.write("invalid json content")
        
        loader = ConfigLoader()
        
        with self.assertRaises(json.JSONDecodeError):
            loader.load_config(self.config_file)

    def test_validate_config(self):
        """Test config validation."""
        loader = ConfigLoader()
        
        # Valid config
        valid_config = {
            "grid_size": 64,
            "box_size": 4.0,
            "max_iterations": 1000
        }
        
        self.assertTrue(loader.validate_config(valid_config))
        
        # Invalid config
        invalid_config = {
            "grid_size": -1,  # Invalid negative value
            "box_size": 4.0
        }
        
        self.assertFalse(loader.validate_config(invalid_config))

    def test_merge_configs(self):
        """Test merging configs."""
        loader = ConfigLoader()
        
        base_config = {
            "grid_size": 64,
            "box_size": 4.0,
            "max_iterations": 1000
        }
        
        override_config = {
            "grid_size": 128,
            "validation_enabled": True
        }
        
        merged_config = loader.merge_configs(base_config, override_config)
        
        self.assertEqual(merged_config["grid_size"], 128)  # Overridden
        self.assertEqual(merged_config["box_size"], 4.0)  # From base
        self.assertEqual(merged_config["max_iterations"], 1000)  # From base
        self.assertEqual(merged_config["validation_enabled"], True)  # From override


class TestFileManager(unittest.TestCase):
    """Test FileManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.file_manager = FileManager(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_file_manager_initialization(self):
        """Test file manager initialization."""
        self.assertEqual(self.file_manager.base_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_create_directory(self):
        """Test directory creation."""
        dir_path = os.path.join(self.temp_dir, "test_dir")
        
        self.file_manager.create_directory(dir_path)
        
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

    def test_create_nested_directory(self):
        """Test nested directory creation."""
        dir_path = os.path.join(self.temp_dir, "level1", "level2", "level3")
        
        self.file_manager.create_directory(dir_path)
        
        self.assertTrue(os.path.exists(dir_path))
        self.assertTrue(os.path.isdir(dir_path))

    def test_file_exists(self):
        """Test file existence check."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        
        # File doesn't exist
        self.assertFalse(self.file_manager.file_exists(test_file))
        
        # Create file
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # File exists
        self.assertTrue(self.file_manager.file_exists(test_file))

    def test_directory_exists(self):
        """Test directory existence check."""
        test_dir = os.path.join(self.temp_dir, "test_dir")
        
        # Directory doesn't exist
        self.assertFalse(self.file_manager.directory_exists(test_dir))
        
        # Create directory
        os.makedirs(test_dir)
        
        # Directory exists
        self.assertTrue(self.file_manager.directory_exists(test_dir))

    def test_get_file_size(self):
        """Test getting file size."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "test content"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        file_size = self.file_manager.get_file_size(test_file)
        
        self.assertEqual(file_size, len(test_content))

    def test_list_files(self):
        """Test listing files."""
        # Create some test files
        test_files = ["file1.txt", "file2.txt", "file3.txt"]
        for filename in test_files:
            filepath = os.path.join(self.temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write("test content")
        
        listed_files = self.file_manager.list_files(self.temp_dir)
        
        self.assertEqual(len(listed_files), 3)
        for filename in test_files:
            self.assertIn(filename, listed_files)

    def test_list_directories(self):
        """Test listing directories."""
        # Create some test directories
        test_dirs = ["dir1", "dir2", "dir3"]
        for dirname in test_dirs:
            dirpath = os.path.join(self.temp_dir, dirname)
            os.makedirs(dirpath)
        
        listed_dirs = self.file_manager.list_directories(self.temp_dir)
        
        self.assertEqual(len(listed_dirs), 3)
        for dirname in test_dirs:
            self.assertIn(dirname, listed_dirs)

    def test_cleanup_directory(self):
        """Test directory cleanup."""
        # Create some test files and directories
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_dir = os.path.join(self.temp_dir, "test_dir")
        
        with open(test_file, 'w') as f:
            f.write("test content")
        os.makedirs(test_dir)
        
        # Cleanup directory
        self.file_manager.cleanup_directory(self.temp_dir)
        
        # Directory should be empty
        self.assertEqual(len(os.listdir(self.temp_dir)), 0)

    def test_backup_file(self):
        """Test file backup."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        test_content = "test content"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        backup_path = self.file_manager.backup_file(test_file)
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(os.path.exists(test_file))  # Original should still exist
        
        # Check backup content
        with open(backup_path, 'r') as f:
            backup_content = f.read()
            self.assertEqual(backup_content, test_content)


class TestDataExporter(unittest.TestCase):
    """Test DataExporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = DataExporter(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_data_exporter_initialization(self):
        """Test data exporter initialization."""
        self.assertEqual(self.exporter.output_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_export_numpy_array(self):
        """Test exporting numpy array."""
        test_array = np.array([1, 2, 3, 4, 5])
        
        filepath = self.exporter.export_numpy_array(test_array, "test_array")
        
        self.assertTrue(os.path.exists(filepath))
        
        # Load and check data
        loaded_array = np.load(filepath)
        np.testing.assert_array_equal(loaded_array, test_array)

    def test_export_dataframe(self):
        """Test exporting dataframe."""
        import pandas as pd
        
        test_data = {
            "column1": [1, 2, 3, 4, 5],
            "column2": ["a", "b", "c", "d", "e"],
            "column3": [1.1, 2.2, 3.3, 4.4, 5.5]
        }
        df = pd.DataFrame(test_data)
        
        filepath = self.exporter.export_dataframe(df, "test_dataframe")
        
        self.assertTrue(os.path.exists(filepath))
        
        # Load and check data
        loaded_df = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_export_plot(self):
        """Test exporting plot."""
        import matplotlib.pyplot as plt
        
        # Create simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
        
        filepath = self.exporter.export_plot(fig, "test_plot")
        
        self.assertTrue(os.path.exists(filepath))
        
        plt.close(fig)

    def test_export_metadata(self):
        """Test exporting metadata."""
        metadata = {
            "experiment_name": "test_experiment",
            "timestamp": "2024-01-15T14:30:25",
            "parameters": {
                "grid_size": 64,
                "box_size": 4.0
            }
        }
        
        filepath = self.exporter.export_metadata(metadata, "test_metadata")
        
        self.assertTrue(os.path.exists(filepath))
        
        # Load and check data
        with open(filepath, 'r') as f:
            loaded_metadata = json.load(f)
            self.assertEqual(loaded_metadata, metadata)


class TestDataImporter(unittest.TestCase):
    """Test DataImporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.importer = DataImporter(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_data_importer_initialization(self):
        """Test data importer initialization."""
        self.assertEqual(self.importer.input_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))

    def test_import_numpy_array(self):
        """Test importing numpy array."""
        test_array = np.array([1, 2, 3, 4, 5])
        filepath = os.path.join(self.temp_dir, "test_array.npy")
        
        np.save(filepath, test_array)
        
        loaded_array = self.importer.import_numpy_array("test_array.npy")
        
        np.testing.assert_array_equal(loaded_array, test_array)

    def test_import_dataframe(self):
        """Test importing dataframe."""
        import pandas as pd
        
        test_data = {
            "column1": [1, 2, 3, 4, 5],
            "column2": ["a", "b", "c", "d", "e"],
            "column3": [1.1, 2.2, 3.3, 4.4, 5.5]
        }
        df = pd.DataFrame(test_data)
        filepath = os.path.join(self.temp_dir, "test_dataframe.csv")
        
        df.to_csv(filepath, index=False)
        
        loaded_df = self.importer.import_dataframe("test_dataframe.csv")
        
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_import_metadata(self):
        """Test importing metadata."""
        metadata = {
            "experiment_name": "test_experiment",
            "timestamp": "2024-01-15T14:30:25",
            "parameters": {
                "grid_size": 64,
                "box_size": 4.0
            }
        }
        filepath = os.path.join(self.temp_dir, "test_metadata.json")
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f)
        
        loaded_metadata = self.importer.import_metadata("test_metadata.json")
        
        self.assertEqual(loaded_metadata, metadata)

    def test_import_file_not_found(self):
        """Test importing non-existent file."""
        with self.assertRaises(FileNotFoundError):
            self.importer.import_numpy_array("nonexistent.npy")

    def test_import_invalid_file(self):
        """Test importing invalid file."""
        filepath = os.path.join(self.temp_dir, "invalid.txt")
        with open(filepath, 'w') as f:
            f.write("invalid content")
        
        with self.assertRaises(ValueError):
            self.importer.import_numpy_array("invalid.txt")


if __name__ == '__main__':
    unittest.main()
