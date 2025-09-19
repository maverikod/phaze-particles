#!/usr/bin/env python3
"""
Simple unit tests for IO utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

from phaze_particles.utils.io import (
    save_results,
    load_config,
    ensure_output_directory,
    generate_report_filename,
)


class TestSaveResults(unittest.TestCase):
    """Test save_results function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.json_file = os.path.join(self.temp_dir, "test.json")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_results_json(self):
        """Test saving results to JSON file."""
        results = {"test": "value", "number": 42}
        save_results(results, self.json_file, format="json")
        
        # Check if file was created
        self.assertTrue(os.path.exists(self.json_file))
        
        # Check content
        with open(self.json_file, 'r') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data, results)

    def test_save_results_creates_directory(self):
        """Test that save_results creates directory if it doesn't exist."""
        nested_file = os.path.join(self.temp_dir, "nested", "test.json")
        results = {"test": "value"}
        save_results(results, nested_file, format="json")
        
        # Check if file was created
        self.assertTrue(os.path.exists(nested_file))

    def test_save_results_invalid_format(self):
        """Test saving with invalid format."""
        results = {"test": "value"}
        with self.assertRaises(ValueError):
            save_results(results, self.json_file, format="invalid")


class TestLoadConfig(unittest.TestCase):
    """Test load_config function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        
        # Create test config
        test_config = {"grid_size": 32, "box_size": 2.0}
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_config_success(self):
        """Test loading valid config file."""
        config = load_config(self.config_file)
        self.assertEqual(config["grid_size"], 32)
        self.assertEqual(config["box_size"], 2.0)

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent.json")

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON file."""
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with self.assertRaises(json.JSONDecodeError):
            load_config(invalid_file)


class TestEnsureOutputDirectory(unittest.TestCase):
    """Test ensure_output_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_ensure_output_directory_creates_dir(self):
        """Test that function creates directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "new_directory")
        result = ensure_output_directory(new_dir)
        
        self.assertTrue(os.path.exists(new_dir))
        self.assertIsInstance(result, Path)

    def test_ensure_output_directory_existing_dir(self):
        """Test that function works with existing directory."""
        result = ensure_output_directory(self.temp_dir)
        
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertIsInstance(result, Path)


class TestGenerateReportFilename(unittest.TestCase):
    """Test generate_report_filename function."""

    def test_generate_report_filename(self):
        """Test filename generation."""
        filename = generate_report_filename("test", "json")
        
        self.assertIsInstance(filename, str)
        self.assertIn("test", filename)
        self.assertIn("report.md", filename)  # Function generates .md files


if __name__ == '__main__':
    unittest.main()
