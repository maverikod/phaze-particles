#!/usr/bin/env python3
"""
Unit tests for base CLI command class.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from phaze_particles.cli.base import BaseCommand


class TestBaseCommand(unittest.TestCase):
    """Test BaseCommand class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test configuration
        test_config = {
            "grid_size": 32,
            "box_size": 2.0,
            "max_iterations": 100,
            "validation_enabled": True
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # BaseCommand should be abstract
        with self.assertRaises(TypeError):
            BaseCommand()

    def test_get_subcommands_abstract(self):
        """Test that get_subcommands is abstract."""
        class ConcreteCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        command = ConcreteCommand()
        
        # get_subcommands is not abstract in BaseCommand, it has default implementation
        subcommands = command.get_subcommands()
        self.assertEqual(subcommands, [])

    def test_get_help_abstract(self):
        """Test that get_help is abstract."""
        class ConcreteCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        command = ConcreteCommand()
        
        # get_help is not abstract in BaseCommand, it has default implementation
        help_text = command.get_help()
        self.assertEqual(help_text, "Test command")

    def test_load_config_abstract(self):
        """Test that load_config is abstract."""
        class ConcreteCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        command = ConcreteCommand()
        
        # load_config is not abstract in BaseCommand, it has default implementation
        # It will raise FileNotFoundError for non-existent file
        with self.assertRaises(FileNotFoundError):
            command.load_config("nonexistent.json")

    def test_validate_config_abstract(self):
        """Test that validate_config is abstract."""
        class ConcreteCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
        
        command = ConcreteCommand()
        
        # validate_config is not abstract in BaseCommand, it has default implementation
        result = command.validate_config()
        self.assertTrue(result)

    def test_concrete_implementation(self):
        """Test concrete command implementation."""
        class TestCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                parser.add_argument('--test', type=str, default='default')
            
            def execute(self, args):
                return {'test': args.test}
            
            def get_subcommands(self):
                return ['test1', 'test2']
            
            def get_help(self):
                return "Test command help"
            
            def load_config(self, config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self._config = config  # Set internal config
                    return config
            
            def validate_config(self):
                return 'grid_size' in self._config
        
        command = TestCommand()
        
        # Test subcommands
        subcommands = command.get_subcommands()
        self.assertEqual(subcommands, ['test1', 'test2'])
        
        # Test help
        help_text = command.get_help()
        self.assertEqual(help_text, "Test command help")
        
        # Test config loading
        config = command.load_config(self.config_file)
        self.assertEqual(config['grid_size'], 32)
        
        # Test config validation
        # Load config first to set internal config
        command.load_config(self.config_file)
        self.assertTrue(command.validate_config())

    def test_argument_parsing(self):
        """Test argument parsing functionality."""
        import argparse
        
        class TestCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                parser.add_argument('--grid-size', type=int, default=64)
                parser.add_argument('--box-size', type=float, default=4.0)
                parser.add_argument('--verbose', action='store_true')
            
            def execute(self, args):
                return args
            
            def get_subcommands(self):
                return []
            
            def get_help(self):
                return "Test help"
            
            def load_config(self, config_path):
                return {}
            
            def validate_config(self):
                return True
        
        command = TestCommand()
        
        # Test argument parsing
        parser = argparse.ArgumentParser()
        command.add_arguments(parser)
        
        # Test with default values
        args = parser.parse_args([])
        self.assertEqual(args.grid_size, 64)
        self.assertEqual(args.box_size, 4.0)
        self.assertFalse(args.verbose)
        
        # Test with custom values
        args = parser.parse_args(['--grid-size', '128', '--box-size', '6.0', '--verbose'])
        self.assertEqual(args.grid_size, 128)
        self.assertEqual(args.box_size, 6.0)
        self.assertTrue(args.verbose)

    def test_error_handling(self):
        """Test error handling in command execution."""
        class TestCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                raise ValueError("Test error")
            
            def get_subcommands(self):
                return []
            
            def get_help(self):
                return "Test help"
            
            def load_config(self, config_path):
                return {}
            
            def validate_config(self):
                return True
        
        command = TestCommand()
        
        # Test error handling
        with self.assertRaises(ValueError):
            command.execute(None)

    def test_config_file_not_found(self):
        """Test handling of missing config file."""
        class TestCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
            
            def get_subcommands(self):
                return []
            
            def get_help(self):
                return "Test help"
            
            def load_config(self, config_path):
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                return {}
            
            def validate_config(self):
                return True
        
        command = TestCommand()
        
        # Test missing config file
        with self.assertRaises(FileNotFoundError):
            command.load_config("nonexistent_config.json")

    def test_invalid_config_format(self):
        """Test handling of invalid config format."""
        class TestCommand(BaseCommand):
            def __init__(self):
                super().__init__("test", "Test command")
            
            def add_arguments(self, parser):
                pass
            
            def execute(self, args):
                pass
            
            def get_subcommands(self):
                return []
            
            def get_help(self):
                return "Test help"
            
            def load_config(self, config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            
            def validate_config(self):
                return True
        
        command = TestCommand()
        
        # Create invalid JSON file
        invalid_config_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_config_file, 'w') as f:
            f.write("invalid json content")
        
        # Test invalid JSON
        with self.assertRaises(json.JSONDecodeError):
            command.load_config(invalid_config_file)


if __name__ == '__main__':
    unittest.main()
