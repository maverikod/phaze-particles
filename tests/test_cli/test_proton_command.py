#!/usr/bin/env python3
"""
Unit tests for proton CLI command.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import tempfile
import os
import json
import argparse
from unittest.mock import Mock, patch, MagicMock

from phaze_particles.cli.commands.proton import ProtonCommand


class TestProtonCommand(unittest.TestCase):
    """Test ProtonCommand class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "proton_config.json")
        
        # Create test configuration
        test_config = {
            "grid_size": 32,
            "box_size": 2.0,
            "torus_config": "120deg",
            "max_iterations": 100,
            "validation_enabled": True,
            "c2": 1.0,
            "c4": 1.0,
            "c6": 1.0
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_command_initialization(self):
        """Test command initialization."""
        command = ProtonCommand()
        self.assertIsNotNone(command)

    def test_get_subcommands(self):
        """Test getting subcommands."""
        command = ProtonCommand()
        subcommands = command.get_subcommands()
        
        self.assertIsInstance(subcommands, list)
        self.assertIn('static', subcommands)
        self.assertIn('dynamic', subcommands)

    def test_get_help(self):
        """Test getting help text."""
        command = ProtonCommand()
        help_text = command.get_help()
        
        self.assertIsInstance(help_text, str)
        self.assertIn('proton', help_text.lower())
        self.assertIn('static', help_text.lower())
        self.assertIn('dynamic', help_text.lower())

    def test_add_arguments(self):
        """Test adding command line arguments."""
        command = ProtonCommand()
        parser = argparse.ArgumentParser()
        
        # Should not raise any exceptions
        command.add_arguments(parser)
        
        # Test parsing arguments
        args = parser.parse_args(['--grid-size', '64', '--box-size', '4.0'])
        self.assertEqual(args.grid_size, 64)
        self.assertEqual(args.box_size, 4.0)

    def test_load_config(self):
        """Test loading configuration from file."""
        command = ProtonCommand()
        config = command.load_config(self.config_file)
        
        self.assertIsInstance(config, dict)
        self.assertEqual(config['grid_size'], 32)
        self.assertEqual(config['box_size'], 2.0)
        self.assertEqual(config['torus_config'], '120deg')

    def test_validate_config(self):
        """Test configuration validation."""
        command = ProtonCommand()
        
        # Valid config
        valid_config = {
            "grid_size": 32,
            "box_size": 2.0,
            "torus_config": "120deg",
            "max_iterations": 100,
            "validation_enabled": True
        }
        self.assertTrue(command.validate_config(valid_config))
        
        # Invalid config - missing required fields
        invalid_config = {
            "grid_size": 32
        }
        self.assertFalse(command.validate_config(invalid_config))
        
        # Invalid config - negative grid size
        invalid_config2 = {
            "grid_size": -1,
            "box_size": 2.0,
            "torus_config": "120deg",
            "max_iterations": 100,
            "validation_enabled": True
        }
        self.assertFalse(command.validate_config(invalid_config2))

    @patch('phaze_particles.cli.commands.proton.ProtonModel')
    def test_execute_static_subcommand(self, mock_model_class):
        """Test executing static subcommand."""
        # Mock the model
        mock_model = Mock()
        mock_results = Mock()
        mock_results.status.value = "optimized"
        mock_results.converged = True
        mock_results.proton_mass = 938.272
        mock_results.charge_radius = 0.841
        mock_results.magnetic_moment = 2.793
        mock_results.electric_charge = 1.0
        mock_results.baryon_number = 1.0
        mock_model.run.return_value = mock_results
        mock_model_class.return_value = mock_model
        
        command = ProtonCommand()
        
        # Create mock args
        args = Mock()
        args.subcommand = 'static'
        args.grid_size = 32
        args.box_size = 2.0
        args.torus_config = '120deg'
        args.max_iterations = 100
        args.validation_enabled = True
        args.config_file = None
        args.output_dir = None
        args.verbose = False
        
        # Execute command
        result = command.execute(args)
        
        # Verify model was created and run
        mock_model_class.assert_called_once()
        mock_model.run.assert_called_once()
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'optimized')

    @patch('phaze_particles.cli.commands.proton.ProtonModel')
    def test_execute_dynamic_subcommand(self, mock_model_class):
        """Test executing dynamic subcommand."""
        # Mock the model
        mock_model = Mock()
        mock_results = Mock()
        mock_results.status.value = "optimized"
        mock_results.converged = True
        mock_model.run.return_value = mock_results
        mock_model_class.return_value = mock_model
        
        command = ProtonCommand()
        
        # Create mock args
        args = Mock()
        args.subcommand = 'dynamic'
        args.grid_size = 32
        args.box_size = 2.0
        args.torus_config = '120deg'
        args.max_iterations = 100
        args.validation_enabled = True
        args.config_file = None
        args.output_dir = None
        args.verbose = False
        
        # Execute command
        result = command.execute(args)
        
        # Verify model was created and run
        mock_model_class.assert_called_once()
        mock_model.run.assert_called_once()
        
        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result['status'], 'optimized')

    def test_execute_invalid_subcommand(self):
        """Test executing invalid subcommand."""
        command = ProtonCommand()
        
        # Create mock args with invalid subcommand
        args = Mock()
        args.subcommand = 'invalid'
        
        # Execute command should raise ValueError
        with self.assertRaises(ValueError):
            command.execute(args)

    def test_config_file_override(self):
        """Test that config file overrides command line arguments."""
        command = ProtonCommand()
        
        # Create mock args
        args = Mock()
        args.subcommand = 'static'
        args.config_file = self.config_file
        args.grid_size = 128  # This should be overridden by config file
        args.box_size = 6.0   # This should be overridden by config file
        args.torus_config = 'clover'  # This should be overridden by config file
        args.max_iterations = 1000    # This should be overridden by config file
        args.validation_enabled = False  # This should be overridden by config file
        args.output_dir = None
        args.verbose = False
        
        # Mock the model
        with patch('phaze_particles.cli.commands.proton.ProtonModel') as mock_model_class:
            mock_model = Mock()
            mock_results = Mock()
            mock_results.status.value = "optimized"
            mock_results.converged = True
            mock_model.run.return_value = mock_results
            mock_model_class.return_value = mock_model
            
            # Execute command
            result = command.execute(args)
            
            # Verify model was created with config file values
            call_args = mock_model_class.call_args[1]
            self.assertEqual(call_args['grid_size'], 32)  # From config file
            self.assertEqual(call_args['box_size'], 2.0)  # From config file
            self.assertEqual(call_args['torus_config'], '120deg')  # From config file
            self.assertEqual(call_args['max_iterations'], 100)  # From config file
            self.assertEqual(call_args['validation_enabled'], True)  # From config file

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        command = ProtonCommand()
        
        # Create non-existent output directory
        output_dir = os.path.join(self.temp_dir, "nonexistent", "output")
        
        # Create mock args
        args = Mock()
        args.subcommand = 'static'
        args.config_file = None
        args.grid_size = 32
        args.box_size = 2.0
        args.torus_config = '120deg'
        args.max_iterations = 100
        args.validation_enabled = True
        args.output_dir = output_dir
        args.verbose = False
        
        # Mock the model
        with patch('phaze_particles.cli.commands.proton.ProtonModel') as mock_model_class:
            mock_model = Mock()
            mock_results = Mock()
            mock_results.status.value = "optimized"
            mock_results.converged = True
            mock_model.run.return_value = mock_results
            mock_model_class.return_value = mock_model
            
            # Execute command
            result = command.execute(args)
            
            # Verify output directory was created
            self.assertTrue(os.path.exists(output_dir))

    def test_verbose_output(self):
        """Test verbose output mode."""
        command = ProtonCommand()
        
        # Create mock args with verbose enabled
        args = Mock()
        args.subcommand = 'static'
        args.config_file = None
        args.grid_size = 32
        args.box_size = 2.0
        args.torus_config = '120deg'
        args.max_iterations = 100
        args.validation_enabled = True
        args.output_dir = None
        args.verbose = True
        
        # Mock the model
        with patch('phaze_particles.cli.commands.proton.ProtonModel') as mock_model_class:
            mock_model = Mock()
            mock_results = Mock()
            mock_results.status.value = "optimized"
            mock_results.converged = True
            mock_model.run.return_value = mock_results
            mock_model_class.return_value = mock_model
            
            # Execute command
            result = command.execute(args)
            
            # Verify model was created with verbose logging
            call_args = mock_model_class.call_args[1]
            self.assertTrue(call_args.get('verbose', False))

    def test_error_handling(self):
        """Test error handling during execution."""
        command = ProtonCommand()
        
        # Create mock args
        args = Mock()
        args.subcommand = 'static'
        args.config_file = None
        args.grid_size = 32
        args.box_size = 2.0
        args.torus_config = '120deg'
        args.max_iterations = 100
        args.validation_enabled = True
        args.output_dir = None
        args.verbose = False
        
        # Mock the model to raise an exception
        with patch('phaze_particles.cli.commands.proton.ProtonModel') as mock_model_class:
            mock_model_class.side_effect = RuntimeError("Model creation failed")
            
            # Execute command should handle the error gracefully
            with self.assertRaises(RuntimeError):
                command.execute(args)

    def test_config_validation_errors(self):
        """Test handling of config validation errors."""
        command = ProtonCommand()
        
        # Create invalid config file
        invalid_config_file = os.path.join(self.temp_dir, "invalid_config.json")
        invalid_config = {
            "grid_size": -1,  # Invalid negative value
            "box_size": 2.0,
            "torus_config": "invalid_config"  # Invalid config type
        }
        
        with open(invalid_config_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # Create mock args
        args = Mock()
        args.subcommand = 'static'
        args.config_file = invalid_config_file
        args.output_dir = None
        args.verbose = False
        
        # Execute command should handle validation error
        with self.assertRaises(ValueError):
            command.execute(args)


if __name__ == '__main__':
    unittest.main()
