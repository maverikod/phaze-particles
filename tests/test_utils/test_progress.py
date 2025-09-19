#!/usr/bin/env python3
"""
Unit tests for progress utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from phaze_particles.utils.progress import (
    ProgressBar,
    PerformanceMonitor,
    create_progress_bar,
    create_performance_monitor,
    TimeEstimator,
    ProgressCallback,
)


class TestProgressBar(unittest.TestCase):
    """Test ProgressBar class."""

    def setUp(self):
        """Set up test fixtures."""
        self.progress_bar = ProgressBar(total=100, description="Test Progress")

    def test_progress_bar_initialization(self):
        """Test progress bar initialization."""
        self.assertEqual(self.progress_bar.total, 100)
        self.assertEqual(self.progress_bar.description, "Test Progress")
        self.assertEqual(self.progress_bar.current, 0)
        self.assertEqual(self.progress_bar.percentage, 0.0)

    def test_progress_update(self):
        """Test progress update."""
        self.progress_bar.update(50)
        self.assertEqual(self.progress_bar.current, 50)
        self.assertEqual(self.progress_bar.percentage, 50.0)

    def test_progress_increment(self):
        """Test progress increment."""
        self.progress_bar.increment()
        self.assertEqual(self.progress_bar.current, 1)
        self.assertEqual(self.progress_bar.percentage, 1.0)

    def test_progress_increment_by_amount(self):
        """Test progress increment by amount."""
        self.progress_bar.increment(25)
        self.assertEqual(self.progress_bar.current, 25)
        self.assertEqual(self.progress_bar.percentage, 25.0)

    def test_progress_completion(self):
        """Test progress completion."""
        self.progress_bar.update(100)
        self.assertTrue(self.progress_bar.is_complete())
        self.assertEqual(self.progress_bar.percentage, 100.0)

    def test_progress_reset(self):
        """Test progress reset."""
        self.progress_bar.update(50)
        self.progress_bar.reset()
        self.assertEqual(self.progress_bar.current, 0)
        self.assertEqual(self.progress_bar.percentage, 0.0)

    def test_progress_overflow_protection(self):
        """Test progress overflow protection."""
        self.progress_bar.update(150)  # More than total
        self.assertEqual(self.progress_bar.current, 100)
        self.assertEqual(self.progress_bar.percentage, 100.0)

    def test_progress_negative_protection(self):
        """Test progress negative protection."""
        self.progress_bar.update(-10)  # Negative value
        self.assertEqual(self.progress_bar.current, 0)
        self.assertEqual(self.progress_bar.percentage, 0.0)

    def test_progress_string_representation(self):
        """Test progress string representation."""
        self.progress_bar.update(50)
        progress_str = str(self.progress_bar)
        self.assertIn("50.0%", progress_str)
        self.assertIn("Test Progress", progress_str)

    def test_progress_bar_with_callback(self):
        """Test progress bar with callback."""
        callback_called = False
        callback_value = None

        def test_callback(current, total, percentage):
            nonlocal callback_called, callback_value
            callback_called = True
            callback_value = current

        progress_bar = ProgressBar(total=100, callback=test_callback)
        progress_bar.update(50)

        self.assertTrue(callback_called)
        self.assertEqual(callback_value, 50)


class TestTimeEstimator(unittest.TestCase):
    """Test TimeEstimator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.time_estimator = TimeEstimator()

    def test_time_estimator_initialization(self):
        """Test time estimator initialization."""
        self.assertIsNotNone(self.time_estimator.start_time)
        self.assertEqual(len(self.time_estimator.checkpoints), 1)
        self.assertEqual(self.time_estimator.last_progress, 0.0)

    def test_time_estimation(self):
        """Test time estimation."""
        # Start timing
        self.time_estimator.start()
        self.assertIsNotNone(self.time_estimator.start_time)

        # Simulate some progress
        time.sleep(0.1)  # 100ms
        self.time_estimator.update_progress(0.5)  # 50% complete

        # Check estimates
        self.assertIsNotNone(self.time_estimator.estimated_total_time)
        self.assertIsNotNone(self.time_estimator.estimated_remaining_time)
        self.assertGreater(self.time_estimator.estimated_total_time, 0)
        self.assertGreater(self.time_estimator.estimated_remaining_time, 0)

    def test_time_estimation_with_zero_progress(self):
        """Test time estimation with zero progress."""
        self.time_estimator.start()
        self.time_estimator.update_progress(0.0)

        # Should handle zero progress gracefully
        self.assertIsNone(self.time_estimator.estimated_total_time)
        self.assertIsNone(self.time_estimator.estimated_remaining_time)

    def test_time_estimation_with_complete_progress(self):
        """Test time estimation with complete progress."""
        self.time_estimator.start()
        time.sleep(0.1)
        self.time_estimator.update_progress(1.0)

        # Should have total time but no remaining time
        self.assertIsNotNone(self.time_estimator.estimated_total_time)
        self.assertEqual(self.time_estimator.estimated_remaining_time, 0)

    def test_time_estimation_reset(self):
        """Test time estimation reset."""
        self.time_estimator.start()
        self.time_estimator.update_progress(0.5)
        self.time_estimator.reset()

        self.assertIsNotNone(self.time_estimator.start_time)
        self.assertIsNone(self.time_estimator.estimated_total_time)
        self.assertIsNone(self.time_estimator.estimated_remaining_time)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        self.monitor.start()
        self.assertIsNotNone(self.monitor.start_time)
        self.assertEqual(len(self.monitor.metrics), 0)

    def test_metric_recording(self):
        """Test metric recording."""
        self.monitor.record_metric("test_metric", 42.0)
        self.assertIn("test_metric", self.monitor.metrics)
        self.assertEqual(self.monitor.metrics["test_metric"], 42.0)

    def test_multiple_metrics(self):
        """Test multiple metrics recording."""
        self.monitor.record_metric("metric1", 10.0)
        self.monitor.record_metric("metric2", 20.0)
        self.monitor.record_metric("metric3", 30.0)

        self.assertEqual(len(self.monitor.metrics), 3)
        self.assertEqual(self.monitor.metrics["metric1"], 10.0)
        self.assertEqual(self.monitor.metrics["metric2"], 20.0)
        self.assertEqual(self.monitor.metrics["metric3"], 30.0)

    def test_metric_overwriting(self):
        """Test metric overwriting."""
        self.monitor.record_metric("test_metric", 42.0)
        self.monitor.record_metric("test_metric", 84.0)

        self.assertEqual(self.monitor.metrics["test_metric"], 84.0)

    def test_execution_time_tracking(self):
        """Test execution time tracking."""
        # Start timing
        self.monitor.start_timing("test_operation")
        time.sleep(0.1)  # 100ms
        self.monitor.end_timing("test_operation")

        # Check that timing was recorded
        self.assertIn("test_operation_duration", self.monitor.metrics)
        self.assertGreater(self.monitor.metrics["test_operation_duration"], 0.09)  # Should be ~100ms

    def test_nested_timing(self):
        """Test nested timing operations."""
        self.monitor.start_timing("outer_operation")
        time.sleep(0.05)
        
        self.monitor.start_timing("inner_operation")
        time.sleep(0.05)
        self.monitor.end_timing("inner_operation")
        
        self.monitor.end_timing("outer_operation")

        # Both operations should be recorded
        self.assertIn("outer_operation_duration", self.monitor.metrics)
        self.assertIn("inner_operation_duration", self.monitor.metrics)
        self.assertGreater(self.monitor.metrics["outer_operation_duration"], self.monitor.metrics["inner_operation_duration"])

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        self.monitor.record_memory_usage("initial")
        self.monitor.record_memory_usage("after_operation")

        # Check that memory usage was recorded
        self.assertIn("initial_usage_mb", self.monitor.metrics)
        self.assertIn("after_operation_usage_mb", self.monitor.metrics)
        self.assertGreater(self.monitor.metrics["initial_usage_mb"], 0)
        self.assertGreater(self.monitor.metrics["after_operation_usage_mb"], 0)

    def test_performance_report_generation(self):
        """Test performance report generation."""
        # Start monitoring
        self.monitor.start()
        
        # Record some metrics
        self.monitor.record_metric("cpu_usage", 75.5)
        self.monitor.record_metric("memory_usage", 512.0)
        self.monitor.record_metric("execution_time", 1.25)

        # Generate report
        report = self.monitor.generate_report()

        # Check report content
        self.assertIsInstance(report, str)
        self.assertIn("cpu_usage", report)
        self.assertIn("memory_usage", report)
        self.assertIn("execution_time", report)
        self.assertIn("75.5", report)
        self.assertIn("512.0", report)
        self.assertIn("1.25", report)

    def test_monitor_reset(self):
        """Test monitor reset."""
        # Record some metrics
        self.monitor.record_metric("test_metric", 42.0)
        self.monitor.start_timing("test_operation")
        self.monitor.end_timing("test_operation")

        # Reset monitor
        self.monitor.reset()

        # Check that metrics are cleared
        self.assertEqual(len(self.monitor.metrics), 0)
        self.assertIsNone(self.monitor.start_time)  # Start time should be reset

    def test_context_manager(self):
        """Test monitor as context manager."""
        with self.monitor:
            time.sleep(0.1)

        # Check that timing was recorded
        self.assertIsNotNone(self.monitor.start_time)
        self.assertIsNotNone(self.monitor.end_time)
        self.assertGreater(self.monitor.metrics.get("duration", 0), 0.09)


class TestProgressCallback(unittest.TestCase):
    """Test ProgressCallback class."""

    def test_callback_initialization(self):
        """Test callback initialization."""
        callback = ProgressCallback()
        self.assertIsNotNone(callback)

    def test_callback_execution(self):
        """Test callback execution."""
        callback_called = False
        callback_value = None

        def test_function(current, total, percentage):
            nonlocal callback_called, callback_value
            callback_called = True
            callback_value = current

        callback = ProgressCallback()
        callback.add_callback(test_function)
        callback.execute_callbacks(50.0)

        self.assertTrue(callback_called)
        self.assertEqual(callback_value, 50.0)

    def test_multiple_callbacks(self):
        """Test multiple callbacks."""
        callback1_called = False
        callback2_called = False

        def callback1(current, total, percentage):
            nonlocal callback1_called
            callback1_called = True

        def callback2(current, total, percentage):
            nonlocal callback2_called
            callback2_called = True

        callback = ProgressCallback()
        callback.add_callback(callback1)
        callback.add_callback(callback2)
        callback.execute_callbacks(50.0)

        self.assertTrue(callback1_called)
        self.assertTrue(callback2_called)

    def test_callback_removal(self):
        """Test callback removal."""
        callback_called = False

        def test_function(current, total, percentage):
            nonlocal callback_called
            callback_called = True

        callback = ProgressCallback()
        callback.add_callback(test_function)
        callback.remove_callback(test_function)
        callback.execute_callbacks(50.0)

        self.assertFalse(callback_called)

    def test_callback_error_handling(self):
        """Test callback error handling."""
        def error_callback(current, total, percentage):
            raise ValueError("Test error")

        def normal_callback(current, total, percentage):
            pass

        callback = ProgressCallback()
        callback.add_callback(error_callback)
        callback.add_callback(normal_callback)

        # Should not raise exception, should handle error gracefully
        callback.execute_callbacks(50.0)


class TestCreateProgressBar(unittest.TestCase):
    """Test create_progress_bar function."""

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        progress_bar = create_progress_bar(total=100, description="Test")
        
        self.assertIsInstance(progress_bar, ProgressBar)
        self.assertEqual(progress_bar.total, 100)
        self.assertEqual(progress_bar.description, "Test")

    def test_create_progress_bar_with_callback(self):
        """Test progress bar creation with callback."""
        callback_called = False

        def test_callback(current, total, percentage):
            nonlocal callback_called
            callback_called = True

        progress_bar = create_progress_bar(
            total=100, 
            description="Test", 
            callback=test_callback
        )
        
        self.assertIsInstance(progress_bar, ProgressBar)
        progress_bar.update(50)
        self.assertTrue(callback_called)


class TestCreatePerformanceMonitor(unittest.TestCase):
    """Test create_performance_monitor function."""

    def test_create_performance_monitor(self):
        """Test performance monitor creation."""
        monitor = create_performance_monitor()
        
        self.assertIsInstance(monitor, PerformanceMonitor)
        self.assertIsNotNone(monitor.start_time)

    def test_create_performance_monitor_with_name(self):
        """Test performance monitor creation with name."""
        monitor = create_performance_monitor(name="Test Monitor")
        
        self.assertIsInstance(monitor, PerformanceMonitor)
        self.assertEqual(monitor.name, "Test Monitor")


class TestProgressIntegration(unittest.TestCase):
    """Test progress utilities integration."""

    def test_progress_bar_with_time_estimator(self):
        """Test progress bar with time estimator."""
        progress_bar = ProgressBar(total=100, description="Test")
        time_estimator = TimeEstimator()
        
        # Start timing
        time_estimator.start()
        
        # Update progress
        progress_bar.update(50)
        time_estimator.update_progress(0.5)
        
        # Check that both work together
        self.assertEqual(progress_bar.percentage, 50.0)
        self.assertIsNotNone(time_estimator.estimated_remaining_time)

    def test_performance_monitor_with_progress_bar(self):
        """Test performance monitor with progress bar."""
        monitor = PerformanceMonitor()
        progress_bar = ProgressBar(total=100, description="Test")
        
        # Record progress in monitor
        monitor.record_metric("progress", progress_bar.percentage)
        
        # Update progress
        progress_bar.update(75)
        monitor.record_metric("progress", progress_bar.percentage)
        
        # Check that both work together
        self.assertEqual(progress_bar.percentage, 75.0)
        self.assertEqual(monitor.metrics["progress"], 75.0)

    def test_callback_with_monitor(self):
        """Test callback with monitor."""
        monitor = PerformanceMonitor()
        callback = ProgressCallback()
        
        def progress_callback(current, total, percentage):
            monitor.record_metric("current_progress", current)
        
        callback.add_callback(progress_callback)
        progress_bar = ProgressBar(total=100, callback=callback.execute_callbacks)
        
        # Update progress
        progress_bar.update(60)
        
        # Check that callback updated monitor
        self.assertIn("current_progress", monitor.metrics)
        self.assertEqual(monitor.metrics["current_progress"], 60.0)


if __name__ == '__main__':
    unittest.main()
