#!/usr/bin/env python3
"""
Unit tests for CUDA utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from phaze_particles.utils.cuda import (
    CUDAManager,
    get_cuda_manager,
    CUDADevice,
    CUDAMemoryManager,
    CUDAOperations,
)


class TestCUDADevice(unittest.TestCase):
    """Test CUDADevice class."""

    def test_device_initialization(self):
        """Test device initialization."""
        device = CUDADevice(
            id=0,
            name="Test GPU",
            memory_total=8192,
            memory_free=4096,
            compute_capability=(7, 5),
            multiprocessors=64,
            max_threads_per_block=1024
        )
        self.assertEqual(device.id, 0)
        self.assertEqual(device.name, "Test GPU")
        self.assertEqual(device.compute_capability, (7, 5))
        self.assertEqual(device.memory_total, 8192)

    def test_device_properties(self):
        """Test device properties."""
        device = CUDADevice(
            id=0,
            name="Test GPU",
            memory_total=8192,
            memory_free=4096,
            compute_capability=(7, 5),
            multiprocessors=64,
            max_threads_per_block=1024
        )
        
        # Test property access
        self.assertEqual(device.id, 0)
        self.assertEqual(device.name, "Test GPU")
        self.assertEqual(device.compute_capability, (7, 5))
        self.assertEqual(device.memory_total, 8192)

    def test_device_string_representation(self):
        """Test device string representation."""
        device = CUDADevice(
            id=0,
            name="Test GPU",
            memory_total=8192,
            memory_free=4096,
            compute_capability=(7, 5),
            multiprocessors=64,
            max_threads_per_block=1024
        )
        
        device_str = str(device)
        self.assertIn("Test GPU", device_str)
        self.assertIn("(7, 5)", device_str)
        self.assertIn("8192", device_str)


class TestCUDAMemoryManager(unittest.TestCase):
    """Test CUDAMemoryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = CUDAMemoryManager()

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.assertIsNotNone(self.memory_manager)
        self.assertEqual(self.memory_manager.allocated_memory, 0)
        self.assertEqual(len(self.memory_manager._allocations), 0)

    def test_memory_allocation_tracking(self):
        """Test memory allocation tracking."""
        # Simulate memory allocation
        allocation_id = self.memory_manager.allocate(1024)
        
        self.assertIsNotNone(allocation_id)
        self.assertEqual(self.memory_manager.allocated_memory, 1024)
        self.assertEqual(len(self.memory_manager.allocations), 1)
        self.assertIn(allocation_id, self.memory_manager.allocations)

    def test_memory_deallocation_tracking(self):
        """Test memory deallocation tracking."""
        # Allocate memory
        allocation_id = self.memory_manager.allocate(1024)
        
        # Deallocate memory
        self.memory_manager.deallocate(allocation_id)
        
        self.assertEqual(self.memory_manager.allocated_memory, 0)
        self.assertEqual(len(self.memory_manager._allocations), 0)
        self.assertNotIn(allocation_id, self.memory_manager.allocations)

    def test_memory_usage_reporting(self):
        """Test memory usage reporting."""
        # Allocate some memory
        self.memory_manager.allocate(1024)
        self.memory_manager.allocate(2048)
        
        usage = self.memory_manager.get_memory_usage()
        
        self.assertEqual(usage['allocated'], 3072)
        self.assertEqual(usage['allocations_count'], 2)

    def test_memory_cleanup(self):
        """Test memory cleanup."""
        # Allocate some memory
        self.memory_manager.allocate(1024)
        self.memory_manager.allocate(2048)
        
        # Cleanup all memory
        self.memory_manager.cleanup()
        
        self.assertEqual(self.memory_manager.allocated_memory, 0)
        self.assertEqual(len(self.memory_manager._allocations), 0)


class TestCUDAOperations(unittest.TestCase):
    """Test CUDAOperations class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_ops = CUDAOperations()

    def test_operations_initialization(self):
        """Test operations initialization."""
        self.assertIsNotNone(self.cuda_ops)

    def test_array_to_gpu(self):
        """Test array transfer to GPU."""
        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Mock CUDA operations
        with patch.object(self.cuda_ops, '_is_cuda_available', return_value=False):
            # Should fallback to CPU
            result = self.cuda_ops.array_to_gpu(test_array)
            self.assertIsNotNone(result)

    def test_array_from_gpu(self):
        """Test array transfer from GPU."""
        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Mock CUDA operations
        with patch.object(self.cuda_ops, '_is_cuda_available', return_value=False):
            # Should fallback to CPU
            result = self.cuda_ops.array_from_gpu(test_array)
            self.assertIsNotNone(result)

    def test_elementwise_operations(self):
        """Test elementwise operations."""
        # Create test arrays
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np.array([2, 3, 4, 5, 6], dtype=np.float32)
        
        # Mock CUDA operations
        with patch.object(self.cuda_ops, '_is_cuda_available', return_value=False):
            # Test addition
            result = self.cuda_ops.add(a, b)
            expected = a + b
            np.testing.assert_array_equal(result, expected)
            
            # Test multiplication
            result = self.cuda_ops.multiply(a, b)
            expected = a * b
            np.testing.assert_array_equal(result, expected)

    def test_reduction_operations(self):
        """Test reduction operations."""
        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Mock CUDA operations
        with patch.object(self.cuda_ops, '_is_cuda_available', return_value=False):
            # Test sum
            result = self.cuda_ops.sum(test_array)
            expected = np.sum(test_array)
            self.assertEqual(result, expected)
            
            # Test mean
            result = self.cuda_ops.mean(test_array)
            expected = np.mean(test_array)
            self.assertEqual(result, expected)

    def test_matrix_operations(self):
        """Test matrix operations."""
        # Create test matrices
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[2, 0], [1, 3]], dtype=np.float32)
        
        # Mock CUDA operations
        with patch.object(self.cuda_ops, '_is_cuda_available', return_value=False):
            # Test matrix multiplication
            result = self.cuda_ops.matmul(a, b)
            expected = np.matmul(a, b)
            np.testing.assert_array_equal(result, expected)


class TestCUDAManager(unittest.TestCase):
    """Test CUDAManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_manager = CUDAManager()

    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertIsNotNone(self.cuda_manager)
        self.assertIsNotNone(self.cuda_manager.memory_manager)
        self.assertIsNotNone(self.cuda_manager.operations)

    @patch('phaze_particles.utils.cuda.cuda_available', return_value=False)
    def test_cuda_availability_check(self, mock_cuda_available):
        """Test CUDA availability check."""
        self.assertFalse(self.cuda_manager.is_cuda_available)
        mock_cuda_available.assert_called_once()

    @patch('phaze_particles.utils.cuda.cuda_available', return_value=True)
    @patch('phaze_particles.utils.cuda.cuda.device_count', return_value=1)
    def test_cuda_availability_with_devices(self, mock_device_count, mock_cuda_available):
        """Test CUDA availability with devices."""
        self.assertTrue(self.cuda_manager.is_cuda_available())
        self.assertEqual(self.cuda_manager.get_device_count(), 1)

    def test_device_discovery(self):
        """Test device discovery."""
        # Mock CUDA availability
        with patch.object(self.cuda_manager, 'is_cuda_available', return_value=False):
            devices = self.cuda_manager.discover_devices()
            self.assertEqual(len(devices), 0)

    def test_device_selection(self):
        """Test device selection."""
        # Mock CUDA availability
        with patch.object(self.cuda_manager, 'is_cuda_available', return_value=False):
            # Should not raise exception when CUDA is not available
            self.cuda_manager.select_device(0)

    def test_memory_management(self):
        """Test memory management."""
        # Test memory allocation
        allocation_id = self.cuda_manager.allocate_memory(1024)
        self.assertIsNotNone(allocation_id)
        
        # Test memory deallocation
        self.cuda_manager.deallocate_memory(allocation_id)
        
        # Test memory cleanup
        self.cuda_manager.cleanup_memory()

    def test_operations_delegation(self):
        """Test operations delegation."""
        # Create test array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Test operations delegation
        result = self.cuda_manager.array_to_gpu(test_array)
        self.assertIsNotNone(result)
        
        result = self.cuda_manager.array_from_gpu(test_array)
        self.assertIsNotNone(result)

    def test_performance_monitoring(self):
        """Test performance monitoring."""
        # Test performance metrics
        metrics = self.cuda_manager.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('memory_usage', metrics)
        self.assertIn('operations_count', metrics)

    def test_error_handling(self):
        """Test error handling."""
        # Test handling of invalid device ID
        with patch.object(self.cuda_manager, 'is_cuda_available', return_value=True):
            with patch.object(self.cuda_manager, 'get_device_count', return_value=1):
                # Should handle invalid device ID gracefully
                self.cuda_manager.select_device(999)

    def test_context_management(self):
        """Test context management."""
        # Test context creation and cleanup
        with self.cuda_manager.get_context():
            # Context should be active
            pass
        # Context should be cleaned up

    def test_synchronization(self):
        """Test synchronization."""
        # Test synchronization operations
        self.cuda_manager.synchronize()
        # Should not raise any exceptions

    def test_stream_management(self):
        """Test stream management."""
        # Test stream creation
        stream = self.cuda_manager.create_stream()
        self.assertIsNotNone(stream)
        
        # Test stream synchronization
        self.cuda_manager.synchronize_stream(stream)
        
        # Test stream cleanup
        self.cuda_manager.destroy_stream(stream)


class TestGetCUDAManager(unittest.TestCase):
    """Test get_cuda_manager function."""

    def test_singleton_behavior(self):
        """Test that get_cuda_manager returns singleton instance."""
        manager1 = get_cuda_manager()
        manager2 = get_cuda_manager()
        
        self.assertIs(manager1, manager2)

    def test_manager_type(self):
        """Test that get_cuda_manager returns CUDAManager instance."""
        manager = get_cuda_manager()
        self.assertIsInstance(manager, CUDAManager)


class TestCUDAIntegration(unittest.TestCase):
    """Test CUDA integration with other components."""

    def setUp(self):
        """Set up test fixtures."""
        self.cuda_manager = get_cuda_manager()

    def test_numpy_integration(self):
        """Test integration with NumPy arrays."""
        # Create NumPy array
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Test array operations
        result = self.cuda_manager.operations.add(test_array, test_array)
        expected = test_array + test_array
        np.testing.assert_array_equal(result, expected)

    def test_memory_integration(self):
        """Test memory management integration."""
        # Test memory allocation and deallocation
        allocation_id = self.cuda_manager.allocate_memory(1024)
        self.assertIsNotNone(allocation_id)
        
        # Test memory usage reporting
        usage = self.cuda_manager.get_performance_metrics()
        self.assertIn('memory_usage', usage)
        
        # Cleanup
        self.cuda_manager.deallocate_memory(allocation_id)

    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test graceful handling of CUDA errors
        with patch.object(self.cuda_manager, 'is_cuda_available', return_value=False):
            # Should fallback to CPU operations
            test_array = np.array([1, 2, 3], dtype=np.float32)
            result = self.cuda_manager.operations.sum(test_array)
            self.assertEqual(result, 6.0)

    def test_performance_comparison(self):
        """Test performance comparison between CUDA and CPU."""
        # Create test data
        test_array = np.random.rand(1000, 1000).astype(np.float32)
        
        # Test operations
        with patch.object(self.cuda_manager, 'is_cuda_available', return_value=False):
            # CPU operations
            cpu_result = self.cuda_manager.operations.sum(test_array)
            self.assertIsNotNone(cpu_result)
        
        # Test that operations complete successfully
        self.assertIsNotNone(cpu_result)


if __name__ == '__main__':
    unittest.main()
