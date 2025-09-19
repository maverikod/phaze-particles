#!/usr/bin/env python3
"""
CUDA detection and management utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CUDADevice:
    """CUDA device information."""

    id: int
    name: str
    memory_total: int  # MB
    memory_free: int  # MB
    compute_capability: Tuple[int, int]
    multiprocessors: int
    max_threads_per_block: int


@dataclass
class MemoryAllocation:
    """CUDA memory allocation information."""
    
    allocation_id: str
    size_bytes: int
    device_id: int
    timestamp: float
    is_active: bool = True


class CUDAMemoryManager:
    """
    CUDA memory management utilities.
    
    Provides memory allocation tracking, usage monitoring,
    and automatic cleanup capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize CUDA memory manager."""
        self._allocations: Dict[str, MemoryAllocation] = {}
        self._total_allocated = 0
        self._logger = logging.getLogger(__name__)
    
    def allocate_memory(self, size_bytes: int, device_id: int = 0) -> str:
        """
        Allocate CUDA memory.
        
        Args:
            size_bytes: Size to allocate in bytes
            device_id: CUDA device ID
            
        Returns:
            Allocation ID for tracking
        """
        allocation_id = str(uuid.uuid4())
        allocation = MemoryAllocation(
            allocation_id=allocation_id,
            size_bytes=size_bytes,
            device_id=device_id,
            timestamp=time.time()
        )
        
        self._allocations[allocation_id] = allocation
        self._total_allocated += size_bytes
        
        self._logger.debug(f"Allocated {size_bytes} bytes on device {device_id}")
        return allocation_id
    
    def deallocate_memory(self, allocation_id: str) -> bool:
        """
        Deallocate CUDA memory.
        
        Args:
            allocation_id: Allocation ID to deallocate
            
        Returns:
            True if successful, False otherwise
        """
        if allocation_id not in self._allocations:
            return False
        
        allocation = self._allocations[allocation_id]
        if allocation.is_active:
            self._total_allocated -= allocation.size_bytes
            allocation.is_active = False
            self._logger.debug(f"Deallocated {allocation.size_bytes} bytes")
        
        return True
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        active_allocations = [a for a in self._allocations.values() if a.is_active]
        
        return {
            "total_allocated_bytes": self._total_allocated,
            "active_allocations": len(active_allocations),
            "total_allocations": len(self._allocations),
            "allocations_by_device": self._get_allocations_by_device()
        }
    
    def _get_allocations_by_device(self) -> Dict[int, Dict[str, Any]]:
        """Get allocations grouped by device."""
        by_device = {}
        for allocation in self._allocations.values():
            if allocation.is_active:
                if allocation.device_id not in by_device:
                    by_device[allocation.device_id] = {
                        "count": 0,
                        "total_bytes": 0
                    }
                by_device[allocation.device_id]["count"] += 1
                by_device[allocation.device_id]["total_bytes"] += allocation.size_bytes
        
        return by_device
    
    def cleanup_all(self) -> None:
        """Clean up all active allocations."""
        for allocation_id in list(self._allocations.keys()):
            self.deallocate_memory(allocation_id)
        self._logger.info("Cleaned up all CUDA memory allocations")


class CUDAOperations:
    """
    CUDA operations for array manipulation.
    
    Provides high-level operations for transferring arrays
    between CPU and GPU, and performing computations.
    """
    
    def __init__(self) -> None:
        """Initialize CUDA operations."""
        self._logger = logging.getLogger(__name__)
        self._cuda_available = False
        
        try:
            import cupy as cp
            self._cuda_available = cp.cuda.is_available()
        except ImportError:
            self._logger.info("CuPy not available - CUDA operations disabled")
    
    def array_to_gpu(self, array: np.ndarray) -> Any:
        """
        Transfer numpy array to GPU.
        
        Args:
            array: NumPy array to transfer
            
        Returns:
            GPU array or original array if CUDA not available
        """
        if not self._cuda_available:
            return array
        
        try:
            import cupy as cp
            return cp.asarray(array)
        except Exception as e:
            self._logger.error(f"Error transferring array to GPU: {e}")
            return array
    
    def array_from_gpu(self, gpu_array: Any) -> np.ndarray:
        """
        Transfer GPU array to CPU.
        
        Args:
            gpu_array: GPU array to transfer
            
        Returns:
            NumPy array
        """
        if not self._cuda_available:
            return gpu_array
        
        try:
            import cupy as cp
            if isinstance(gpu_array, cp.ndarray):
                return cp.asnumpy(gpu_array)
            else:
                return gpu_array
        except Exception as e:
            self._logger.error(f"Error transferring array from GPU: {e}")
            return gpu_array
    
    def add(self, a: Any, b: Any) -> Any:
        """
        Element-wise addition.
        
        Args:
            a, b: Arrays to add
            
        Returns:
            Result of addition
        """
        if not self._cuda_available:
            return np.add(a, b)
        
        try:
            import cupy as cp
            if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
                return cp.add(a, b)
            else:
                return np.add(a, b)
        except Exception as e:
            self._logger.error(f"Error in GPU addition: {e}")
            return np.add(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        """
        Element-wise multiplication.
        
        Args:
            a, b: Arrays to multiply
            
        Returns:
            Result of multiplication
        """
        if not self._cuda_available:
            return np.multiply(a, b)
        
        try:
            import cupy as cp
            if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
                return cp.multiply(a, b)
            else:
                return np.multiply(a, b)
        except Exception as e:
            self._logger.error(f"Error in GPU multiplication: {e}")
            return np.multiply(a, b)
    
    def matrix_multiply(self, a: Any, b: Any) -> Any:
        """
        Matrix multiplication.
        
        Args:
            a, b: Matrices to multiply
            
        Returns:
            Result of matrix multiplication
        """
        if not self._cuda_available:
            return np.dot(a, b)
        
        try:
            import cupy as cp
            if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
                return cp.dot(a, b)
            else:
                return np.dot(a, b)
        except Exception as e:
            self._logger.error(f"Error in GPU matrix multiplication: {e}")
            return np.dot(a, b)
    
    def sum(self, array: Any, axis: Optional[int] = None) -> Any:
        """
        Sum array elements.
        
        Args:
            array: Array to sum
            axis: Axis along which to sum
            
        Returns:
            Sum result
        """
        if not self._cuda_available:
            return np.sum(array, axis=axis)
        
        try:
            import cupy as cp
            if isinstance(array, cp.ndarray):
                return cp.sum(array, axis=axis)
            else:
                return np.sum(array, axis=axis)
        except Exception as e:
            self._logger.error(f"Error in GPU sum: {e}")
            return np.sum(array, axis=axis)
    
    def max(self, array: Any, axis: Optional[int] = None) -> Any:
        """
        Maximum of array elements.
        
        Args:
            array: Array to find maximum
            axis: Axis along which to find maximum
            
        Returns:
            Maximum result
        """
        if not self._cuda_available:
            return np.max(array, axis=axis)
        
        try:
            import cupy as cp
            if isinstance(array, cp.ndarray):
                return cp.max(array, axis=axis)
            else:
                return np.max(array, axis=axis)
        except Exception as e:
            self._logger.error(f"Error in GPU max: {e}")
            return np.max(array, axis=axis)


class CUDAManager:
    """
    Centralized CUDA detection and management.

    Provides unified interface for CUDA availability detection,
    device information, and performance monitoring.
    """

    def __init__(self) -> None:
        """Initialize CUDA manager."""
        self._cuda_available = False
        self._devices: List[CUDADevice] = []
        self._current_device: Optional[CUDADevice] = None
        self._logger = logging.getLogger(__name__)
        
        # Initialize sub-components
        self.memory_manager = CUDAMemoryManager()
        self.operations = CUDAOperations()
        self._performance_metrics: Dict[str, Any] = {}
        self._streams: List[Any] = []

        # Try to detect CUDA
        self._detect_cuda()

    def _detect_cuda(self) -> None:
        """Detect CUDA availability and devices."""
        try:
            # Try to import CUDA libraries
            import cupy as cp

            # Check if CUDA is available
            if cp.cuda.is_available():
                self._cuda_available = True
                self._logger.info("CUDA is available")

                # Get device information
                device_count = cp.cuda.runtime.getDeviceCount()
                self._logger.info(f"Found {device_count} CUDA device(s)")

                for device_id in range(device_count):
                    device = self._get_device_info(device_id)
                    self._devices.append(device)
                    self._logger.info(f"Device {device_id}: {device.name}")

                # Set current device to first available
                if self._devices:
                    self._current_device = self._devices[0]
                    cp.cuda.Device(0).use()

            else:
                self._logger.warning("CUDA runtime is not available")

        except ImportError:
            self._logger.info(
                "CUDA libraries (CuPy) not installed - using CPU"
            )
        except Exception as e:
            self._logger.error(f"Error detecting CUDA: {e}")

    def _get_device_info(self, device_id: int) -> CUDADevice:
        """
        Get information about a CUDA device.

        Args:
            device_id: CUDA device ID

        Returns:
            CUDADevice object with device information
        """
        import cupy as cp

        with cp.cuda.Device(device_id):
            # Get device properties
            props = cp.cuda.runtime.getDeviceProperties(device_id)

            # Get memory information
            mem_info = cp.cuda.runtime.memGetInfo()
            memory_total = mem_info[1] // (1024 * 1024)  # Convert to MB
            memory_free = mem_info[0] // (1024 * 1024)  # Convert to MB

            return CUDADevice(
                id=device_id,
                name=props["name"].decode("utf-8"),
                memory_total=memory_total,
                memory_free=memory_free,
                compute_capability=(props["major"], props["minor"]),
                multiprocessors=props["multiProcessorCount"],
                max_threads_per_block=props["maxThreadsPerBlock"],
            )

    @property
    def is_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def device_count(self) -> int:
        """Get number of available CUDA devices."""
        return len(self._devices)

    @property
    def devices(self) -> List[CUDADevice]:
        """Get list of available CUDA devices."""
        return self._devices.copy()

    @property
    def current_device(self) -> Optional[CUDADevice]:
        """Get current CUDA device."""
        return self._current_device

    def get_device_info(self, device_id: int = 0) -> Optional[CUDADevice]:
        """
        Get information about a specific CUDA device.

        Args:
            device_id: CUDA device ID

        Returns:
            CUDADevice object or None if not available
        """
        if not self._cuda_available or device_id >= len(self._devices):
            return None
        return self._devices[device_id]

    def set_device(self, device_id: int) -> bool:
        """
        Set current CUDA device.

        Args:
            device_id: CUDA device ID

        Returns:
            True if successful, False otherwise
        """
        if not self._cuda_available or device_id >= len(self._devices):
            return False

        try:
            import cupy as cp

            cp.cuda.Device(device_id).use()
            self._current_device = self._devices[device_id]
            self._logger.info(
                f"Switched to CUDA device {device_id}: "
                f"{self._current_device.name}"
            )
            return True
        except Exception as e:
            self._logger.error(f"Error setting CUDA device {device_id}: {e}")
            return False

    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available (alias for compatibility)."""
        return self._cuda_available

    @contextmanager
    def get_context(self):
        """Get CUDA context manager."""
        if not self._cuda_available:
            yield None
            return
        
        try:
            import cupy as cp
            with cp.cuda.Device(self._current_device.id if self._current_device else 0):
                yield self._current_device
        except Exception as e:
            self._logger.error(f"Error creating CUDA context: {e}")
            yield None

    def allocate_memory(self, size_bytes: int, device_id: int = 0) -> str:
        """
        Allocate CUDA memory.
        
        Args:
            size_bytes: Size to allocate in bytes
            device_id: CUDA device ID
            
        Returns:
            Allocation ID for tracking
        """
        return self.memory_manager.allocate_memory(size_bytes, device_id)

    def array_to_gpu(self, array: np.ndarray) -> Any:
        """
        Transfer numpy array to GPU.
        
        Args:
            array: NumPy array to transfer
            
        Returns:
            GPU array or original array if CUDA not available
        """
        return self.operations.array_to_gpu(array)

    def array_from_gpu(self, gpu_array: Any) -> np.ndarray:
        """
        Transfer GPU array to CPU.
        
        Args:
            gpu_array: GPU array to transfer
            
        Returns:
            NumPy array
        """
        return self.operations.array_from_gpu(gpu_array)

    def synchronize(self) -> None:
        """Synchronize CUDA operations."""
        if not self._cuda_available:
            return
        
        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except Exception as e:
            self._logger.error(f"Error synchronizing CUDA: {e}")

    def create_stream(self) -> Any:
        """
        Create CUDA stream.
        
        Returns:
            CUDA stream or None if not available
        """
        if not self._cuda_available:
            return None
        
        try:
            import cupy as cp
            stream = cp.cuda.Stream()
            self._streams.append(stream)
            return stream
        except Exception as e:
            self._logger.error(f"Error creating CUDA stream: {e}")
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "cuda_available": self._cuda_available,
            "device_count": len(self._devices),
            "current_device": self._current_device.id if self._current_device else None,
            "memory_usage": self.memory_manager.get_memory_usage(),
            "stream_count": len(self._streams)
        }
        
        # Add device-specific metrics
        if self._current_device:
            metrics.update({
                "device_name": self._current_device.name,
                "device_memory_total": self._current_device.memory_total,
                "device_memory_free": self._current_device.memory_free,
                "compute_capability": self._current_device.compute_capability
            })
        
        return metrics

    def get_status_string(self) -> str:
        """
        Get CUDA status as formatted string.

        Returns:
            Formatted status string
        """
        if not self._cuda_available:
            return "CUDA: ❌ Not Available (CPU mode)"

        if not self._devices:
            return "CUDA: ⚠️  Available but no devices found"

        device = self._current_device or self._devices[0]
        return (
            f"CUDA: ✅ Available - {device.name} "
            f"(Compute {device.compute_capability[0]}."
            f"{device.compute_capability[1]}, "
            f"{device.memory_total}MB total, "
            f"{device.memory_free}MB free)"
        )

    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed CUDA status information.

        Returns:
            Dictionary with detailed status information
        """
        status: Dict[str, Any] = {
            "available": self._cuda_available,
            "device_count": self.device_count,
            "devices": [],
        }

        for device in self._devices:
            device_info = {
                "id": device.id,
                "name": device.name,
                "memory_total_mb": device.memory_total,
                "memory_free_mb": device.memory_free,
                "compute_capability": (
                    f"{device.compute_capability[0]}."
                    f"{device.compute_capability[1]}"
                ),
                "multiprocessors": device.multiprocessors,
                "max_threads_per_block": device.max_threads_per_block,
                "is_current": device == self._current_device,
            }
            status["devices"].append(device_info)

        return status

    def log_status(self) -> None:
        """Log CUDA status information."""
        if self._cuda_available:
            self._logger.info("CUDA Status:")
            self._logger.info(f"  Available: {self._cuda_available}")
            self._logger.info(f"  Device count: {self.device_count}")

            for device in self._devices:
                current_marker = (
                    " (current)" if device == self._current_device else ""
                )
                self._logger.info(
                    f"  Device {device.id}: "
                    f"{device.name}{current_marker}"
                )
                self._logger.info(
                    f"    Memory: {device.memory_free}/"
                    f"{device.memory_total} MB free"
                )
                self._logger.info(
                    f"    Compute capability: "
                    f"{device.compute_capability[0]}."
                    f"{device.compute_capability[1]}"
                )
        else:
            self._logger.info("CUDA not available - using CPU")


# Global CUDA manager instance
cuda_manager = CUDAManager()


def get_cuda_manager() -> CUDAManager:
    """
    Get global CUDA manager instance.

    Returns:
        Global CUDAManager instance
    """
    return cuda_manager


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    return cuda_manager.is_available


def get_cuda_status() -> str:
    """
    Get CUDA status string.

    Returns:
        Formatted CUDA status string
    """
    return cuda_manager.get_status_string()


# Global variables for compatibility with tests
cuda_available = cuda_manager.is_available
cuda = cuda_manager if cuda_manager.is_available else None
