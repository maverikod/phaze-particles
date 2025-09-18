#!/usr/bin/env python3
"""
CUDA detection and management utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


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
