#!/usr/bin/env python3
"""
Optimization utilities for proton model.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import numpy as np
import time
import psutil
import os
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# CUDA imports
try:
    import cupy as cp

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Import existing modules
from .cuda import get_cuda_manager, CUDAManager


class OptimizationLevel(Enum):
    """Optimization levels."""

    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Performance metrics."""

    execution_time: float
    memory_usage: float
    gpu_utilization: float
    cpu_utilization: float
    iterations: int
    convergence_rate: float
    throughput: float  # operations per second


@dataclass
class OptimizationConfig:
    """Optimization configuration."""

    use_cuda: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.ADVANCED
    memory_optimization: bool = True
    algorithm_optimization: bool = True
    adaptive_parameters: bool = True
    profiling_enabled: bool = True
    cache_enabled: bool = True
    parallel_processing: bool = True


class MemoryOptimizer:
    """Memory optimizer."""

    def __init__(self, cuda_manager: CUDAManager):
        """
        Initialize memory optimizer.

        Args:
            cuda_manager: CUDA manager instance
        """
        self.cuda_manager = cuda_manager
        self.memory_pool: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage information.

        Returns:
            Memory usage information
        """
        memory_info = {}

        # CPU memory
        process = psutil.Process(os.getpid())
        memory_info["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

        # GPU memory
        if self.cuda_manager.is_available:
            try:
                mempool = cp.get_default_memory_pool()
                memory_info["gpu_memory_mb"] = mempool.used_bytes() / 1024 / 1024
                memory_info["gpu_memory_total_mb"] = mempool.total_bytes() / 1024 / 1024
            except Exception:
                memory_info["gpu_memory_mb"] = 0
                memory_info["gpu_memory_total_mb"] = 0
        else:
            memory_info["gpu_memory_mb"] = 0
            memory_info["gpu_memory_total_mb"] = 0

        return memory_info

    def optimize_memory_layout(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize memory layout of arrays.

        Args:
            arrays: List of arrays

        Returns:
            Optimized arrays
        """
        optimized_arrays = []

        for array in arrays:
            # Ensure memory contiguity
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)

            # Optimize data type
            if array.dtype == np.float64:
                array = array.astype(np.float32)

            optimized_arrays.append(array)

        return optimized_arrays

    def cache_result(self, key: str, result: Any) -> None:
        """
        Cache a result.

        Args:
            key: Cache key
            result: Result to cache
        """
        self.cache[key] = result

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get cached result.

        Args:
            key: Cache key

        Returns:
            Cached result or None
        """
        return self.cache.get(key)


class AlgorithmOptimizer:
    """Algorithm optimizer."""

    def __init__(self, cuda_manager: CUDAManager):
        """
        Initialize algorithm optimizer.

        Args:
            cuda_manager: CUDA manager instance
        """
        self.cuda_manager = cuda_manager
        self.optimized_kernels: Dict[str, Any] = {}

    def optimize_gradient_calculation(
        self, field: np.ndarray, dx: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimized gradient calculation.

        Args:
            field: Scalar field
            dx: Grid step

        Returns:
            Gradient components
        """
        if self.cuda_manager.is_available and CUDA_AVAILABLE:
            # CUDA optimized calculation
            field_gpu = cp.asarray(field)

            # Use CuPy for gradient
            grad_x = cp.gradient(field_gpu, dx, axis=0)
            grad_y = cp.gradient(field_gpu, dx, axis=1)
            grad_z = cp.gradient(field_gpu, dx, axis=2)

            # Transfer back to CPU
            grad_x = grad_x.get()
            grad_y = grad_y.get()
            grad_z = grad_z.get()

            return grad_x, grad_y, grad_z
        else:
            # CPU calculation
            grad_x = np.gradient(field, dx, axis=0)
            grad_y = np.gradient(field, dx, axis=1)
            grad_z = np.gradient(field, dx, axis=2)

            return grad_x, grad_y, grad_z

    def optimize_integration(self, field: np.ndarray, dx: float) -> float:
        """
        Optimized integration.

        Args:
            field: Field to integrate
            dx: Grid step

        Returns:
            Integration result
        """
        if self.cuda_manager.is_available and CUDA_AVAILABLE:
            # CUDA optimized integration
            field_gpu = cp.asarray(field)
            result = cp.sum(field_gpu) * dx**3
            return float(result.get())
        else:
            # CPU integration
            return float(np.sum(field) * dx**3)

    def optimize_matrix_operations(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        Optimized matrix operations.

        Args:
            matrices: List of matrices

        Returns:
            Operation result
        """
        if self.cuda_manager.is_available and CUDA_AVAILABLE:
            # CUDA optimized operations
            matrices_gpu = [cp.asarray(m) for m in matrices]

            result = matrices_gpu[0]
            for matrix in matrices_gpu[1:]:
                result = cp.dot(result, matrix)

            return result.get()
        else:
            # CPU operations
            result = matrices[0]
            for matrix in matrices[1:]:
                result = np.dot(result, matrix)

            return result


class AdaptiveParameterOptimizer:
    """Adaptive parameter optimizer."""

    def __init__(self) -> None:
        """Initialize parameter optimizer."""
        self.parameter_history = {}
        self.performance_history = {}

    def optimize_step_size(
        self, current_step: float, convergence_rate: float, iteration: int
    ) -> float:
        """
        Optimize step size.

        Args:
            current_step: Current step size
            convergence_rate: Convergence rate
            iteration: Iteration number

        Returns:
            Optimized step size
        """
        # Adaptive step size adjustment
        if convergence_rate > 0.1:
            # Fast convergence - increase step
            new_step = current_step * 1.1
        elif convergence_rate < 0.01:
            # Slow convergence - decrease step
            new_step = current_step * 0.9
        else:
            new_step = current_step

        # Constraints
        new_step = max(0.001, min(0.1, new_step))

        return new_step

    def optimize_tolerance(self, current_tolerance: float, error: float) -> float:
        """
        Optimize tolerance.

        Args:
            current_tolerance: Current tolerance
            error: Current error

        Returns:
            Optimized tolerance
        """
        # Adaptive tolerance adjustment
        if error < current_tolerance * 0.1:
            # Very small error - can increase tolerance
            new_tolerance = current_tolerance * 1.5
        elif error > current_tolerance * 10:
            # Large error - decrease tolerance
            new_tolerance = current_tolerance * 0.5
        else:
            new_tolerance = current_tolerance

        # Constraints
        new_tolerance = max(1e-8, min(1e-3, new_tolerance))

        return new_tolerance

    def optimize_constraints(
        self, current_constraints: Dict[str, float], violations: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Optimize constraints.

        Args:
            current_constraints: Current constraints
            violations: Constraint violations

        Returns:
            Optimized constraints
        """
        new_constraints = {}

        for key, current_value in current_constraints.items():
            violation = violations.get(key, 0.0)

            if violation > 0.1:
                # Large violation - increase weight
                new_constraints[key] = current_value * 2.0
            elif violation < 0.01:
                # Small violation - decrease weight
                new_constraints[key] = current_value * 0.5
            else:
                new_constraints[key] = current_value

        return new_constraints


class PerformanceProfiler:
    """Performance profiler."""

    def __init__(self) -> None:
        """Initialize profiler."""
        self.profiles = {}
        self.current_profile = None

    def start_profile(self, name: str) -> None:
        """
        Start profiling.

        Args:
            name: Profile name
        """
        self.current_profile = {
            "name": name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": psutil.cpu_percent(),
        }

    def end_profile(self, name: str) -> Dict[str, Any]:
        """
        End profiling.

        Args:
            name: Profile name

        Returns:
            Profiling results
        """
        if not self.current_profile or self.current_profile["name"] != name:
            raise ValueError(f"Profile {name} not started")

        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_cpu = psutil.cpu_percent()

        profile_result = {
            "name": name,
            "execution_time": end_time - self.current_profile["start_time"],
            "memory_delta": end_memory - self.current_profile["start_memory"],
            "cpu_usage": (self.current_profile["start_cpu"] + end_cpu) / 2,
            "timestamp": end_time,
        }

        self.profiles[name] = profile_result
        self.current_profile = None

        return profile_result

    def _get_memory_usage(self) -> float:
        """
        Get memory usage.

        Returns:
            Memory usage in MB
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get profiling summary.

        Returns:
            Profiling summary
        """
        if not self.profiles:
            return {}

        total_time = sum(p["execution_time"] for p in self.profiles.values())
        total_memory = sum(p["memory_delta"] for p in self.profiles.values())

        return {
            "total_execution_time": total_time,
            "total_memory_delta": total_memory,
            "profile_count": len(self.profiles),
            "profiles": self.profiles,
        }


class OptimizationSuite:
    """Optimization suite."""

    def __init__(self) -> None:
        """Initialize optimization suite."""
        self.optimization_levels = {
            OptimizationLevel.NONE: OptimizationConfig(
                use_cuda=False,
                optimization_level=OptimizationLevel.NONE,
                memory_optimization=False,
                algorithm_optimization=False,
                adaptive_parameters=False,
                profiling_enabled=False,
                cache_enabled=False,
                parallel_processing=False,
            ),
            OptimizationLevel.BASIC: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.BASIC,
                memory_optimization=True,
                algorithm_optimization=False,
                adaptive_parameters=False,
                profiling_enabled=True,
                cache_enabled=False,
                parallel_processing=False,
            ),
            OptimizationLevel.ADVANCED: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.ADVANCED,
                memory_optimization=True,
                algorithm_optimization=True,
                adaptive_parameters=True,
                profiling_enabled=True,
                cache_enabled=True,
                parallel_processing=True,
            ),
            OptimizationLevel.MAXIMUM: OptimizationConfig(
                use_cuda=True,
                optimization_level=OptimizationLevel.MAXIMUM,
                memory_optimization=True,
                algorithm_optimization=True,
                adaptive_parameters=True,
                profiling_enabled=True,
                cache_enabled=True,
                parallel_processing=True,
            ),
        }

    def run_optimization_comparison(self, config: Any) -> Dict[str, Any]:
        """
        Run optimization comparison.

        Args:
            config: Model configuration

        Returns:
            Comparison results
        """
        comparison_results = {}

        for level, optimization_config in self.optimization_levels.items():
            print(f"Running optimization level: {level.value}")

            # Create optimized model
            optimized_model = OptimizedProtonModel(config, optimization_config)

            # Run model
            start_time = time.time()
            results = optimized_model.run_optimized()
            end_time = time.time()

            # Save results
            comparison_results[level.value] = {
                "execution_time": end_time - start_time,
                "performance_metrics": results["performance_metrics"],
                "device_info": results["device_info"],
                "optimization_report": optimized_model.get_optimization_report(),
            }

        return comparison_results

    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate comparison report.

        Args:
            comparison_results: Comparison results

        Returns:
            Comparison report
        """
        report = []
        report.append("=" * 80)
        report.append("OPTIMIZATION COMPARISON REPORT")
        report.append("=" * 80)

        # Comparison table
        report.append("COMPARISON TABLE:")
        report.append("-" * 80)
        header = (
            f"{'Level':<15} {'Time (sec)':<12} {'Memory (MB)':<12} "
            f"{'GPU (MB)':<10} {'Throughput':<20}"
        )
        report.append(header)
        report.append("-" * 80)

        for level, results in comparison_results.items():
            metrics = results["performance_metrics"]
            row = (
                f"{level:<15} {metrics.execution_time:<12.2f} "
                f"{metrics.memory_usage:<12.2f} {metrics.gpu_utilization:<10.2f} "
                f"{metrics.throughput:<20.2e}"
            )
            report.append(row)

        report.append("")

        # Analysis
        report.append("ANALYSIS:")
        report.append("-" * 40)

        # Find best results
        best_time = min(
            results["performance_metrics"].execution_time
            for results in comparison_results.values()
        )
        best_memory = min(
            results["performance_metrics"].memory_usage
            for results in comparison_results.values()
        )
        best_throughput = max(
            results["performance_metrics"].throughput
            for results in comparison_results.values()
        )

        for level, results in comparison_results.items():
            metrics = results["performance_metrics"]
            if metrics.execution_time == best_time:
                report.append(f"Best execution time: {level}")
            if metrics.memory_usage == best_memory:
                report.append(f"Best memory usage: {level}")
            if metrics.throughput == best_throughput:
                report.append(f"Best throughput: {level}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


class OptimizedProtonModel:
    """Optimized proton model."""

    def __init__(self, config: Any, optimization_config: OptimizationConfig):
        """
        Initialize optimized model.

        Args:
            config: Model configuration
            optimization_config: Optimization configuration
        """
        self.config = config
        self.optimization_config = optimization_config

        # Initialize optimizers
        self.cuda_manager = get_cuda_manager()
        self.memory_optimizer = MemoryOptimizer(self.cuda_manager)
        self.algorithm_optimizer = AlgorithmOptimizer(self.cuda_manager)
        self.parameter_optimizer = AdaptiveParameterOptimizer()
        self.profiler = PerformanceProfiler()

        # Device information
        self.device_info = self.cuda_manager.get_detailed_status()

        # Performance metrics
        self.performance_metrics: Optional[PerformanceMetrics] = None

    def optimize_configuration(self) -> Any:
        """
        Optimize model configuration.

        Returns:
            Optimized configuration
        """
        optimized_config = self.config

        if self.optimization_config.adaptive_parameters:
            # Adaptive parameter adjustment
            if self.device_info["available"]:
                # GPU available - increase grid
                optimized_config.grid_size = min(128, self.config.grid_size * 2)
                optimized_config.max_iterations = min(
                    2000, self.config.max_iterations * 2
                )
            else:
                # CPU only - optimize for CPU
                optimized_config.grid_size = min(64, self.config.grid_size)
                optimized_config.max_iterations = min(1000, self.config.max_iterations)

        return optimized_config

    def run_optimized(self) -> Dict[str, Any]:
        """
        Run optimized model.

        Returns:
            Optimized model results
        """
        # Start profiling
        self.profiler.start_profile("total_execution")

        # Optimize configuration
        self.optimize_configuration()

        # Simulate model execution (placeholder)
        time.sleep(0.1)  # Simulate work

        # End profiling
        total_profile = self.profiler.end_profile("total_execution")

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(total_profile)

        # Profiling summary
        profile_summary = self.profiler.get_profile_summary()

        return {
            "results": {"iterations": 100, "converged": True},  # Placeholder
            "performance_metrics": self.performance_metrics,
            "profile_summary": profile_summary,
            "device_info": self.device_info,
            "optimization_config": self.optimization_config,
        }

    def _calculate_performance_metrics(
        self, total_profile: Dict[str, Any]
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.

        Args:
            total_profile: Total execution profile

        Returns:
            Performance metrics
        """
        # Memory usage
        memory_info = self.memory_optimizer.get_memory_usage()

        # Calculate throughput
        total_operations = self.config.grid_size**3 * 100  # Placeholder
        throughput = total_operations / total_profile["execution_time"]

        # Convergence rate
        convergence_rate = 1.0 / 100  # Placeholder

        return PerformanceMetrics(
            execution_time=total_profile["execution_time"],
            memory_usage=memory_info["cpu_memory_mb"],
            gpu_utilization=memory_info.get("gpu_memory_mb", 0),
            cpu_utilization=total_profile["cpu_usage"],
            iterations=100,  # Placeholder
            convergence_rate=convergence_rate,
            throughput=throughput,
        )

    def get_optimization_report(self) -> str:
        """
        Get optimization report.

        Returns:
            Optimization report
        """
        report = []
        report.append("=" * 80)
        report.append("PROTON MODEL OPTIMIZATION REPORT")
        report.append("=" * 80)

        # Device information
        report.append("DEVICE INFORMATION:")
        report.append("-" * 40)
        if self.device_info["available"]:
            report.append("CUDA available: YES")
            if self.device_info["devices"]:
                device = self.device_info["devices"][0]
                report.append(f"Device: {device['name']}")
                report.append(f"Compute capability: {device['compute_capability']}")
                report.append(
                    f"Total memory: {device['memory_total_mb'] / 1024:.2f} GB"
                )
        else:
            report.append("CUDA available: NO")
        report.append("")

        # Performance metrics
        if self.performance_metrics:
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 40)
            report.append(
                f"Execution time: {self.performance_metrics.execution_time:.2f} sec"
            )
            report.append(
                f"Memory usage: {self.performance_metrics.memory_usage:.2f} MB"
            )
            report.append(
                f"GPU utilization: {self.performance_metrics.gpu_utilization:.2f} MB"
            )
            report.append(
                f"CPU utilization: {self.performance_metrics.cpu_utilization:.2f}%"
            )
            report.append(f"Iterations: {self.performance_metrics.iterations}")
            report.append(
                f"Convergence rate: {self.performance_metrics.convergence_rate:.6f}"
            )
            report.append(
                f"Throughput: {self.performance_metrics.throughput:.2e} ops/sec"
            )
            report.append("")

        # Optimization configuration
        report.append("OPTIMIZATION CONFIGURATION:")
        report.append("-" * 40)
        cuda_use = "YES" if self.optimization_config.use_cuda else "NO"
        report.append(f"Use CUDA: {cuda_use}")
        report.append(
            f"Optimization level: {self.optimization_config.optimization_level.value}"
        )
        mem_opt = "YES" if self.optimization_config.memory_optimization else "NO"
        report.append(f"Memory optimization: {mem_opt}")
        alg_opt = "YES" if self.optimization_config.algorithm_optimization else "NO"
        report.append(f"Algorithm optimization: {alg_opt}")
        adapt_params = "YES" if self.optimization_config.adaptive_parameters else "NO"
        report.append(f"Adaptive parameters: {adapt_params}")
        profiling = "YES" if self.optimization_config.profiling_enabled else "NO"
        report.append(f"Profiling: {profiling}")
        caching = "YES" if self.optimization_config.cache_enabled else "NO"
        report.append(f"Caching: {caching}")
        parallel = "YES" if self.optimization_config.parallel_processing else "NO"
        report.append(f"Parallel processing: {parallel}")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)


def run_optimization_benchmark() -> Dict[str, Any]:
    """Run optimization benchmark."""
    # Create configuration
    config = type(
        "Config",
        (),
        {
            "grid_size": 64,
            "box_size": 4.0,
            "max_iterations": 500,
            "validation_enabled": True,
        },
    )()

    # Create optimization suite
    optimization_suite = OptimizationSuite()

    # Run comparison
    comparison_results = optimization_suite.run_optimization_comparison(config)

    # Generate report
    report = optimization_suite.generate_comparison_report(comparison_results)

    print(report)

    return comparison_results


if __name__ == "__main__":
    run_optimization_benchmark()
