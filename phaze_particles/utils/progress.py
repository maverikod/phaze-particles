#!/usr/bin/env python3
"""
Progress tracking and logging utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import logging
import sys
import time
import psutil
from typing import Any, Callable, Dict, List, Optional
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class ProgressInfo:
    """Progress information container."""

    current: int
    total: int
    percentage: float
    elapsed_time: float
    estimated_remaining: float
    rate: float  # items per second


class ProgressBar:
    """
    Progress bar with time estimation and percentage display.

    Provides visual progress indication for long-running operations
    with time estimation and percentage completion.
    """

    def __init__(
        self, total: int, description: str = "Progress", width: int = 50, callback: Optional[Callable] = None
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items to process
            description: Description of the operation
            width: Width of the progress bar in characters
            callback: Optional callback function for progress updates
        """
        self.total = total
        self.description = description
        self.width = width
        self.current: int = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms
        self.callback = callback

    def update(self, increment: int = 1) -> None:
        """
        Update progress by increment.

        Args:
            increment: Number of items completed
        """
        self.current = min(self.current + increment, self.total)
        self._display()

    def set_progress(self, current: int) -> None:
        """
        Set current progress.

        Args:
            current: Current progress value
        """
        self.current = min(max(current, 0), self.total)  # Ensure non-negative and not exceeding total
        self._display()

    def increment(self, amount: int = 1) -> None:
        """
        Increment progress by amount.

        Args:
            amount: Amount to increment
        """
        self.current = min(self.current + amount, self.total)
        self._display()

    @property
    def percentage(self) -> float:
        """Get current percentage."""
        return (self.current / self.total) * 100 if self.total > 0 else 0

    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.current >= self.total

    def reset(self) -> None:
        """Reset progress to zero."""
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0

    def __str__(self) -> str:
        """String representation of progress bar."""
        percentage = self.percentage
        return f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)"

    def _display(self) -> None:
        """Display progress bar."""
        current_time = time.time()

        # Only update if enough time has passed
        if (
            current_time - self.last_update < self.update_interval
            and self.current < self.total
        ):
            return

        self.last_update = current_time

        # Calculate progress information
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed_time = current_time - self.start_time

        # Calculate rate and estimated remaining time
        if elapsed_time > 0:
            rate = self.current / elapsed_time
            if rate > 0:
                estimated_remaining = (self.total - self.current) / rate
            else:
                estimated_remaining = 0
        else:
            rate = 0
            estimated_remaining = 0

        # Create progress bar
        filled_width = int(self.width * percentage / 100)
        bar = "█" * filled_width + "░" * (self.width - filled_width)

        # Format time strings
        elapsed_str = self._format_time(elapsed_time)
        remaining_str = self._format_time(estimated_remaining)

        # Create status line
        status = (
            f"\r{self.description}: |{bar}| "
            f"{self.current}/{self.total} ({percentage:.1f}%) "
            f"Elapsed: {elapsed_str} "
            f"Remaining: {remaining_str} "
            f"Rate: {rate:.1f} items/s"
        )

        # Print status
        sys.stdout.write(status)
        sys.stdout.flush()

        # Print newline when complete
        if self.current >= self.total:
            print()  # Newline

        # Call callback if provided
        if self.callback:
            try:
                self.callback(self.current, self.total, percentage)
            except Exception:
                pass  # Ignore callback errors

    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.1f}s"

    def finish(self) -> None:
        """Finish progress bar."""
        self.current = self.total
        self._display()


class ProgressLogger:
    """
    Logger with progress tracking capabilities.

    Provides logging functionality with progress indication
    for long-running operations.
    """

    def __init__(self, name: str = __name__, level: int = logging.INFO) -> None:
        """
        Initialize progress logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def progress(self, message: str, current: int, total: int) -> None:
        """
        Log progress message.

        Args:
            message: Progress message
            current: Current progress
            total: Total items
        """
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(f"{message} - {current}/{total} ({percentage:.1f}%)")


@contextmanager
def progress_tracker(
    total: int, description: str = "Processing", use_bar: bool = True
):
    """
    Context manager for progress tracking.

    Args:
        total: Total number of items to process
        description: Description of the operation
        use_bar: Whether to use progress bar or logging

    Yields:
        Progress tracker object
    """
    if use_bar:
        tracker: Any = ProgressBar(total, description)
    else:
        tracker: Any = ProgressLogger()

    try:
        yield tracker
    finally:
        if hasattr(tracker, "finish"):
            tracker.finish()


class PerformanceMonitor:
    """
    Performance monitoring and timing utilities.

    Tracks execution time, memory usage, and performance metrics
    for operations and provides detailed reporting.
    """

    def __init__(self, name: str = "Operation"):
        """
        Initialize performance monitor.

        Args:
            name: Name of the operation being monitored
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.name}")

    def stop(self) -> None:
        """Stop performance monitoring."""
        if self.start_time is None:
            return

        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.metrics["duration"] = duration

        self.logger.info(f"Completed {self.name} in {duration:.2f} seconds")

    def add_metric(self, name: str, value: Any) -> None:
        """
        Add performance metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value

    def get_report(self) -> str:
        """
        Get performance report.

        Returns:
            Formatted performance report
        """
        if self.start_time is None:
            return f"{self.name}: Not started"

        duration = self.metrics.get("duration", 0)
        report = f"\n{self.name} Performance Report:\n"
        report += f"{'='*50}\n"
        report += f"Duration: {duration:.2f} seconds\n"

        for name, value in self.metrics.items():
            if name != "duration":
                report += f"{name}: {value}\n"

        report += f"{'='*50}\n"
        return report

    def __enter__(self) -> "PerformanceMonitor":
        """Enter context manager."""
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.stop()

    def start_timing(self, name: str) -> None:
        """
        Start timing a specific operation.
        
        Args:
            name: Name of the operation
        """
        self.metrics[f"{name}_start"] = time.time()

    def stop_timing(self, name: str) -> float:
        """
        Stop timing a specific operation.
        
        Args:
            name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        start_key = f"{name}_start"
        if start_key not in self.metrics:
            return 0.0
        
        duration = time.time() - self.metrics[start_key]
        self.metrics[f"{name}_duration"] = duration
        return duration

    def record_metric(self, name: str, value: Any) -> None:
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value

    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics["memory_usage_mb"] = memory_info.rss / 1024 / 1024
        except Exception:
            pass  # Ignore memory monitoring errors

    @property
    def timing(self) -> Dict[str, float]:
        """Get timing metrics."""
        return {k: v for k, v in self.metrics.items() if k.endswith('_duration')}


class TimeEstimator:
    """
    Time estimation utility for progress tracking.
    
    Provides accurate time estimation based on current progress
    and historical data.
    """
    
    def __init__(self):
        """Initialize time estimator."""
        self.start_time = time.time()
        self.checkpoints: List[Tuple[float, float]] = []  # (progress, timestamp)
        self.last_progress = 0.0
        self.last_time = self.start_time
    
    def update(self, progress: float) -> None:
        """
        Update progress for time estimation.
        
        Args:
            progress: Current progress (0.0 to 1.0)
        """
        current_time = time.time()
        
        # Only add checkpoint if progress has increased significantly
        if progress > self.last_progress + 0.01:  # 1% threshold
            self.checkpoints.append((progress, current_time))
            self.last_progress = progress
            self.last_time = current_time
    
    def estimate_remaining(self, current_progress: float) -> float:
        """
        Estimate remaining time.
        
        Args:
            current_progress: Current progress (0.0 to 1.0)
            
        Returns:
            Estimated remaining time in seconds
        """
        if current_progress <= 0:
            return float('inf')
        
        if current_progress >= 1.0:
            return 0.0
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Simple linear estimation
        if current_progress > 0:
            total_estimated = elapsed / current_progress
            remaining = total_estimated - elapsed
            return max(0, remaining)
        
        return 0.0
    
    def estimate_total(self, current_progress: float) -> float:
        """
        Estimate total time.
        
        Args:
            current_progress: Current progress (0.0 to 1.0)
            
        Returns:
            Estimated total time in seconds
        """
        if current_progress <= 0:
            return float('inf')
        
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        return elapsed / current_progress
    
    def reset(self) -> None:
        """Reset time estimator."""
        self.start_time = time.time()
        self.checkpoints.clear()
        self.last_progress = 0.0
        self.last_time = self.start_time


class ProgressCallback:
    """
    Progress callback system for monitoring progress updates.
    
    Allows registration of multiple callbacks that are called
    when progress is updated.
    """
    
    def __init__(self):
        """Initialize progress callback system."""
        self.callbacks: List[Callable[[int, int, float], None]] = []
        self.error_handlers: List[Callable[[Exception], None]] = []
    
    def add_callback(self, callback: Callable[[int, int, float], None]) -> None:
        """
        Add progress callback.
        
        Args:
            callback: Callback function (current, total, percentage)
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[int, int, float], None]) -> bool:
        """
        Remove progress callback.
        
        Args:
            callback: Callback function to remove
            
        Returns:
            True if callback was removed, False if not found
        """
        try:
            self.callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def add_error_handler(self, handler: Callable[[Exception], None]) -> None:
        """
        Add error handler for callback errors.
        
        Args:
            handler: Error handler function
        """
        self.error_handlers.append(handler)
    
    def execute_callbacks(self, current: int, total: int, percentage: float) -> None:
        """
        Execute all registered callbacks.
        
        Args:
            current: Current progress
            total: Total progress
            percentage: Progress percentage
        """
        for callback in self.callbacks:
            try:
                callback(current, total, percentage)
            except Exception as e:
                for handler in self.error_handlers:
                    try:
                        handler(e)
                    except Exception:
                        pass  # Ignore handler errors


def create_progress_bar(
    total: int, description: str = "Progress", callback: Optional[Callable] = None
) -> ProgressBar:
    """
    Create a progress bar.

    Args:
        total: Total number of items
        description: Description of the operation
        callback: Optional callback function

    Returns:
        ProgressBar instance
    """
    return ProgressBar(total, description, callback=callback)


def create_progress_logger(name: str = __name__) -> ProgressLogger:
    """
    Create a progress logger.

    Args:
        name: Logger name

    Returns:
        ProgressLogger instance
    """
    return ProgressLogger(name)


def create_performance_monitor(name: str = "Operation") -> PerformanceMonitor:
    """
    Create a performance monitor.

    Args:
        name: Operation name

    Returns:
        PerformanceMonitor instance
    """
    monitor = PerformanceMonitor(name)
    monitor.start()  # Auto-start the monitor
    return monitor
