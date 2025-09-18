#!/usr/bin/env python3
"""
Progress tracking and logging utilities.

Author: Vasiliy Zdanovskiy
Email: vasilyvz@gmail.com
"""

import logging
import sys
import time
from typing import Any, Dict, Optional
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
        self, total: int, description: str = "Progress", width: int = 50
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items to process
            description: Description of the operation
            width: Width of the progress bar in characters
        """
        self.total = total
        self.description = description
        self.width = width
        self.current: int = 0
        self.start_time = time.time()
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms

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
        self.current = min(current, self.total)
        self._display()

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


def create_progress_bar(
    total: int, description: str = "Progress"
) -> ProgressBar:
    """
    Create a progress bar.

    Args:
        total: Total number of items
        description: Description of the operation

    Returns:
        ProgressBar instance
    """
    return ProgressBar(total, description)


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
    return PerformanceMonitor(name)
