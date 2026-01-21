#!/usr/bin/env python3
"""
Performance Tracker Module

Comprehensive performance tracking for inference operations.
Tracks timing, memory usage, and operation counts to identify bottlenecks.
Includes bounds checking to prevent unlimited memory growth.
"""

import time
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Comprehensive performance tracking for inference operations.

    Tracks timing, memory usage, and operation counts to identify bottlenecks.
    Includes bounds checking to prevent unlimited memory growth.
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize performance tracker with configurable bounds.

        Args:
            max_history_size: Maximum number of entries to keep in history lists
        """
        self.max_history_size = max_history_size
        self.reset()

    def reset(self):
        """Reset all performance metrics."""
        self.start_time = time.perf_counter()
        self.step_times = []
        self.operation_times = {}
        self.strategy_times = {}
        self.spatial_index_times = {}
        self.memory_usage = []
        self.step_metrics = []

    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        # Store start time
        self._current_ops = getattr(self, '_current_ops', {})
        self._current_ops[operation_name] = time.perf_counter()

    def end_operation(self, operation_name: str):
        """End timing an operation."""
        if operation_name in getattr(self, '_current_ops', {}):
            start_time = self._current_ops[operation_name]
            duration = time.perf_counter() - start_time
            self.operation_times[operation_name].append(duration)
            del self._current_ops[operation_name]

    def record_strategy_time(self, strategy_name: str, duration: float):
        """Record time spent in a strategy with bounds checking."""
        if strategy_name not in self.strategy_times:
            self.strategy_times[strategy_name] = []
        self.strategy_times[strategy_name].append(duration)

        # Bounds checking: keep only the most recent entries
        if len(self.strategy_times[strategy_name]) > self.max_history_size:
            self.strategy_times[strategy_name] = self.strategy_times[strategy_name][-self.max_history_size:]

    def record_spatial_index_time(self, operation: str, duration: float):
        """Record time spent in spatial index operations with bounds checking."""
        if operation not in self.spatial_index_times:
            self.spatial_index_times[operation] = []
        self.spatial_index_times[operation].append(duration)

        # Bounds checking: keep only the most recent entries
        if len(self.spatial_index_times[operation]) > self.max_history_size:
            self.spatial_index_times[operation] = self.spatial_index_times[operation][-self.max_history_size:]

    def record_step_metrics(self, step: int, streets_before: int, streets_after: int,
                          candidates_count: int, strategy_stats: Dict[str, int]):
        """Record metrics for a completed inference step with bounds checking."""
        step_duration = time.perf_counter() - self.start_time
        self.step_times.append(step_duration)

        self.step_metrics.append({
            'step': step,
            'duration': step_duration,
            'streets_before': streets_before,
            'streets_after': streets_after,
            'candidates_count': candidates_count,
            'strategy_stats': strategy_stats.copy()
        })

        # Bounds checking: keep only the most recent entries
        if len(self.step_times) > self.max_history_size:
            self.step_times = self.step_times[-self.max_history_size:]
        if len(self.step_metrics) > self.max_history_size:
            self.step_metrics = self.step_metrics[-self.max_history_size:]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_time = time.perf_counter() - self.start_time

        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                operation_stats[op_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'pct_total': sum(times) / total_time * 100
                }

        # Calculate strategy statistics
        strategy_stats = {}
        for strategy_name, times in self.strategy_times.items():
            if times:
                strategy_stats[strategy_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'pct_total': sum(times) / total_time * 100
                }

        # Calculate spatial index statistics
        spatial_stats = {}
        for op_name, times in self.spatial_index_times.items():
            if times:
                spatial_stats[op_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times)
                }

        # Calculate throughput metrics
        total_steps = len(self.step_times)
        total_streets_processed = sum(m['streets_before'] - m['streets_after'] for m in self.step_metrics)

        return {
            'total_time': total_time,
            'total_steps': total_steps,
            'total_streets_processed': total_streets_processed,
            'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            'streets_per_second': total_streets_processed / total_time if total_time > 0 else 0,
            'steps_per_second': total_steps / total_time if total_time > 0 else 0,
            'operation_stats': operation_stats,
            'strategy_stats': strategy_stats,
            'spatial_index_stats': spatial_stats,
            'step_metrics': self.step_metrics
        }

    def log_performance_summary(self):
        """Log a human-readable performance summary."""
        stats = self.get_summary_stats()

        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total time: {stats['total_time']:.2f}s")
        logger.info(f"Total steps: {stats['total_steps']}")
        logger.info(f"Streets processed: {stats['total_streets_processed']}")
        logger.info(f"Avg step time: {stats['avg_step_time']:.2f}s")
        logger.info(f"Streets per second: {stats['streets_per_second']:.2f}")
        logger.info(f"Steps per second: {stats['steps_per_second']:.2f}")

        # Log top time-consuming operations
        if stats['operation_stats']:
            logger.info("Top operations by time:")
            sorted_ops = sorted(stats['operation_stats'].items(),
                              key=lambda x: x[1]['total_time'], reverse=True)
            for op_name, op_stats in sorted_ops[:5]:
                logger.info(f"  {op_name}: {op_stats['total_time']:.3f}s ({op_stats['pct_total']:.1f}%)")

        # Log strategy performance
        if stats['strategy_stats']:
            logger.info("Strategy performance:")
            sorted_strategies = sorted(stats['strategy_stats'].items(),
                                     key=lambda x: x[1]['total_time'], reverse=True)
            for strategy_name, strategy_stats in sorted_strategies:
                logger.info(f"  {strategy_name}: {strategy_stats['total_time']:.3f}s ({strategy_stats['pct_total']:.1f}%)")
