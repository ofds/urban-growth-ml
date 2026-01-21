#!/usr/bin/env python3
"""
Unit tests for PerformanceTracker module.
"""

import time
import unittest
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inverse.metrics.performance_tracker import PerformanceTracker


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for PerformanceTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PerformanceTracker(max_history_size=10)

    def test_initialization(self):
        """Test PerformanceTracker initialization."""
        tracker = PerformanceTracker(max_history_size=500)
        self.assertEqual(tracker.max_history_size, 500)
        self.assertIsInstance(tracker.step_times, list)
        self.assertIsInstance(tracker.operation_times, dict)
        self.assertIsInstance(tracker.strategy_times, dict)
        self.assertIsInstance(tracker.spatial_index_times, dict)

    def test_reset(self):
        """Test reset functionality."""
        # Add some data
        self.tracker.step_times.append(1.0)
        self.tracker.operation_times['test'] = [0.5]
        self.tracker.strategy_times['test_strategy'] = [0.3]

        # Reset
        self.tracker.reset()

        # Verify reset
        self.assertEqual(len(self.tracker.step_times), 0)
        self.assertEqual(len(self.tracker.operation_times), 0)
        self.assertEqual(len(self.tracker.strategy_times), 0)
        self.assertEqual(len(self.tracker.spatial_index_times), 0)

    def test_operation_timing(self):
        """Test operation timing functionality."""
        # Start operation
        self.tracker.start_operation("test_operation")

        # Simulate some work
        time.sleep(0.01)

        # End operation
        self.tracker.end_operation("test_operation")

        # Verify timing was recorded
        self.assertIn("test_operation", self.tracker.operation_times)
        self.assertEqual(len(self.tracker.operation_times["test_operation"]), 1)
        self.assertGreater(self.tracker.operation_times["test_operation"][0], 0)

    def test_operation_timing_without_start(self):
        """Test ending operation that was never started."""
        # This should not raise an error
        self.tracker.end_operation("nonexistent_operation")
        # Should not have recorded anything
        self.assertNotIn("nonexistent_operation", self.tracker.operation_times)

    def test_strategy_time_recording(self):
        """Test strategy time recording with bounds checking."""
        # Record some strategy times
        for i in range(15):  # More than max_history_size (10)
            self.tracker.record_strategy_time("test_strategy", float(i))

        # Should only keep the most recent entries
        self.assertEqual(len(self.tracker.strategy_times["test_strategy"]), 10)
        # Should keep the last 10 entries
        expected = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        self.assertEqual(self.tracker.strategy_times["test_strategy"], expected)

    def test_spatial_index_time_recording(self):
        """Test spatial index time recording with bounds checking."""
        # Record some spatial index times
        for i in range(12):  # More than max_history_size (10)
            self.tracker.record_spatial_index_time("build_index", float(i) * 0.1)

        # Should only keep the most recent entries
        self.assertEqual(len(self.tracker.spatial_index_times["build_index"]), 10)
        # Should keep the last 10 entries
        actual = self.tracker.spatial_index_times["build_index"]
        expected = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        for a, e in zip(actual, expected):
            self.assertAlmostEqual(a, e, places=7)

    def test_step_metrics_recording(self):
        """Test step metrics recording with bounds checking."""
        strategy_stats = {"strategy1": 5, "strategy2": 3}

        # Record step metrics
        for i in range(12):  # More than max_history_size (10)
            self.tracker.record_step_metrics(
                step=i,
                streets_before=100 + i,
                streets_after=99 + i,
                candidates_count=10 + i,
                strategy_stats=strategy_stats
            )

        # Should only keep the most recent entries
        self.assertEqual(len(self.tracker.step_times), 10)
        self.assertEqual(len(self.tracker.step_metrics), 10)

        # Check the last recorded step
        last_metric = self.tracker.step_metrics[-1]
        self.assertEqual(last_metric['step'], 11)
        self.assertEqual(last_metric['streets_before'], 111)
        self.assertEqual(last_metric['streets_after'], 110)
        self.assertEqual(last_metric['candidates_count'], 21)
        self.assertEqual(last_metric['strategy_stats'], strategy_stats)

    def test_get_summary_stats_empty(self):
        """Test summary stats with no data."""
        stats = self.tracker.get_summary_stats()

        expected_keys = [
            'total_time', 'total_steps', 'total_streets_processed',
            'avg_step_time', 'streets_per_second', 'steps_per_second',
            'operation_stats', 'strategy_stats', 'spatial_index_stats', 'step_metrics'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        self.assertEqual(stats['total_steps'], 0)
        self.assertEqual(stats['total_streets_processed'], 0)
        self.assertEqual(stats['avg_step_time'], 0)
        self.assertEqual(stats['streets_per_second'], 0)
        self.assertEqual(stats['steps_per_second'], 0)

    def test_get_summary_stats_with_data(self):
        """Test summary stats with recorded data."""
        # Add some test data
        self.tracker.step_times = [1.0, 2.0, 3.0]
        self.tracker.step_metrics = [
            {'streets_before': 100, 'streets_after': 99},
            {'streets_before': 99, 'streets_after': 98},
            {'streets_before': 98, 'streets_after': 97}
        ]
        self.tracker.operation_times = {
            'op1': [0.5, 1.0],
            'op2': [0.2, 0.3, 0.4]
        }
        self.tracker.strategy_times = {
            'strategy1': [0.1, 0.2],
            'strategy2': [0.3, 0.4, 0.5]
        }
        self.tracker.spatial_index_times = {
            'build': [0.05, 0.06]
        }

        stats = self.tracker.get_summary_stats()

        # Test basic metrics
        self.assertEqual(stats['total_steps'], 3)
        self.assertEqual(stats['total_streets_processed'], 3)  # 100->99, 99->98, 98->97
        self.assertAlmostEqual(stats['avg_step_time'], 2.0, places=1)

        # Test operation stats
        self.assertIn('op1', stats['operation_stats'])
        self.assertIn('op2', stats['operation_stats'])
        op1_stats = stats['operation_stats']['op1']
        self.assertEqual(op1_stats['count'], 2)
        self.assertAlmostEqual(op1_stats['total_time'], 1.5)
        self.assertAlmostEqual(op1_stats['avg_time'], 0.75)

        # Test strategy stats
        self.assertIn('strategy1', stats['strategy_stats'])
        self.assertIn('strategy2', stats['strategy_stats'])
        strategy2_stats = stats['strategy_stats']['strategy2']
        self.assertEqual(strategy2_stats['count'], 3)
        self.assertAlmostEqual(strategy2_stats['total_time'], 1.2)

        # Test spatial index stats
        self.assertIn('build', stats['spatial_index_stats'])
        build_stats = stats['spatial_index_stats']['build']
        self.assertEqual(build_stats['count'], 2)
        self.assertAlmostEqual(build_stats['avg_time'], 0.055)

    @patch('inverse.metrics.performance_tracker.logger')
    def test_log_performance_summary(self, mock_logger):
        """Test performance summary logging."""
        # Add some test data
        self.tracker.step_times = [1.0, 2.0]
        self.tracker.step_metrics = [
            {'streets_before': 100, 'streets_after': 99},
            {'streets_before': 99, 'streets_after': 98}
        ]
        self.tracker.operation_times = {'op1': [1.0]}
        self.tracker.strategy_times = {'strategy1': [0.5]}

        # Call log function
        self.tracker.log_performance_summary()

        # Verify logging calls were made
        mock_logger.info.assert_called()
        calls = mock_logger.info.call_args_list

        # Check that summary header was logged
        self.assertTrue(any("PERFORMANCE SUMMARY" in str(call) for call in calls))

        # Check that key metrics were logged
        logged_messages = [str(call) for call in calls]
        logged_text = ' '.join(logged_messages)

        self.assertIn("Total time:", logged_text)
        self.assertIn("Total steps: 2", logged_text)
        self.assertIn("Streets processed: 2", logged_text)

    def test_multiple_operations_timing(self):
        """Test timing multiple concurrent operations."""
        # Start multiple operations
        self.tracker.start_operation("op1")
        self.tracker.start_operation("op2")

        time.sleep(0.01)

        # End them in different order
        self.tracker.end_operation("op1")
        self.tracker.end_operation("op2")

        # Both should be recorded
        self.assertIn("op1", self.tracker.operation_times)
        self.assertIn("op2", self.tracker.operation_times)
        self.assertEqual(len(self.tracker.operation_times["op1"]), 1)
        self.assertEqual(len(self.tracker.operation_times["op2"]), 1)


if __name__ == '__main__':
    unittest.main()
