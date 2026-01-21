#!/usr/bin/env python3
"""
Unit tests for PeripheralStrategy class.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from src.inverse.strategies.peripheral_strategy import PeripheralStrategy
from src.inverse.core.config import InferenceConfig
from src.core.contracts import GrowthState
from src.inverse.data_structures import InverseGrowthAction, ActionType


class TestPeripheralStrategy(unittest.TestCase):
    """Test cases for PeripheralStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        self.strategy = PeripheralStrategy(weight=0.8, config=self.config)

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "peripheral")
        self.assertEqual(self.strategy.weight, 0.8)
        self.assertIsInstance(self.strategy.config, InferenceConfig)

    def test_initialization_defaults(self):
        """Test strategy initialization with defaults."""
        strategy = PeripheralStrategy()
        self.assertEqual(strategy.name, "peripheral")
        self.assertEqual(strategy.weight, 0.8)  # Default weight

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        strategy = PeripheralStrategy(config=self.config)
        self.assertIsNotNone(strategy)

        # Invalid spatial query radius should raise
        invalid_config = InferenceConfig()
        invalid_config.spatial.spatial_query_radius = 0

        with self.assertRaises(ValueError):
            PeripheralStrategy(config=invalid_config)

    def test_generate_candidates_no_frontiers(self):
        """Test candidate generation with no frontiers."""
        mock_state = Mock()
        mock_state.frontiers = []

        candidates = self.strategy.generate_candidates(mock_state, set())

        self.assertEqual(len(candidates), 0)

    def test_generate_candidates_no_dead_end_frontiers(self):
        """Test candidate generation with frontiers but no dead-ends."""
        mock_state = Mock()
        mock_frontier = Mock()
        mock_frontier.frontier_type = "block_edge"  # Not dead_end
        mock_state.frontiers = [mock_frontier]

        candidates = self.strategy.generate_candidates(mock_state, set())

        self.assertEqual(len(candidates), 0)

    def test_generate_candidates_with_dead_end_frontiers(self):
        """Test candidate generation with dead-end frontiers."""
        # Create mock state
        mock_state = Mock()

        # Create mock frontiers
        frontier1 = Mock()
        frontier1.frontier_type = "dead_end"
        frontier1.geometry = LineString([(0, 0), (1, 0)])  # Horizontal, close to center
        frontier1.edge_id = (1, 2)

        frontier2 = Mock()
        frontier2.frontier_type = "dead_end"
        frontier2.geometry = LineString([(10, 10), (11, 10)])  # Farther from center
        frontier2.edge_id = (3, 4)

        mock_state.frontiers = [frontier1, frontier2]

        # Mock city center calculation
        with patch.object(self.strategy, '_get_city_center', return_value=Point(0, 0)):
            # Mock action creation to return valid actions
            with patch.object(self.strategy, '_create_action_from_frontier') as mock_create:
                mock_action1 = Mock()
                mock_action1.intent_params = {'strategy': 'peripheral'}
                mock_create.return_value = mock_action1

                candidates = self.strategy.generate_candidates(mock_state, set())

                # Should create actions for both frontiers
                self.assertEqual(len(candidates), 2)
                self.assertEqual(mock_create.call_count, 2)

                # Check that farther frontier gets higher confidence
                calls = mock_create.call_args_list
                # After sorting by confidence (descending), first call should be farther frontier
                # calls[0] is first processed (farther/higher confidence), calls[1] is second processed (closer/lower confidence)
                far_confidence = calls[0][0][1]    # First call (farther frontier, higher confidence)
                close_confidence = calls[1][0][1]  # Second call (closer frontier, lower confidence)
                self.assertGreater(far_confidence, close_confidence)

    def test_generate_candidates_skip_skeleton_edges(self):
        """Test that skeleton edges are skipped."""
        mock_state = Mock()

        frontier = Mock()
        frontier.frontier_type = "dead_end"
        frontier.geometry = LineString([(10, 10), (11, 10)])
        frontier.edge_id = (1, 2)  # This is a skeleton edge

        mock_state.frontiers = [frontier]
        skeleton_edges = {(1, 2)}  # Mark as skeleton

        with patch.object(self.strategy, '_get_city_center', return_value=Point(0, 0)):
            candidates = self.strategy.generate_candidates(mock_state, skeleton_edges)

            # Should skip skeleton edge
            self.assertEqual(len(candidates), 0)

    def test_generate_candidates_respects_max_candidates(self):
        """Test that max_candidates limit is respected."""
        mock_state = Mock()

        # Create many frontiers
        frontiers = []
        for i in range(10):
            frontier = Mock()
            frontier.frontier_type = "dead_end"
            frontier.geometry = LineString([(i*10, i*10), (i*10+1, i*10)])
            frontier.edge_id = (i, i+1)
            frontiers.append(frontier)

        mock_state.frontiers = frontiers

        with patch.object(self.strategy, '_get_city_center', return_value=Point(0, 0)):
            with patch.object(self.strategy, '_create_action_from_frontier') as mock_create:
                mock_create.return_value = Mock()

                candidates = self.strategy.generate_candidates(mock_state, set())

                # Should respect max_candidates limit
                max_candidates = self.config.limits.max_candidates
                self.assertLessEqual(len(candidates), max_candidates)
                self.assertLessEqual(mock_create.call_count, max_candidates)

    def test_get_city_center_with_bounds(self):
        """Test city center calculation with city bounds."""
        mock_state = Mock()
        mock_bounds = Mock()
        mock_bounds.centroid = Point(5, 5)
        mock_state.city_bounds = mock_bounds
        mock_state.streets = Mock()  # Not used when bounds exist

        center = self.strategy._get_city_center(mock_state)

        self.assertEqual(center, Point(5, 5))

    def test_get_city_center_without_bounds(self):
        """Test city center calculation without city bounds."""
        mock_state = Mock()
        mock_state.city_bounds = None

        # Create mock streets DataFrame
        mock_streets = []

        # Mock street with coordinates
        class MockStreet:
            def __init__(self, coords):
                self.geometry = Mock()
                self.geometry.coords = coords

        mock_streets_data = [
            ("street1", MockStreet([(0, 0), (1, 1)])),
            ("street2", MockStreet([(2, 2), (3, 3)])),
        ]

        mock_state.streets.iterrows.return_value = mock_streets_data

        center = self.strategy._get_city_center(mock_state)

        # Should be centroid of all coordinates: (0,0), (1,1), (2,2), (3,3)
        # Mean x = (0+1+2+3)/4 = 1.5, Mean y = (0+1+2+3)/4 = 1.5
        self.assertAlmostEqual(center.x, 1.5)
        self.assertAlmostEqual(center.y, 1.5)

    def test_get_city_center_empty_streets(self):
        """Test city center calculation with empty streets."""
        mock_state = Mock()
        mock_state.city_bounds = None
        mock_state.streets.iterrows.return_value = []

        center = self.strategy._get_city_center(mock_state)

        # Should return fallback
        self.assertEqual(center.x, 0)
        self.assertEqual(center.y, 0)

    def test_distance_from_center(self):
        """Test distance calculation from city center."""
        center = Point(0, 0)

        # Test with LineString
        line = LineString([(3, 4), (5, 6)])  # Centroid at (4, 5)
        distance = self.strategy._distance_from_center(line, center)

        # Distance from (0,0) to (4,5) = sqrt(16 + 25) = sqrt(41) ≈ 6.403
        self.assertAlmostEqual(distance, 6.403, places=3)

        # Test with Point
        point = Point(3, 4)
        distance = self.strategy._distance_from_center(point, center)

        # Distance from (0,0) to (3,4) = sqrt(9 + 16) = 5
        self.assertEqual(distance, 5.0)

    def test_distance_from_center_no_centroid(self):
        """Test distance calculation with geometry that has no centroid."""
        center = Point(0, 0)

        # Mock geometry without centroid
        mock_geom = Mock()
        del mock_geom.centroid  # Remove centroid attribute

        distance = self.strategy._distance_from_center(mock_geom, center)

        self.assertEqual(distance, 0.0)

    def test_generate_candidates_intent_params(self):
        """Test that correct intent parameters are set."""
        mock_state = Mock()

        frontier = Mock()
        frontier.frontier_type = "dead_end"
        frontier.geometry = LineString([(10, 10), (11, 10)])
        frontier.edge_id = (1, 2)

        mock_state.frontiers = [frontier]

        with patch.object(self.strategy, '_get_city_center', return_value=Point(0, 0)):
            with patch.object(self.strategy, '_create_action_from_frontier') as mock_create:
                mock_action = Mock()
                mock_create.return_value = mock_action

                self.strategy.generate_candidates(mock_state, set())

                # Check that _create_action_from_frontier was called with correct intent params
                call_args = mock_create.call_args
                intent_params = call_args[0][3]  # 4th positional argument

                self.assertEqual(intent_params['direction'], 'peripheral_expansion')
                # Distance from (0,0) to centroid of LineString([(10, 10), (11, 10)]) which is (10.5, 10)
                # sqrt(10.5^2 + 10^2) = sqrt(110.25 + 100) = sqrt(210.25) ≈ 14.5
                self.assertAlmostEqual(intent_params['distance_from_center'], 14.5, places=1)
                self.assertEqual(intent_params['frontier_type'], 'dead_end')

    def test_strategy_info(self):
        """Test strategy info reporting."""
        info = self.strategy.get_strategy_info()

        self.assertEqual(info['name'], 'peripheral')
        self.assertEqual(info['weight'], 0.8)
        self.assertEqual(info['type'], 'PeripheralStrategy')
        self.assertTrue(info['config_valid'])
        self.assertIn('Peripheral Expansion Strategy', info['description'])


class TestPeripheralStrategyIntegration(unittest.TestCase):
    """Integration tests for PeripheralStrategy with realistic data."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.strategy = PeripheralStrategy()

    def test_realistic_city_scenario(self):
        """Test with a realistic city layout."""
        # Create a mock city with streets in a grid pattern
        mock_state = Mock()

        # Create frontiers at different distances from center
        frontiers = []

        # Close to center
        close_frontier = Mock()
        close_frontier.frontier_type = "dead_end"
        close_frontier.geometry = LineString([(0, 0), (1, 0)])
        close_frontier.edge_id = (1, 2)
        frontiers.append(close_frontier)

        # Medium distance
        medium_frontier = Mock()
        medium_frontier.frontier_type = "dead_end"
        medium_frontier.geometry = LineString([(5, 5), (6, 5)])
        medium_frontier.edge_id = (3, 4)
        frontiers.append(medium_frontier)

        # Far distance
        far_frontier = Mock()
        far_frontier.frontier_type = "dead_end"
        far_frontier.geometry = LineString([(10, 10), (11, 10)])
        far_frontier.edge_id = (5, 6)
        frontiers.append(far_frontier)

        mock_state.frontiers = frontiers

        with patch.object(self.strategy, '_get_city_center', return_value=Point(0, 0)):
            with patch.object(self.strategy, '_create_action_from_frontier') as mock_create:
                mock_create.return_value = Mock()

                candidates = self.strategy.generate_candidates(mock_state, set())

                # Should generate 3 candidates
                self.assertEqual(len(candidates), 3)

                # Verify that _create_action_from_frontier was called 3 times
                self.assertEqual(mock_create.call_count, 3)

                # The calls should be ordered by distance (farthest first)
                calls = mock_create.call_args_list

                # Extract confidence values (second positional argument)
                confidences = [call[0][1] for call in calls]

                # Confidences should be in descending order (farthest first)
                self.assertGreater(confidences[0], confidences[1])
                self.assertGreater(confidences[1], confidences[2])


if __name__ == '__main__':
    unittest.main()
