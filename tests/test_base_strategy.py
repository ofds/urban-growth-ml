#!/usr/bin/env python3
"""
Unit tests for BaseInferenceStrategy class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from shapely.geometry import LineString, Point

from src.inverse.strategies.base_strategy import BaseInferenceStrategy
from src.inverse.core.config import InferenceConfig
from src.core.contracts import GrowthState
from src.inverse.data_structures import InverseGrowthAction, ActionType


class TestBaseInferenceStrategy(unittest.TestCase):
    """Test cases for BaseInferenceStrategy abstract class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a concrete implementation for testing
        class ConcreteStrategy(BaseInferenceStrategy):
            def generate_candidates(self, state, skeleton_edges, spatial_index=None):
                return []

        self.strategy_class = ConcreteStrategy
        self.config = InferenceConfig()

    def test_initialization_with_defaults(self):
        """Test strategy initialization with default parameters."""
        strategy = self.strategy_class("test_strategy")

        self.assertEqual(strategy.name, "test_strategy")
        self.assertEqual(strategy.weight, 1.0)
        self.assertIsInstance(strategy.config, InferenceConfig)

    def test_initialization_with_custom_params(self):
        """Test strategy initialization with custom parameters."""
        custom_config = InferenceConfig()
        strategy = self.strategy_class("test_strategy", weight=2.5, config=custom_config)

        self.assertEqual(strategy.name, "test_strategy")
        self.assertEqual(strategy.weight, 2.5)
        self.assertEqual(strategy.config, custom_config)

    def test_negative_weight_validation(self):
        """Test that negative weights raise ValueError."""
        with self.assertRaises(ValueError):
            self.strategy_class("test_strategy", weight=-1.0)

    @patch('src.inverse.strategies.base_strategy.logger')
    def test_disabled_strategy_warning(self, mock_logger):
        """Test warning when strategy is disabled in config."""
        config = InferenceConfig()
        # Disable our test strategy
        config.strategies.enabled_strategies = {'test_strategy': False}

        strategy = self.strategy_class("test_strategy", config=config)

        # Should warn for our disabled test strategy
        mock_logger.warning.assert_called_once_with("Strategy 'test_strategy' is disabled in configuration")

    def test_get_strategy_info(self):
        """Test getting strategy information."""
        strategy = self.strategy_class("test_strategy", weight=1.5)

        info = strategy.get_strategy_info()

        self.assertEqual(info['name'], "test_strategy")
        self.assertEqual(info['weight'], 1.5)
        self.assertEqual(info['type'], "ConcreteStrategy")
        self.assertTrue(info['config_valid'])

    def test_abstract_method_enforcement(self):
        """Test that abstract method must be implemented."""
        # This should work since we implemented generate_candidates
        strategy = self.strategy_class("test_strategy")
        self.assertIsNotNone(strategy)

    def test_street_angle_calculation(self):
        """Test street angle calculation utility."""
        strategy = self.strategy_class("test_strategy")

        # Horizontal line (0 degrees)
        line = LineString([(0, 0), (1, 0)])
        angle = strategy._get_street_angle(line)
        self.assertAlmostEqual(angle, 0.0, places=1)

        # Vertical line (90 degrees)
        line = LineString([(0, 0), (0, 1)])
        angle = strategy._get_street_angle(line)
        self.assertAlmostEqual(angle, 90.0, places=1)

        # Diagonal line (45 degrees)
        line = LineString([(0, 0), (1, 1)])
        angle = strategy._get_street_angle(line)
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_street_angle_empty_geometry(self):
        """Test street angle calculation with invalid geometry."""
        strategy = self.strategy_class("test_strategy")

        # Empty line - this should work
        line = LineString([])
        angle = strategy._get_street_angle(line)
        self.assertEqual(angle, 0.0)

        # Test with a mock LineString that has coords attribute but insufficient points
        mock_line = Mock()
        mock_line.coords = [(0, 0)]  # Only one point
        angle = strategy._get_street_angle(mock_line)
        self.assertEqual(angle, 0.0)

    def test_skeleton_edge_checking(self):
        """Test skeleton edge validation."""
        strategy = self.strategy_class("test_strategy")

        skeleton_edges = {(1, 2), (3, 4), (5, 6)}

        # Test existing edges
        self.assertTrue(strategy._is_skeleton_edge((1, 2), skeleton_edges))
        self.assertTrue(strategy._is_skeleton_edge((2, 1), skeleton_edges))  # Reversed

        # Test non-existing edges
        self.assertFalse(strategy._is_skeleton_edge((1, 3), skeleton_edges))
        self.assertFalse(strategy._is_skeleton_edge((7, 8), skeleton_edges))

    def test_stable_frontier_id_generation(self):
        """Test stable frontier ID generation."""
        strategy = self.strategy_class("test_strategy")

        # Mock frontier with geometry
        mock_frontier = Mock()
        mock_frontier.geometry = LineString([(0, 0), (1, 1)])
        mock_frontier.frontier_type = "dead_end"

        stable_id = strategy._compute_stable_frontier_id(mock_frontier)

        # Should be a 16-character hex string
        self.assertEqual(len(stable_id), 16)
        self.assertTrue(all(c in '0123456789abcdef' for c in stable_id))

        # Same geometry should produce same ID
        stable_id2 = strategy._compute_stable_frontier_id(mock_frontier)
        self.assertEqual(stable_id, stable_id2)

    def test_stable_frontier_id_invalid_geometry(self):
        """Test stable frontier ID with invalid geometry."""
        strategy = self.strategy_class("test_strategy")

        # Mock frontier with no geometry
        mock_frontier = Mock()
        mock_frontier.geometry = None

        stable_id = strategy._compute_stable_frontier_id(mock_frontier)
        self.assertEqual(stable_id, "invalid_frontier")

        # Mock frontier with invalid LineString
        mock_frontier.geometry = LineString([])
        stable_id = strategy._compute_stable_frontier_id(mock_frontier)
        self.assertEqual(stable_id, "invalid_geometry")

    def test_find_street_for_edge(self):
        """Test finding street ID for a given edge."""
        strategy = self.strategy_class("test_strategy")

        # Create mock state with streets
        mock_state = Mock()
        mock_streets = []

        # Create mock streets DataFrame
        class MockStreet:
            def __init__(self, idx, u, v):
                self.idx = idx
                self.u = u
                self.v = v

            def get(self, key):
                return getattr(self, key, None)

        # Mock iterrows to return street data
        mock_streets_data = [
            ("street1", MockStreet("street1", 1, 2)),
            ("street2", MockStreet("street2", 3, 4)),
            ("street3", MockStreet("street3", 1, 2)),  # Same edge as street1
        ]

        mock_state.streets.iterrows.return_value = mock_streets_data

        # Test finding existing edges
        street_id = strategy._find_street_for_edge((1, 2), mock_state)
        self.assertEqual(street_id, "street1")

        street_id = strategy._find_street_for_edge((2, 1), mock_state)  # Reversed
        self.assertEqual(street_id, "street1")

        # Test non-existing edge
        street_id = strategy._find_street_for_edge((5, 6), mock_state)
        self.assertIsNone(street_id)

    def test_morphological_similarity_calculation(self):
        """Test morphological similarity calculation."""
        strategy = self.strategy_class("test_strategy")

        # Create test geometries
        frontier_geom = LineString([(0, 0), (10, 0)])  # 10 units long, 0 degrees
        frontier_length = 10.0
        frontier_centroid = Point(5, 0)
        frontier_angle = 0.0

        street_geom = LineString([(0, 1), (10, 1)])  # Parallel, same length, 1 unit away

        similarity = strategy._calculate_morphological_similarity(
            frontier_geom, frontier_length, frontier_centroid, frontier_angle, street_geom
        )

        # Should be high similarity (same length, same angle, close proximity)
        self.assertGreater(similarity, 0.5)

        # Test with very different geometry
        different_geom = LineString([(0, 0), (0, 10)])  # Perpendicular, same length
        similarity2 = strategy._calculate_morphological_similarity(
            frontier_geom, frontier_length, frontier_centroid, frontier_angle, different_geom
        )

        # Should be lower similarity (different angle)
        self.assertLess(similarity2, similarity)


class TestBaseStrategyActionCreation(unittest.TestCase):
    """Test action creation methods in BaseInferenceStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        class ConcreteStrategy(BaseInferenceStrategy):
            def generate_candidates(self, state, skeleton_edges, spatial_index=None):
                return []

        self.strategy = ConcreteStrategy("test_strategy")
        self.mock_state = Mock()
        self.mock_frontier = Mock()

    @patch('src.inverse.data_structures.compute_frontier_signature')
    def test_create_action_from_frontier_success(self, mock_compute_sig):
        """Test successful action creation from frontier."""
        # Setup mocks
        mock_compute_sig.return_value = "test_signature"

        self.mock_frontier.geometry = LineString([(0, 0), (1, 1)])
        self.mock_frontier.edge_id = (1, 2)
        self.mock_frontier.frontier_type = "dead_end"

        # Mock state with graph that has the edge
        self.mock_state.graph.has_edge.return_value = True

        # Mock streets DataFrame
        mock_street = Mock()
        mock_street.get.side_effect = lambda key: {"u": 1, "v": 2, "osmid": None, "highway": "residential"}.get(key)
        mock_street.geometry = LineString([(0, 0), (1, 1)])  # Real LineString for morphological similarity

        self.mock_state.streets.iterrows.return_value = [("street1", mock_street)]
        # Mock loc to return the street when accessed by index
        self.mock_state.streets.loc.__getitem__ = Mock(return_value=mock_street)
        self.mock_state.streets.__len__ = Mock(return_value=10)  # Mock len() method

        # Enable the test strategy in config
        self.strategy.config.strategies.enabled_strategies['test_strategy'] = True

        # Test action creation
        action = self.strategy._create_action_from_frontier(self.mock_frontier, 0.8, self.mock_state)

        self.assertIsInstance(action, InverseGrowthAction)
        self.assertEqual(action.action_type, ActionType.REMOVE_STREET)
        self.assertEqual(action.confidence, 0.8)
        self.assertEqual(action.intent_params['strategy'], 'test_strategy')

    def test_create_action_from_frontier_stale_edge(self):
        """Test action creation fails with stale edge."""
        self.mock_frontier.edge_id = (1, 2)
        self.mock_state.graph.has_edge.return_value = False  # Edge doesn't exist

        action = self.strategy._create_action_from_frontier(self.mock_frontier, 0.8, self.mock_state)

        self.assertIsNone(action)

    def test_create_action_from_street_success(self):
        """Test successful action creation from street."""
        # Setup mock street
        mock_street = Mock()
        mock_street.get.side_effect = lambda key: {"u": 1, "v": 2, "osmid": None, "highway": "residential"}.get(key)
        mock_street.geometry = LineString([(0, 0), (1, 1)])  # Real LineString

        # Mock state
        self.mock_state.graph.has_edge.return_value = True
        self.mock_state.streets.__len__ = Mock(return_value=10)  # Mock len() method

        action = self.strategy._create_action_from_street("street1", mock_street, 0.7, self.mock_state)

        self.assertIsInstance(action, InverseGrowthAction)
        self.assertEqual(action.action_type, ActionType.REMOVE_STREET)
        self.assertEqual(action.street_id, "street1")
        self.assertEqual(action.confidence, 0.7)

    def test_create_action_from_street_missing_edge_info(self):
        """Test action creation fails with missing edge info."""
        mock_street = Mock()
        mock_street.get.return_value = None  # No u/v info

        action = self.strategy._create_action_from_street("street1", mock_street, 0.7, self.mock_state)

        self.assertIsNone(action)

    def test_create_action_from_street_edge_not_in_graph(self):
        """Test action creation fails when edge not in graph."""
        mock_street = Mock()
        mock_street.get.side_effect = lambda key: {"u": 1, "v": 2}.get(key)
        self.mock_state.graph.has_edge.return_value = False

        action = self.strategy._create_action_from_street("street1", mock_street, 0.7, self.mock_state)

        self.assertIsNone(action)


if __name__ == '__main__':
    unittest.main()
