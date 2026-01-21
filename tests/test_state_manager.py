#!/usr/bin/env python3
"""
Unit tests for StateManager module.
"""

import unittest
import sys
import os
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inverse.core.state_manager import StateManager
from inverse.data_structures import InverseGrowthAction, ActionType
from src.core.contracts import GrowthState


class TestStateManager(unittest.TestCase):
    """Test cases for StateManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_manager = StateManager()

        # Create mock states
        self.mock_state = self._create_mock_state()
        self.mock_action = self._create_mock_action()

    def _create_mock_state(self):
        """Create a mock GrowthState for testing."""
        # Create test street data
        streets = pd.DataFrame({
            'geometry': [
                LineString([(0, 0), (10, 0)]),
                LineString([(10, 10), (20, 10)]),
            ],
            'u': [1, 2],
            'v': [2, 3],
            'highway': ['residential', 'primary'],
            'osmid': [100, 200]
        })

        # Create mock frontiers
        frontiers = [
            Mock(
                geometry=LineString([(0, 0), (5, 0)]),
                edge_id=(1, 2),
                frontier_id='f1',
                frontier_type='dead_end'
            ),
            Mock(
                geometry=LineString([(10, 10), (15, 10)]),
                edge_id=(2, 3),
                frontier_id='f2',
                frontier_type='block_edge'
            )
        ]

        # Create mock graph
        graph = Mock()
        graph.number_of_nodes.return_value = 4
        graph.number_of_edges.return_value = 3

        # Create mock blocks
        blocks = pd.DataFrame({
            'geometry': [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
        })

        return GrowthState(
            streets=streets,
            frontiers=frontiers,
            graph=graph,
            blocks=blocks,
            city_bounds=Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),
            iteration=0
        )

    def _create_mock_action(self):
        """Create a mock InverseGrowthAction for testing."""
        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id='0',  # Index of first street (string to avoid falsy check)
            intent_params={
                'strategy': 'test_strategy',
                'edge_u': 1,
                'edge_v': 2
            },
            confidence=0.8,
            timestamp=100,
            state_diff=None,
            action_metadata={'test': True}
        )

    def test_initialization(self):
        """Test StateManager initialization."""
        state_manager = StateManager()
        self.assertIsInstance(state_manager, StateManager)

    def test_compute_state_diff_remove_street(self):
        """Test computing state diff for REMOVE_STREET action."""
        state_diff = self.state_manager.compute_state_diff(self.mock_state, self.mock_action)

        # Check structure
        self.assertIn('added_streets', state_diff)
        self.assertIn('removed_streets', state_diff)
        self.assertIn('graph_changes', state_diff)
        self.assertIn('frontier_changes', state_diff)

        # Check content
        self.assertEqual(len(state_diff['added_streets']), 1)
        self.assertEqual(len(state_diff['removed_streets']), 1)
        self.assertEqual(state_diff['removed_streets'][0], 0)  # Should be the converted integer index

        # Check street data
        street_data = state_diff['added_streets'][0]
        self.assertEqual(street_data['edge_id'], (1, 2))
        self.assertEqual(street_data['u'], 1)
        self.assertEqual(street_data['v'], 2)
        self.assertEqual(street_data['highway'], 'residential')
        self.assertEqual(street_data['osmid'], 100)
        self.assertIsNotNone(street_data['geometry_wkt'])

    def test_compute_state_diff_extend_frontier(self):
        """Test computing state diff for EXTEND_FRONTIER action."""
        # Note: InverseGrowthAction only allows REMOVE_STREET, so we test the logic path
        # by using REMOVE_STREET with intent_params that would come from EXTEND_FRONTIER
        extend_action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id='0',  # Index of first street (string)
            intent_params={
                'strategy': 'extend_frontier',
                'edge_u': 1,
                'edge_v': 2
            },
            confidence=0.9,
            timestamp=200,
            state_diff=None,
            action_metadata={}
        )

        state_diff = self.state_manager.compute_state_diff(self.mock_state, extend_action)

        # Should find the matching street
        self.assertEqual(len(state_diff['added_streets']), 1)
        self.assertEqual(len(state_diff['removed_streets']), 1)
        self.assertEqual(state_diff['removed_streets'][0], 0)  # Converted to integer

    def test_compute_state_diff_action_without_matching_street(self):
        """Test computing state diff for action without matching street."""
        # Create action that references a non-existent street
        no_match_action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id='999',  # Non-existent street ID
            intent_params={
                'strategy': 'test',
                'edge_u': 999,
                'edge_v': 999
            },
            confidence=0.5,
            timestamp=300,
            state_diff=None,
            action_metadata={}
        )

        state_diff = self.state_manager.compute_state_diff(self.mock_state, no_match_action)

        # Should still have basic structure but no streets added/removed
        self.assertEqual(len(state_diff['added_streets']), 0)
        self.assertEqual(len(state_diff['removed_streets']), 0)
        self.assertIn('graph_changes', state_diff)
        self.assertIn('frontier_changes', state_diff)

    def test_validate_state_transition_valid(self):
        """Test validating a valid state transition."""
        before_state = self.mock_state

        # Create after state with one less street
        after_streets = before_state.streets.drop(0)  # Remove first street
        after_graph = Mock()
        after_graph.number_of_edges.return_value = 2  # One less edge

        after_state = GrowthState(
            streets=after_streets,
            frontiers=before_state.frontiers,
            graph=after_graph,
            blocks=before_state.blocks,
            city_bounds=before_state.city_bounds,
            iteration=1
        )

        expected_changes = {
            'removed_streets': ['0'],
            'added_streets': [],
            'graph_changes': {
                'edges_before': 3,
                'edges_after': 2
            }
        }

        is_valid = self.state_manager.validate_state_transition(before_state, after_state, expected_changes)
        self.assertTrue(is_valid)

    def test_validate_state_transition_invalid_streets(self):
        """Test validating an invalid state transition (street count mismatch)."""
        before_state = self.mock_state
        after_state = self.mock_state  # Same state, no change

        expected_changes = {
            'removed_streets': ['0'],  # Expected one street removed
            'added_streets': [],
            'graph_changes': {
                'edges_before': 3,
                'edges_after': 3
            }
        }

        is_valid = self.state_manager.validate_state_transition(before_state, after_state, expected_changes)
        self.assertFalse(is_valid)

    def test_validate_state_transition_invalid_edges(self):
        """Test validating an invalid state transition (edge count mismatch)."""
        before_state = self.mock_state

        after_graph = Mock()
        after_graph.number_of_edges.return_value = 1  # Two edges removed, but expected one

        after_state = GrowthState(
            streets=before_state.streets.drop(0),
            frontiers=before_state.frontiers,
            graph=after_graph,
            blocks=before_state.blocks,
            city_bounds=before_state.city_bounds,
            iteration=1
        )

        expected_changes = {
            'removed_streets': ['0'],
            'added_streets': [],
            'graph_changes': {
                'edges_before': 3,
                'edges_after': 2  # Expected one edge removed
            }
        }

        is_valid = self.state_manager.validate_state_transition(before_state, after_state, expected_changes)
        self.assertFalse(is_valid)

    def test_get_state_summary(self):
        """Test getting state summary."""
        summary = self.state_manager.get_state_summary(self.mock_state)

        # Check basic counts
        self.assertEqual(summary['streets_count'], 2)
        self.assertEqual(summary['frontiers_count'], 2)
        self.assertEqual(summary['blocks_count'], 1)
        self.assertEqual(summary['graph_nodes'], 4)
        self.assertEqual(summary['graph_edges'], 3)

        # Check city bounds
        self.assertIsNotNone(summary['city_bounds'])
        self.assertIn('area', summary['city_bounds'])
        self.assertIn('centroid', summary['city_bounds'])

        # Check street types
        self.assertEqual(summary['street_types']['residential'], 1)
        self.assertEqual(summary['street_types']['primary'], 1)

        # Check frontier types
        self.assertEqual(summary['frontier_types']['dead_end'], 1)
        self.assertEqual(summary['frontier_types']['block_edge'], 1)

    def test_get_state_summary_empty_blocks(self):
        """Test getting state summary when blocks is empty."""
        empty_blocks = pd.DataFrame(columns=['geometry'])
        state_empty_blocks = GrowthState(
            streets=self.mock_state.streets,
            frontiers=self.mock_state.frontiers,
            graph=self.mock_state.graph,
            blocks=empty_blocks,
            city_bounds=self.mock_state.city_bounds,
            iteration=0
        )

        summary = self.state_manager.get_state_summary(state_empty_blocks)
        self.assertEqual(summary['blocks_count'], 0)

    def test_compute_transition_metrics(self):
        """Test computing transition metrics."""
        before_state = self.mock_state

        # Create after state with changes
        after_streets = before_state.streets.drop(0)  # Remove one street
        after_frontiers = before_state.frontiers[:-1]  # Remove one frontier

        after_graph = Mock()
        after_graph.number_of_edges.return_value = 2  # Remove one edge

        after_state = GrowthState(
            streets=after_streets,
            frontiers=after_frontiers,
            graph=after_graph,
            blocks=before_state.blocks,
            city_bounds=before_state.city_bounds,
            iteration=1
        )

        metrics = self.state_manager.compute_transition_metrics(before_state, after_state)

        # Check individual changes
        self.assertEqual(metrics['streets_removed'], 1)
        self.assertEqual(metrics['streets_added'], -1)  # We removed 1 street, so added is -1
        self.assertEqual(metrics['graph_edges_removed'], 1)
        self.assertEqual(metrics['graph_edges_added'], -1)  # We removed 1 edge, so added is -1
        self.assertEqual(metrics['frontiers_removed'], 1)
        self.assertEqual(metrics['frontiers_added'], -1)  # We removed 1 frontier, so added is -1

        # Check net changes
        self.assertEqual(metrics['net_streets'], -1)  # 1 - 2 = -1
        self.assertEqual(metrics['net_edges'], -1)    # 2 - 3 = -1
        self.assertEqual(metrics['net_frontiers'], -1)  # 1 - 2 = -1

    def test_create_action_with_state_diff(self):
        """Test creating action with state diff."""
        original_action = self.mock_action
        state_diff = {'test': 'data', 'added_streets': [], 'removed_streets': []}

        new_action = self.state_manager.create_action_with_state_diff(original_action, state_diff)

        # Check that it's a new instance
        self.assertIsNot(new_action, original_action)

        # Check that state_diff is included
        self.assertEqual(new_action.state_diff, state_diff)

        # Check that other fields are preserved
        self.assertEqual(new_action.action_type, original_action.action_type)
        self.assertEqual(new_action.street_id, original_action.street_id)
        self.assertEqual(new_action.intent_params, original_action.intent_params)
        self.assertEqual(new_action.confidence, original_action.confidence)
        self.assertEqual(new_action.timestamp, original_action.timestamp)
        self.assertEqual(new_action.action_metadata, original_action.action_metadata)


if __name__ == '__main__':
    unittest.main()
