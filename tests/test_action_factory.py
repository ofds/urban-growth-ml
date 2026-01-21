#!/usr/bin/env python3
"""
Unit tests for ActionFactory module.
"""

import pytest
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Point
from unittest.mock import Mock, MagicMock

from src.inverse.core.action_factory import ActionFactory
from src.inverse.core.config import InferenceConfig
from src.inverse.data_structures import InverseGrowthAction, ActionType
from src.core.contracts import GrowthState


class TestActionFactory:
    """Test suite for ActionFactory."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return InferenceConfig()

    @pytest.fixture
    def factory(self, config):
        """Create ActionFactory instance."""
        return ActionFactory(config)

    @pytest.fixture
    def mock_frontier(self):
        """Create mock frontier."""
        frontier = Mock()
        frontier.geometry = LineString([(0, 0), (10, 0)])
        frontier.frontier_type = "dead_end"
        return frontier

    @pytest.fixture
    def mock_street(self):
        """Create mock street DataFrame row."""
        street_data = {
            'u': 1,
            'v': 2,
            'geometry': LineString([(0, 0), (10, 0)]),
            'osmid': 123,
            'highway': 'residential',
            'length': 10.0
        }
        return pd.Series(street_data)

    @pytest.fixture
    def mock_state(self, mock_street, mock_frontier):
        """Create mock growth state."""
        state = Mock(spec=GrowthState)

        # Mock streets DataFrame
        streets_df = pd.DataFrame([mock_street.to_dict()])
        streets_df.index = ['street_1']
        state.streets = streets_df

        # Mock graph with specific edge behavior
        graph = Mock()
        def has_edge_side_effect(u, v=None):
            # Return True only for the existing edge (1, 2) or (2, 1)
            if v is None:
                return False  # Single node check
            return (u == 1 and v == 2) or (u == 2 and v == 1)
        graph.has_edge.side_effect = has_edge_side_effect
        state.graph = graph

        # Mock frontiers
        state.frontiers = [mock_frontier]

        return state

    def test_initialization(self, factory, config):
        """Test ActionFactory initialization."""
        assert factory.config == config
        assert factory.config is not None

    def test_create_action_from_frontier_success(self, factory, mock_frontier, mock_state):
        """Test successful action creation from frontier."""
        # Mock the resolution to return a valid edge
        factory._resolve_frontier_to_current_edge = Mock(return_value=(1, 2))
        factory._find_street_for_edge = Mock(return_value="street_1")

        action = factory.create_action_from_frontier(
            mock_frontier, "test_strategy", 0.8, mock_state
        )

        assert action is not None
        assert action.action_type == ActionType.REMOVE_STREET
        assert action.confidence == 0.8
        assert action.intent_params['strategy'] == 'test_strategy'
        assert 'state_diff' in action.__dict__

    def test_create_action_from_frontier_resolution_failure(self, factory, mock_frontier, mock_state):
        """Test action creation failure when frontier resolution fails."""
        factory._resolve_frontier_to_current_edge = Mock(return_value=None)

        action = factory.create_action_from_frontier(
            mock_frontier, "test_strategy", 0.8, mock_state
        )

        assert action is None

    def test_create_action_from_street_success(self, factory, mock_street, mock_state):
        """Test successful action creation from street."""
        action = factory.create_action_from_street(
            "street_1", mock_street, "test_strategy", 0.7, mock_state
        )

        assert action is not None
        assert action.action_type == ActionType.REMOVE_STREET
        assert action.street_id == "street_1"
        assert action.confidence == 0.7
        assert action.intent_params['strategy'] == 'test_strategy'

    def test_create_action_from_street_missing_edge(self, factory, mock_street, mock_state):
        """Test action creation failure when street has no edge."""
        mock_street_no_edge = mock_street.copy()
        mock_street_no_edge['u'] = None
        mock_street_no_edge['v'] = None

        action = factory.create_action_from_street(
            "street_1", mock_street_no_edge, "test_strategy", 0.7, mock_state
        )

        assert action is None

    def test_validate_action_success(self, factory, mock_state):
        """Test successful action validation."""
        action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id="street_1",
            intent_params={
                'strategy': 'test',
                'edge_u': '1',
                'edge_v': '2',
                'stable_id': 'abc123'
            },
            confidence=0.8,
            timestamp=100,
            state_diff={
                'added_streets': [{'edge_id': (1, 2), 'u': 1, 'v': 2, 'geometry_wkt': 'LINESTRING (0 0, 10 0)'}],
                'removed_streets': ['street_1'],
                'geometry_wkt': 'LINESTRING (0 0, 10 0)',
                'edgeid': (1, 2)
            }
        )

        is_valid = factory.validate_action(action, mock_state)
        assert is_valid is True

    def test_validate_action_invalid_type(self, factory, mock_state):
        """Test that invalid action types cannot be created."""
        # InverseGrowthAction dataclass validation prevents invalid action types
        with pytest.raises(ValueError, match="Invalid action_type.*Only REMOVE_STREET allowed"):
            InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,  # Invalid for inverse actions
                street_id="street_1",
                intent_params={'strategy': 'test', 'edge_u': '1', 'edge_v': '2', 'stable_id': 'abc'},
                confidence=0.8,
                timestamp=100,
                state_diff={'removed_streets': ['street_1']}
            )

    def test_validate_action_missing_street(self, factory, mock_state):
        """Test action validation failure when street doesn't exist."""
        action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id="nonexistent_street",
            intent_params={'strategy': 'test', 'edge_u': '1', 'edge_v': '2', 'stable_id': 'abc'},
            confidence=0.8,
            timestamp=100,
            state_diff={'removed_streets': ['nonexistent_street']}
        )

        is_valid = factory.validate_action(action, mock_state)
        assert is_valid is False

    def test_validate_intent_params(self, factory):
        """Test intent parameters validation."""
        # Valid params
        valid_params = {
            'strategy': 'test',
            'edge_u': '1',
            'edge_v': '2',
            'stable_id': 'abc123'
        }
        assert factory._validate_intent_params(valid_params) is True

        # Missing required param
        invalid_params = {
            'strategy': 'test',
            'edge_u': '1',
            'edge_v': '2'
            # missing stable_id
        }
        assert factory._validate_intent_params(invalid_params) is False

    def test_validate_state_diff(self, factory, mock_state):
        """Test state diff validation."""
        # Valid state diff
        valid_diff = {
            'added_streets': [{'edge_id': (1, 2), 'u': 1, 'v': 2, 'geometry_wkt': 'LINESTRING (0 0, 10 0)'}],
            'removed_streets': ['street_1'],
            'geometry_wkt': 'LINESTRING (0 0, 10 0)',
            'edgeid': (1, 2)
        }
        assert factory._validate_state_diff(valid_diff, mock_state) is True

        # Invalid - missing required field
        invalid_diff = {
            'added_streets': [{'edge_id': (1, 2), 'u': 1, 'v': 2, 'geometry_wkt': 'LINESTRING (0 0, 10 0)'}],
            'removed_streets': ['street_1']
            # missing geometry_wkt and edgeid
        }
        assert factory._validate_state_diff(invalid_diff, mock_state) is False

        # Invalid - nonexistent street
        invalid_diff2 = {
            'added_streets': [{'edge_id': (1, 2), 'u': 1, 'v': 2, 'geometry_wkt': 'LINESTRING (0 0, 10 0)'}],
            'removed_streets': ['nonexistent'],
            'geometry_wkt': 'LINESTRING (0 0, 10 0)',
            'edgeid': (1, 2)
        }
        assert factory._validate_state_diff(invalid_diff2, mock_state) is False

    def test_resolve_frontier_to_current_edge(self, factory, mock_frontier, mock_state):
        """Test frontier to edge resolution."""
        # Mock morphological similarity to return high score
        factory._calculate_morphological_similarity = Mock(return_value=0.9)

        edge = factory._resolve_frontier_to_current_edge(mock_frontier, mock_state)
        assert edge == (1, 2)  # Should find the street's edge

    def test_find_street_for_edge(self, factory, mock_state):
        """Test finding street ID for edge."""
        street_id = factory._find_street_for_edge((1, 2), mock_state)
        assert street_id == "street_1"

        # Test nonexistent edge
        street_id = factory._find_street_for_edge((99, 100), mock_state)
        assert street_id is None

    def test_validate_edge_exists(self, factory, mock_state):
        """Test edge existence validation."""
        assert factory._validate_edge_exists((1, 2), mock_state) is True
        assert factory._validate_edge_exists((99, 100), mock_state) is False

    def test_prepare_intent_params(self, factory):
        """Test intent parameters preparation."""
        params = factory._prepare_intent_params(
            "test_strategy", (1, 2), "stable123",
            {"extra": "value"}
        )

        expected = {
            'strategy': 'test_strategy',
            'edge_u': '1',
            'edge_v': '2',
            'stable_id': 'stable123',
            'extra': 'value'
        }
        assert params == expected

    def test_compute_stable_frontier_id(self, factory, mock_frontier):
        """Test stable ID computation."""
        stable_id = factory._compute_stable_frontier_id(mock_frontier)
        assert isinstance(stable_id, str)
        assert len(stable_id) == 16  # MD5 hex length

    def test_get_street_angle(self, factory):
        """Test street angle calculation."""
        # Horizontal line
        line = LineString([(0, 0), (10, 0)])
        angle = factory._get_street_angle(line)
        assert abs(angle) < 0.01  # Should be close to 0

        # Vertical line
        line = LineString([(0, 0), (0, 10)])
        angle = factory._get_street_angle(line)
        assert abs(angle - 90) < 0.01

        # Diagonal line
        line = LineString([(0, 0), (10, 10)])
        angle = factory._get_street_angle(line)
        assert abs(angle - 45) < 0.01

    def test_calculate_morphological_similarity(self, factory):
        """Test morphological similarity calculation."""
        frontier_geom = LineString([(0, 0), (10, 0)])
        street_geom = LineString([(0, 0), (10, 0)])

        similarity = factory._calculate_morphological_similarity(
            frontier_geom, 10.0, Point(5, 0), 0.0, street_geom
        )

        assert 0 <= similarity <= 1
        # Should be high similarity for identical geometries
        assert similarity > 0.8
