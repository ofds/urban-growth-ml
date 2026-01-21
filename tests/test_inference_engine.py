#!/usr/bin/env python3
"""
Unit tests for InferenceEngine module.
"""

import pytest
import pandas as pd
from shapely.geometry import LineString
from unittest.mock import Mock, MagicMock, patch

from src.inverse.core.inference_engine import InferenceEngine
from src.inverse.core.config import InferenceConfig
from src.inverse.strategies.base_strategy import BaseInferenceStrategy
from src.inverse.data_structures import GrowthTrace, InverseGrowthAction, ActionType
from src.core.contracts import GrowthState


class MockStrategy(BaseInferenceStrategy):
    """Mock strategy for testing."""

    def __init__(self, name="mock_strategy", weight=1.0):
        super().__init__(name, weight)

    def generate_candidates(self, state, skeleton_edges, spatial_index=None):
        """Return mock candidates."""
        action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id="street_1",
            intent_params={'strategy': self.name, 'edge_u': '1', 'edge_v': '2', 'stable_id': 'abc'},
            confidence=0.8,
            timestamp=100,
            state_diff={'removed_streets': ['street_1']}
        )
        return [(action, 0.8)]


class TestInferenceEngine:
    """Test suite for InferenceEngine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = InferenceConfig()
        config.limits.max_steps = 10  # Limit for testing
        return config

    @pytest.fixture
    def mock_strategies(self):
        """Create mock strategies."""
        strategy1 = Mock(spec=MockStrategy)
        strategy1.name = "strategy1"
        strategy1.weight = 1.0
        strategy1.generate_candidates = Mock(return_value=[])

        strategy2 = Mock(spec=MockStrategy)
        strategy2.name = "strategy2"
        strategy2.weight = 1.0
        strategy2.generate_candidates = Mock(return_value=[])

        return [strategy1, strategy2]

    @pytest.fixture
    def engine(self, mock_strategies, config):
        """Create InferenceEngine instance with mocked dependencies."""
        with patch('src.inverse.core.inference_engine.PerformanceTracker'), \
             patch('src.inverse.core.inference_engine.SpatialIndex'), \
             patch('src.inverse.core.inference_engine.StateManager'), \
             patch('src.inverse.core.inference_engine.ActionFactory'), \
             patch('src.inverse.core.inference_engine.RewindEngine'), \
             patch('src.inverse.core.inference_engine.ArterialSkeletonExtractor'):

            engine = InferenceEngine(mock_strategies, config)
            return engine

    @pytest.fixture
    def mock_final_state(self):
        """Create mock final growth state."""
        state = Mock(spec=GrowthState)

        # Create streets DataFrame with 5 streets
        streets_data = []
        for i in range(5):
            streets_data.append({
                'u': i,
                'v': i + 1,
                'geometry': LineString([(i, 0), (i+1, 0)]),
                'osmid': 100 + i,
                'highway': 'residential',
                'length': 1.0
            })

        streets_df = pd.DataFrame(streets_data)
        streets_df.index = [f"street_{i}" for i in range(5)]
        state.streets = streets_df

        # Mock other required attributes
        state.graph = Mock()
        state.frontiers = []
        state.blocks = pd.DataFrame()

        return state

    @pytest.fixture
    def mock_initial_state(self):
        """Create mock initial growth state."""
        state = Mock(spec=GrowthState)

        # Create streets DataFrame with 2 streets (skeleton)
        streets_data = []
        for i in range(2):
            streets_data.append({
                'u': i,
                'v': i + 1,
                'geometry': LineString([(i, 0), (i+1, 0)]),
                'osmid': 100 + i,
                'highway': 'primary',
                'length': 1.0
            })

        streets_df = pd.DataFrame(streets_data)
        streets_df.index = [f"skeleton_{i}" for i in range(2)]
        state.streets = streets_df

        state.graph = Mock()
        state.frontiers = []
        state.blocks = pd.DataFrame()

        return state

    def test_initialization(self, engine, mock_strategies, config):
        """Test InferenceEngine initialization."""
        assert engine.strategies == mock_strategies
        assert engine.config == config
        assert hasattr(engine, 'performance_tracker')
        assert hasattr(engine, 'spatial_index')
        assert hasattr(engine, 'state_manager')
        assert hasattr(engine, 'action_factory')

    def test_initialization_validation(self, config):
        """Test initialization validation."""
        # Empty strategies should raise error
        with pytest.raises(ValueError, match="At least one strategy must be provided"):
            InferenceEngine([], config)

    def test_prepare_initial_state_with_provided(self, engine, mock_final_state, mock_initial_state):
        """Test initial state preparation when initial state is provided."""
        # Mock skeleton extractor
        engine.skeleton_extractor.extract_skeleton.return_value = (set(), pd.DataFrame())

        initial_state, skeleton_edges = engine._prepare_initial_state(mock_final_state, mock_initial_state)

        assert initial_state == mock_initial_state
        assert isinstance(skeleton_edges, set)

    def test_prepare_initial_state_extraction(self, engine, mock_final_state):
        """Test initial state preparation with skeleton extraction."""
        # Mock skeleton extraction
        mock_skeleton_streets = pd.DataFrame([{'u': 0, 'v': 1, 'geometry': LineString([(0, 0), (1, 0)])}])
        mock_skeleton_streets.index = ['skeleton_0']
        mock_skeleton_state = Mock()
        mock_skeleton_state.streets = mock_skeleton_streets

        engine.skeleton_extractor.extract_skeleton.return_value = (set([(0, 1)]), mock_skeleton_streets)
        engine.skeleton_extractor.create_skeleton_state.return_value = mock_skeleton_state

        initial_state, skeleton_edges = engine._prepare_initial_state(mock_final_state, None)

        assert initial_state == mock_skeleton_state
        assert skeleton_edges == set([(0, 1)])
        engine.skeleton_extractor.extract_skeleton.assert_called_once()
        engine.skeleton_extractor.create_skeleton_state.assert_called_once()

    def test_build_spatial_indexes(self, engine, mock_final_state):
        """Test spatial index building."""
        engine._build_spatial_indexes(mock_final_state)

        engine.spatial_index.build_street_index.assert_called_once_with(mock_final_state.streets)
        engine.spatial_index.build_frontier_index.assert_called_once_with(mock_final_state.frontiers)
        engine.spatial_index.build_block_index.assert_called_once_with(mock_final_state.blocks)

    def test_generate_candidates(self, engine, mock_final_state):
        """Test candidate generation from strategies."""
        skeleton_edges = set()

        # Mock strategy candidate generation
        mock_candidates = [(Mock(), 0.8)]
        for strategy in engine.strategies:
            strategy.generate_candidates.return_value = mock_candidates

        candidates = engine._generate_candidates(mock_final_state, skeleton_edges)

        assert len(candidates) == len(engine.strategies)  # One candidate per strategy
        assert all(isinstance(c, tuple) and len(c) == 2 for c in candidates)

    def test_apply_strategy_weighting(self, engine):
        """Test strategy weighting application."""
        candidates = [(Mock(), 0.8)]
        strategy = engine.strategies[0]
        strategy.weight = 1.5

        weighted = engine._apply_strategy_weighting(candidates, strategy)

        assert len(weighted) == 1
        assert weighted[0][1] == 0.8 * 1.5  # confidence * weight

    def test_execute_rewind_success(self, engine):
        """Test successful rewind execution."""
        action = Mock()
        current_state = Mock()
        prev_state = Mock()

        engine.rewind_engine.rewind_action.return_value = prev_state

        result = engine._execute_rewind(action, current_state)

        assert result == prev_state
        engine.rewind_engine.rewind_action.assert_called_once_with(action, current_state)

    def test_execute_rewind_failure(self, engine):
        """Test rewind execution failure."""
        action = Mock()
        current_state = Mock()

        engine.rewind_engine.rewind_action.return_value = None

        result = engine._execute_rewind(action, current_state)

        assert result is None

    def test_finalize_action(self, engine):
        """Test action finalization with state diff."""
        action = Mock()
        before_state = Mock()
        after_state = Mock()

        # Mock graph methods
        after_state.graph.number_of_nodes.return_value = 10
        after_state.graph.number_of_edges.return_value = 15
        after_state.frontiers = []  # Mock frontiers list

        # Mock state diff computation with proper structure
        state_diff = {
            'removed_streets': ['street_1'],
            'graph_changes': {'nodes_before': 12, 'edges_before': 18, 'nodes_after': None, 'edges_after': None},
            'frontier_changes': {'frontiers_before': 5, 'frontiers_after': None}
        }
        engine.state_manager.compute_state_diff.return_value = state_diff

        # Mock action creation
        final_action = Mock()
        engine.state_manager.create_action_with_state_diff.return_value = final_action

        result = engine._finalize_action(action, before_state, after_state)

        assert result == final_action
        engine.state_manager.compute_state_diff.assert_called_once_with(before_state, action)
        engine.state_manager.create_action_with_state_diff.assert_called_once()

        # Check that state_diff was updated with after-state info
        assert state_diff['graph_changes']['nodes_after'] == 10
        assert state_diff['graph_changes']['edges_after'] == 15

    def test_record_step_metrics(self, engine):
        """Test step metrics recording."""
        before_state = Mock()
        before_state.streets.__len__ = Mock(return_value=10)

        after_state = Mock()
        after_state.streets.__len__ = Mock(return_value=9)

        strategy_stats = {'strategy1': 5, 'strategy2': 3}

        engine._record_step_metrics(1, before_state, after_state, 8, strategy_stats)

        engine.performance_tracker.record_step_metrics.assert_called_once_with(
            1, 10, 9, 8, strategy_stats
        )

    def test_check_progress_making_progress(self, engine):
        """Test progress checking when making progress."""
        state = Mock()
        state.streets.__len__ = Mock(return_value=8)

        result = engine._check_progress(state, 10, 0)

        assert result is True

    def test_check_progress_no_progress(self, engine):
        """Test progress checking when no progress is made."""
        state = Mock()
        state.streets.__len__ = Mock(return_value=10)

        # When no_progress_count >= max_no_progress_steps, should return False
        result = engine._check_progress(state, 10, engine.config.validation.max_no_progress_steps)

        assert result is False

    def test_log_step_progress(self, engine, mock_final_state, mock_initial_state):
        """Test step progress logging."""
        with patch('src.inverse.core.inference_engine.logger') as mock_logger:
            engine._log_step_progress(5, mock_final_state, mock_initial_state, 10, 0.8, "test_strategy")

            # Should log since step 5 < 10 (first 10 steps always logged)
            mock_logger.info.assert_called_once()

    def test_create_progress_trace(self, engine, mock_initial_state, mock_final_state):
        """Test progress trace creation."""
        actions = [Mock()]
        steps_taken = 3

        trace = engine._create_progress_trace(actions, mock_initial_state, mock_final_state, steps_taken)

        assert isinstance(trace, GrowthTrace)
        assert trace.actions == actions
        assert trace.initial_state == mock_initial_state
        assert trace.final_state == mock_final_state
        assert trace.metadata['interrupted'] is True
        assert trace.metadata['steps_taken'] == steps_taken

    def test_create_trace_metadata(self, engine):
        """Test final trace metadata creation."""
        steps_taken = 42
        skeleton_edges = set([(0, 1), (1, 2)])

        metadata = engine._create_trace_metadata(steps_taken, skeleton_edges)

        assert metadata['inference_method'] == 'modular_inference_engine'
        assert metadata['steps_taken'] == steps_taken
        assert metadata['skeleton_streets'] == len(skeleton_edges)
        assert 'performance_stats' in metadata
        assert 'engine_config' in metadata

    @patch('src.inverse.core.inference_engine.time')
    def test_infer_trace_full_pipeline(self, mock_time, engine, mock_final_state, mock_initial_state):
        """Test full inference pipeline."""
        # Mock time.perf_counter
        mock_time.perf_counter.return_value = 0.0

        # Setup mocks for successful inference
        with patch.object(engine, '_prepare_initial_state', return_value=(mock_initial_state, set())), \
             patch.object(engine, '_build_spatial_indexes'), \
             patch.object(engine, '_generate_candidates', return_value=[(Mock(), 0.8)]), \
             patch.object(engine, '_execute_rewind', return_value=mock_final_state), \
             patch.object(engine, '_finalize_action', return_value=Mock()), \
             patch.object(engine, '_check_progress', return_value=True):

            # Mock streets length to simulate reaching initial state
            mock_final_state.streets.__len__ = Mock(return_value=2)  # Same as initial
            mock_initial_state.streets.__len__ = Mock(return_value=2)

            trace = engine.infer_trace(mock_final_state, max_steps=5)

            assert isinstance(trace, GrowthTrace)
            assert trace.final_state == mock_final_state
            assert trace.initial_state == mock_initial_state

    def test_infer_trace_no_candidates(self, engine, mock_final_state, mock_initial_state):
        """Test inference when no candidates are generated."""
        # Mock setup
        with patch.object(engine, '_prepare_initial_state', return_value=(mock_initial_state, set())), \
             patch.object(engine, '_build_spatial_indexes'), \
             patch.object(engine, '_generate_candidates', return_value=[]):  # No candidates

            trace = engine.infer_trace(mock_final_state, max_steps=1)

            assert isinstance(trace, GrowthTrace)
            assert len(trace.actions) == 0  # No actions inferred

    def test_infer_trace_rewind_failure(self, engine, mock_final_state, mock_initial_state):
        """Test inference when rewind fails."""
        # Mock setup
        with patch.object(engine, '_prepare_initial_state', return_value=(mock_initial_state, set())), \
             patch.object(engine, '_build_spatial_indexes'), \
             patch.object(engine, '_generate_candidates', return_value=[(Mock(), 0.8)]), \
             patch.object(engine, '_execute_rewind', return_value=None):  # Rewind fails

            trace = engine.infer_trace(mock_final_state, max_steps=1)

            assert isinstance(trace, GrowthTrace)
            assert len(trace.actions) == 0  # No actions inferred due to rewind failure
