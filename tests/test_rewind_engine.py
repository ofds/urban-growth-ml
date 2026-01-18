#!/usr/bin/env python3
"""
Comprehensive test suite for RewindEngine.
Tests correctness, performance, and invariants.
"""

import pytest
import time
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from core.contracts import GrowthState, FrontierEdge
from inverse.rewind import RewindEngine
from inverse.data_structures import InverseGrowthAction, ActionType


@pytest.fixture
def sample_city():
    """Create a small test city for unit tests."""
    streets_data = {
        'geometry': [
            LineString([(0, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(200, 0), (200, 100)]),
        ],
        'highway': ['primary', 'secondary', 'residential'],
        'osmid': ['street_0', 'street_1', 'street_2'],
        'u': ['0', '1', '2'],
        'v': ['1', '2', '3']
    }

    graph = nx.Graph()
    graph.add_node('0', geometry=Point(0, 0), x=0, y=0)
    graph.add_node('1', geometry=Point(100, 0), x=100, y=0)
    graph.add_node('2', geometry=Point(200, 0), x=200, y=0)
    graph.add_node('3', geometry=Point(200, 100), x=200, y=100)

    graph.add_edge('0', '1', geometry=streets_data['geometry'][0])
    graph.add_edge('1', '2', geometry=streets_data['geometry'][1])
    graph.add_edge('2', '3', geometry=streets_data['geometry'][2])

    streets = gpd.GeoDataFrame(streets_data, crs='EPSG:4326')
    blocks = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

    frontiers = [
        FrontierEdge(
            frontier_id='dead_end_0_1',
            edge_id=('0', '1'),
            block_id=None,
            geometry=LineString([(0, 0), (100, 0)]),
            frontier_type='dead_end',
            expansion_weight=0.8,
            spatial_hash=""
        ),
        FrontierEdge(
            frontier_id='block_edge_1_2',
            edge_id=('1', '2'),
            block_id=None,
            geometry=LineString([(100, 0), (200, 0)]),
            frontier_type='block_edge',
            expansion_weight=0.5,
            spatial_hash=""
        ),
        FrontierEdge(
            frontier_id='dead_end_2_3',
            edge_id=('2', '3'),
            block_id=None,
            geometry=LineString([(200, 0), (200, 100)]),
            frontier_type='dead_end',
            expansion_weight=0.8,
            spatial_hash=""
        )
    ]

    city_bounds = Polygon([(-50, -50), (250, -50), (250, 150), (-50, 150)])

    return GrowthState(
        streets=streets,
        blocks=blocks,
        frontiers=frontiers,
        graph=graph,
        iteration=1,
        city_bounds=city_bounds
    )


@pytest.fixture
def rewind_engine():
    """Create a fresh RewindEngine for each test."""
    return RewindEngine()


class TestRewindCorrectness:
    """Test that rewind operations produce correct results."""

    def test_rewind_extend_frontier_removes_street(self, rewind_engine, sample_city):
        """Test that EXTEND_FRONTIER rewind removes the correct street."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        # Should have one less street
        assert len(result.streets) == len(sample_city.streets) - 1
        # Should not contain the removed edge
        assert not result.graph.has_edge('2', '3')
        # Should not have frontier for removed edge
        frontier_edge_ids = {(f.edge_id[0], f.edge_id[1]) for f in result.frontiers}
        assert ('2', '3') not in frontier_edge_ids
        assert ('3', '2') not in frontier_edge_ids

    def test_rewind_preserves_graph_integrity(self, rewind_engine, sample_city):
        """Test that rewind maintains graph invariants."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        # Graph should remain connected (or empty)
        assert nx.is_connected(result.graph) or len(result.graph) == 0

        # All streets in GeoDataFrame should exist in graph
        for _, street in result.streets.iterrows():
            u, v = str(street['u']), str(street['v'])
            assert result.graph.has_edge(u, v), f"Street ({u}, {v}) missing from graph"

    def test_rewind_fails_gracefully_on_invalid_action(self, rewind_engine, sample_city):
        """Test that invalid actions return unchanged state."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='nonexistent_street',
            intent_params={},
            realized_geometry={'edgeid': ('999', '1000')}  # Non-existent edge
        )

        result = rewind_engine.rewind_action(action, sample_city)

        # Should return unchanged state
        assert result is sample_city

    def test_rewind_maintains_city_bounds(self, rewind_engine, sample_city):
        """Test that city bounds are preserved."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        assert result.city_bounds.equals(sample_city.city_bounds)

    def test_rewind_decrements_iteration(self, rewind_engine, sample_city):
        """Test that iteration counter is decremented."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        assert result.iteration == sample_city.iteration - 1


class TestRewindInvariants:
    """Test that rewind operations maintain system invariants."""

    @pytest.mark.parametrize("edge_id", [('0', '1'), ('1', '2'), ('2', '3')])
    def test_rewind_preserves_street_graph_consistency(self, rewind_engine, sample_city, edge_id):
        """Test that streets GeoDataFrame and graph remain consistent."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id=f'street_{edge_id[0]}_{edge_id[1]}',
            intent_params={},
            realized_geometry={'edgeid': edge_id}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        # Every edge in streets should exist in graph
        for _, street in result.streets.iterrows():
            u, v = str(street['u']), str(street['v'])
            assert result.graph.has_edge(u, v), f"Street ({u}, {v}) not in graph"

        # Every graph edge should have geometry
        for u, v, data in result.graph.edges(data=True):
            assert 'geometry' in data, f"Edge ({u}, {v}) missing geometry"
            assert isinstance(data['geometry'], LineString)

    def test_rewind_frontiers_reference_valid_edges(self, rewind_engine, sample_city):
        """Test that all frontiers reference edges that exist in the graph."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = rewind_engine.rewind_action(action, sample_city)

        for frontier in result.frontiers:
            u, v = frontier.edge_id
            assert result.graph.has_edge(u, v), f"Frontier references non-existent edge ({u}, {v})"


class TestRewindPerformance:
    """Performance tests and benchmarks."""

    @pytest.mark.benchmark
    def test_rewind_performance_baseline(self, benchmark, rewind_engine, sample_city):
        """Baseline performance test for rewind operations."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        result = benchmark(rewind_engine.rewind_action, action, sample_city)

        # Basic correctness check
        assert len(result.streets) < len(sample_city.streets)
        assert not result.graph.has_edge('2', '3')

    def test_rewind_memory_efficiency(self, rewind_engine, sample_city):
        """Test that rewind doesn't leak memory or create excessive copies."""
        import gc
        import psutil
        import os

        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )

        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform rewind
        result = rewind_engine.rewind_action(action, sample_city)

        # Force garbage collection
        gc.collect()

        # Check memory after operation
        final_memory = process.memory_info().rss
        memory_delta = final_memory - initial_memory

        # Memory increase should be reasonable (less than 10MB)
        assert memory_delta < 10 * 1024 * 1024, f"Memory leak detected: {memory_delta} bytes"


class TestRewindEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_rewind_empty_graph(self, rewind_engine):
        """Test rewind on empty graph."""
        empty_state = GrowthState(
            streets=gpd.GeoDataFrame(columns=['geometry', 'u', 'v'], crs='EPSG:4326'),
            blocks=gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326'),
            frontiers=[],
            graph=nx.Graph(),
            iteration=0,
            city_bounds=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        )

        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='test',
            intent_params={},
            realized_geometry={'edgeid': ('0', '1')}
        )

        result = rewind_engine.rewind_action(action, empty_state)
        assert result is empty_state  # Should return unchanged

    def test_rewind_single_edge_graph(self, rewind_engine):
        """Test rewind on graph with single edge."""
        streets_data = {
            'geometry': [LineString([(0, 0), (100, 0)])],
            'highway': ['primary'],
            'osmid': ['street_0'],
            'u': ['0'],
            'v': ['1']
        }

        graph = nx.Graph()
        graph.add_edge('0', '1', geometry=streets_data['geometry'][0])

        single_edge_state = GrowthState(
            streets=gpd.GeoDataFrame(streets_data, crs='EPSG:4326'),
            blocks=gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326'),
            frontiers=[],
            graph=graph,
            iteration=1,
            city_bounds=Polygon([(-50, -50), (150, -50), (150, 50), (-50, 50)])
        )

        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_0',
            intent_params={},
            realized_geometry={'edgeid': ('0', '1')}
        )

        result = rewind_engine.rewind_action(action, single_edge_state)

        assert len(result.streets) == 0
        assert len(result.graph.edges()) == 0

    def test_rewind_invalid_action_type(self, rewind_engine, sample_city):
        """Test handling of unsupported action types."""
        # Create action with invalid type
        action = InverseGrowthAction(
            action_type="INVALID_TYPE",  # Not in ActionType enum
            target_id='test',
            intent_params={},
            realized_geometry={}
        )

        result = rewind_engine.rewind_action(action, sample_city)
        assert result is sample_city  # Should return unchanged


class TestRewindRegression:
    """Regression tests for previously fixed bugs."""

    def test_rewind_doesnt_corrupt_edge_index(self, rewind_engine, sample_city):
        """Test that edge index doesn't get corrupted between rewinds."""
        # First rewind
        action1 = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_2',
            intent_params={},
            realized_geometry={'edgeid': ('2', '3')}
        )
        result1 = rewind_engine.rewind_action(action1, sample_city)

        # Second rewind on the result
        action2 = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id='street_1',
            intent_params={},
            realized_geometry={'edgeid': ('1', '2')}
        )
        result2 = rewind_engine.rewind_action(action2, result1)

        # Should still maintain consistency
        for _, street in result2.streets.iterrows():
            u, v = str(street['u']), str(street['v'])
            assert result2.graph.has_edge(u, v)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
