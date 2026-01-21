#!/usr/bin/env python3
"""
Unit Tests for FractalPatternStrategy

Comprehensive test suite for the fractal pattern strategy implementation,
following SOLID principles and testing all major functionality.
"""

import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon
from unittest.mock import Mock, patch

from src.inverse.strategies.fractal_strategy import FractalPatternStrategy
from src.inverse.core.config import InferenceConfig
from src.core.contracts import GrowthState
from src.inverse.data_structures import InverseGrowthAction, ActionType


class TestFractalPatternStrategy(unittest.TestCase):
    """Test suite for FractalPatternStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        self.strategy = FractalPatternStrategy(weight=1.2, config=self.config)

        # Create mock growth state
        self.mock_state = self._create_mock_growth_state()

    def _create_mock_growth_state(self) -> GrowthState:
        """Create a mock growth state for testing."""
        # Create sample streets with different angles and patterns
        streets_data = []

        # Create a grid-like pattern with some fractal characteristics
        for i in range(5):
            for j in range(5):
                if i < 4:  # Horizontal streets
                    geom = LineString([(i*100, j*100), ((i+1)*100, j*100)])
                    streets_data.append({
                        'geometry': geom,
                        'u': i*5 + j,
                        'v': (i+1)*5 + j,
                        'osmid': f"h_{i}_{j}",
                        'highway': 'residential'
                    })

                if j < 4:  # Vertical streets
                    geom = LineString([(i*100, j*100), (i*100, (j+1)*100)])
                    streets_data.append({
                        'geometry': geom,
                        'u': i*5 + j,
                        'v': i*5 + j + 1,
                        'osmid': f"v_{i}_{j}",
                        'highway': 'residential'
                    })

        streets_df = gpd.GeoDataFrame(streets_data, geometry='geometry')

        # Create mock graph
        graph = Mock()
        graph.has_edge = Mock(return_value=True)
        graph.number_of_nodes = Mock(return_value=25)
        graph.number_of_edges = Mock(return_value=len(streets_df))

        # Create mock frontiers
        frontiers = []

        return GrowthState(
            streets=streets_df,
            graph=graph,
            frontiers=frontiers,
            blocks=gpd.GeoDataFrame(columns=['geometry']).set_geometry('geometry'),
            city_bounds=Polygon([(0, 0), (500, 0), (500, 500), (0, 500)]),
            iteration=0
        )

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = FractalPatternStrategy()
        self.assertEqual(strategy.name, "fractal_pattern")
        self.assertEqual(strategy.weight, 1.2)
        self.assertIsInstance(strategy.config, InferenceConfig)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        self.strategy._validate_strategy_config()

        # Test invalid config
        invalid_config = InferenceConfig()
        invalid_config.strategies.fractal_pattern_min_similarity = -0.1

        with self.assertRaises(ValueError):
            strategy = FractalPatternStrategy(config=invalid_config)

    def test_fractal_dimension_computation(self):
        """Test fractal dimension estimation."""
        fractal_dim = self.strategy._compute_fractal_dimension(self.mock_state.streets)

        # Should be between 1.0 and 2.0 for street networks
        self.assertGreaterEqual(fractal_dim, 1.0)
        self.assertLessEqual(fractal_dim, 2.0)

        # Test with minimal streets
        minimal_streets = pd.DataFrame([{
            'geometry': LineString([(0, 0), (10, 0)]),
            'u': 0, 'v': 1
        }])

        fractal_dim_minimal = self.strategy._compute_fractal_dimension(minimal_streets)
        self.assertEqual(fractal_dim_minimal, 1.5)  # Default for small networks

    def test_angle_scoring(self):
        """Test angle-based fractal pattern scoring."""
        # Create angle cache
        angle_cache = {'angles': np.array([0, 90, 45, 135])}

        # Test angle 0: only 1 out of 4 angles (25%) are similar, so disruption score = 1.0 - 0.25 = 0.75
        angle_0_score = self.strategy._score_angle_fractal_fit(0, angle_cache)
        self.assertAlmostEqual(angle_0_score, 0.75, places=2)  # Moderately disruptive

        # Test angle 30: 0 out of 4 angles are similar (0%), so disruption score = 1.0 - 0.0 = 1.0
        angle_30_score = self.strategy._score_angle_fractal_fit(30, angle_cache)
        self.assertAlmostEqual(angle_30_score, 1.0, places=2)  # Highly disruptive

        # Test angle 45: 1 out of 4 angles (25%) are similar, so disruption score = 1.0 - 0.25 = 0.75
        angle_45_score = self.strategy._score_angle_fractal_fit(45, angle_cache)
        self.assertAlmostEqual(angle_45_score, 0.75, places=2)  # Moderately disruptive

    def test_length_scoring(self):
        """Test length-based fractal pattern scoring."""
        # Create streets with different lengths
        lengths = [50, 100, 100, 150, 200]
        streets_data = []
        for i, length in enumerate(lengths):
            geom = LineString([(0, 0), (length, 0)])
            streets_data.append({
                'geometry': geom,
                'u': i, 'v': i+1
            })

        streets_df = pd.DataFrame(streets_data)

        # Pre-compute length statistics
        length_stats = self.strategy._precompute_length_statistics(streets_df)

        # Test outlier length (should have high disruption score)
        outlier_score = self.strategy._score_length_fractal_fit(300, length_stats)
        self.assertGreater(outlier_score, 0.5)

        # Test typical length (should have lower disruption score)
        typical_score = self.strategy._score_length_fractal_fit(125, length_stats)
        self.assertLess(typical_score, 0.5)

    def test_connectivity_scoring(self):
        """Test connectivity-based fractal pattern scoring."""
        # Create connectivity map for the test streets
        # Street 0: u=0, v=1 -> connects to nodes 0,1
        # Street 1: u=1, v=2 -> connects to nodes 1,2
        # Street 2: u=3, v=4 -> connects to nodes 3,4
        connectivity_map = {
            '0': 1,  # node 0 has 1 connection (to street 0)
            '1': 2,  # node 1 has 2 connections (to streets 0,1)
            '2': 1,  # node 2 has 1 connection (to street 1)
            '3': 1,  # node 3 has 1 connection (to street 2)
            '4': 1,  # node 4 has 1 connection (to street 2)
        }

        # Mock street objects
        isolated_street = Mock()
        isolated_street.get.side_effect = lambda key: {'u': '3', 'v': '4'}.get(key)  # Street 2: isolated

        connected_street = Mock()
        connected_street.get.side_effect = lambda key: {'u': '0', 'v': '1'}.get(key)  # Street 0: connected

        # Isolated street: nodes 3,4 have 1 connection each
        # Total connections = 1 + 1 - 2 = 0, so score = 1.0 - 0/6 = 1.0
        isolated_score = self.strategy._score_connectivity_fractal_impact(isolated_street, connectivity_map)
        self.assertAlmostEqual(isolated_score, 1.0, places=1)  # Maximum disruption score

        # Connected street: nodes 0,1 have 1 and 2 connections respectively
        # Total connections = 1 + 2 - 2 = 1, so score = 1.0 - 1/6 ≈ 0.83
        connected_score = self.strategy._score_connectivity_fractal_impact(connected_street, connectivity_map)
        self.assertAlmostEqual(connected_score, 0.83, places=2)  # Moderate disruption score

    def test_fractal_disruption_scoring(self):
        """Test overall fractal disruption scoring."""
        # Create a test street
        test_street = Mock()
        test_street.geometry = LineString([(0, 0), (100, 0)])  # Horizontal, 100m

        # Create angle cache
        angle_cache = {'angles': np.array([0, 90, 45, 135])}  # Mix of angles

        # Create connectivity map and length stats for the mock state
        connectivity_map = self.strategy._build_connectivity_map(self.mock_state.streets)
        length_stats = self.strategy._precompute_length_statistics(self.mock_state.streets)

        fractal_dim = 1.7  # Typical fractal dimension

        score = self.strategy._score_fractal_disruption(
            test_street, fractal_dim, angle_cache, connectivity_map, length_stats
        )

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_generate_candidates(self):
        """Test candidate generation."""
        skeleton_edges = set()  # No skeleton edges

        candidates = self.strategy.generate_candidates(
            self.mock_state, skeleton_edges
        )

        # Should generate some candidates
        self.assertIsInstance(candidates, list)
        for action, confidence in candidates:
            self.assertIsInstance(action, InverseGrowthAction)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_generate_candidates_insufficient_streets(self):
        """Test candidate generation with insufficient streets."""
        # Create state with very few streets
        minimal_streets = pd.DataFrame([{
            'geometry': LineString([(0, 0), (10, 0)]),
            'u': 0, 'v': 1
        }])

        minimal_state = GrowthState(
            streets=minimal_streets,
            graph=Mock(),
            frontiers=[],
            blocks=gpd.GeoDataFrame(columns=['geometry']).set_geometry('geometry'),
            city_bounds=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            iteration=0
        )

        candidates = self.strategy.generate_candidates(minimal_state, set())
        self.assertEqual(len(candidates), 0)

    def test_generate_candidates_with_skeleton(self):
        """Test candidate generation respects skeleton edges."""
        # Mark some edges as skeleton
        skeleton_edges = {(0, 1)}  # First horizontal edge

        candidates = self.strategy.generate_candidates(
            self.mock_state, skeleton_edges
        )

        # Should not generate candidates for skeleton edges
        for action, confidence in candidates:
            edge_u = action.intent_params.get('edge_u')
            edge_v = action.intent_params.get('edge_v')
            self.assertNotEqual((edge_u, edge_v), (0, 1))

    def test_street_angle_calculation(self):
        """Test street angle calculation."""
        # Horizontal line
        horizontal = LineString([(0, 0), (100, 0)])
        angle_h = self.strategy._get_street_angle(horizontal)
        self.assertAlmostEqual(angle_h, 0.0, places=1)

        # Vertical line
        vertical = LineString([(0, 0), (0, 100)])
        angle_v = self.strategy._get_street_angle(vertical)
        self.assertAlmostEqual(angle_v, 90.0, places=1)

        # Diagonal line
        diagonal = LineString([(0, 0), (100, 100)])
        angle_d = self.strategy._get_street_angle(diagonal)
        self.assertAlmostEqual(angle_d, 45.0, places=1)

    def test_angle_vectorized_extraction(self):
        """Test vectorized angle extraction."""
        cache = self.strategy._extract_street_angles_vectorized(self.mock_state.streets)

        self.assertIn('angles', cache)
        self.assertIn('lengths', cache)
        self.assertIsInstance(cache['angles'], np.ndarray)
        self.assertIsInstance(cache['lengths'], np.ndarray)
        self.assertEqual(len(cache['angles']), len(self.mock_state.streets))
        self.assertEqual(len(cache['lengths']), len(self.mock_state.streets))

    def test_strategy_info(self):
        """Test strategy information retrieval."""
        info = self.strategy.get_strategy_info()

        self.assertEqual(info['name'], 'fractal_pattern')
        self.assertEqual(info['weight'], 1.2)
        self.assertIn('fractal_min_similarity', info)
        self.assertIn('description', info)

    def test_edge_case_empty_streets(self):
        """Test behavior with empty streets DataFrame."""
        empty_streets = gpd.GeoDataFrame(columns=['geometry']).set_geometry('geometry')
        empty_state = GrowthState(
            streets=empty_streets,
            graph=Mock(),
            frontiers=[],
            blocks=gpd.GeoDataFrame(columns=['geometry']).set_geometry('geometry'),
            city_bounds=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            iteration=0
        )

        candidates = self.strategy.generate_candidates(empty_state, set())
        self.assertEqual(len(candidates), 0)

    def test_edge_case_invalid_geometry(self):
        """Test behavior with invalid geometry."""
        # Create street with invalid geometry
        invalid_streets = gpd.GeoDataFrame([{
            'geometry': Point(0, 0),  # Point instead of LineString
            'u': 0, 'v': 1
        }])

        invalid_state = GrowthState(
            streets=invalid_streets,
            graph=Mock(),
            frontiers=[],
            blocks=gpd.GeoDataFrame(columns=['geometry']).set_geometry('geometry'),
            city_bounds=Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            iteration=0
        )

        candidates = self.strategy.generate_candidates(invalid_state, set())
        self.assertEqual(len(candidates), 0)

    @patch('src.inverse.strategies.fractal_strategy.logger')
    def test_logging_debug_output(self, mock_logger):
        """Test that debug logging works correctly."""
        # Enable debug logging in config
        debug_config = InferenceConfig()
        debug_config.logging.enable_debug_logging = True

        debug_strategy = FractalPatternStrategy(config=debug_config)

        # Generate candidates (should trigger debug logs)
        candidates = debug_strategy.generate_candidates(self.mock_state, set())

        # Verify debug logging was called
        mock_logger.debug.assert_called()


if __name__ == '__main__':
    unittest.main()
