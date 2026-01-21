#!/usr/bin/env python3
"""
Fractal Pattern Strategy Module for Urban Growth Inference

Implements fractal dimension analysis and pattern continuation logic
for detecting self-similar growth patterns in urban street networks.
"""

import logging
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict, Any, Set

from .base_strategy import BaseInferenceStrategy
from ..spatial.spatial_index import SpatialIndex
from ..data_structures import InverseGrowthAction
from src.core.contracts import GrowthState

logger = logging.getLogger(__name__)


class FractalPatternStrategy(BaseInferenceStrategy):
    """
    Fractal Pattern Strategy for Urban Growth Inference.

    This strategy analyzes the fractal properties of street networks to detect
    self-similar growth patterns. It uses fractal dimension estimation and
    morphological similarity scoring to identify streets that disrupt or
    continue fractal patterns.

    Mathematical Approach:
    - Estimates fractal dimension using box-counting method
    - Analyzes angle distributions for pattern continuation
    - Scores streets based on their contribution to fractal patterns
    """

    def __init__(self, weight: float = 1.2, config=None):
        """
        Initialize the fractal pattern strategy.

        Args:
            weight: Strategy weight for candidate ranking (default 1.2)
            config: Configuration instance
        """
        super().__init__("fractal_pattern", weight, config)

    def _validate_strategy_config(self):
        """Validate fractal-specific configuration."""
        # Check fractal dimension parameters
        if self.config.strategies.fractal_pattern_min_similarity < 0:
            raise ValueError("fractal_pattern_min_similarity must be non-negative")
        if self.config.strategies.fractal_pattern_min_similarity > 1:
            raise ValueError("fractal_pattern_min_similarity must be <= 1.0")

    def generate_candidates(self, state: GrowthState, skeleton_edges: set = None,
                          spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """
        Generate candidate actions using fractal pattern analysis.

        Analyzes fractal properties and identifies streets that disrupt
        self-similar patterns for removal. Uses internal heuristics to
        determine which streets are candidates for removal.

        Args:
            state: Current growth state
            skeleton_edges: Optional arterial skeleton edges (not to be removed)
            spatial_index: Optional spatial index for efficiency

        Returns:
            List of (action, confidence) tuples
        """
        candidates = []

        # Skip if insufficient streets for fractal analysis
        min_streets_for_fractal = 5  # Minimum streets needed for meaningful fractal analysis
        if len(state.streets) < min_streets_for_fractal:
            logger.debug(f"FractalPatternStrategy: Insufficient streets ({len(state.streets)}) for analysis")
            return candidates

        # Compute fractal dimension for pattern analysis
        fractal_dim = self._compute_fractal_dimension(state.streets)

        # Pre-compute caches for O(1) lookups during scoring
        angle_cache = self._extract_street_angles_vectorized(state.streets)
        connectivity_map = self._build_connectivity_map_vectorized(state.streets)
        length_stats = self._precompute_length_statistics(state.streets)

        # Pre-compute connectivity constraints (bridges) - O(V+E) one-time cost
        bridge_streets = self._compute_connectivity_constraints(state)

        # Score existing streets for removal based on fractal pattern disruption
        for street_id in state.streets.index:
            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, self._get_geometry_type()):
                continue

            # Optional: Check if this street's edge is part of external skeleton
            if skeleton_edges:
                u, v = street.get('u'), street.get('v')
                if u is not None and v is not None:
                    edge_key = (min(u, v), max(u, v))
                    if edge_key in skeleton_edges:
                        logger.debug(f"FractalPatternStrategy: Skipping skeleton street {street_id}")
                        continue

            # CRITICAL: Skip bridge streets (would disconnect graph)
            if bridge_streets.get(street_id, False):
                logger.debug(f"FractalPatternStrategy: Skipping bridge street {street_id}")
                continue

            # Internal filtering: Skip streets that are likely core infrastructure
            if not self._is_street_removable(street, connectivity_map, state.streets):
                logger.debug(f"FractalPatternStrategy: Skipping core street {street_id}")
                continue

            # Score based on fractal pattern disruption
            confidence = self._score_fractal_disruption(
                street, fractal_dim, angle_cache, connectivity_map, length_stats
            )

            if confidence > 0.1:
                # Create action directly from street
                action = self._create_action_from_street(street_id, street, confidence, state)
                if action is not None:
                    candidates.append((action, confidence))
                    logger.debug(f"FractalPatternStrategy: Created candidate for street {street_id} (confidence: {confidence:.3f})")

        logger.debug(f"FractalPatternStrategy: Generated {len(candidates)} candidates from {len(state.streets)} total streets")
        return candidates

    def _score_fractal_disruption(self, street, fractal_dim: float,
                                angle_cache: Optional[Dict] = None, connectivity_map: Optional[Dict[str, int]] = None,
                                length_stats: Optional[Dict[str, float]] = None) -> float:
        """
        Score how disruptive removing this street would be to fractal patterns.

        Higher scores indicate streets that are more likely to be recent additions
        that disrupt fractal patterns.
        """
        if not isinstance(street.geometry, self._get_geometry_type()):
            return 0.0

        # Get street properties
        street_angle = self._get_street_angle(street.geometry)
        street_length = street.geometry.length

        # Score based on multiple fractal criteria
        angle_score = self._score_angle_fractal_fit(street_angle, angle_cache)
        length_score = self._score_length_fractal_fit(street_length, length_stats)
        connectivity_score = self._score_connectivity_fractal_impact(street, connectivity_map)

        # Combine scores (weighted average)
        # Angle patterns are most important for fractal analysis
        combined_score = 0.5 * angle_score + 0.3 * length_score + 0.2 * connectivity_score

        # Apply fractal dimension adjustment
        # Streets in networks with higher fractal dimension are more likely to be pattern-disrupting
        fractal_adjustment = min(1.0, fractal_dim / 1.8)  # Normalize around typical fractal dimension

        return combined_score * fractal_adjustment

    def _compute_fractal_dimension(self, streets_gdf) -> float:
        """
        Estimate fractal dimension using box-counting method with coordinate sampling.

        Reduces memory usage by 5-10x and CPU time proportionally by limiting
        coordinate processing to MAX_COORDS samples.

        Returns the estimated fractal dimension D where 1 ≤ D ≤ 2.
        """
        if len(streets_gdf) < 10:
            return 1.5  # Default for small networks

        # Get all street coordinates with sampling to reduce memory usage
        all_coords = []
        MAX_COORDS = 5000  # Configurable limit for memory efficiency

        for idx, street in streets_gdf.iterrows():
            if hasattr(street.geometry, 'coords'):
                coords = list(street.geometry.coords)
                all_coords.extend(coords)

                # Early termination if we exceed the limit
                if len(all_coords) >= MAX_COORDS:
                    # Random sample for representative distribution
                    indices = np.random.choice(len(all_coords), MAX_COORDS, replace=False)
                    all_coords = [all_coords[i] for i in sorted(indices)]
                    break

        if len(all_coords) < 20:
            return 1.5

        points = np.array(all_coords)

        # Simple box-counting with different box sizes
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        diagonal = np.linalg.norm(max_coord - min_coord)

        if diagonal == 0:
            return 1.5

        # Use fewer box sizes for performance, but enough for reliable estimation
        box_sizes = [diagonal / (2 ** i) for i in range(3, 7)]  # 4 box sizes: 8, 16, 32, 64 boxes
        box_counts = []

        for box_size in box_sizes:
            if box_size <= 0:
                continue

            # Count occupied boxes using histogram
            x_bins = np.arange(min_coord[0], max_coord[0] + box_size, box_size)
            y_bins = np.arange(min_coord[1], max_coord[1] + box_size, box_size)

            # Digitize coordinates into bins
            x_indices = np.digitize(points[:, 0], x_bins) - 1
            y_indices = np.digitize(points[:, 1], y_bins) - 1

            # Count unique (x,y) bin combinations
            occupied = set(zip(x_indices, y_indices))
            box_counts.append(len(occupied))

        # Linear regression for fractal dimension
        if len(box_counts) >= 3:
            log_sizes = np.log(1 / np.array(box_sizes[:len(box_counts)]))
            log_counts = np.log(box_counts)

            # Simple linear regression
            try:
                slope = np.polyfit(log_sizes, log_counts, 1)[0]
                # Clamp to reasonable range for street networks
                return max(1.0, min(2.0, slope))
            except (np.RankWarning, ValueError):
                pass

        return 1.5  # Fallback

    def _score_angle_fractal_fit(self, street_angle: float,
                               angle_cache: Optional[Dict] = None) -> float:
        """
        Score how well the street angle fits fractal angle patterns.

        Optimized angle calculations using vectorized operations.
        Streets with uncommon angles are more likely to disrupt fractal patterns.
        """
        if angle_cache is None or 'angles' not in angle_cache:
            return 0.5

        angles = angle_cache['angles']
        if len(angles) < 3:
            return 0.5

        # Optimized angle difference calculation
        angle_diffs = np.abs(angles - street_angle)
        # Handle circular nature of angles (0° = 180°)
        min_diffs = np.where(angle_diffs > 90, 180 - angle_diffs, angle_diffs)

        # Count angles within tolerance (15 degrees for fractal patterns)
        similar_count = np.sum(min_diffs < 15)
        commonality_score = similar_count / len(angles)

        # Streets with uncommon angles get higher scores (more likely recent additions)
        return 1.0 - commonality_score

    def _precompute_length_statistics(self, streets_gdf) -> Dict[str, float]:
        """
        Pre-compute length statistics for O(1) length scoring.

        Eliminates O(N²) bottleneck by computing statistics once instead of
        iterating through all streets for each street being scored.

        Returns:
            Dict with 'mean', 'std', and 'valid' flag
        """
        lengths = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, self._get_geometry_type()):
                lengths.append(street.geometry.length)

        if len(lengths) < 5:
            return {'mean': 0.0, 'std': 0.0, 'valid': False}

        lengths_array = np.array(lengths)
        return {
            'mean': np.mean(lengths_array),
            'std': np.std(lengths_array),
            'valid': True
        }

    def _score_length_fractal_fit(self, street_length: float, length_stats: Dict[str, float]) -> float:
        """
        Score street length fit for fractal patterns using pre-computed statistics.

        O(1) operation instead of O(N) - eliminates the major O(N²) bottleneck.
        """
        if not length_stats.get('valid', False):
            return 0.5

        mean_length = length_stats['mean']
        std_length = length_stats['std']

        if std_length == 0:
            return 0.5

        # Z-score for length deviation
        z_score = abs(street_length - mean_length) / std_length

        # Higher z-scores (more unusual lengths) get higher disruption scores
        return min(1.0, z_score / 2.0)

    def _score_connectivity_fractal_impact(self, street, connectivity_map: Dict[str, int]) -> float:
        """
        Score connectivity impact in fractal context using pre-computed connectivity map.

        In fractal networks, connectivity patterns follow specific scaling laws.
        Uses O(1) dictionary lookups instead of O(N) iteration for ~100x speedup.
        """
        # Get node connectivity from pre-computed map
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return 0.1

        # O(1) dictionary lookups instead of O(N) iteration
        u_connections = connectivity_map.get(u, 0)
        v_connections = connectivity_map.get(v, 0)

        # Total connections to these nodes (subtract self-connections)
        # Each street contributes 2 connections (one to each endpoint)
        total_connections = u_connections + v_connections - 2

        # Lower connectivity = higher score (more likely recent addition)
        connectivity_score = 1.0 - min(1.0, total_connections / 6.0)  # Normalize by typical connectivity

        return connectivity_score

    def _extract_street_angles_vectorized(self, streets_gdf) -> Dict[str, np.ndarray]:
        """
        Extract all street angles and lengths in a batch for 2-3x speedup.

        Reduces pandas overhead by extracting all geometries upfront instead
        of repeated .loc operations during scoring.

        Returns dict with 'angles', 'lengths', and 'valid_mask' arrays.
        """
        # Batch extract geometries to avoid repeated pandas operations
        geometries = streets_gdf.geometry.values
        valid_mask = np.array([isinstance(g, self._get_geometry_type()) for g in geometries])

        if not np.any(valid_mask):
            return {'angles': np.array([]), 'lengths': np.array([]), 'valid_mask': valid_mask}

        # Extract angles and lengths for valid geometries only
        valid_geoms = geometries[valid_mask]
        angles = np.array([self._get_street_angle(g) for g in valid_geoms])
        lengths = np.array([g.length for g in valid_geoms])

        return {
            'angles': angles,
            'lengths': lengths,
            'valid_mask': valid_mask
        }

    def _get_street_angle(self, geometry) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180  # Normalize to 0-180

    def _build_connectivity_map_vectorized(self, streets_gdf) -> Dict[str, int]:
        """
        Build connectivity map using vectorized operations for 3-5x speedup.

        Uses numpy unique counting instead of iterative dictionary updates.
        """
        # Extract all node IDs (both u and v columns)
        u_nodes = streets_gdf['u'].dropna().values
        v_nodes = streets_gdf['v'].dropna().values
        all_nodes = np.concatenate([u_nodes, v_nodes])

        # Vectorized counting using numpy
        unique_nodes, counts = np.unique(all_nodes, return_counts=True)

        # Convert back to dictionary
        return dict(zip(unique_nodes, counts))

    def _build_connectivity_map(self, streets_gdf) -> Dict[str, int]:
        """
        Build a connectivity map showing how many connections each node has.

        Returns dict mapping node_id -> connection_count
        """
        connectivity = {}

        # Count connections for each node
        for idx, street in streets_gdf.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None:
                connectivity[u] = connectivity.get(u, 0) + 1
            if v is not None:
                connectivity[v] = connectivity.get(v, 0) + 1

        return connectivity

    def _is_street_removable(self, street, connectivity_map: Dict[str, int], streets_gdf) -> bool:
        """
        Determine if a street should be considered for removal based on internal heuristics.

        Uses connectivity, length, and centrality to identify core vs. peripheral streets.
        """
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return True  # Can't determine connectivity, assume removable

        # Get connectivity scores for endpoints
        u_connectivity = connectivity_map.get(u, 0)
        v_connectivity = connectivity_map.get(v, 0)
        avg_connectivity = (u_connectivity + v_connectivity) / 2.0

        # Get street length
        street_length = street.geometry.length

        # Heuristics for core streets (should NOT be removed):

        # 1. High connectivity: Streets connecting highly connected nodes are likely core
        if avg_connectivity >= 4:  # Nodes with 4+ connections are likely intersections
            return False

        # 2. Very long streets: Long streets are likely major arterials
        if street_length >= 500:  # 500m+ streets are likely major roads
            return False

        # 3. Bridge connections: Streets that connect high-connectivity nodes
        if u_connectivity >= 3 and v_connectivity >= 3:
            return False

        # For ML-generated cities starting from one edge:
        # - Allow removal of peripheral streets (low connectivity)
        # - Allow removal of shorter streets (likely recent additions)
        # - Focus on streets that could be growth artifacts

        return True  # Default: consider for removal

    def _build_graph_from_state(self, state: GrowthState) -> nx.Graph:
        """
        Build NetworkX graph from current street network state.

        Returns:
            NetworkX Graph with nodes and edges representing street network
        """
        G = nx.Graph()

        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                G.add_edge(u, v, street_id=idx)

        return G

    def _would_disconnect_graph(self, street_id, state: GrowthState) -> bool:
        """
        Return True if removing this street disconnects the graph.

        Uses NetworkX is_connected() to verify graph remains connected
        after hypothetical street removal.

        Args:
            street_id: ID of street to check
            state: Current growth state

        Returns:
            True if removal would disconnect graph, False otherwise
        """
        street = state.streets.loc[street_id]
        u, v = street.get('u'), street.get('v')

        if u is None or v is None:
            return False  # Can't determine, assume safe

        # Build graph from current state
        G = self._build_graph_from_state(state)

        # Check if graph is currently connected
        if not nx.is_connected(G):
            return True  # Already disconnected, don't make it worse

        # Temporarily remove edge and check connectivity
        G.remove_edge(u, v)

        # If graph becomes disconnected, this street is critical
        return not nx.is_connected(G)

    def _is_bridge_street(self, street_id, state: GrowthState) -> bool:
        """
        Return True if this street is a bridge (critical connectivity link).

        A bridge is an edge whose removal increases the number of
        connected components. Uses NetworkX bridge detection algorithm
        for O(V+E) efficiency.

        Args:
            street_id: ID of street to check
            state: Current growth state

        Returns:
            True if street is a bridge, False otherwise
        """
        street = state.streets.loc[street_id]
        u, v = street.get('u'), street.get('v')

        if u is None or v is None:
            return False

        # Build graph from current state
        G = self._build_graph_from_state(state)

        # Get all bridges in the graph (efficient O(V+E) algorithm)
        bridges = set(nx.bridges(G))

        # Check if this edge is a bridge (order-independent)
        return (u, v) in bridges or (v, u) in bridges

    def _compute_connectivity_constraints(self, state: GrowthState) -> Dict[int, bool]:
        """
        Pre-compute all bridge streets in O(V+E) time for the entire graph.

        This is more efficient than checking each street individually,
        avoiding O(N*(V+E)) complexity.

        Returns:
            Dict mapping street_id -> is_bridge_or_articulation
        """
        G = self._build_graph_from_state(state)

        # Get all bridges in one pass - O(V+E)
        bridges = set(nx.bridges(G))

        # Build reverse mapping: edge -> street_id
        edge_to_street = {}
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                edge_to_street[(min(u, v), max(u, v))] = idx

        # Mark which streets are bridges
        bridge_streets = {}
        for u, v in bridges:
            edge_key = (min(u, v), max(u, v))
            if edge_key in edge_to_street:
                street_id = edge_to_street[edge_key]
                bridge_streets[street_id] = True

        return bridge_streets

    def _get_geometry_type(self):
        """Get the geometry type for streets (for import safety)."""
        from shapely.geometry import LineString
        return LineString

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy for debugging and monitoring."""
        info = super().get_strategy_info()
        info.update({
            'fractal_min_similarity': self.config.strategies.fractal_pattern_min_similarity,
            'description': 'Analyzes fractal dimension and self-similar patterns in street networks'
        })
        return info
