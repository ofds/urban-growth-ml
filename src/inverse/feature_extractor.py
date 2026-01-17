#!/usr/bin/env python3
"""
State Feature Extractor
Phase 2: Extract fixed-size feature vectors from GrowthState for ML training.
"""

from typing import List, Dict, Any
import numpy as np
import networkx as nx
import logging
from shapely.geometry import LineString

from ..core.contracts import GrowthState

logger = logging.getLogger(__name__)

class StateFeatureExtractor:
    """Extract fixed-size feature vectors from GrowthState."""

    FEATURE_DIM = 128  # FROZEN - do not change after dataset creation

    def __init__(self, use_per_city_normalization: bool = True):
        """
        Args:
            use_per_city_normalization: Enable per-city z-score normalization
        """
        self.use_per_city_normalization = use_per_city_normalization
        self.city_stats = {}  # Cache for per-city statistics

    def extract_features(self, state: GrowthState, city_name: str = None) -> np.ndarray:
        """
        Extract 128-dim feature vector from state.

        Args:
            state: Growth state to extract features from
            city_name: Optional city identifier for per-city normalization
        """
        features = []

        # Frontier features (32 dims)
        features.extend(self._frontier_features(state))

        # Block features (32 dims)
        features.extend(self._block_features(state))

        # Street network features (32 dims)
        features.extend(self._network_features(state))

        # Global context (32 dims)
        features.extend(self._global_features(state))

        # Validate feature dimension (debug mode can be enabled via environment variable)
        if len(features) != self.FEATURE_DIM:
            logger.error(f"CRITICAL: Feature dimension mismatch! Got {len(features)}, expected {self.FEATURE_DIM}")
            raise ValueError(f"Feature extraction failed: expected {self.FEATURE_DIM} features, got {len(features)}")

        feature_vector = np.array(features, dtype=np.float32)
        
        # OPTIMIZATION: Apply per-city normalization
        if self.use_per_city_normalization and city_name:
            feature_vector = self._normalize_per_city(feature_vector, city_name)
        
        return feature_vector

    def _normalize_per_city(self, features: np.ndarray, city_name: str) -> np.ndarray:
        """
        OPTIMIZATION: Apply z-score normalization per city.
        
        Ensures features have consistent ranges across different city scales.
        """
        if city_name not in self.city_stats:
            # Initialize stats for new city
            self.city_stats[city_name] = {
                'mean': np.zeros(self.FEATURE_DIM, dtype=np.float32),
                'std': np.ones(self.FEATURE_DIM, dtype=np.float32),
                'count': 0
            }
        
        stats = self.city_stats[city_name]
        
        # Update running statistics (Welford's online algorithm)
        stats['count'] += 1
        delta = features - stats['mean']
        stats['mean'] += delta / stats['count']
        
        if stats['count'] > 1:
            stats['std'] = np.sqrt(
                ((stats['count'] - 2) * stats['std']**2 + delta**2) / (stats['count'] - 1)
            )
        
        # Prevent division by zero and NaN propagation
        safe_std = np.where(stats['std'] > 1e-6, stats['std'], 1.0)
        normalized = (features - stats['mean']) / safe_std

        # Replace any remaining NaN/Inf with zeros
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        
        return normalized.astype(np.float32)

    def reset_city_stats(self, city_name: str = None):
        """Reset normalization statistics for a specific city or all cities."""
        if city_name:
            if city_name in self.city_stats:
                del self.city_stats[city_name]
        else:
            self.city_stats.clear()

    def _frontier_features(self, state: GrowthState) -> List[float]:
        """Frontier-specific features (32 dims)."""
        # Angle histogram (16 bins)
        angles = [self._compute_frontier_angle(f) for f in state.frontiers]
        angle_hist, _ = np.histogram(angles, bins=16, range=(0, 360))

        # Normalize with epsilon to prevent division by zero
        angle_hist_normalized = angle_hist / (np.sum(angle_hist) + 1e-6)

        # Length statistics (8 dims)
        lengths = [f.geometry.length for f in state.frontiers]
        length_stats = [
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0,
            np.min(lengths) if lengths else 0,
            np.max(lengths) if lengths else 0,
            np.percentile(lengths, 25) if lengths else 0,
            np.percentile(lengths, 50) if lengths else 0,
            np.percentile(lengths, 75) if lengths else 0,
            len(lengths)
        ]

        # Frontier type distribution (8 dims)
        type_counts = np.zeros(8)
        for f in state.frontiers:
            if hasattr(f, 'frontier_type'):
                if f.frontier_type == 'dead_end':
                    type_counts[0] += 1
                elif f.frontier_type == 'boundary':
                    type_counts[1] += 1
                elif f.frontier_type == 'block_edge':
                    type_counts[2] += 1

        return list(angle_hist_normalized) + length_stats + list(type_counts)

    def _block_features(self, state: GrowthState) -> List[float]:
        """Block-specific features (32 dims)."""
        if state.blocks.empty:
            return [0.0] * 32

        # Area statistics (8 dims)
        areas = [geom.area for geom in state.blocks.geometry if geom is not None]
        area_stats = [
            np.mean(areas) if areas else 0,
            np.std(areas) if areas else 0,
            np.min(areas) if areas else 0,
            np.max(areas) if areas else 0,
            np.percentile(areas, 25) if areas else 0,
            np.percentile(areas, 50) if areas else 0,
            np.percentile(areas, 75) if areas else 0,
            len(areas)
        ]

        # Compactness statistics (8 dims)
        compactness = []
        for geom in state.blocks.geometry:
            if geom is not None and geom.area > 0:
                perimeter = geom.length
                compactness.append(4 * np.pi * geom.area / (perimeter ** 2))

        compactness_stats = [
            np.mean(compactness) if compactness else 0,
            np.std(compactness) if compactness else 0,
            np.min(compactness) if compactness else 0,
            np.max(compactness) if compactness else 0,
            np.percentile(compactness, 25) if compactness else 0,
            np.percentile(compactness, 50) if compactness else 0,
            np.percentile(compactness, 75) if compactness else 0,
            len(compactness)
        ]

        # Block adjacency (8 dims) - simplified
        adjacency_stats = [0.0] * 8

        # Block orientation histogram (8 dims)
        orientations = [self._compute_block_orientation(geom) for geom in state.blocks.geometry if geom is not None]
        orient_hist, _ = np.histogram(orientations, bins=8, range=(0, 180))

        return area_stats + compactness_stats + adjacency_stats + list(orient_hist / (np.sum(orient_hist) + 1e-6))

    def _network_features(self, state: GrowthState) -> List[float]:
        """Street network features (32 dims)."""
        if state.streets.empty:
            return [0.0] * 32

        # Street length statistics (8 dims)
        lengths = [geom.length for geom in state.streets.geometry if isinstance(geom, LineString)]
        length_stats = [
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0,
            np.min(lengths) if lengths else 0,
            np.max(lengths) if lengths else 0,
            np.percentile(lengths, 25) if lengths else 0,
            np.percentile(lengths, 50) if lengths else 0,
            np.percentile(lengths, 75) if lengths else 0,
            len(lengths)
        ]

        # Degree distribution (8 dims)
        degrees = [state.graph.degree[node] for node in state.graph.nodes()]
        degree_hist, _ = np.histogram(degrees, bins=8, range=(0, 8))

        # Curvature statistics (8 dims)
        curvatures = [self._compute_street_curvature(geom) for geom in state.streets.geometry
                     if isinstance(geom, LineString)]
        curvature_stats = [
            np.mean(curvatures) if curvatures else 0,
            np.std(curvatures) if curvatures else 0,
            np.min(curvatures) if curvatures else 0,
            np.max(curvatures) if curvatures else 0,
            np.percentile(curvatures, 25) if curvatures else 0,
            np.percentile(curvatures, 50) if curvatures else 0,
            np.percentile(curvatures, 75) if curvatures else 0,
            len(curvatures)
        ]

        # Highway type distribution (8 dims)
        highway_counts = {}
        for _, street in state.streets.iterrows():
            highway = street.get('highway', 'unknown')
            highway_counts[highway] = highway_counts.get(highway, 0) + 1

        highway_features = [highway_counts.get(t, 0) for t in ['primary', 'secondary', 'tertiary', 'residential', 'unknown']] + [0.0] * 3

        return length_stats + list(degree_hist) + curvature_stats + highway_features

    def _global_features_bounds(self, state: GrowthState) -> List[float]:
        """City bounds statistics (4 dims)."""
        if state.city_bounds:
            bounds_area = state.city_bounds.area
            bounds_perimeter = state.city_bounds.length
            bounds_compactness = 4 * np.pi * bounds_area / (bounds_perimeter ** 2) if bounds_perimeter > 0 else 0
            bounds_aspect = state.city_bounds.bounds[2] - state.city_bounds.bounds[0], state.city_bounds.bounds[3] - state.city_bounds.bounds[1]
            bounds_aspect_ratio = max(bounds_aspect) / min(bounds_aspect) if min(bounds_aspect) > 0 else 0
        else:
            bounds_area = bounds_perimeter = bounds_compactness = bounds_aspect_ratio = 0

        return [bounds_area, bounds_perimeter, bounds_compactness, bounds_aspect_ratio]

    def _global_features_iteration(self, state: GrowthState) -> List[float]:
        """Iteration and growth progress (4 dims)."""
        return [
            float(state.iteration),
            float(len(state.streets)),
            float(len(state.blocks)),
            float(len(state.frontiers))
        ]

    def _global_features_connectivity(self, state: GrowthState) -> List[float]:
        """Network topology (8 dims)."""
        if state.graph:
            num_nodes = len(state.graph.nodes())
            num_edges = len(state.graph.edges())
            avg_degree = num_edges * 2 / num_nodes if num_nodes > 0 else 0
            try:
                num_components = nx.number_connected_components(state.graph) if num_nodes > 0 else 0
                avg_clustering = nx.average_clustering(state.graph) if num_nodes > 0 else 0
            except:
                num_components = 0
                avg_clustering = 0.0
            connectivity_features = [
                float(num_nodes),
                float(num_edges),
                float(avg_degree),
                float(num_components),
                float(avg_clustering),
            ] + [0.0] * 3
        else:
            connectivity_features = [0.0] * 8

        return connectivity_features

    def _global_features_spatial(self, state: GrowthState) -> List[float]:
        """Spatial distribution (8 dims)."""
        return [0.0] * 8

    def _global_features_phase(self, state: GrowthState) -> List[float]:
        """Growth phase indicators (8 dims)."""
        return [0.0] * 8

    def _global_features(self, state: GrowthState) -> List[float]:
        """Global context features (32 dims)."""
        bounds_features = self._global_features_bounds(state)
        iteration_features = self._global_features_iteration(state)
        connectivity_features = self._global_features_connectivity(state)
        spatial_features = self._global_features_spatial(state)
        phase_features = self._global_features_phase(state)

        result = bounds_features + iteration_features + connectivity_features + spatial_features + phase_features
        logger.debug(f"Global features: {len(result)} dims")
        return result

    def _compute_frontier_angle(self, frontier) -> float:
        """Compute the angle of a frontier edge."""
        if not hasattr(frontier, 'geometry') or not isinstance(frontier.geometry, LineString):
            return 0.0

        coords = list(frontier.geometry.coords)
        if len(coords) < 2:
            return 0.0

        dx = coords[1][0] - coords[0][0]
        dy = coords[1][1] - coords[0][1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        return (angle % 360 + 360) % 360

    def _compute_block_orientation(self, geom) -> float:
        """Compute the dominant orientation of a block."""
        if not geom or geom.is_empty:
            return 0.0

        mbr = geom.minimum_rotated_rectangle
        coords = list(mbr.exterior.coords)
        if len(coords) < 4:
            return 0.0

        max_length = 0
        max_angle = 0
        for i in range(len(coords) - 1):
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            length = np.sqrt(dx**2 + dy**2)
            if length > max_length:
                max_length = length
                max_angle = np.arctan2(dy, dx) * 180 / np.pi

        return (max_angle % 180 + 180) % 180

    def _compute_street_curvature(self, geom: LineString) -> float:
        """Compute curvature metric for a street."""
        if not isinstance(geom, LineString) or geom.length == 0:
            return 0.0

        coords = list(geom.coords)
        if len(coords) < 3:
            return 0.0

        total_angle = 0.0
        for i in range(1, len(coords) - 1):
            v1 = (coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
            v2 = (coords[i+1][0] - coords[i][0], coords[i+1][1] - coords[i][1])

            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if len1 > 0 and len2 > 0:
                v1 = (v1[0]/len1, v1[1]/len1)
                v2 = (v2[0]/len2, v2[1]/len2)

                dot = v1[0]*v2[0] + v1[1]*v2[1]
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                total_angle += angle

        return total_angle / geom.length
