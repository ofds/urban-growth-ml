#!/usr/bin/env python3
"""
Base Strategy Module for Urban Growth Inference

Provides the abstract base class for all inference strategies,
following SOLID principles with proper separation of concerns.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import LineString

from ..core.config import InferenceConfig
from ..spatial.spatial_index import SpatialIndex
from ..data_structures import InverseGrowthAction, ActionType
from src.core.contracts import GrowthState

logger = logging.getLogger(__name__)


class BaseInferenceStrategy(ABC):
    """
    Abstract base class for all inference strategies.

    Defines the common interface and provides shared functionality
    for strategies that generate candidate actions for urban growth inference.

    This follows the Strategy pattern and Interface Segregation Principle (ISP)
    by providing a focused interface for inference strategies.
    """

    def __init__(self, name: str, weight: float = 1.0, config: Optional[InferenceConfig] = None):
        """
        Initialize the strategy.

        Args:
            name: Unique name identifier for the strategy
            weight: Relative importance weight for candidate ranking
            config: Configuration instance (uses default if None)
        """
        self.name = name
        self.weight = weight
        self.config = config or InferenceConfig()

        # Validate configuration compatibility
        self._validate_config()

        logger.debug(f"Initialized strategy '{self.name}' with weight {self.weight}")

    def _validate_config(self):
        """Validate that configuration is compatible with this strategy."""
        # Check if strategy is enabled in config
        if not self.config.is_strategy_enabled(self.name):
            logger.warning(f"Strategy '{self.name}' is disabled in configuration")

        # Validate weight is reasonable
        if self.weight < 0:
            raise ValueError(f"Strategy weight must be non-negative, got {self.weight}")

        # Strategy-specific validation can be overridden by subclasses
        self._validate_strategy_config()

    def _validate_strategy_config(self):
        """Strategy-specific configuration validation. Override in subclasses."""
        pass

    @abstractmethod
    def generate_candidates(self, state: GrowthState, skeleton_edges: set,
                          spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """
        Generate candidate actions with confidence scores.

        This is the main strategy interface that all concrete strategies must implement.

        Args:
            state: Current growth state
            skeleton_edges: Set of edges that form the arterial skeleton (should not be removed)
            spatial_index: Optional spatial index for efficient geometric queries

        Returns:
            List of (action, confidence) tuples, where confidence is in [0, 1]
        """
        raise NotImplementedError("Subclasses must implement generate_candidates")

    def _create_action_from_frontier(self, frontier, confidence: float,
                                   state: GrowthState, intent_params: Optional[Dict[str, Any]] = None) -> Optional[InverseGrowthAction]:
        """
        Create an action from a frontier with freshness validation.

        This is a common helper method used by most strategies.

        Args:
            frontier: Frontier object to create action from
            confidence: Confidence score for the action
            state: Current growth state
            intent_params: Additional intent parameters

        Returns:
            InverseGrowthAction or None if creation fails
        """
        # FRONTIER FRESHNESS GATE: Validate that frontier's edge exists in current graph
        if hasattr(frontier, 'edge_id') and frontier.edge_id:
            if not self._validate_frontier_freshness(frontier.edge_id, state):
                logger.debug(f"Frontier freshness validation failed for edge {frontier.edge_id} - skipping")
                return None

        # GEOMETRY-BASED RESOLUTION: Resolve frontier geometry to current graph edge
        current_edge_id = self._resolve_frontier_to_current_edge(frontier.geometry, state)

        # If we can't resolve to a current edge, skip this frontier
        if current_edge_id is None:
            logger.debug("Could not resolve frontier geometry to current graph edge - skipping")
            return None

        # Create stable ID and geometric signature
        stable_id = self._compute_stable_frontier_id(frontier)
        geometric_signature = self._compute_geometric_signature(frontier)

        # Prepare intent parameters
        if intent_params is None:
            intent_params = {}

        intent_params.update({
            'strategy': self.name,
            'edge_u': current_edge_id[0],
            'edge_v': current_edge_id[1],
            'stable_id': stable_id
        })

        # Create state diff for the street that will be removed
        state_diff = self._create_state_diff_for_frontier(frontier, current_edge_id, state)

        # Find the street that corresponds to the resolved edge
        street_id = self._find_street_for_edge(current_edge_id, state)
        if street_id is None:
            logger.debug(f"No street found for resolved edge {current_edge_id} - skipping")
            return None

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=str(street_id),
            intent_params=intent_params,
            confidence=confidence,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={
                'geometric_signature': geometric_signature,
                'strategy_weight': self.weight
            }
        )

    def _create_action_from_street(self, street_id, street, confidence: float,
                                 state: GrowthState, intent_params: Optional[Dict[str, Any]] = None) -> Optional[InverseGrowthAction]:
        """
        Create an action directly from street data.

        Alternative to frontier-based action creation for strategies that work directly with streets.

        Args:
            street_id: ID of the street to remove
            street: Street data (DataFrame row)
            confidence: Confidence score
            state: Current growth state
            intent_params: Additional intent parameters

        Returns:
            InverseGrowthAction or None if creation fails
        """
        # Get edge information
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            logger.debug(f"Street {street_id} missing edge information (u={u}, v={v})")
            return None

        # CRITICAL VALIDATION: Ensure the street's edge exists in the current graph
        edge_exists = state.graph.has_edge(u, v) or state.graph.has_edge(v, u)
        if not edge_exists:
            logger.debug(f"Street {street_id} edge ({u}, {v}) does not exist in current graph - skipping")
            return None

        # Create stable ID from street geometry
        stable_id = self._compute_stable_frontier_id(type('MockFrontier', (), {'geometry': street.geometry})())

        # Prepare intent parameters
        if intent_params is None:
            intent_params = {}

        intent_params.update({
            'strategy': self.name,
            'edge_u': str(u),
            'edge_v': str(v),
            'stable_id': stable_id
        })

        # Create state diff with street data
        state_diff = self._create_state_diff_for_street(street, street_id)

        # Compute geometric signature
        geometric_signature = self._compute_geometric_signature(type('MockFrontier', (), {'geometry': street.geometry})())

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=str(street_id),
            intent_params=intent_params,
            confidence=confidence,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={
                'geometric_signature': geometric_signature,
                'strategy_weight': self.weight
            }
        )

    def _validate_frontier_freshness(self, edge_id: Tuple[int, int], state: GrowthState) -> bool:
        """
        Validate that a frontier's edge ID is still valid in current state.

        This is the freshness gate that prevents stale references.
        """
        if edge_id is None:
            return False
        u, v = edge_id
        return state.graph.has_edge(u, v) or state.graph.has_edge(v, u)

    def _resolve_frontier_to_current_edge(self, frontier_geometry, state: GrowthState) -> Optional[Tuple[int, int]]:
        """
        Resolve frontier geometry to current graph edge using morphological similarity.

        Uses geometric similarity (length, angle, position) rather than exact matching.
        Returns (u, v) tuple of the most similar valid street.
        """
        if not isinstance(frontier_geometry, LineString):
            return None

        logger.debug(f"Resolving frontier geometry: {frontier_geometry.wkt[:100]}...")

        # Calculate frontier properties
        frontier_length = frontier_geometry.length
        frontier_centroid = frontier_geometry.centroid
        frontier_angle = self._get_street_angle(frontier_geometry)

        best_score = 0
        best_street_id = None

        # Find most morphologically similar street
        for idx, street in state.streets.iterrows():
            if not isinstance(street.geometry, LineString):
                continue

            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue

            # Verify this edge exists in current graph
            if not (state.graph.has_edge(u, v) or state.graph.has_edge(v, u)):
                continue

            # Calculate morphological similarity score
            similarity_score = self._calculate_morphological_similarity(
                frontier_geometry, frontier_length, frontier_centroid, frontier_angle,
                street.geometry
            )

            if similarity_score > best_score:
                best_score = similarity_score
                best_street_id = idx

        # Return the best matching street's edge
        if best_street_id is not None and best_score > self.config.spatial.geometry_similarity_threshold:
            street = state.streets.loc[best_street_id]
            u, v = street.get('u'), street.get('v')
            actual_edge = (min(u, v), max(u, v))
            logger.debug(f"Frontier resolved to street {best_street_id} with edge {actual_edge} (similarity {best_score:.3f})")
            return actual_edge

        logger.debug(f"Frontier resolution failed - best similarity {best_score:.3f} (threshold {self.config.spatial.geometry_similarity_threshold})")
        return None

    def _calculate_morphological_similarity(self, frontier_geom, frontier_length, frontier_centroid, frontier_angle, street_geom):
        """Calculate morphological similarity between frontier and street geometries."""
        # Length similarity (0-1 scale)
        street_length = street_geom.length
        length_diff = abs(frontier_length - street_length)
        max_length = max(frontier_length, street_length)
        length_score = 1.0 - (length_diff / max_length) if max_length > 0 else 1.0

        # Position similarity (distance-based)
        street_centroid = street_geom.centroid
        distance = frontier_centroid.distance(street_centroid)
        position_score = max(0, 1.0 - distance / 50.0)  # 50m radius

        # Angle similarity (0-1 scale)
        street_angle = self._get_street_angle(street_geom)
        angle_diff = abs(frontier_angle - street_angle) % 180
        min_angle_diff = min(angle_diff, 180 - angle_diff)
        angle_score = max(0, 1.0 - min_angle_diff / self.config.spatial.angle_tolerance_degrees)

        # Combined score (weighted average)
        return 0.4 * length_score + 0.4 * position_score + 0.2 * angle_score

    def _find_street_for_edge(self, edge_id: Tuple[int, int], state: GrowthState) -> Optional[str]:
        """Find the street ID that corresponds to the given edge."""
        # Normalize the input edge_id
        normalized_edge = (min(edge_id[0], edge_id[1]), max(edge_id[0], edge_id[1]))

        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                street_edge = (min(u, v), max(u, v))
                if street_edge == normalized_edge:
                    return str(idx)
        return None

    def _create_state_diff_for_frontier(self, frontier, current_edge_id: Tuple[int, int], state: GrowthState) -> Dict[str, Any]:
        """Create state diff for a frontier-based action."""
        from shapely import wkt

        state_diff = {
            'geometry_wkt': wkt.dumps(frontier.geometry),
            'edgeid': current_edge_id,
            'frontier_type': getattr(frontier, 'frontier_type', 'unknown'),
            'stable_id': self._compute_stable_frontier_id(frontier)
        }

        # Find and add the street data
        street_id = self._find_street_for_edge(current_edge_id, state)
        if street_id:
            street = state.streets.loc[street_id]
            state_diff['added_streets'] = [{
                'edge_id': current_edge_id,
                'u': street.get('u'),
                'v': street.get('v'),
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }]
            state_diff['removed_streets'] = [street_id]

        return state_diff

    def _create_state_diff_for_street(self, street, street_id) -> Dict[str, Any]:
        """Create state diff for a street-based action."""
        from shapely import wkt

        u, v = street.get('u'), street.get('v')
        edge_id = (min(u, v), max(u, v)) if u is not None and v is not None else (None, None)

        return {
            'geometry_wkt': wkt.dumps(street.geometry),
            'edgeid': edge_id,
            'frontier_type': 'street_removal',
            'stable_id': self._compute_stable_frontier_id(type('MockFrontier', (), {'geometry': street.geometry})()),
            'added_streets': [{
                'edge_id': edge_id,
                'u': u,
                'v': v,
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }],
            'removed_streets': [str(street_id)]
        }

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry coordinates."""
        import hashlib
        if not hasattr(frontier, 'geometry') or frontier.geometry is None:
            return "invalid_frontier"

        geom = frontier.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            start = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
            end = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
            frontier_type = getattr(frontier, 'frontier_type', 'unknown')

            hash_input = f"{start}_{end}_{frontier_type}".encode('utf-8')
            return hashlib.md5(hash_input).hexdigest()[:16]
        return "invalid_geometry"

    def _compute_geometric_signature(self, frontier) -> str:
        """Compute geometric signature for frontier matching."""
        # Import here to avoid circular imports
        from ..data_structures import compute_frontier_signature
        return compute_frontier_signature(frontier)

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180

    def _is_skeleton_edge(self, edge: Tuple[int, int], skeleton_edges: set) -> bool:
        """Check if an edge is part of the arterial skeleton."""
        return edge in skeleton_edges or (edge[1], edge[0]) in skeleton_edges

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy for debugging and monitoring."""
        return {
            'name': self.name,
            'weight': self.weight,
            'type': self.__class__.__name__,
            'config_valid': True,  # Could be extended to check config compatibility
            'description': self.__doc__.strip().split('\n')[0] if self.__doc__ else "No description"
        }


# Import numpy here to avoid circular import issues
import numpy as np
