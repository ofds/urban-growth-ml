#!/usr/bin/env python3
"""
Action Factory Module

Centralizes action creation and validation logic for urban growth inference.
Provides type-safe action construction with comprehensive validation.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from shapely.geometry import LineString
from shapely import wkt

from ..data_structures import InverseGrowthAction, ActionType, compute_frontier_signature
from ..core.config import InferenceConfig
from src.core.contracts import GrowthState

logger = logging.getLogger(__name__)


class ActionFactory:
    """
    Factory for creating and validating InverseGrowthAction instances.

    Centralizes action creation logic with proper validation and state diff computation.
    Ensures all actions meet the contract invariants and contain complete metadata.
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the action factory.

        Args:
            config: Configuration instance (uses default if None)
        """
        self.config = config or InferenceConfig()

    def create_action_from_frontier(self, frontier, strategy_name: str, confidence: float,
                                  state: GrowthState, intent_params: Optional[Dict[str, Any]] = None) -> Optional[InverseGrowthAction]:
        """
        Create an action from a frontier with comprehensive validation.

        Args:
            frontier: Frontier object to create action from
            strategy_name: Name of the strategy creating this action
            confidence: Confidence score for the action
            state: Current growth state
            intent_params: Additional intent parameters

        Returns:
            Validated InverseGrowthAction or None if creation fails
        """
        try:
            # Validate inputs
            if not self._validate_frontier_input(frontier, confidence, state):
                return None

            # Resolve frontier to current graph edge
            current_edge_id = self._resolve_frontier_to_current_edge(frontier, state)
            if current_edge_id is None:
                logger.debug("Could not resolve frontier to current graph edge")
                return None

            # Find the corresponding street
            street_id = self._find_street_for_edge(current_edge_id, state)
            if street_id is None:
                logger.debug(f"No street found for edge {current_edge_id}")
                return None

            # Create stable identifiers
            stable_id = self._compute_stable_frontier_id(frontier)
            geometric_signature = compute_frontier_signature(frontier)

            # Prepare intent parameters
            final_intent_params = self._prepare_intent_params(
                strategy_name, current_edge_id, stable_id, intent_params
            )

            # Create state diff
            state_diff = self._create_state_diff_for_frontier(frontier, current_edge_id, street_id, state)

            # Create the action
            action = InverseGrowthAction(
                action_type=ActionType.REMOVE_STREET,
                street_id=str(street_id),
                intent_params=final_intent_params,
                confidence=confidence,
                timestamp=len(state.streets),
                state_diff=state_diff,
                action_metadata={
                    'geometric_signature': geometric_signature,
                    'strategy_name': strategy_name,
                    'creation_method': 'frontier_based'
                }
            )

            # Final validation
            if self.validate_action(action, state):
                return action
            else:
                logger.debug("Action failed final validation")
                return None

        except Exception as e:
            logger.warning(f"Error creating action from frontier: {e}")
            return None

    def create_action_from_street(self, street_id, street, strategy_name: str, confidence: float,
                                state: GrowthState, intent_params: Optional[Dict[str, Any]] = None) -> Optional[InverseGrowthAction]:
        """
        Create an action directly from street data.

        Args:
            street_id: ID of the street to remove
            street: Street data (DataFrame row)
            strategy_name: Name of the strategy creating this action
            confidence: Confidence score for the action
            state: Current growth state
            intent_params: Additional intent parameters

        Returns:
            Validated InverseGrowthAction or None if creation fails
        """
        try:
            # Validate inputs
            if not self._validate_street_input(street_id, street, confidence, state):
                return None

            # Get edge information
            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                logger.debug(f"Street {street_id} missing edge information")
                return None

            # Validate edge exists in graph
            if not self._validate_edge_exists((u, v), state):
                logger.debug(f"Street {street_id} edge ({u}, {v}) does not exist in graph")
                return None

            # Create stable identifiers
            stable_id = self._compute_stable_frontier_id(type('MockFrontier', (), {'geometry': street.geometry})())
            geometric_signature = compute_frontier_signature(type('MockFrontier', (), {'geometry': street.geometry})())

            # Prepare intent parameters
            edge_id = (min(u, v), max(u, v))
            final_intent_params = self._prepare_intent_params(
                strategy_name, edge_id, stable_id, intent_params
            )

            # Create state diff
            state_diff = self._create_state_diff_for_street(street, street_id)

            # Create the action
            action = InverseGrowthAction(
                action_type=ActionType.REMOVE_STREET,
                street_id=str(street_id),
                intent_params=final_intent_params,
                confidence=confidence,
                timestamp=len(state.streets),
                state_diff=state_diff,
                action_metadata={
                    'geometric_signature': geometric_signature,
                    'strategy_name': strategy_name,
                    'creation_method': 'street_based'
                }
            )

            # Final validation
            if self.validate_action(action, state):
                return action
            else:
                logger.debug("Action failed final validation")
                return None

        except Exception as e:
            logger.warning(f"Error creating action from street {street_id}: {e}")
            return None

    def validate_action(self, action: InverseGrowthAction, state: GrowthState) -> bool:
        """
        Comprehensive validation of an action against the current state.

        Args:
            action: Action to validate
            state: Current growth state

        Returns:
            True if action is valid, False otherwise
        """
        try:
            # Basic contract validation
            if action.action_type != ActionType.REMOVE_STREET:
                logger.debug(f"Invalid action type: {action.action_type}")
                return False

            if not action.street_id:
                logger.debug("Empty street_id")
                return False

            if not (0.0 <= action.confidence <= 1.0):
                logger.debug(f"Invalid confidence: {action.confidence}")
                return False

            # State-specific validation
            if action.street_id not in state.streets.index:
                logger.debug(f"Street {action.street_id} not found in state")
                return False

            # Validate intent parameters
            if not self._validate_intent_params(action.intent_params):
                return False

            # Validate state diff
            if not self._validate_state_diff(action.state_diff, state):
                return False

            return True

        except Exception as e:
            logger.warning(f"Error validating action: {e}")
            return False

    def _validate_frontier_input(self, frontier, confidence: float, state: GrowthState) -> bool:
        """Validate frontier-based action inputs."""
        if not hasattr(frontier, 'geometry') or frontier.geometry is None:
            return False

        if not isinstance(frontier.geometry, LineString):
            return False

        if not (0.0 <= confidence <= 1.0):
            return False

        return True

    def _validate_street_input(self, street_id, street, confidence: float, state: GrowthState) -> bool:
        """Validate street-based action inputs."""
        if street_id not in state.streets.index:
            return False

        if not hasattr(street, 'geometry') or street.geometry is None:
            return False

        if not isinstance(street.geometry, LineString):
            return False

        if not (0.0 <= confidence <= 1.0):
            return False

        return True

    def _resolve_frontier_to_current_edge(self, frontier, state: GrowthState) -> Optional[Tuple[int, int]]:
        """
        Resolve frontier geometry to current graph edge using morphological similarity.

        Uses geometric similarity (length, angle, position) rather than exact matching.
        """
        if not isinstance(frontier.geometry, LineString):
            return None

        # Calculate frontier properties
        frontier_length = frontier.geometry.length
        frontier_centroid = frontier.geometry.centroid
        frontier_angle = self._get_street_angle(frontier.geometry)

        best_score = 0
        best_edge = None

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
                frontier.geometry, frontier_length, frontier_centroid, frontier_angle,
                street.geometry
            )

            if similarity_score > best_score:
                best_score = similarity_score
                best_edge = (min(u, v), max(u, v))

        # Return the best matching edge if similarity is sufficient
        if best_edge is not None and best_score > self.config.spatial.geometry_similarity_threshold:
            logger.debug(f"Frontier resolved to edge {best_edge} (similarity {best_score:.3f})")
            return best_edge

        logger.debug(f"Frontier resolution failed - best similarity {best_score:.3f}")
        return None

    def _find_street_for_edge(self, edge_id: Tuple[int, int], state: GrowthState) -> Optional[str]:
        """Find the street ID that corresponds to the given edge."""
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                street_edge = (min(u, v), max(u, v))
                if street_edge == edge_id:
                    return str(idx)
        return None

    def _validate_edge_exists(self, edge: Tuple[int, int], state: GrowthState) -> bool:
        """Check if an edge exists in the current graph."""
        u, v = edge
        return state.graph.has_edge(u, v) or state.graph.has_edge(v, u)

    def _prepare_intent_params(self, strategy_name: str, edge_id: Tuple[int, int],
                             stable_id: str, additional_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare intent parameters with required fields."""
        params = {
            'strategy': strategy_name,
            'edge_u': str(edge_id[0]),
            'edge_v': str(edge_id[1]),
            'stable_id': stable_id
        }

        if additional_params:
            params.update(additional_params)

        return params

    def _create_state_diff_for_frontier(self, frontier, current_edge_id: Tuple[int, int],
                                       street_id: str, state: GrowthState) -> Dict[str, Any]:
        """Create state diff for a frontier-based action."""
        # Get street data
        street = state.streets.loc[street_id]

        return {
            'geometry_wkt': wkt.dumps(frontier.geometry),
            'edgeid': current_edge_id,
            'frontier_type': getattr(frontier, 'frontier_type', 'unknown'),
            'stable_id': self._compute_stable_frontier_id(frontier),
            'added_streets': [{
                'edge_id': current_edge_id,
                'u': street.get('u'),
                'v': street.get('v'),
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }],
            'removed_streets': [str(street_id)]
        }

    def _create_state_diff_for_street(self, street, street_id) -> Dict[str, Any]:
        """Create state diff for a street-based action."""
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

    def _validate_intent_params(self, intent_params: Dict[str, Any]) -> bool:
        """Validate intent parameters contain required fields."""
        required_fields = ['strategy', 'edge_u', 'edge_v', 'stable_id']
        return all(field in intent_params for field in required_fields)

    def _validate_state_diff(self, state_diff: Optional[Dict[str, Any]], state: GrowthState) -> bool:
        """Validate state diff structure and content."""
        if state_diff is None:
            return False

        # Check required fields
        required_fields = ['added_streets', 'removed_streets', 'geometry_wkt', 'edgeid']
        if not all(field in state_diff for field in required_fields):
            return False

        # Validate removed streets exist
        removed_streets = state_diff.get('removed_streets', [])
        if not removed_streets or not all(street_id in state.streets.index for street_id in removed_streets):
            return False

        # Validate added streets structure
        added_streets = state_diff.get('added_streets', [])
        if not added_streets:
            return False

        for street_data in added_streets:
            required_street_fields = ['edge_id', 'u', 'v', 'geometry_wkt']
            if not all(field in street_data for field in required_street_fields):
                return False

        return True

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

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        import math
        angle = math.degrees(math.atan2(dy, dx))
        return angle % 180
