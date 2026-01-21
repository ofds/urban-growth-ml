#!/usr/bin/env python3
"""
Peripheral Strategy Module for Urban Growth Inference

Implements the peripheral expansion strategy that identifies and removes
the most distant streets from the city center, following the simple
"peeling" heuristic from the original inference engine.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

from .base_strategy import BaseInferenceStrategy
from ..spatial.spatial_index import SpatialIndex
from ..data_structures import InverseGrowthAction
from src.core.contracts import GrowthState

logger = logging.getLogger(__name__)


class PeripheralStrategy(BaseInferenceStrategy):
    """
    Peripheral Expansion Strategy for Urban Growth Inference.

    This strategy implements the simple "peeling" heuristic that identifies
    streets most distant from the city center and removes them as likely
    candidates for recent growth actions.

    The strategy focuses on dead-end frontiers (cul-de-sacs and street ends)
    and selects the one farthest from the city center for removal.
    """

    def __init__(self, weight: float = 0.8, config=None):
        """
        Initialize the peripheral strategy.

        Args:
            weight: Strategy weight for candidate ranking (default 0.8)
            config: Configuration instance
        """
        super().__init__("peripheral", weight, config)

    def _validate_strategy_config(self):
        """Validate peripheral-specific configuration."""
        # Check spatial query radius is reasonable
        if self.config.spatial.spatial_query_radius <= 0:
            raise ValueError("Spatial query radius must be positive")

    def generate_candidates(self, state: GrowthState, skeleton_edges: set,
                          spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """
        Generate candidate actions using peripheral expansion heuristic.

        Focuses on dead-end frontiers farthest from city center.

        Args:
            state: Current growth state
            skeleton_edges: Arterial skeleton edges (not to be removed)
            spatial_index: Optional spatial index for efficiency

        Returns:
            List of (action, confidence) tuples
        """
        candidates = []

        # Calculate city center
        city_center = self._get_city_center(state)

        # Find dead-end frontiers (most likely recent additions)
        dead_end_frontiers = [
            f for f in state.frontiers
            if getattr(f, 'frontier_type', None) == "dead_end"
        ]

        if not dead_end_frontiers:
            logger.debug("PeripheralStrategy: No dead-end frontiers found")
            return candidates

        # Score frontiers by distance from center
        scored_frontiers = []
        for frontier in dead_end_frontiers:
            # Skip if this frontier corresponds to a skeleton edge
            if hasattr(frontier, 'edge_id') and frontier.edge_id:
                if self._is_skeleton_edge(frontier.edge_id, skeleton_edges):
                    continue

            # Calculate distance from city center
            distance = self._distance_from_center(frontier.geometry, city_center)

            # Calculate confidence based on distance (farther = more confident)
            confidence = min(0.8, distance / 1000.0)  # Normalize by 1km, cap at 0.8

            scored_frontiers.append((frontier, confidence))

        # Sort by confidence (highest first)
        scored_frontiers.sort(key=lambda x: x[1], reverse=True)

        # Create actions for top candidates
        max_candidates = min(len(scored_frontiers), self.config.limits.max_candidates)
        for frontier, confidence in scored_frontiers[:max_candidates]:
            # Create action with peripheral expansion intent
            intent_params = {
                'direction': 'peripheral_expansion',
                'distance_from_center': self._distance_from_center(frontier.geometry, city_center),
                'frontier_type': getattr(frontier, 'frontier_type', 'unknown')
            }

            action = self._create_action_from_frontier(
                frontier, confidence, state, intent_params
            )

            if action is not None:
                candidates.append((action, confidence))
            else:
                logger.debug(f"PeripheralStrategy: Failed to create action for frontier")

        logger.debug(f"PeripheralStrategy: Generated {len(candidates)} candidates from {len(dead_end_frontiers)} dead-end frontiers")
        return candidates

    def _get_city_center(self, state: GrowthState):
        """
        Calculate the city center from street geometries.

        Uses the centroid of all street endpoints as a simple approximation.
        """
        if state.city_bounds:
            return state.city_bounds.centroid

        # Fallback: calculate center from street coordinates
        all_coords = []
        for idx, street in state.streets.iterrows():
            if hasattr(street.geometry, 'coords'):
                all_coords.extend(street.geometry.coords)

        if all_coords:
            x_coords = [c[0] for c in all_coords]
            y_coords = [c[1] for c in all_coords]
            return type('Point', (), {'x': sum(x_coords)/len(x_coords), 'y': sum(y_coords)/len(y_coords)})()

        # Ultimate fallback
        return type('Point', (), {'x': 0, 'y': 0})()

    def _distance_from_center(self, geometry, center) -> float:
        """
        Calculate distance from geometry to city center.

        Args:
            geometry: Shapely geometry object
            center: Point-like object with x, y attributes

        Returns:
            Distance in coordinate units
        """
        if hasattr(geometry, 'centroid'):
            geom_center = geometry.centroid
            return ((geom_center.x - center.x)**2 + (geom_center.y - center.y)**2)**0.5
        return 0.0
