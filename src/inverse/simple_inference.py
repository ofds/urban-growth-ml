#!/usr/bin/env python3
"""
Simple Priority Queue Inference Engine
Core insight: Recent streets are short, peripheral, and terminate in dead-ends.
"""

import logging
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from shapely.geometry import LineString, Point

from src.core.contracts import GrowthState
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType
from .skeleton import ArterialSkeletonExtractor
from .rewind import RewindEngine

logger = logging.getLogger(__name__)


class SimpleInferenceEngine:
    """
    Priority queue inference using geometric likelihood scoring.

    Core hypothesis: Recent streets score high on composite metric of:
    - Length (shorter = more recent) - 40% weight
    - Distance from center (farther = more recent) - 30% weight
    - Dead-end status (dead-end = most recent) - 30% weight
    """

    def __init__(self):
        self.skeleton_extractor = ArterialSkeletonExtractor()
        self.rewind_engine = RewindEngine()

    def infer_trace(self, final_state: GrowthState, max_steps: int = 10000000,
                    initial_state: Optional[GrowthState] = None,
                    progress_callback: Optional[callable] = None) -> GrowthTrace:
        """
        Infer growth trace using simple priority queue scoring.
        """
        logger.info("Starting simple priority queue inference...")

        # Extract skeleton for center calculation and exclusion
        if initial_state is None:
            skeleton_edges, skeleton_streets = self.skeleton_extractor.extract_skeleton(
                final_state.streets, final_state.graph
            )
            initial_state = self.skeleton_extractor.create_skeleton_state(skeleton_streets, final_state)
            skeleton_edge_set = set(skeleton_edges)
        else:
            logger.info(f"Using provided initial state with {len(initial_state.streets)} streets")
            skeleton_edge_set = set()
            for idx, street in initial_state.streets.iterrows():
                u, v = street.get('u'), street.get('v')
                if u and v:
                    skeleton_edge_set.add((min(u, v), max(u, v)))

        # Calculate city center from skeleton
        city_center = self._calculate_city_center(initial_state.streets)

        current_state = final_state
        actions = []
        step = 0

        logger.info(f"Inference setup: {len(final_state.streets)} -> {len(initial_state.streets)} streets")

        while step < max_steps:
            if len(current_state.streets) <= len(initial_state.streets):
                logger.info(f"Reached initial state at step {step}")
                break

            # Score all candidate streets
            candidates = []
            for street_id, street in current_state.streets.iterrows():
                # Skip skeleton streets
                u, v = street.get('u'), street.get('v')
                if u and v and (min(u, v), max(u, v)) in skeleton_edge_set:
                    continue

                if not isinstance(street.geometry, LineString):
                    continue

                # Calculate composite score
                score = self._calculate_removal_score(street, city_center, current_state.graph)
                candidates.append((score, street_id, street))

            if not candidates:
                logger.info(f"No more candidates at step {step}")
                break

            # Select highest scoring street
            candidates.sort(reverse=True)  # Highest score first
            best_score, best_street_id, best_street = candidates[0]

            logger.debug(f"Step {step}: Selected street {best_street_id} with score {best_score:.3f}")

            # Create action
            action = self._create_action_from_street(best_street_id, best_street, current_state)
            if action is None:
                logger.warning(f"Failed to create action for street {best_street_id}")
                break

            # Compute state diff
            state_diff = self._compute_state_diff(current_state, action)

            # Rewind
            prev_state = self.rewind_engine.rewind_action(action, current_state)

            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed at step {step}")
                break

            # Record action
            action_with_diff = self._add_state_diff_to_action(action, state_diff)
            actions.insert(0, action_with_diff)

            if step % 50 == 0 or step < 5:
                logger.info(f"Step {step}: streets {len(current_state.streets)} -> {len(prev_state.streets)}")

            current_state = prev_state
            step += 1

        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                "inference_method": "simple_priority_queue",
                "max_steps": max_steps,
                "steps_taken": step,
                "skeleton_streets": len(skeleton_edge_set)
            }
        )

        logger.info(f"Simple inference complete: {len(actions)} actions")
        return trace

    def _calculate_city_center(self, streets_gdf) -> Point:
        """Calculate city center from street geometries."""
        if streets_gdf.empty:
            return Point(0, 0)

        # Use centroid of all street geometries
        all_geoms = streets_gdf.geometry.unary_union
        return all_geoms.centroid

    def _calculate_removal_score(self, street, city_center: Point, graph) -> float:
        """
        Calculate composite score for street removal likelihood.

        Higher score = more likely to be recent addition.
        """
        # Length score (shorter = higher score)
        length = street.geometry.length
        length_score = 1.0 / (1.0 + length / 100.0)  # Normalize to 0-1

        # Distance score (farther from center = higher score)
        centroid = street.geometry.centroid
        distance = city_center.distance(centroid)
        distance_score = min(1.0, distance / 1000.0)  # Normalize, cap at 1km

        # Dead-end score (dead-end = 1.0, connected = 0.0)
        u, v = street.get('u'), street.get('v')
        dead_end_score = 0.0
        if u and v:
            u_degree = graph.degree[u] if graph.has_node(u) else 0
            v_degree = graph.degree[v] if graph.has_node(v) else 0
            # Street is dead-end if either endpoint has degree 1
            if u_degree == 1 or v_degree == 1:
                dead_end_score = 1.0

        # Composite score with weights
        composite_score = (
            0.4 * length_score +
            0.3 * distance_score +
            0.3 * dead_end_score
        )

        return composite_score

    def _create_action_from_street(self, street_id, street, state: GrowthState) -> Optional[InverseGrowthAction]:
        """Create action from street data."""
        from shapely import wkt

        # Get edge information
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return None

        # Ensure street_id is string
        street_id_str = str(street_id)

        # Create state diff
        state_diff = {
            'geometry_wkt': wkt.dumps(street.geometry),
            'edgeid': (min(u, v), max(u, v)),
            'frontier_type': 'street_removal',
            'added_streets': [{
                'edge_id': (min(u, v), max(u, v)),
                'u': u,
                'v': v,
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }],
            'removed_streets': [street_id_str]
        }

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=street_id_str,
            intent_params={
                'strategy': 'simple_priority_queue',
                'edge_u': str(u),
                'edge_v': str(v)
            },
            confidence=0.8,  # Fixed confidence for simple approach
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={},
            realized_geometry={
                'geometry_wkt': wkt.dumps(street.geometry),
                'edgeid': (min(u, v), max(u, v)),
                'frontier_type': 'street_removal'
            }
        )

    def _compute_state_diff(self, current_state: GrowthState, action: InverseGrowthAction) -> Dict[str, Any]:
        """Compute state diff for action."""
        state_diff = {
            'added_streets': [],
            'removed_streets': [],
            'graph_changes': {},
            'frontier_changes': {}
        }

        if action.action_type == ActionType.REMOVE_STREET:
            street_id = action.street_id
            if street_id in current_state.streets.index:
                street = current_state.streets.loc[street_id]
                street_data = {
                    'edge_id': (min(street.get('u'), street.get('v')), max(street.get('u'), street.get('v'))),
                    'u': street.get('u'),
                    'v': street.get('v'),
                    'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                    'osmid': street.get('osmid'),
                    'highway': street.get('highway'),
                    'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                }
                state_diff['added_streets'].append(street_data)
                state_diff['removed_streets'].append(street_id)

        state_diff['graph_changes'] = {
            'nodes_before': current_state.graph.number_of_nodes(),
            'edges_before': current_state.graph.number_of_edges(),
            'nodes_after': None,
            'edges_after': None
        }

        state_diff['frontier_changes'] = {
            'frontiers_before': len(current_state.frontiers),
            'frontiers_after': None
        }

        return state_diff

    def _add_state_diff_to_action(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
        """Add state diff to action."""
        return InverseGrowthAction(
            action_type=action.action_type,
            street_id=action.street_id,
            intent_params=action.intent_params,
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=state_diff,
            action_metadata=action.action_metadata
        )
