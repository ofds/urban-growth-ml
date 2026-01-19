#!/usr/bin/env python3
"""
Correctly Simplified Inference Engine
Keeps essential geometric reasoning (frontiers, blocks, morphology) but simplifies scoring.
"""

import logging
from typing import List, Optional, Any, Dict, Tuple
import numpy as np
from shapely.geometry import LineString, Point

from src.core.contracts import GrowthState, FrontierEdge
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType
from .skeleton import ArterialSkeletonExtractor
from .rewind import RewindEngine

logger = logging.getLogger(__name__)


class CorrectlySimplifiedInferenceEngine:
    """
    Simplified inference that keeps geometric reasoning but uses frontier-type scoring.

    Core insight: Different frontier types have different growth likelihoods:
    - dead_end frontiers: Almost always recent growth (high priority)
    - block_edge frontiers: Close incomplete blocks (medium priority)
    - Other frontiers: Less likely (low priority)

    Keeps: Frontiers, blocks, morphological validation, replay capability
    Simplifies: Single scoring function instead of 9 complex strategies
    """

    def __init__(self):
        self.skeleton_extractor = ArterialSkeletonExtractor()
        self.rewind_engine = RewindEngine()

    def infer_trace(self, final_state: GrowthState, max_steps: int = 10000000,
                    initial_state: Optional[GrowthState] = None,
                    progress_callback: Optional[callable] = None) -> GrowthTrace:
        """
        Infer growth trace using frontier-type-based scoring.
        """
        logger.info("Starting correctly simplified inference...")

        # Extract skeleton for exclusion (same as before)
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

            # Score frontiers using simplified frontier-type logic
            frontier_candidates = self._score_frontiers(current_state, skeleton_edge_set, city_center)

            if not frontier_candidates:
                logger.info(f"No more frontier candidates at step {step}")
                break

            # Select highest scoring frontier
            frontier_candidates.sort(key=lambda x: x[0], reverse=True)  # Sort by score (first element)
            best_score, best_frontier = frontier_candidates[0]

            logger.debug(f"Step {step}: Selected frontier {best_frontier.frontier_id} (type: {best_frontier.frontier_type}) with score {best_score:.3f}")

            # Create action from frontier (keeps full geometric reasoning)
            action = self._create_action_from_frontier(best_frontier, current_state)
            if action is None:
                logger.warning(f"Failed to create action for frontier {best_frontier.frontier_id}")
                break

            # Compute state diff (essential for replay)
            state_diff = self._compute_state_diff(current_state, action)

            # Rewind
            prev_state = self.rewind_engine.rewind_action(action, current_state)

            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed at step {step}")
                break

            # Record action with complete state diff
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
                "inference_method": "correctly_simplified_frontier_based",
                "max_steps": max_steps,
                "steps_taken": step,
                "skeleton_streets": len(skeleton_edge_set)
            }
        )

        logger.info(f"Correctly simplified inference complete: {len(actions)} actions")
        return trace

    def _score_frontiers(self, state: GrowthState, skeleton_edges: set, city_center: Point) -> List[Tuple[float, FrontierEdge]]:
        """
        Score frontiers using simplified frontier-type-based logic.

        Instead of 9 complex strategies, use frontier type + simple geometric modifiers.
        """
        candidates = []

        for frontier in state.frontiers:
            # Base score from frontier type (core insight)
            if frontier.frontier_type == "dead_end":
                base_score = 0.8  # Dead-ends almost always recent growth
            elif frontier.frontier_type == "block_edge":
                base_score = 0.7  # Block-closing streets are recent
            elif frontier.frontier_type == "intersection":
                base_score = 0.2  # Less likely to be growth edges
            else:
                base_score = 0.3  # Default

            # Simple geometric modifiers (much simpler than original 9 strategies)
            geometric_modifier = self._calculate_geometric_modifier(frontier, state, city_center)

            # Skip skeleton edges
            if hasattr(frontier, 'edge_id') and frontier.edge_id:
                u, v = frontier.edge_id
                if (min(u, v), max(u, v)) in skeleton_edges:
                    continue

            final_score = base_score * geometric_modifier
            candidates.append((final_score, frontier))

        return candidates

    def _calculate_geometric_modifier(self, frontier: FrontierEdge, state: GrowthState, city_center: Point) -> float:
        """
        Simple geometric modifier combining length and distance.

        Much simpler than original strategies but captures key factors.
        """
        # Length modifier: shorter frontiers slightly preferred (but not dominant)
        if hasattr(frontier.geometry, 'length'):
            length = frontier.geometry.length
            length_modifier = 1.0 / (1.0 + length / 200.0)  # Mild preference for shorter
        else:
            length_modifier = 1.0

        # Distance modifier: farther from center slightly preferred
        if hasattr(frontier.geometry, 'centroid'):
            centroid = frontier.geometry.centroid
            distance = city_center.distance(centroid)
            distance_modifier = min(1.0, distance / 500.0)  # Mild preference for peripheral
        else:
            distance_modifier = 1.0

        return length_modifier * distance_modifier

    def _calculate_city_center(self, streets_gdf) -> Point:
        """Calculate city center from street geometries."""
        if streets_gdf.empty:
            return Point(0, 0)

        # Use centroid of all street geometries
        all_geoms = streets_gdf.geometry.unary_union
        return all_geoms.centroid

    def _create_action_from_frontier(self, frontier: FrontierEdge, state: GrowthState) -> Optional[InverseGrowthAction]:
        """
        Create action from frontier using geometric resolution (keeps full reasoning).
        """
        from shapely import wkt

        # Resolve frontier geometry to current graph edge (essential for replay)
        current_edge_id = self._resolve_frontier_to_current_edge(frontier.geometry, state)
        if current_edge_id is None:
            logger.debug(f"Could not resolve frontier geometry to current graph edge")
            return None

        # Find the street that corresponds to the resolved edge
        street_id = None
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                street_edge = (min(u, v), max(u, v))
                if street_edge == current_edge_id:
                    street_id = str(idx)  # Convert to string
                    break

        if street_id is None:
            logger.debug(f"No street found for resolved edge {current_edge_id}")
            return None

        # Get the actual street for geometry
        street = state.streets.loc[int(street_id)]

        stable_id = self._compute_stable_frontier_id(frontier)

        # Create state diff with complete street data (essential for replay)
        state_diff = {
            'geometry_wkt': wkt.dumps(frontier.geometry),
            'edgeid': current_edge_id,
            'frontier_type': getattr(frontier, 'frontier_type', 'unknown'),
            'stable_id': stable_id,
            'added_streets': [{
                'edge_id': current_edge_id,
                'u': street.get('u'),
                'v': street.get('v'),
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }],
            'removed_streets': [street_id]
        }

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=street_id,
            intent_params={
                'strategy': 'simplified_frontier_based',
                'edge_u': current_edge_id[0],
                'edge_v': current_edge_id[1],
                'stable_id': stable_id,
                'frontier_type': frontier.frontier_type
            },
            confidence=0.8,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={},
            realized_geometry={
                'geometry_wkt': wkt.dumps(frontier.geometry),
                'edgeid': current_edge_id,
                'frontier_type': frontier.frontier_type,
                'stable_id': stable_id
            }
        )

    def _resolve_frontier_to_current_edge(self, frontier_geometry, state: GrowthState) -> Optional[Tuple[str, str]]:
        """
        Resolve frontier geometry to current graph edge using similarity.
        """
        if not isinstance(frontier_geometry, LineString):
            return None

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

            # Simple similarity score
            similarity_score = self._calculate_similarity(frontier_geometry, street.geometry)

            if similarity_score > best_score:
                best_score = similarity_score
                best_edge = (min(u, v), max(u, v))

        return best_edge if best_score > 0.5 else None

    def _calculate_similarity(self, geom1: LineString, geom2: LineString) -> float:
        """Simple geometric similarity."""
        # Length similarity
        len1, len2 = geom1.length, geom2.length
        length_sim = 1.0 - abs(len1 - len2) / max(len1, len2)

        # Position similarity
        centroid1, centroid2 = geom1.centroid, geom2.centroid
        distance = centroid1.distance(centroid2)
        position_sim = max(0, 1.0 - distance / 50.0)

        return 0.6 * length_sim + 0.4 * position_sim

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry."""
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
