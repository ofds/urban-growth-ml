#!/usr/bin/env python3
"""
Basic Inference Engine
Phase A: Simple backward inference for synthetic cities.
"""

from typing import List, Optional, Any, Dict
import logging
from shapely.geometry import LineString

from core.contracts import GrowthState
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType, compute_frontier_signature
from .skeleton import ArterialSkeletonExtractor
from .rewind import RewindEngine

logger = logging.getLogger(__name__)


class BasicInferenceEngine:
    """
    Basic backward inference engine for synthetic cities.
    
    Implements simplified "peeling" heuristics for proof-of-concept.
    """
    
    def __init__(self):
        self.skeleton_extractor = ArterialSkeletonExtractor()
        self.rewind_engine = RewindEngine()
    
    def infer_trace(self, final_state: GrowthState, max_steps: int = 10000000,
                    initial_state: Optional[GrowthState] = None) -> GrowthTrace:
        """
        Infer growth trace from final city state.
        
        Args:
            final_state: Final grown city state
            max_steps: Maximum inference steps  
            initial_state: Optional known initial state for testing
        
        Returns:
            Inferred GrowthTrace
        """
        logger.info("Starting basic inference...")
        
        if initial_state is None:
            # Extract arterial skeleton
            skeleton_edges, skeleton_streets = self.skeleton_extractor.extract_skeleton(
                final_state.streets, final_state.graph
            )
            initial_state = self.skeleton_extractor.create_skeleton_state(skeleton_streets, final_state)
            skeleton_edges_set = skeleton_edges
        else:
            logger.info(f"Using provided initial state with {len(initial_state.streets)} streets")
            skeleton_edges_set = set()
            for idx, street in initial_state.streets.iterrows():
                u, v = street.get('u'), street.get('v')
                if u and v:
                    skeleton_edges_set.add((min(u, v), max(u, v)))
        
        # Initialize inference
        current_state = final_state
        actions = []
        step = 0

        # BATCH COMPUTATION CACHING: Cache expensive computations
        cache_iteration = 0
        CACHE_INTERVAL = 50  # Recalculate every N steps
        cached_city_center = None
        cached_candidate_streets = None
        cached_frontier_distances = None

        logger.info(f"Inference setup: final={len(final_state.streets)} streets, initial={len(initial_state.streets)} streets, skeleton={len(skeleton_edges_set)} edges")

        # Main inference loop - rewind until we reach initial state
        while step < max_steps:
            # Check if we've reached the initial state
            if len(current_state.streets) <= len(initial_state.streets):
                logger.info(f"Reached initial state size at step {step}")
                break

            # BATCH COMPUTATION CACHING: Update cached values periodically
            if step % CACHE_INTERVAL == 0 or cached_city_center is None:
                cached_city_center = self._get_city_center(current_state)
                # Precompute frontier distances for dead-end frontiers
                dead_end_frontiers = [f for f in current_state.frontiers if f.frontier_type == "dead_end"]
                cached_frontier_distances = {}
                for frontier in dead_end_frontiers:
                    cached_frontier_distances[frontier] = self.distance_from_center(frontier.geometry, cached_city_center)
                cache_iteration = step
                logger.debug(f"Updated cached computations at step {step}")

            # Try to infer next action
            action = self.infer_most_recent_action(current_state, skeleton_edges_set)
            if action is None:
                logger.info(f"No more actions to infer at step {step}")
                break

            # PHASE 2: Capture complete state diff before rewind
            state_diff = self._compute_state_diff(current_state, action)

            # Try to rewind
            prev_state = self.rewind_engine.rewind_action(action, current_state)

            # Check if rewind actually worked
            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed at step {step} - state unchanged ({len(prev_state.streets)} >= {len(current_state.streets)})")
                break

            # PHASE 2: Update action with complete state diff
            action_with_diff = self._add_state_diff_to_action(action, state_diff)

            # Only record action if rewind succeeded
            actions.insert(0, action_with_diff)
            if step % 100 == 0 or step < 10:
                logger.info(f"Inference step {step}: streets {len(current_state.streets)} -> {len(prev_state.streets)}")
            current_state = prev_state
            step += 1
        
        # Create trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                "inference_method": "basic_peeling",
                "max_steps": max_steps,
                "steps_taken": step,
                "skeleton_streets": len(skeleton_edges_set)
            }
        )
        
        logger.info(f"Inference complete: {len(actions)} actions inferred")
        return trace


    def _is_minimal_state(self, state: GrowthState, skeleton_edges: set) -> bool:
        """Check if state is minimal (just skeleton or seed)."""
        # Simplified: check if we have very few streets
        return len(state.streets) <= max(5, len(skeleton_edges) * 2)
    

    def infer_most_recent_action(self, state: GrowthState, skeleton_edges: set):
        """Infer the most recently added action using simple heuristics."""
        center = self._get_city_center(state)

        # FRONTIER LOOKUP OPTIMIZATION: Build dictionary mapping edge tuples to frontiers
        frontier_lookup = {}
        for frontier in state.frontiers:
            edge_key = (min(frontier.edge_id[0], frontier.edge_id[1]),
                       max(frontier.edge_id[0], frontier.edge_id[1]))
            frontier_lookup[edge_key] = frontier

        # Find dead-end frontiers
        dead_end_frontiers = [f for f in state.frontiers if f.frontier_type == "dead_end"]

        if dead_end_frontiers:
            peripheral_frontier = max(
                dead_end_frontiers,
                key=lambda f: self.distance_from_center(f.geometry, center)
            )


            from shapely import wkt

            # CRITICAL FIX: Compute and store stable geometric ID
            stable_id = self._compute_stable_frontier_id(peripheral_frontier)

            # Compute geometric signature for stable frontier matching
            geometric_signature = compute_frontier_signature(peripheral_frontier)

            return InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=peripheral_frontier.frontier_id,  # Keep for debugging
                intent_params={
                    "direction": "peripheral_expansion",
                    "edge_u": str(peripheral_frontier.edge_id[0]),
                    "edge_v": str(peripheral_frontier.edge_id[1]),
                    "stable_id": stable_id  # ← ADD THIS
                },
                realized_geometry={
                    "geometry_wkt": wkt.dumps(peripheral_frontier.geometry),
                    "edgeid": peripheral_frontier.edge_id,
                    "frontier_type": peripheral_frontier.frontier_type,
                    "stable_id": stable_id  # ← ADD THIS TOO
                },
                confidence=0.8,
                timestamp=len(state.streets),
                geometric_signature=geometric_signature  # ← ADD GEOMETRIC SIGNATURE
            )

        # Fallback for short streets
        candidate_streets = []
        for idx, street in state.streets.iterrows():
            geometry = street.geometry
            if not isinstance(geometry, LineString):
                continue

            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue

            edge_key = (min(u, v), max(u, v))
            if edge_key in skeleton_edges:
                continue

            # OPTIMIZED FRONTIER LOOKUP: O(1) dictionary lookup instead of O(n) linear search
            matching_frontier = frontier_lookup.get(edge_key)

            if matching_frontier:
                length = geometry.length
                candidate_streets.append((idx, length, street, matching_frontier))


        if candidate_streets:
            shortest_idx, length, street, frontier = min(candidate_streets, key=lambda x: x[1])

            from shapely import wkt

            # CRITICAL FIX: Compute and store stable geometric ID
            stable_id = self._compute_stable_frontier_id(frontier)

            # Compute geometric signature for stable frontier matching
            geometric_signature = compute_frontier_signature(frontier)

            return InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=frontier.frontier_id,
                intent_params={
                    'strategy': 'short_segment',
                    'edge_u': str(frontier.edge_id[0]),
                    'edge_v': str(frontier.edge_id[1]),
                    'stable_id': stable_id  # ← ADD THIS
                },
                realized_geometry={
                    'geometry_wkt': wkt.dumps(frontier.geometry),
                    'edgeid': frontier.edge_id,
                    'frontier_type': frontier.frontier_type,
                    'stable_id': stable_id  # ← ADD THIS TOO
                },
                confidence=0.6,
                timestamp=len(state.streets),
                geometric_signature=geometric_signature  # ← ADD GEOMETRIC SIGNATURE
            )

        logger.warning(f"DEBUG: No actions found - cannot infer further")
        return None


    def _get_city_center(self, state: GrowthState):
        """Get approximate city center."""
        if state.city_bounds:
            return state.city_bounds.centroid
        # Fallback to mean of all street coordinates
        all_coords = []
        for idx, street in state.streets.iterrows():
            if hasattr(street.geometry, 'coords'):
                all_coords.extend(street.geometry.coords)
        if all_coords:
            x_coords = [c[0] for c in all_coords]
            y_coords = [c[1] for c in all_coords]
            return type('Point', (), {'x': sum(x_coords)/len(x_coords), 'y': sum(y_coords)/len(y_coords)})()
        return type('Point', (), {'x': 0, 'y': 0})()
    
    def distance_from_center(self, geometry: LineString, center) -> float:
        """Calculate distance from geometry to city center."""
        if hasattr(geometry, 'centroid'):
            geom_center = geometry.centroid
            return ((geom_center.x - center.x)**2 + (geom_center.y - center.y)**2)**0.5
        return 0.0

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry coordinates with consistent rounding."""
        import hashlib
        if not hasattr(frontier, 'geometry') or frontier.geometry is None:
            return "invalid_frontier"

        geom = frontier.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            # FIX: Use consistent 2-decimal precision for both coordinates
            start = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
            end = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
            frontier_type = getattr(frontier, 'frontier_type', 'unknown')

            hash_input = f"{start}_{end}_{frontier_type}".encode('utf-8')
            return hashlib.md5(hash_input).hexdigest()[:16]

        return "invalid_geometry"

    def _compute_state_diff(self, current_state: GrowthState, action: InverseGrowthAction) -> Dict[str, Any]:
        """
        Compute the complete state diff that this action represents.

        This captures exactly what streets are added/removed and how the graph changes,
        eliminating the need for frontier matching during replay.

        Args:
            current_state: State before rewind (contains the street to be removed)
            action: Action being rewound

        Returns:
            Dict containing complete state changes
        """
        state_diff = {
            'added_streets': [],    # Streets that will be added during replay
            'removed_streets': [],  # Streets that are removed during rewind
            'graph_changes': {},    # Node/edge changes
            'frontier_changes': {}  # Frontier state changes
        }

        # For EXTEND_FRONTIER actions, the street being removed during rewind
        # is the one that was added during forward growth
        if action.action_type == ActionType.EXTEND_FRONTIER:
            # Find the street that matches this action's edge
            edge_u = action.realized_geometry.get('edgeid', (None, None))[0] if action.realized_geometry else None
            edge_v = action.realized_geometry.get('edgeid', (None, None))[1] if action.realized_geometry else None

            if edge_u is not None and edge_v is not None:
                # Find the street with this edge
                for idx, street in current_state.streets.iterrows():
                    u, v = street.get('u'), street.get('v')
                    if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                        # MEMORY OPTIMIZATION: Store only edge IDs and essential metadata
                        # Geometry will be reconstructed from final state during replay
                        street_data = {
                            'edge_id': (min(u, v), max(u, v)),  # Normalized edge tuple for lookup
                            'u': u,
                            'v': v,
                            # Remove heavy WKT geometry storage - reconstruct from final state
                            # 'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                            'osmid': street.get('osmid'),
                            'highway': street.get('highway'),
                            'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                        }
                        state_diff['added_streets'].append(street_data)
                        state_diff['removed_streets'].append(idx)
                        break

        # Store graph state information
        state_diff['graph_changes'] = {
            'nodes_before': current_state.graph.number_of_nodes(),
            'edges_before': current_state.graph.number_of_edges(),
            'nodes_after': None,  # Will be filled after rewind
            'edges_after': None   # Will be filled after rewind
        }

        # Store frontier state information
        state_diff['frontier_changes'] = {
            'frontiers_before': len(current_state.frontiers),
            'frontiers_after': None  # Will be filled after rewind
        }

        return state_diff

    def _add_state_diff_to_action(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
        """
        Create a new action with the state diff included.

        Since InverseGrowthAction is frozen, we need to create a new instance.

        Args:
            action: Original action
            state_diff: Computed state diff

        Returns:
            New action with state_diff populated
        """
        return InverseGrowthAction(
            action_type=action.action_type,
            target_id=action.target_id,
            intent_params=action.intent_params,
            realized_geometry=action.realized_geometry,
            confidence=action.confidence,
            timestamp=action.timestamp,
            geometric_signature=action.geometric_signature,
            state_diff=state_diff,  # Add the state diff
            action_metadata=action.action_metadata
        )
