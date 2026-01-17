#!/usr/bin/env python3
"""
Basic Inference Engine
Phase A: Simple backward inference for synthetic cities.
"""

from typing import List, Optional, Any
import logging
from shapely.geometry import LineString

from core.contracts import GrowthState
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType
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
    
    def infer_trace(self, final_state: GrowthState, max_steps: int = 50) -> GrowthTrace:
        """
        Infer growth trace from final city state.
        
        Args:
            final_state: Final grown city state
            max_steps: Maximum inference steps
        
        Returns:
            Inferred GrowthTrace
        """
        logger.info("Starting basic inference...")
        
        # Extract arterial skeleton
        skeleton_edges, skeleton_streets = self.skeleton_extractor.extract_skeleton(
            final_state.streets, final_state.graph
        )
        
        # Create initial state with skeleton
        initial_state = self.skeleton_extractor.create_skeleton_state(skeleton_streets, final_state)
        
        # Initialize inference
        current_state = final_state
        actions = []
        step = 0
        
        while step < max_steps and not self._is_minimal_state(current_state, skeleton_edges):
            action = self._infer_most_recent_action(current_state, skeleton_edges)
            if action is None:
                break
            
            # Record action (in reverse chronological order)
            actions.insert(0, action)
            
            # Rewind state
            current_state = self.rewind_engine.rewind_action(action, current_state)
            step += 1
            
            logger.debug(f"Inference step {step}: {action.action_type.value}")
        
        # Create trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                'inference_method': 'basic_peeling',
                'max_steps': max_steps,
                'steps_taken': step,
                'skeleton_streets': len(skeleton_streets)
            }
        )
        
        logger.info(f"Inference complete: {len(actions)} actions inferred")
        return trace
    
    def _is_minimal_state(self, state: GrowthState, skeleton_edges: set) -> bool:
        """Check if state is minimal (just skeleton or seed)."""
        # Simplified: check if we have very few streets
        return len(state.streets) <= max(5, len(skeleton_edges) * 2)
    
    def _infer_most_recent_action(self, state: GrowthState, skeleton_edges: set):
        """Infer the most recently added action using simple heuristics."""
        
        center = self._get_city_center(state)
        
        # Find dead-end frontiers
        dead_end_frontiers = [f for f in state.frontiers if f.frontier_type == 'dead_end']
        
        if dead_end_frontiers:
            peripheral_frontier = max(
                dead_end_frontiers,
                key=lambda f: self._distance_from_center(f.geometry, center)
            )
            
            # CRITICAL FIX: Store geometry for replay matching
            from shapely import wkt
            return InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=peripheral_frontier.frontier_id,
                intent_params={
                    'direction': 'peripheral_expansion',
                    'edge_u': str(peripheral_frontier.edge_id[0]),
                    'edge_v': str(peripheral_frontier.edge_id[1])
                },
                realized_geometry={  # ← USE THIS FIELD!
                    'geometry_wkt': wkt.dumps(peripheral_frontier.geometry),
                    'edge_id': peripheral_frontier.edge_id,
                    'frontier_type': peripheral_frontier.frontier_type
                },
                confidence=0.8,
                timestamp=len(state.streets)
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
            
            # Find matching frontier
            matching_frontier = None
            for frontier in state.frontiers:
                if frontier.edge_id == (u, v) or frontier.edge_id == (v, u):
                    matching_frontier = frontier
                    break
            
            if matching_frontier:
                length = geometry.length
                candidate_streets.append((idx, length, street, matching_frontier))
        
        if candidate_streets:
            shortest_idx, length, street, frontier = min(candidate_streets, key=lambda x: x[1])
            
            from shapely import wkt
            return InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=frontier.frontier_id,
                intent_params={
                    'strategy': 'short_segment',
                    'edge_u': str(frontier.edge_id[0]),
                    'edge_v': str(frontier.edge_id[1])
                },
                realized_geometry={  # ← STORE GEOMETRY HERE!
                    'geometry_wkt': wkt.dumps(frontier.geometry),
                    'edge_id': frontier.edge_id,
                    'frontier_type': frontier.frontier_type
                },
                confidence=0.6,
                timestamp=len(state.streets)
            )
        
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
    
    def _distance_from_center(self, geometry: LineString, center) -> float:
        """Calculate distance from geometry to city center."""
        if hasattr(geometry, 'centroid'):
            geom_center = geometry.centroid
            return ((geom_center.x - center.x)**2 + (geom_center.y - center.y)**2)**0.5
        return 0.0
