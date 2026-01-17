#!/usr/bin/env python3
"""
State Rewind Operations
Phase A: Basic rewind operations for state manipulation during inverse inference.
"""

from typing import Optional, Any, Dict, List
from shapely.geometry import Point, LineString
import logging

from core.contracts import GrowthState, FrontierEdge
from .data_structures import InverseGrowthAction, ActionType

logger = logging.getLogger(__name__)


class RewindEngine:
    """
    Engine for rewinding growth states during inverse inference.
    
    Provides operations to undo growth actions and restore previous states.
    """
    
    def __init__(self):
        self.action_handlers = {
            ActionType.EXTEND_FRONTIER: self._rewind_extend_frontier,
            ActionType.SUBDIVIDE_BLOCK: self._rewind_subdivide_block,
            ActionType.REALIGN_STREET: self._rewind_realign_street,
            ActionType.REMOVE_STREET: self._rewind_remove_street,
        }
    
    def rewind_action(self, action: InverseGrowthAction, current_state: GrowthState) -> GrowthState:
        """Rewind a single action from the current state."""
        
        # Remove this check - we want to rewind FROM final state TO initial state
        # if current_state.iteration == 0:
        #     logger.warning(f"Cannot rewind action {action.action_type}: already at initial state (iteration {current_state.iteration})")
        #     return current_state
        
        # Check if we can actually remove this street
        handler = self.action_handlers.get(action.action_type)
        if handler is None:
            logger.warning(f"No rewind handler for action type: {action.action_type}")
            return current_state
        
        try:
            new_state = handler(action, current_state)
            # Verify state actually changed
            if len(new_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed to remove street - state unchanged")
                return current_state
            return new_state
        except Exception as e:
            logger.error(f"Failed to rewind action {action.action_type}: {e}")
            return current_state
    
    def _rewind_extend_frontier(self, action: InverseGrowthAction, state: GrowthState) -> GrowthState:
        """Rewind an EXTEND_FRONTIER action by removing the added street segment."""
        
        # Get edge info from action
        edge_u = action.realized_geometry.get('edgeid', (None, None))[0] if action.realized_geometry else None
        edge_v = action.realized_geometry.get('edgeid', (None, None))[1] if action.realized_geometry else None
        
        if edge_u is None or edge_v is None:
            logger.warning(f"Cannot rewind: no edge_id in action")
            return state
        
        # Find street(s) matching this edge
        streets_to_remove = []
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                streets_to_remove.append(idx)
                break  # Found it
        
        if not streets_to_remove:
            logger.warning(f"Cannot rewind: street with edge ({edge_u}, {edge_v}) not found in current state")
            return state
        
        # Remove streets (rest of code stays the same)
        new_streets = state.streets.drop(streets_to_remove)
        
        # Update graph efficiently
        new_graph = state.graph.copy()
        nodes_to_check = set()
        
        for idx in streets_to_remove:
            street = state.streets.loc[idx]
            u, v = street['u'], street['v']
            
            # Remove edge
            if new_graph.has_edge(u, v):
                new_graph.remove_edge(u, v)
                nodes_to_check.update([u, v])
        
        # Batch remove isolated nodes
        isolated_nodes = [node for node in nodes_to_check if new_graph.degree[node] == 0]
        new_graph.remove_nodes_from(isolated_nodes)
        
        # Rebuild frontiers
        new_frontiers = self._rebuild_frontiers_simple(new_streets, new_graph)
        
        return GrowthState(
            streets=new_streets,
            blocks=state.blocks,
            frontiers=new_frontiers,
            graph=new_graph,
            iteration=state.iteration - 1,
            city_bounds=state.city_bounds
        )
    
    def _rewind_subdivide_block(self, action: InverseGrowthAction, state: GrowthState) -> GrowthState:
        """Rewind a SUBDIVIDE_BLOCK action by merging the subdivided blocks."""
        # This is complex - would need to identify which blocks were created
        # For now, return unchanged state with warning
        logger.warning("SUBDIVIDE_BLOCK rewind not fully implemented")
        return state
    
    def _rewind_realign_street(self, action: InverseGrowthAction, state: GrowthState) -> GrowthState:
        """Rewind a REALIGN_STREET action by restoring original geometry."""
        # Would need to store original geometry in action metadata
        logger.warning("REALIGN_STREET rewind not fully implemented")
        return state
    
    def _rewind_remove_street(self, action: InverseGrowthAction, state: GrowthState) -> GrowthState:
        """Rewind a REMOVE_STREET action by restoring the removed street."""
        # This is the inverse of EXTEND_FRONTIER rewind
        # Would need stored geometry in action metadata
        logger.warning("REMOVE_STREET rewind not fully implemented")
        return state
    
    def _rebuild_frontiers_simple(self, streets, graph) -> List[FrontierEdge]:
        """Simple frontier rebuilding after rewind operations."""
        frontiers = []
        
        # Detect dead-end edges
        for u, v, data in graph.edges(data=True):
            u_degree = graph.degree[u]
            v_degree = graph.degree[v]
            
            if u_degree == 1 or v_degree == 1:
                geometry = data.get('geometry')
                if geometry and isinstance(geometry, LineString):
                    edge_tuple = (min(u, v), max(u, v))
                    frontier_id = f"dead_end_{edge_tuple[0]}_{edge_tuple[1]}"
                    
                    frontier = FrontierEdge(
                        frontier_id=frontier_id,
                        edge_id=(u, v),
                        block_id=None,
                        geometry=geometry,
                        frontier_type='dead_end',
                        expansion_weight=0.8,
                        spatial_hash=""
                    )
                    frontiers.append(frontier)
        
        return frontiers
    
    def can_rewind_action(self, action: InverseGrowthAction, state: GrowthState) -> bool:
        """
        Check if an action can be safely rewound from the current state.
        
        Args:
            action: Action to check
            state: Current state
        
        Returns:
            True if rewind is possible
        """
        # Basic checks - can be extended
        if action.action_type == ActionType.EXTEND_FRONTIER:
            # Check if target street exists
            target_id = action.target_id
            for idx, street in state.streets.iterrows():
                if str(idx) in target_id or street.get('osmid', '') == target_id:
                    return True
            return False
        
        # For other actions, assume possible for now
        return True
