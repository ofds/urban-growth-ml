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
        
        # Remove streets
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
        new_frontiers = self._rebuild_frontiers_simple(new_streets, new_graph, state.blocks)
        
        # FIXED: Keep iteration at 0 during inference rewind
        # During inverse inference, we're rewinding from a final state (iteration=0)
        # back towards an initial state. We don't need negative iterations.
        new_iteration = max(0, state.iteration - 1)
        
        return GrowthState(
            streets=new_streets,
            blocks=state.blocks,
            frontiers=new_frontiers,
            graph=new_graph,
            iteration=new_iteration,  # ← FIXED: Use max(0, iteration-1)
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
    
    def _rebuild_frontiers_simple(self, streets, graph, blocks=None) -> List[FrontierEdge]:
        """
        Enhanced frontier rebuilding after rewind operations.
        
        Detects both dead-end frontiers and block-edge frontiers by checking
        spatial adjacency to block boundaries.
        
        Args:
            streets: GeoDataFrame of streets
            graph: NetworkX graph of street network
            blocks: Optional GeoDataFrame of blocks (from state.blocks)
        
        Returns:
            List of FrontierEdge objects
        """
        from shapely.ops import unary_union
        from shapely.geometry import MultiLineString
        
        frontiers = []
        
        # Step 1: Detect dead-end frontiers (existing logic)
        dead_end_edges = set()
        for u, v, data in graph.edges(data=True):
            u_degree = graph.degree[u]
            v_degree = graph.degree[v]
            
            geometry = data.get('geometry')
            if not geometry or not isinstance(geometry, LineString):
                continue
                
            if u_degree == 1 or v_degree == 1:
                edge_tuple = (min(u, v), max(u, v))
                dead_end_edges.add(edge_tuple)
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
        
        logger.info(f"Detected {len(dead_end_edges)} dead-end frontiers")
        
        # Step 2: Detect block-edge frontiers (NEW LOGIC)
        if blocks is not None and len(blocks) > 0:
            # Create unified block boundary for spatial queries
            try:
                all_blocks_union = unary_union(blocks.geometry)
                
                # Get block boundary as LineString or MultiLineString
                if hasattr(all_blocks_union, 'boundary'):
                    block_boundaries = all_blocks_union.boundary
                else:
                    logger.warning("Could not extract block boundaries")
                    block_boundaries = None
                
                if block_boundaries:
                    # Check each edge against block boundaries
                    for u, v, data in graph.edges(data=True):
                        edge_tuple = (min(u, v), max(u, v))
                        
                        # Skip if already classified as dead-end
                        if edge_tuple in dead_end_edges:
                            continue
                        
                        geometry = data.get('geometry')
                        if not geometry or not isinstance(geometry, LineString):
                            continue
                        
                        # Check if edge is adjacent to block boundary
                        # Use small buffer for numerical tolerance
                        is_block_edge = geometry.buffer(0.1).intersects(block_boundaries)
                        
                        if is_block_edge:
                            # Find which block this edge belongs to
                            block_id = None
                            for idx, block in blocks.iterrows():
                                if geometry.buffer(0.1).intersects(block.geometry.boundary):
                                    block_id = int(idx)
                                    break
                            
                            frontier_id = f"block_edge_{edge_tuple[0]}_{edge_tuple[1]}"
                            
                            frontier = FrontierEdge(
                                frontier_id=frontier_id,
                                edge_id=(u, v),
                                block_id=block_id,
                                geometry=geometry,
                                frontier_type='block_edge',
                                expansion_weight=0.5,
                                spatial_hash=""
                            )
                            frontiers.append(frontier)
                
                logger.info(f"Detected {len(frontiers) - len(dead_end_edges)} block-edge frontiers")
                
            except Exception as e:
                logger.error(f"Error detecting block-edge frontiers: {e}")
                logger.info("Falling back to simple heuristic")
                
                # Fallback: treat non-dead-end edges as potential block edges
                for u, v, data in graph.edges(data=True):
                    edge_tuple = (min(u, v), max(u, v))
                    if edge_tuple in dead_end_edges:
                        continue
                    
                    geometry = data.get('geometry')
                    if not geometry or not isinstance(geometry, LineString):
                        continue
                    
                    frontier_id = f"block_edge_{edge_tuple[0]}_{edge_tuple[1]}"
                    
                    frontier = FrontierEdge(
                        frontier_id=frontier_id,
                        edge_id=(u, v),
                        block_id=None,
                        geometry=geometry,
                        frontier_type='block_edge',
                        expansion_weight=0.5,
                        spatial_hash=""
                    )
                    frontiers.append(frontier)
        
        logger.info(f"Total frontiers rebuilt: {len(frontiers)}")
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
