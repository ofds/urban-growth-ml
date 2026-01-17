#!/usr/bin/env python3
"""
State Rewind Operations
Phase A: Basic rewind operations for state manipulation during inverse inference.
"""

from typing import Optional, Any, Dict, List, Set, Tuple
from shapely.geometry import Point, LineString, Polygon
import logging
import networkx as nx

from src.core.contracts import GrowthState, FrontierEdge
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

        # Remove streets from GeoDataFrame
        new_streets = state.streets.drop(streets_to_remove)

        # INCREMENTAL GRAPH UPDATE: Remove specific edge instead of full rebuild
        new_graph = state.graph.copy()

        # Safely remove the edge using has_edge check
        if new_graph.has_edge(edge_u, edge_v):
            new_graph.remove_edge(edge_u, edge_v)
            
            # Remove isolated nodes (nodes with no edges)
            nodes_to_remove = [node for node in [edge_u, edge_v] 
                             if node in new_graph and new_graph.degree[node] == 0]
            for node in nodes_to_remove:
                new_graph.remove_node(node)
            
            logger.info(f"Incremental graph update: removed edge ({edge_u}, {edge_v}), removed {len(nodes_to_remove)} isolated nodes")
        else:
            logger.warning(f"Edge ({edge_u}, {edge_v}) not found in graph during incremental update")

        # DELTA-BASED FRONTIER UPDATE: Only update affected frontiers
        new_frontiers = self._update_frontiers_delta(state.frontiers, new_graph, edge_u, edge_v)

        # Keep iteration at 0 during inference rewind
        new_iteration = max(0, state.iteration - 1)

        return GrowthState(
            streets=new_streets,
            blocks=state.blocks,
            frontiers=new_frontiers,
            graph=new_graph,
            iteration=new_iteration,
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
    
    def _extract_block_edge_segments(self, blocks) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Extract all edge segments from block polygons as coordinate pairs.
        
        Args:
            blocks: GeoDataFrame of block polygons
        
        Returns:
            Set of edge segments as ((x1,y1), (x2,y2)) tuples (normalized)
        """
        edge_segments = set()
        
        for idx, block in blocks.iterrows():
            geom = block.geometry
            if not isinstance(geom, Polygon):
                continue
            
            # Extract exterior ring coordinates
            coords = list(geom.exterior.coords)
            
            # Create edge segments from consecutive points
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i + 1]
                
                # Normalize segment (always store as (min, max) to handle direction)
                segment = tuple(sorted([p1, p2]))
                edge_segments.add(segment)
        
        return edge_segments
    
    def _edge_matches_block_segment(self, edge_geom: LineString, 
                                    block_segments: Set[Tuple[Tuple[float, float], Tuple[float, float]]],
                                    tolerance: float = 1.0) -> Tuple[bool, float]:
        """
        Check if an edge geometry matches any block boundary segment.
        
        Args:
            edge_geom: LineString geometry of the edge
            block_segments: Set of block edge segments
            tolerance: Distance tolerance for matching (meters)
        
        Returns:
            Tuple of (is_match, min_distance)
        """
        edge_coords = list(edge_geom.coords)
        if len(edge_coords) < 2:
            return False, float('inf')
        
        # Get edge endpoints
        edge_start = edge_coords[0]
        edge_end = edge_coords[-1]
        
        # Normalize edge segment
        edge_segment = tuple(sorted([edge_start, edge_end]))
        
        min_total_dist = float('inf')
        
        # Check for exact or near matches
        for block_seg in block_segments:
            # Check if endpoints match within tolerance
            start_dist = ((edge_segment[0][0] - block_seg[0][0])**2 + 
                        (edge_segment[0][1] - block_seg[0][1])**2)**0.5
            end_dist = ((edge_segment[1][0] - block_seg[1][0])**2 + 
                    (edge_segment[1][1] - block_seg[1][1])**2)**0.5
            
            total_dist = start_dist + end_dist
            min_total_dist = min(min_total_dist, total_dist)
            
            if start_dist < tolerance and end_dist < tolerance:
                return True, total_dist
        
        return False, min_total_dist

    def _rebuild_frontiers_simple(self, streets, graph, blocks=None) -> List[FrontierEdge]:
        """
        Simple frontier rebuilding after rewind operations.
        
        Classifies edges as:
        - Dead-end: if one endpoint has degree 1
        - Block-edge: all other edges (default)
        
        Args:
            streets: GeoDataFrame of streets
            graph: NetworkX graph of street network
            blocks: Optional GeoDataFrame of blocks
        
        Returns:
            List of FrontierEdge objects
        """
        frontiers = []
        
        # Debug logging
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        logger.info(f"Rebuilding frontiers from graph with {num_nodes} nodes, {num_edges} edges")
        
        if num_edges == 0:
            logger.warning("Graph has no edges after rewind - this shouldn't happen!")
            return frontiers
        
        # Classify all edges
        dead_end_count = 0
        block_edge_count = 0
        
        for u, v, data in graph.edges(data=True):
            geometry = data.get('geometry')
            if not geometry or not isinstance(geometry, LineString):
                logger.warning(f"Edge ({u}, {v}) missing geometry, skipping")
                continue
            
            u_degree = graph.degree[u]
            v_degree = graph.degree[v]
            
            # Determine frontier type
            if u_degree == 1 or v_degree == 1:
                # Dead-end frontier
                frontier_type = 'dead_end'
                expansion_weight = 0.8
                dead_end_count += 1
            else:
                # Block-edge frontier (default for all other edges)
                frontier_type = 'block_edge'
                expansion_weight = 0.5
                block_edge_count += 1
            
            edge_tuple = (min(u, v), max(u, v))
            frontier_id = f"{frontier_type}_{edge_tuple[0]}_{edge_tuple[1]}"
            
            frontier = FrontierEdge(
                frontier_id=frontier_id,
                edge_id=(u, v),
                block_id=None,  # Could add block lookup if needed
                geometry=geometry,
                frontier_type=frontier_type,
                expansion_weight=expansion_weight,
                spatial_hash=""
            )
            frontiers.append(frontier)
        
        logger.info(f"Detected {dead_end_count} dead-end + {block_edge_count} block-edge = {len(frontiers)} total frontiers")
        
        return frontiers

    def _update_frontiers_delta(self, current_frontiers: List[FrontierEdge], graph, removed_u: int, removed_v: int) -> List[FrontierEdge]:
        """
        Delta-based frontier update: only modify frontiers affected by edge removal.

        Instead of rebuilding all frontiers, identify which frontiers are affected by the
        removed edge and update only those, keeping others unchanged.

        Args:
            current_frontiers: Current list of frontiers
            graph: Updated graph after edge removal
            removed_u, removed_v: Node IDs of the removed edge

        Returns:
            Updated list of frontiers
        """
        # Create a copy of current frontiers
        new_frontiers = current_frontiers.copy()

        # Find frontiers that need updating (those connected to removed nodes)
        affected_frontiers = []
        affected_indices = []

        for i, frontier in enumerate(current_frontiers):
            u, v = frontier.edge_id
            # Check if this frontier is connected to the removed edge's nodes
            if u in [removed_u, removed_v] or v in [removed_u, removed_v]:
                affected_frontiers.append(frontier)
                affected_indices.append(i)

        # Remove affected frontiers from the list
        for i in reversed(affected_indices):
            new_frontiers.pop(i)

        # Rebuild only the affected frontiers
        rebuilt_frontiers = []

        # Get all edges connected to the affected nodes
        affected_nodes = set([removed_u, removed_v])
        affected_edges = []

        for node in affected_nodes:
            if node in graph:
                # Get all edges connected to this node
                for neighbor in graph.neighbors(node):
                    edge_key = tuple(sorted([node, neighbor]))
                    if edge_key not in affected_edges:
                        affected_edges.append(edge_key)

        # Rebuild frontiers for affected edges
        for u, v in affected_edges:
            if graph.has_edge(u, v):
                data = graph.get_edge_data(u, v)
                geometry = data.get('geometry')

                if geometry and isinstance(geometry, LineString):
                    u_degree = graph.degree[u]
                    v_degree = graph.degree[v]

                    # Determine frontier type
                    if u_degree == 1 or v_degree == 1:
                        frontier_type = 'dead_end'
                        expansion_weight = 0.8
                    else:
                        frontier_type = 'block_edge'
                        expansion_weight = 0.5

                    edge_tuple = (min(u, v), max(u, v))
                    frontier_id = f"{frontier_type}_{edge_tuple[0]}_{edge_tuple[1]}"

                    frontier = FrontierEdge(
                        frontier_id=frontier_id,
                        edge_id=(u, v),
                        block_id=None,
                        geometry=geometry,
                        frontier_type=frontier_type,
                        expansion_weight=expansion_weight,
                        spatial_hash=""
                    )
                    rebuilt_frontiers.append(frontier)

        # Add rebuilt frontiers to the list
        new_frontiers.extend(rebuilt_frontiers)

        logger.info(f"Delta frontier update: affected {len(affected_frontiers)} frontiers, rebuilt {len(rebuilt_frontiers)} frontiers")
        return new_frontiers

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
