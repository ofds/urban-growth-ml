#!/usr/bin/env python3
"""
State Rewind Operations
Phase A: Basic rewind operations for state manipulation during inverse inference.
"""

from typing import Optional, Any, Dict, List, Set, Tuple
from shapely.geometry import Point, LineString, Polygon
import logging
import networkx as nx
import numpy as np

from src.core.contracts import GrowthState, FrontierEdge
from .data_structures import InverseGrowthAction, ActionType

logger = logging.getLogger(__name__)


class RewindEngine:
    """
    Engine for rewinding growth states during inverse inference.

    Provides operations to undo growth actions and restore previous states.

    OPTIMIZATION: Maintains edge index for O(1) street lookups.
    """

    # Performance constants
    GEOMETRY_TOLERANCE = 1.0
    DEAD_END_WEIGHT = 0.8
    BLOCK_EDGE_WEIGHT = 0.5

    def __init__(self):
        self.action_handlers = {
            ActionType.EXTEND_FRONTIER: self._rewind_extend_frontier,
            ActionType.SUBDIVIDE_BLOCK: self._rewind_subdivide_block,
            ActionType.REALIGN_STREET: self._rewind_realign_street,
            ActionType.REMOVE_STREET: self._rewind_remove_street,
        }
        # OPTIMIZATION: Edge index for O(1) lookups instead of O(n) DataFrame scans
        self._edge_index = {}  # Maps (u,v) tuples to DataFrame indices

    def _build_edge_index(self, streets_gdf) -> None:
        """
        Build O(1) edge lookup index from streets GeoDataFrame.

        OPTIMIZATION: Pre-compute edge-to-index mapping for fast lookups.

        Args:
            streets_gdf: GeoDataFrame with 'u' and 'v' columns
        """
        self._edge_index.clear()
        for idx, row in streets_gdf.iterrows():
            u, v = str(row['u']), str(row['v'])
            # Store both directions for bidirectional lookup
            self._edge_index[(u, v)] = idx
            self._edge_index[(v, u)] = idx

    def _find_street_index(self, edge_u: int, edge_v: int) -> Optional[int]:
        """
        Find DataFrame index for edge using O(1) lookup.

        Args:
            edge_u, edge_v: Edge node IDs

        Returns:
            DataFrame index if found, None otherwise
        """
        return self._edge_index.get((str(edge_u), str(edge_v)))

    def rewind_action(self, action: InverseGrowthAction, current_state: GrowthState) -> GrowthState:
        """Rewind a single action from the current state."""

        # OPTIMIZATION: Build edge index for O(1) lookups
        self._build_edge_index(current_state.streets)

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

        # OPTIMIZATION: Single boolean array for edge matching (replaces 4 arrays + drop operation)
        # Convert to strings to match DataFrame dtypes
        edge_u_str = str(edge_u)
        edge_v_str = str(edge_v)

        # Create single boolean mask for matching edges (bidirectional)
        edge_mask = ((state.streets['u'] == edge_u_str) & (state.streets['v'] == edge_v_str)) | \
                   ((state.streets['u'] == edge_v_str) & (state.streets['v'] == edge_u_str))

        # Check if any streets match
        if not edge_mask.any():
            logger.warning(f"Cannot rewind: street with edge ({edge_u}, {edge_v}) not found in current state")
            return state

        # Remove streets using boolean indexing (more efficient than inverted mask)
        new_streets = state.streets[~edge_mask]

        # INCREMENTAL GRAPH UPDATE: Modify graph in-place (assumes throw-away state)
        new_graph = state.graph

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
        edge_coords = np.array(list(edge_geom.coords))
        if len(edge_coords) < 2:
            return False, float('inf')

        # Get edge endpoints and sort
        edge_segment = np.sort(edge_coords[[0, -1]], axis=0)

        min_dist = float('inf')
        for block_seg in block_segments:
            block_array = np.array(block_seg)
            dist = np.sum(np.sqrt(np.sum((edge_segment - block_array)**2, axis=1)))
            min_dist = min(min_dist, dist)
            if dist < tolerance * 2:  # Total distance for both endpoints
                return True, dist

        return False, min_dist

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
        
        # OPTIMIZATION: Move expensive logging behind debug guard
        if logger.isEnabledFor(logging.DEBUG):
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            logger.debug(f"Rebuilding frontiers from graph with {num_nodes} nodes, {num_edges} edges")

        if graph.number_of_edges() == 0:
            logger.warning("Graph has no edges after rewind - this shouldn't happen!")
            return frontiers

        # Classify all edges
        dead_end_count = 0
        block_edge_count = 0

        for u, v, data in graph.edges(data=True):
            geometry = data.get('geometry')
            if not geometry or not isinstance(geometry, LineString):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Edge ({u}, {v}) missing geometry, skipping")
                continue

            u_degree = graph.degree[u]
            v_degree = graph.degree[v]

            # Determine frontier type
            if u_degree == 1 or v_degree == 1:
                # Dead-end frontier
                frontier_type = 'dead_end'
                expansion_weight = self.DEAD_END_WEIGHT
                dead_end_count += 1
            else:
                # Block-edge frontier (default for all other edges)
                frontier_type = 'block_edge'
                expansion_weight = self.BLOCK_EDGE_WEIGHT
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

        # OPTIMIZATION: Move expensive logging behind debug guard
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Detected {dead_end_count} dead-end + {block_edge_count} block-edge = {len(frontiers)} total frontiers")
        
        return frontiers

    def _update_frontiers_delta(self, current_frontiers: List[FrontierEdge], graph, removed_u: int, removed_v: int) -> List[FrontierEdge]:
        """
        Delta-based frontier update: only modify frontiers affected by edge removal.

        OPTIMIZATION: Pre-normalize edges and use set-based deduplication for O(1) lookups.

        Args:
            current_frontiers: Current list of frontiers
            graph: Updated graph after edge removal
            removed_u, removed_v: Node IDs of the removed edge

        Returns:
            Updated list of frontiers
        """
        # Use set for O(1) lookups
        affected_nodes = {removed_u, removed_v}

        # Track counts for logging (avoid expensive list comprehensions)
        removed_count = 0
        added_count = 0

        # Remove affected frontiers in-place to avoid O(n) list copy
        i = 0
        while i < len(current_frontiers):
            f = current_frontiers[i]
            if f.edge_id[0] in affected_nodes or f.edge_id[1] in affected_nodes:
                current_frontiers.pop(i)
                removed_count += 1
            else:
                i += 1

        # Rename for clarity - current_frontiers is now filtered
        new_frontiers = current_frontiers

        # OPTIMIZATION: Pre-build all affected edges at once for O(1) deduplication
        # Use normalized edge tuples as keys for automatic deduplication
        affected_edges = set()
        for node in affected_nodes:
            if node in graph:
                for neighbor in graph.neighbors(node):
                    # Pre-normalize edge to avoid repeated calls
                    affected_edges.add((min(node, neighbor), max(node, neighbor)))

        # OPTIMIZATION: Batch process unique edges with single graph query per edge
        for edge_key in affected_edges:
            u, v = edge_key

            # Single graph edge existence check
            if graph.has_edge(u, v):
                node, neighbor = u, v
            elif graph.has_edge(v, u):
                node, neighbor = v, u
            else:
                continue  # Edge no longer exists

            data = graph.get_edge_data(node, neighbor)
            geometry = data.get('geometry')

            if geometry and isinstance(geometry, LineString):
                # OPTIMIZATION: Single degree lookup per node
                u_degree = graph.degree[node]
                v_degree = graph.degree[neighbor]

                # Determine frontier type
                if u_degree == 1 or v_degree == 1:
                    frontier_type = 'dead_end'
                    expansion_weight = self.DEAD_END_WEIGHT
                else:
                    frontier_type = 'block_edge'
                    expansion_weight = self.BLOCK_EDGE_WEIGHT

                frontier_id = f"{frontier_type}_{edge_key[0]}_{edge_key[1]}"

                frontier = FrontierEdge(
                    frontier_id=frontier_id,
                    edge_id=(node, neighbor),
                    block_id=None,
                    geometry=geometry,
                    frontier_type=frontier_type,
                    expansion_weight=expansion_weight,
                    spatial_hash=""
                )
                new_frontiers.append(frontier)
                added_count += 1

        # OPTIMIZATION: Move expensive logging behind debug guard
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Delta frontier update: removed {removed_count} affected frontiers, added {added_count} new frontiers")
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
