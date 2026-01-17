"""Main orchestrator for procedural city growth simulation.

This module implements the GrowthEngine class that coordinates OSM data loading,
frontier selection, action proposal, and state evolution for urban growth.
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from .actions import GrowthAction, ACTION_GROW_TRAJECTORY, ACTION_SUBDIVIDE_BLOCK
from .validators import validate_growth_action
from .state_updater import apply_growth_action
from .geometry_utils import generate_canonical_node_id
from src.core.contracts import GrowthState, FrontierEdge

logger = logging.getLogger(__name__)


class GrowthEngine:
    """Main API for procedural city growth simulation.
    
    Orchestrates the growth process by:
    1. Loading initial city state from OSM data
    2. Selecting frontiers for expansion
    3. Proposing validated growth actions
    4. Applying actions to evolve the city state
    """
    
    def __init__(self, city_name: str, seed: int = 42):
        """Initialize growth engine with city name and random seed.
        
        Args:
            city_name: Name of city for loading OSM data
            seed: Random seed for deterministic selection
        """
        self.city_name = city_name
        self.seed = seed
        self.random = random.Random(seed)
        
        # Set up file paths
        data_dir = Path("data/processed")
        self.streets_path = data_dir / f"{city_name}_streets.gpkg"
        self.blocks_path = data_dir / f"{city_name}_blocks_cleaned.gpkg"
        self.frontier_path = data_dir / f"{city_name}_frontier_edges.gpkg"
        self.graph_path = data_dir / f"{city_name}_street_graph_cleaned.graphml"
        
        # Constants
        self.MIN_STREET_LENGTH = 10.0
        self.MAX_STREET_LENGTH = 100.0
        
        logger.info(f"Initialized GrowthEngine for {city_name} with seed {seed}")
    
    def load_initial_state(self) -> GrowthState:
        """Load initial state from OSM data files.
        
        Returns:
            GrowthState with synchronized streets, blocks, frontiers, and graph
        """
        logger.info(f"Loading OSM state for {self.city_name}")
        
        # Load streets
        streets = gpd.read_file(self.streets_path)
        logger.info(f"Loaded {len(streets)} streets")
        
        # Load blocks
        blocks = gpd.read_file(self.blocks_path)
        logger.info(f"Loaded {len(blocks)} blocks")
        
        # Load frontiers
        frontiers_gdf = gpd.read_file(self.frontier_path, layer='frontier_edges')
        frontiers = []
        for idx, row in frontiers_gdf.iterrows():
            frontier = FrontierEdge(
                frontier_id=row['frontier_id'],
                edge_id=(row['edge_id_u'], row['edge_id_v']),
                block_id=int(row['block_id']) if pd.notna(row['block_id']) else None,
                geometry=row['geometry'],
                frontier_type=row['frontier_type'],
                expansion_weight=row['expansion_weight'],
                spatial_hash=""
            )
            frontiers.append(frontier)
        logger.info(f"Loaded {len(frontiers)} frontiers")
        
        # Load graph
        graph = nx.read_graphml(self.graph_path)
        logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes")
        
        # Convert to canonical IDs
        graph, node_id_mapping = self._convert_to_canonical_ids(graph)
        logger.info("Converted graph to canonical node IDs")
        
        # Update streets u/v to canonical IDs
        streets['u'] = streets['u'].apply(
            lambda x: node_id_mapping.get(str(int(x)), str(x))
        )
        streets['v'] = streets['v'].apply(
            lambda x: node_id_mapping.get(str(int(x)), str(x))
        )
        
        # Update frontier edge_ids to canonical IDs
        updated_frontiers = []
        for frontier in frontiers:
            edge_u, edge_v = frontier.edge_id
            new_edge_u = node_id_mapping.get(str(edge_u), str(edge_u))
            new_edge_v = node_id_mapping.get(str(edge_v), str(edge_v))
            
            updated_frontier = FrontierEdge(
                frontier_id=frontier.frontier_id,
                edge_id=(new_edge_u, new_edge_v),
                block_id=frontier.block_id,
                geometry=frontier.geometry,
                frontier_type=frontier.frontier_type,
                expansion_weight=frontier.expansion_weight,
                spatial_hash=frontier.spatial_hash
            )
            updated_frontiers.append(updated_frontier)
        
        # Compute city bounds
        city_bounds = streets.unary_union.convex_hull
        
        return GrowthState(
            streets=streets,
            blocks=blocks,
            frontiers=updated_frontiers,
            graph=graph,
            iteration=0,
            city_bounds=city_bounds
        )
    
    def select_frontier_edge(
        self, 
        frontiers: List[FrontierEdge], 
        state: GrowthState
    ) -> Optional[FrontierEdge]:
        """Select frontier using weighted random selection.
        
        Args:
            frontiers: List of available frontiers
            state: Current growth state
            
        Returns:
            Selected frontier or None if list is empty
        """
        if not frontiers:
            return None
        
        # Sort by frontier_id for determinism
        frontiers_sorted = sorted(frontiers, key=lambda x: x.frontier_id)
        
        # Weighted random selection
        weights = [f.expansion_weight for f in frontiers_sorted]
        total_weight = sum(weights)
        
        if total_weight == 0:
            logger.warning("All frontiers have zero weight")
            return None
        
        r = self.random.uniform(0, total_weight)
        cumulative = 0
        for frontier, weight in zip(frontiers_sorted, weights):
            cumulative += weight
            if r <= cumulative:
                logger.debug(f"Selected frontier {frontier.frontier_id}")
                return frontier
        
        return frontiers_sorted[-1]
    
    def propose_grow_trajectory(
        self,
        frontier: FrontierEdge,
        state: GrowthState
    ) -> Optional[GrowthAction]:
        """Propose extending a frontier with a straight street segment."""
        
        # DEBUG: Log what we received
        logger.info(f"propose_grow_trajectory called:")
        logger.info(f"  Frontier type: {frontier.frontier_type}")
        logger.info(f"  Expected: 'dead_end'")
        
        if frontier.frontier_type != 'dead_end':
            logger.warning(f"Rejecting frontier: wrong type '{frontier.frontier_type}'")
            return None
        
        # Get frontier endpoint
        endpoint = Point(frontier.geometry.coords[-1])
        
        # Get frontier direction
        direction = self._get_frontier_direction(frontier)
        
        # Random length and angle variation
        length = self.random.uniform(20, 80)
        angle_variation = self.random.uniform(-np.pi/6, np.pi/6)  # ±30°
        new_direction = direction + angle_variation
        
        # Calculate new endpoint
        new_x = endpoint.x + length * np.cos(new_direction)
        new_y = endpoint.y + length * np.sin(new_direction)
        new_endpoint = Point(new_x, new_y)
        
        # Create proposed geometry
        proposed_geometry = LineString([endpoint, new_endpoint])
        
        # Create action
        action = GrowthAction(
            action_type=ACTION_GROW_TRAJECTORY,
            frontier_edge=frontier,
            proposed_geometry=proposed_geometry,
            parameters={'length': length, 'angle': new_direction}
        )
        
        # Validate
        valid, reason = validate_growth_action(action, state)
        if not valid:
            logger.debug(f"Invalid grow_trajectory: {reason}")
            return None
        
        return action
    
    def propose_subdivide_block(
        self, 
        frontier: FrontierEdge, 
        state: GrowthState
    ) -> Optional[GrowthAction]:
        """Propose subdividing a block with a straight street.
        
        Args:
            frontier: The block-edge frontier
            state: Current growth state
            
        Returns:
            Valid GrowthAction or None if invalid
        """
        if frontier.frontier_type != 'block_edge':
            return None
        
        if frontier.block_id is None or frontier.block_id >= len(state.blocks):
            return None
        
        block = state.blocks.iloc[frontier.block_id]
        block_geom = block.geometry
        
        # Check block size
        if block_geom.area < 1000:  # Too small to subdivide
            logger.debug(f"Block {frontier.block_id} too small: {block_geom.area:.1f}m²")
            return None
        
        # Find opposite edge
        opposite_edge = self._find_opposite_block_edge(frontier, block_geom)
        if not opposite_edge:
            logger.debug("Could not find opposite block edge")
            return None
        
        # Create bisecting street from frontier midpoint to opposite edge midpoint
        frontier_midpoint = frontier.geometry.interpolate(0.5, normalized=True)
        opposite_midpoint = opposite_edge.interpolate(0.5, normalized=True)
        
        proposed_geometry = LineString([frontier_midpoint, opposite_midpoint])
        
        # Create action
        action = GrowthAction(
            action_type=ACTION_SUBDIVIDE_BLOCK,
            frontier_edge=frontier,
            proposed_geometry=proposed_geometry,
            parameters={'block_id': frontier.block_id}
        )
        
        # Validate
        valid, reason = validate_growth_action(action, state)
        if not valid:
            logger.debug(f"Invalid subdivide_block: {reason}")
            return None
        
        return action
    
    def apply_growth_action(
        self, 
        action: GrowthAction, 
        state: GrowthState
    ) -> GrowthState:
        """Apply growth action to state, returning new synchronized state.
        
        Args:
            action: Validated growth action to apply
            state: Current growth state
            
        Returns:
            New GrowthState with applied action
        """
        logger.info(f"Applying {action.action_type} action")
        return apply_growth_action(action, state)
    
    def _convert_to_canonical_ids(
        self, 
        graph: nx.Graph
    ) -> Tuple[nx.Graph, Dict[str, str]]:
        """Convert all graph node IDs to canonical format.
        
        Args:
            graph: Input graph with arbitrary node IDs
            
        Returns:
            Tuple of (new_graph, node_id_mapping)
        """
        new_graph = nx.Graph()
        node_id_mapping = {}
        
        # Convert nodes
        for node, data in graph.nodes(data=True):
            # Extract coordinates from node
            if 'x' in data and 'y' in data:
                x, y = float(data['x']), float(data['y'])
            elif 'geometry' in data:
                geom = data['geometry']
                if isinstance(geom, Point):
                    x, y = geom.x, geom.y
                else:
                    continue
            else:
                continue
            
            # Generate canonical ID
            canonical_id = generate_canonical_node_id(x, y)
            node_id_mapping[str(node)] = canonical_id
            
            # Add node with geometry
            new_graph.add_node(canonical_id, geometry=Point(x, y), x=x, y=y)
        
        # Convert edges
        for u, v, data in graph.edges(data=True):
            u_canonical = node_id_mapping.get(str(u))
            v_canonical = node_id_mapping.get(str(v))
            
            if u_canonical and v_canonical:
                new_graph.add_edge(u_canonical, v_canonical, **data)
        
        logger.debug(f"Mapped {len(node_id_mapping)} nodes to canonical IDs")
        return new_graph, node_id_mapping
    
    def _get_frontier_direction(self, frontier: FrontierEdge) -> float:
        """Calculate frontier's outward direction in radians.
        
        Args:
            frontier: Frontier edge to analyze
            
        Returns:
            Direction in radians
        """
        coords = list(frontier.geometry.coords)
        if len(coords) < 2:
            return 0.0
        
        # Use last two points to determine direction
        x1, y1 = coords[-2]
        x2, y2 = coords[-1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        return np.arctan2(dy, dx)
    
    def _find_opposite_block_edge(
        self, 
        frontier: FrontierEdge, 
        block: Polygon
    ) -> Optional[LineString]:
        """Find opposite edge of block for subdivision.
        
        Args:
            frontier: The frontier edge
            block: Block polygon to subdivide
            
        Returns:
            Opposite edge LineString or None
        """
        if not isinstance(block, Polygon):
            return None
        
        # Get block boundary coordinates
        boundary_coords = list(block.exterior.coords)
        if len(boundary_coords) < 4:
            return None
        
        # Find all edges of the block
        edges = []
        for i in range(len(boundary_coords) - 1):
            edge = LineString([boundary_coords[i], boundary_coords[i + 1]])
            edges.append(edge)
        
        # Find edge farthest from frontier
        frontier_centroid = frontier.geometry.centroid
        max_distance = 0
        opposite_edge = None
        
        for edge in edges:
            distance = frontier_centroid.distance(edge.centroid)
            if distance > max_distance:
                max_distance = distance
                opposite_edge = edge
        
        return opposite_edge
