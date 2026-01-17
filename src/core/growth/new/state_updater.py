"""State mutation layer for urban growth simulation.

This module applies validated growth actions to immutable GrowthState objects,
ensuring perfect synchronization between streets, graph, blocks, and frontiers.
"""

from typing import List
import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import polygonize
import hashlib

from .actions import GrowthAction
from .geometry_utils import generate_canonical_node_id, find_or_create_node
from ...contracts import GrowthState, FrontierEdge


def apply_growth_action(action: GrowthAction, state: GrowthState) -> GrowthState:
    """Apply a validated growth action to state, returning new synchronized state.
    
    Args:
        action: The growth action to apply
        state: Current immutable growth state
        
    Returns:
        New GrowthState with all components synchronized
    """
    # Extract endpoints from proposed geometry
    start_point = Point(action.proposed_geometry.coords[0])
    end_point = Point(action.proposed_geometry.coords[-1])
    
    # Determine node IDs (snap to existing or create new)
    start_node_id = find_or_create_node(start_point, state.graph, snap_tolerance=0.5)
    end_node_id = find_or_create_node(end_point, state.graph, snap_tolerance=0.5)
    
    # Add street to GeoDataFrame
    new_streets = add_street_to_gdf(
        state.streets, action.proposed_geometry, start_node_id, end_node_id
    )
    
    # Add nodes/edges to graph
    new_graph = add_street_to_graph(
        state.graph, action.proposed_geometry, start_node_id, end_node_id
    )
    
    # Repolygonize blocks from updated streets
    new_blocks = repolygonize_blocks(new_streets)
    
    # Rebuild frontiers from updated graph
    new_frontiers = rebuild_frontiers(new_streets, new_blocks, new_graph)
    
    # Return new immutable state
    return GrowthState(
        streets=new_streets,
        blocks=new_blocks,
        frontiers=new_frontiers,
        graph=new_graph,
        iteration=state.iteration + 1,
        city_bounds=state.city_bounds
    )


def add_street_to_gdf(
    streets: gpd.GeoDataFrame,
    geometry: LineString,
    start_node_id: str,
    end_node_id: str
) -> gpd.GeoDataFrame:
    """Add new street row to GeoDataFrame immutably.
    
    Args:
        streets: Existing streets GeoDataFrame
        geometry: New street geometry
        start_node_id: Canonical ID of start node
        end_node_id: Canonical ID of end node
        
    Returns:
        New GeoDataFrame with added street
    """
    new_street = {
        'u': start_node_id,
        'v': end_node_id,
        'geometry': geometry,
        'osmid': -1,  # Synthetic street marker
        'highway': 'residential',
        'length': geometry.length
    }
    
    new_street_gdf = gpd.GeoDataFrame(
        [new_street], geometry='geometry', crs=streets.crs
    )
    
    return pd.concat([streets, new_street_gdf], ignore_index=True)


def add_street_to_graph(
    graph: nx.Graph,
    geometry: LineString,
    start_node_id: str,
    end_node_id: str
) -> nx.Graph:
    """Add nodes and edge to graph immutably.
    
    Args:
        graph: Existing NetworkX graph
        geometry: Street geometry
        start_node_id: Canonical ID of start node
        end_node_id: Canonical ID of end node
        
    Returns:
        New graph with added nodes/edge
    """
    new_graph = graph.copy()
    
    # Extract point geometries from LineString
    start_point = Point(geometry.coords[0])
    end_point = Point(geometry.coords[-1])
    
    # Add nodes if they don't exist
    if start_node_id not in new_graph.nodes:
        new_graph.add_node(start_node_id, geometry=start_point)
    
    if end_node_id not in new_graph.nodes:
        new_graph.add_node(end_node_id, geometry=end_point)
    
    # Add edge (avoid duplicates)
    if not new_graph.has_edge(start_node_id, end_node_id):
        new_graph.add_edge(
            start_node_id,
            end_node_id,
            geometry=geometry,
            length=geometry.length,
            highway='residential'
        )
    
    return new_graph


def repolygonize_blocks(streets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Regenerate blocks from street network using polygonize.
    
    Args:
        streets: Street network GeoDataFrame
        
    Returns:
        New blocks GeoDataFrame with polygonized geometries
    """
    if streets.empty:
        return gpd.GeoDataFrame(columns=['geometry'], crs=streets.crs, geometry='geometry')
    
    # Extract all street geometries
    lines = list(streets.geometry)
    
    # Polygonize street network
    polygons = list(polygonize(lines))
    
    if not polygons:
        return gpd.GeoDataFrame(columns=['geometry'], crs=streets.crs, geometry='geometry')
    
    # Remove exterior boundary (largest polygon) if it's significantly larger
    if len(polygons) > 1:
        polygons_sorted = sorted(polygons, key=lambda p: p.area, reverse=True)
        # If largest is 5x bigger than second, it's the exterior
        if polygons_sorted[0].area > 5 * polygons_sorted[1].area:
            polygons = polygons_sorted[1:]
    
    return gpd.GeoDataFrame({'geometry': polygons}, crs=streets.crs, geometry='geometry')


def rebuild_frontiers(
    streets: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    graph: nx.Graph
) -> List[FrontierEdge]:
    """Rebuild frontiers from current graph and blocks.
    
    Args:
        streets: Current streets GeoDataFrame
        blocks: Current blocks GeoDataFrame
        graph: Current NetworkX graph
        
    Returns:
        List of FrontierEdge objects (dead-end + block-edge frontiers)
    """
    frontiers = []
    
    # Dead-end frontiers: nodes with degree == 1
    for node in graph.nodes():
        if graph.degree[node] == 1:
            # Get the single neighbor
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor = neighbors[0]
                edge_data = graph.edges.get((node, neighbor), {})
                geometry = edge_data.get('geometry')
                
                if geometry:
                    # Create frontier ID
                    frontier_id = hashlib.sha256(
                        f"dead_end_{min(node, neighbor)}_{max(node, neighbor)}".encode()
                    ).hexdigest()[:16]
                    
                    frontier = FrontierEdge(
                        frontier_id=frontier_id,
                        edge_id=(node, neighbor),
                        block_id=None,
                        geometry=geometry,
                        frontier_type='dead_end',
                        expansion_weight=0.8,
                        spatial_hash=""
                    )
                    frontiers.append(frontier)
    
    # Block-edge frontiers would be added here based on block boundaries
    # (Simplified for this phase - can be extended later)
    
    return frontiers
