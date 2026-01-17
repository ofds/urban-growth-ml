#!/usr/bin/env python3
"""
Arterial Skeleton Extraction
Phase A: Identify and freeze high-level infrastructure as initial conditions.
"""

import networkx as nx
from typing import List, Set, Tuple, Any
from shapely.geometry import LineString
import logging
import math

logger = logging.getLogger(__name__)


class ArterialSkeletonExtractor:
    """
    Extracts arterial skeleton from street network.
    
    Identifies high-betweenness, long, low-curvature streets that represent
    major infrastructure, treating them as pre-existing rather than inferred.
    """
    
    def __init__(self,
                 min_length: float = 5.0,      # Minimum length for arterial consideration
                 betweenness_threshold: float = 0.00001,  # Relative betweenness centrality
                 max_curvature: float = 0.40):   # Maximum curvature for straight streets
        self.min_length = min_length
        self.betweenness_threshold = betweenness_threshold
        self.max_curvature = max_curvature
    
    def extract_skeleton(self, streets_gdf, graph: nx.Graph) -> Tuple[Set[str], List[dict]]:
        """
        Extract arterial skeleton from street network.
        
        Args:
            streets_gdf: GeoDataFrame of streets
            graph: NetworkX graph of street network
        
        Returns:
            Tuple of (skeleton_edge_ids, skeleton_street_data)
        """
        logger.info("Extracting arterial skeleton...")
        
        # Calculate betweenness centrality once
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        
        skeleton_edges = set()
        skeleton_streets = []
        
        # Pre-filter by length first (cheapest check)
        for idx, street in streets_gdf.iterrows():
            geometry = street.geometry
            if not isinstance(geometry, LineString):
                continue
            
            # Cache length
            length = geometry.length
            if length < self.min_length:
                continue
            
            # Get nodes
            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue
            
            # Check betweenness (second cheapest)
            edge_betweenness = betweenness.get(u, 0) + betweenness.get(v, 0)
            if edge_betweenness < self.betweenness_threshold:
                continue
            
            # Check curvature last (most expensive)
            curvature = self._calculate_curvature(geometry)
            if curvature > self.max_curvature:
                continue
            
            # Qualified as arterial
            edge_key = (min(u, v), max(u, v))
            skeleton_edges.add(edge_key)
            skeleton_streets.append({
                'idx': idx,
                'u': u,
                'v': v,
                'geometry': geometry,
                'length': length,
                'betweenness': edge_betweenness,
                'curvature': curvature,
                'highway': street.get('highway', 'arterial')
            })
        
        logger.info(f"Extracted {len(skeleton_streets)} arterial skeleton streets")
        return skeleton_edges, skeleton_streets
    
    def _calculate_curvature(self, geometry: LineString) -> float:
        """
        Calculate simplified curvature metric for a street geometry.
        
        Returns a value where 0 = perfectly straight, higher = more curved.
        """
        coords = list(geometry.coords)
        if len(coords) < 3:
            return 0.0
        
        # Cache endpoints
        start_point = coords[0]
        end_point = coords[-1]
        
        # Vector from start to end
        dx_total = end_point[0] - start_point[0]
        dy_total = end_point[1] - start_point[1]
        total_length_sq = dx_total**2 + dy_total**2
        
        if total_length_sq == 0:
            return 0.0
        
        total_length = total_length_sq**0.5
        
        # Normalized reference vector
        ref_dx = dx_total / total_length
        ref_dy = dy_total / total_length
        
        total_angle = 0.0
        cumulative_length = 0.0
        
        # Optimized: single pass with cached calculations
        for i in range(1, len(coords) - 1):
            p1, p2, p3 = coords[i-1], coords[i], coords[i+1]
            
            # Segment vectors
            dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
            dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
            
            # Lengths
            len1 = (dx1**2 + dy1**2)**0.5
            len2 = (dx2**2 + dy2**2)**0.5
            
            if len1 > 0 and len2 > 0:
                # Normalized vectors
                vec1_dx, vec1_dy = dx1 / len1, dy1 / len1
                vec2_dx, vec2_dy = dx2 / len2, dy2 / len2
                
                # Dot products with reference (clamped)
                dot1 = max(-1.0, min(1.0, ref_dx * vec1_dx + ref_dy * vec1_dy))
                dot2 = max(-1.0, min(1.0, ref_dx * vec2_dx + ref_dy * vec2_dy))
                
                # Angular deviation
                angle1 = math.acos(dot1)
                angle2 = math.acos(dot2)
                
                total_angle += abs(angle1) + abs(angle2)
                cumulative_length += (len1 + len2) * 0.5
        
        return total_angle / max(cumulative_length, 1.0)
    
    def create_skeleton_state(self, skeleton_streets: List[dict], original_state) -> Any:
        """
        Create a minimal GrowthState containing only the arterial skeleton.
        
        Args:
            skeleton_streets: List of skeleton street data
            original_state: Original GrowthState for reference
        
        Returns:
            Minimal GrowthState with skeleton as initial conditions
        """
        import geopandas as gpd
        import pandas as pd
        from src.core.contracts import GrowthState, FrontierEdge
        import hashlib
        from shapely.geometry import LineString
        
        if not skeleton_streets:
            # No skeleton, return minimal state with just first 2 streets
            logger.info("No skeleton found, using first 2 streets as seed")
            seed_streets = original_state.streets.iloc[:2].copy()
            
            # Create graph with only seed streets
            import networkx as nx
            seed_graph = nx.Graph()
            
            # Build graph from seed streets
            for idx, street in seed_streets.iterrows():
                u, v = street['u'], street['v']
                geom = street.geometry
                
                if u not in seed_graph.nodes:
                    from shapely.geometry import Point
                    seed_graph.add_node(u, geometry=Point(geom.coords[0]))
                if v not in seed_graph.nodes:
                    seed_graph.add_node(v, geometry=Point(geom.coords[-1]))
                
                seed_graph.add_edge(u, v, geometry=geom, length=geom.length)
            
            # Create frontiers from dead-end nodes
            seed_frontiers = []
            for node in seed_graph.nodes():
                if seed_graph.degree(node) == 1:
                    # Get the edge geometry
                    neighbors = list(seed_graph.neighbors(node))
                    if neighbors:
                        neighbor = neighbors[0]
                        edge_data = seed_graph.edges.get((node, neighbor), {})
                        geometry = edge_data.get('geometry')
                        
                        if geometry and isinstance(geometry, LineString) and geometry.is_valid:
                            frontier_id = hashlib.sha256(
                                f"dead_end_{min(node, neighbor)}_{max(node, neighbor)}".encode()
                            ).hexdigest()[:16]
                            
                            frontier = FrontierEdge(
                                frontier_id=frontier_id,
                                edge_id=(node, neighbor),
                                block_id=None,
                                geometry=geometry,
                                frontier_type="dead_end",
                                expansion_weight=0.8,
                                spatial_hash=""
                            )
                            seed_frontiers.append(frontier)
            
            logger.info(f"Created seed state with {len(seed_streets)} streets and {len(seed_frontiers)} frontiers")
            
            return GrowthState(
                streets=seed_streets,
                blocks=gpd.GeoDataFrame(columns=['geometry'], crs=original_state.streets.crs),
                frontiers=seed_frontiers,
                graph=seed_graph,
                iteration=0,
                city_bounds=original_state.city_bounds
            )
        
        # Extract skeleton streets into GeoDataFrame
        skeleton_gdf = gpd.GeoDataFrame(
            [s for s in skeleton_streets],
            crs=original_state.streets.crs
        )
        
        # Build graph from skeleton streets
        import networkx as nx
        skeleton_graph = nx.Graph()
        
        for idx, street in skeleton_gdf.iterrows():
            u, v = street['u'], street['v']
            geom = street['geometry']
            
            if u not in skeleton_graph.nodes:
                from shapely.geometry import Point
                skeleton_graph.add_node(u, geometry=Point(geom.coords[0]))
            if v not in skeleton_graph.nodes:
                skeleton_graph.add_node(v, geometry=Point(geom.coords[-1]))
            
            skeleton_graph.add_edge(u, v, geometry=geom, length=geom.length)
        
        # Create frontiers from dead-end nodes
        skeleton_frontiers = []
        for node in skeleton_graph.nodes():
            if skeleton_graph.degree(node) == 1:
                neighbors = list(skeleton_graph.neighbors(node))
                if neighbors:
                    neighbor = neighbors[0]
                    edge_data = skeleton_graph.edges.get((node, neighbor), {})
                    geometry = edge_data.get('geometry')
                    
                    if geometry and isinstance(geometry, LineString) and geometry.is_valid:
                        frontier_id = hashlib.sha256(
                            f"dead_end_{min(node, neighbor)}_{max(node, neighbor)}".encode()
                        ).hexdigest()[:16]
                        
                        frontier = FrontierEdge(
                            frontier_id=frontier_id,
                            edge_id=(node, neighbor),
                            block_id=None,
                            geometry=geometry,
                            frontier_type="dead_end",
                            expansion_weight=0.8,
                            spatial_hash=""
                        )
                        skeleton_frontiers.append(frontier)
        
        logger.info(f"Created skeleton state with {len(skeleton_gdf)} arterial streets and {len(skeleton_frontiers)} frontiers")
        
        return GrowthState(
            streets=skeleton_gdf,
            blocks=gpd.GeoDataFrame(columns=['geometry'], crs=original_state.streets.crs),
            frontiers=skeleton_frontiers,
            graph=skeleton_graph,
            iteration=0,
            city_bounds=original_state.city_bounds
        )
