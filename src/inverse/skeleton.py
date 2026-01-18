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
                 betweenness_threshold: float = 0.000001,  # Relative betweenness centrality (relaxed)
                 max_curvature: float = 0.70):   # Maximum curvature for straight streets (relaxed)
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
        candidates = []

        # Create mapping from street geometries to graph edges
        # Since streets GDF may use different node IDs than graph, we match by geometry
        street_to_graph_edges = self._map_streets_to_graph_edges(streets_gdf, graph)

        # Collect all candidates with scores
        for idx, street in streets_gdf.iterrows():
            geometry = street.geometry
            if not isinstance(geometry, LineString):
                continue

            # Cache length
            length = geometry.length
            if length < self.min_length:
                continue

            # Get the actual graph edge nodes for this street
            graph_edge = street_to_graph_edges.get(idx)
            if graph_edge is None:
                continue

            u, v = graph_edge

            # Check betweenness using actual graph node IDs
            edge_betweenness = betweenness.get(u, 0) + betweenness.get(v, 0)
            if edge_betweenness < self.betweenness_threshold:
                continue

            # Check curvature
            curvature = self._calculate_curvature(geometry)
            if curvature > self.max_curvature:
                continue

            # Calculate priority score based on highway type and metrics
            highway = street.get('highway', 'residential')
            highway_priority = self._get_highway_priority(highway)

            # Combined score: prioritize highway type, then betweenness, then length
            score = (highway_priority * 1000) + (edge_betweenness * 100) + (length / 10)

            candidates.append({
                'idx': idx,
                'u': u,  # Use actual graph node IDs
                'v': v,
                'geometry': geometry,
                'length': length,
                'betweenness': edge_betweenness,
                'curvature': curvature,
                'highway': highway,
                'score': score
            })

        # Sort by score and take top candidates (limit to reasonable skeleton size)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        max_skeleton_size = min(50, len(candidates))  # Limit skeleton size

        for candidate in candidates[:max_skeleton_size]:
            edge_key = (min(candidate['u'], candidate['v']), max(candidate['u'], candidate['v']))
            skeleton_edges.add(edge_key)
            skeleton_streets.append(candidate)

        logger.info(f"Extracted {len(skeleton_streets)} arterial skeleton streets from {len(candidates)} candidates")
        return skeleton_edges, skeleton_streets

    def _map_streets_to_graph_edges(self, streets_gdf, graph):
        """
        Map streets GeoDataFrame indices to actual graph edge node pairs.

        Since the streets GDF may use renumbered node IDs, we need to match
        streets to graph edges by geometry proximity.
        """
        from shapely.geometry import Point
        from shapely.wkt import loads
        import numpy as np

        mapping = {}

        # For each street, find the closest graph edge by geometry
        for idx, street in streets_gdf.iterrows():
            if not isinstance(street.geometry, LineString):
                continue

            street_geom = street.geometry
            street_start = Point(street_geom.coords[0])
            street_end = Point(street_geom.coords[-1])

            best_match = None
            best_distance = float('inf')

            # Check each graph edge
            for u, v, edge_data in graph.edges(data=True):
                edge_geom = edge_data.get('geometry')

                # Handle geometry stored as WKT string
                if edge_geom and isinstance(edge_geom, str):
                    try:
                        edge_geom = loads(edge_geom)
                    except:
                        continue

                if edge_geom and isinstance(edge_geom, LineString):
                    # Calculate distance between start/end points
                    graph_start = Point(edge_geom.coords[0])
                    graph_end = Point(edge_geom.coords[-1])

                    # Check if start/end points match (within tolerance)
                    start_dist = street_start.distance(graph_start)
                    end_dist = street_end.distance(graph_end)

                    # Also check reverse direction
                    rev_start_dist = street_start.distance(graph_end)
                    rev_end_dist = street_end.distance(graph_start)

                    min_dist = min(start_dist + end_dist, rev_start_dist + rev_end_dist)

                    if min_dist < best_distance and min_dist < 1.0:  # 1 meter tolerance
                        best_distance = min_dist
                        best_match = (u, v)

            if best_match:
                mapping[idx] = best_match

        logger.info(f"Mapped {len(mapping)} streets to graph edges out of {len(streets_gdf)} total streets")
        return mapping

    def _get_highway_priority(self, highway: str) -> int:
        """Get priority score for highway type (higher = more important)."""
        priority_map = {
            'motorway': 10,
            'trunk': 9,
            'primary': 8,
            'secondary': 7,
            'tertiary': 6,
            'unclassified': 5,
            'residential': 4,
            'living_street': 3,
            'service': 2,
            'track': 1
        }
        return priority_map.get(highway, 3)

    def _create_multi_seed_state(self, original_state) -> Any:
        """
        Create a multi-seed initial state using diverse, well-connected streets.

        Instead of just the first 2 streets, select streets that are:
        1. Well-distributed geographically
        2. High betweenness centrality
        3. Different highway types
        4. Connected to form a small network
        """
        import geopandas as gpd
        from shapely.geometry import Point
        import numpy as np

        streets_gdf = original_state.streets
        graph = original_state.graph

        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(graph, normalized=True)

        # Calculate city center
        city_center = self._get_city_center(streets_gdf)

        # Score streets for seed selection
        seed_candidates = []
        for idx, street in streets_gdf.iterrows():
            geometry = street.geometry
            if not isinstance(geometry, LineString):
                continue

            length = geometry.length
            if length < 10:  # Minimum length for seed streets
                continue

            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue

            # Betweenness score
            edge_betweenness = betweenness.get(u, 0) + betweenness.get(v, 0)

            # Distance from center (prefer peripheral streets for diversity)
            centroid = geometry.centroid
            distance_from_center = ((centroid.x - city_center[0])**2 + (centroid.y - city_center[1])**2)**0.5

            # Highway priority
            highway = street.get('highway', 'residential')
            highway_priority = self._get_highway_priority(highway)

            # Connectivity score (prefer streets connected to many others)
            connectivity = graph.degree(u) + graph.degree(v)

            # Combined score
            score = (highway_priority * 100) + (edge_betweenness * 50) + (connectivity * 10) + (distance_from_center * 0.1)

            seed_candidates.append({
                'idx': idx,
                'street': street,
                'score': score,
                'centroid': (centroid.x, centroid.y),
                'highway': highway
            })

        # Sort by score and select diverse seeds
        seed_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Select seeds ensuring geographic diversity
        selected_seeds = []
        min_distance = 100  # Minimum distance between seeds in meters

        for candidate in seed_candidates:
            # Check distance from already selected seeds
            too_close = False
            for selected in selected_seeds:
                dist = ((candidate['centroid'][0] - selected['centroid'][0])**2 +
                       (candidate['centroid'][1] - selected['centroid'][1])**2)**0.5
                if dist < min_distance:
                    too_close = True
                    break

            if not too_close:
                selected_seeds.append(candidate)
                if len(selected_seeds) >= 8:  # Limit to 8 diverse seeds
                    break

        # If we don't have enough diverse seeds, add more from different highway types
        if len(selected_seeds) < 4:
            highway_types = ['secondary', 'tertiary', 'residential']
            for highway_type in highway_types:
                if len(selected_seeds) >= 6:
                    break
                for candidate in seed_candidates:
                    if candidate['highway'] == highway_type and candidate not in selected_seeds:
                        selected_seeds.append(candidate)
                        break

        # Ensure we have at least 2 seeds
        if len(selected_seeds) < 2:
            selected_seeds = seed_candidates[:max(2, min(4, len(seed_candidates)))]

        # Create seed streets GeoDataFrame
        seed_streets = gpd.GeoDataFrame(
            [s['street'] for s in selected_seeds],
            crs=original_state.streets.crs
        )

        # Build graph from seed streets
        seed_graph = nx.Graph()
        for idx, street in seed_streets.iterrows():
            u, v = street['u'], street['v']
            geom = street.geometry

            if u not in seed_graph.nodes:
                seed_graph.add_node(u, geometry=Point(geom.coords[0]))
            if v not in seed_graph.nodes:
                seed_graph.add_node(v, geometry=Point(geom.coords[-1]))

            seed_graph.add_edge(u, v, geometry=geom, length=geom.length)

        # Create frontiers from dead-end nodes
        seed_frontiers = []
        for node in seed_graph.nodes():
            if seed_graph.degree(node) == 1:
                neighbors = list(seed_graph.neighbors(node))
                if neighbors:
                    neighbor = neighbors[0]
                    edge_data = seed_graph.edges.get((node, neighbor), {})
                    geometry = edge_data.get('geometry')

                    if geometry and isinstance(geometry, LineString) and geometry.is_valid:
                        frontier_id = hashlib.sha256(
                            f"seed_dead_end_{min(node, neighbor)}_{max(node, neighbor)}".encode()
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

        logger.info(f"Created multi-seed state with {len(seed_streets)} diverse streets and {len(seed_frontiers)} frontiers")
        logger.info(f"Seed highway types: {[s['highway'] for s in selected_seeds]}")

        return GrowthState(
            streets=seed_streets,
            blocks=gpd.GeoDataFrame(columns=['geometry'], crs=original_state.streets.crs),
            frontiers=seed_frontiers,
            graph=seed_graph,
            iteration=0,
            city_bounds=original_state.city_bounds
        )

    def _get_city_center(self, streets_gdf) -> Tuple[float, float]:
        """Calculate approximate city center."""
        centroids = []
        for idx, street in streets_gdf.iterrows():
            if hasattr(street.geometry, 'centroid'):
                centroid = street.geometry.centroid
                centroids.append((centroid.x, centroid.y))

        if centroids:
            x_coords = [c[0] for c in centroids]
            y_coords = [c[1] for c in centroids]
            return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        return (0.0, 0.0)

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
            # Multi-seed initialization: Use diverse, well-connected streets as seed
            logger.info("No skeleton found, using multi-seed initialization")
            seed_streets = self._create_multi_seed_state(original_state)

            return seed_streets
        
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
