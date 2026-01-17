"""
Geometry utilities for urban growth simulation.
Foundation module with zero external dependencies.
"""

import math
from typing import Tuple
import numpy as np
import networkx as nx
from shapely.geometry import Point, LineString


def generate_canonical_node_id(x: float, y: float) -> str:
    """
    Generate a canonical node ID from projected coordinates.
    
    Args:
        x: X coordinate (meters)
        y: Y coordinate (meters)
        
    Returns:
        Canonical ID string with 0.1m precision (e.g., "100.1_200.5")
    """
    snap_precision = 0.1  # 10cm precision
    x_rounded = round(x / snap_precision) * snap_precision
    y_rounded = round(y / snap_precision) * snap_precision
    return f"{x_rounded:.1f}_{y_rounded:.1f}"


def find_or_create_node(point: Point, graph: nx.Graph, snap_tolerance: float = 0.5) -> str:
    """
    Find existing node within snap_tolerance or return new canonical ID.
    
    Args:
        point: Point to snap
        graph: Current graph with nodes containing 'geometry' attribute
        snap_tolerance: Distance in meters to consider same node
        
    Returns:
        Node ID (existing node ID if found, or new canonical ID)
    """
    # Check for existing nearby nodes
    for node, data in graph.nodes(data=True):
        if 'geometry' in data:
            node_geom = data['geometry']
            if isinstance(node_geom, Point):
                distance = point.distance(node_geom)
                if distance <= snap_tolerance:
                    return node
    
    # No nearby node found, create new canonical ID
    new_id = generate_canonical_node_id(point.x, point.y)
    return new_id


def validate_line_length(
    geometry: LineString, 
    min_length: float = 10.0, 
    max_length: float = 100.0
) -> Tuple[bool, str]:
    """
    Check if LineString length is within bounds.
    
    Args:
        geometry: LineString to validate
        min_length: Minimum acceptable length in meters
        max_length: Maximum acceptable length in meters
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not isinstance(geometry, LineString):
        return False, "Geometry is not a LineString"
    
    length = geometry.length
    
    if length < min_length:
        return False, f"Too short: {length:.2f}m < {min_length:.2f}m"
    
    if length > max_length:
        return False, f"Too long: {length:.2f}m > {max_length:.2f}m"
    
    return True, "Valid"


def calculate_angle_between_lines(line1: LineString, line2: LineString) -> float:
    """
    Calculate angle in radians between two lines' directions.
    
    Uses first→last point vectors to determine line direction.
    
    Args:
        line1: First LineString
        line2: Second LineString
        
    Returns:
        Angle in radians in range [0, π]
    """
    # Get direction vectors (first → last point)
    coords1 = list(line1.coords)
    coords2 = list(line2.coords)
    
    vec1 = np.array([coords1[-1][0] - coords1[0][0], coords1[-1][1] - coords1[0][1]])
    vec2 = np.array([coords2[-1][0] - coords2[0][0], coords2[-1][1] - coords2[0][1]])
    
    # Normalize vectors
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    
    # Calculate angle using dot product
    dot_product = np.dot(vec1_norm, vec2_norm)
    
    # Clamp to [-1, 1] to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Return angle in [0, π]
    angle = math.acos(dot_product)
    
    return angle


def point_to_line_distance(point: Point, line: LineString) -> float:
    """
    Calculate minimum distance from point to line.
    
    Args:
        point: Point geometry
        line: LineString geometry
        
    Returns:
        Minimum distance in same units as input geometries
    """
    return point.distance(line)
