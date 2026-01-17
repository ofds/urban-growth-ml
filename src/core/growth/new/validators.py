"""Validation logic for proposed growth actions.

Pure validation functions that check geometric and topological validity
of proposed street additions without modifying state.
"""

import math
from typing import Tuple
import geopandas as gpd
from shapely.geometry import LineString, Point

from .actions import GrowthAction, ACTION_GROW_TRAJECTORY, ACTION_SUBDIVIDE_BLOCK
from .geometry_utils import calculate_angle_between_lines, point_to_line_distance
from src.core.contracts import GrowthState

# Validation constants
MIN_STREET_LENGTH = 10.0  # meters
MAX_STREET_LENGTH = 100.0  # meters
MIN_SPACING = 5.0  # meters
ENDPOINT_TOLERANCE = 1.0  # meters
MAX_EXTENSION_ANGLE = 150.0  # degrees


def validate_growth_action(action: GrowthAction, state: GrowthState) -> Tuple[bool, str]:
    """Main validation entry point for growth actions.
    
    Orchestrates all sub-validators to ensure proposed action is valid.
    
    Args:
        action: The proposed growth action to validate
        state: Current city growth state
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    # Validate basic geometry
    valid, reason = validate_geometry(action.proposed_geometry, state)
    if not valid:
        return False, reason
    
    # Validate street-street intersections
    valid, reason = validate_street_intersections(
        action.proposed_geometry, 
        state.streets
    )
    if not valid:
        return False, reason
    
    # Validate block interactions based on action type
    valid, reason = validate_block_interior_intersection(
        action.proposed_geometry,
        state.blocks,
        action.action_type
    )
    if not valid:
        return False, reason
    
    # Validate minimum spacing from parallel streets
    valid, reason = validate_minimum_spacing(
        action.proposed_geometry,
        state.streets,
        MIN_SPACING
    )
    if not valid:
        return False, reason
    
    # Validate angle constraint when extending from frontier
    if action.action_type == ACTION_GROW_TRAJECTORY:
        valid, reason = validate_angle_constraint(
            action.proposed_geometry,
            action.frontier_edge.geometry,
            MAX_EXTENSION_ANGLE
        )
        if not valid:
            return False, reason
    
    return True, "Valid"


def validate_geometry(geometry: LineString, state: GrowthState) -> Tuple[bool, str]:
    """Validate basic geometric properties of proposed street.
    
    Args:
        geometry: Proposed LineString geometry
        state: Current city growth state
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    # Check length constraints
    length = geometry.length
    if length < MIN_STREET_LENGTH:
        return False, f"Street too short: {length:.1f}m < {MIN_STREET_LENGTH}m"
    if length > MAX_STREET_LENGTH:
        return False, f"Street too long: {length:.1f}m > {MAX_STREET_LENGTH}m"
    
    # Check if within city bounds
    if not state.city_bounds.contains(geometry):
        return False, "Street extends outside city bounds"
    
    return True, "Valid"


def validate_street_intersections(
    proposed: LineString, 
    streets: gpd.GeoDataFrame
) -> Tuple[bool, str]:
    """Validate that proposed street doesn't illegally cross existing streets.
    
    Allows endpoint connections (future intersections) but forbids
    mid-segment crossings and overlaps.
    
    Args:
        proposed: Proposed street geometry
        streets: GeoDataFrame of existing streets
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    if streets.empty:
        return True, "Valid"
    
    for idx, street_row in streets.iterrows():
        existing = street_row.geometry
        
        # Check if geometries intersect
        if not proposed.intersects(existing):
            continue
        
        # If they intersect, check if it's a valid endpoint connection
        if is_endpoint_connection(proposed, existing, ENDPOINT_TOLERANCE):
            continue
        
        # Invalid intersection detected
        intersection = proposed.intersection(existing)
        return False, f"Illegal street crossing at {intersection}"
    
    return True, "Valid"


def validate_block_interior_intersection(
    proposed: LineString,
    blocks: gpd.GeoDataFrame,
    action_type: str
) -> Tuple[bool, str]:
    """Validate block interior crossing based on action type.
    
    For grow_trajectory: should NOT cross block interiors
    For subdivide_block: SHOULD cross block interior (required)
    
    Args:
        proposed: Proposed street geometry
        blocks: GeoDataFrame of existing blocks
        action_type: Type of action being validated
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    if blocks.empty:
        return True, "Valid"
    
    for idx, block_row in blocks.iterrows():
        block_geom = block_row.geometry
        
        # Check if proposed crosses block interior
        crosses_interior = proposed.crosses(block_geom) or proposed.within(block_geom)
        
        if action_type == ACTION_GROW_TRAJECTORY:
            # Grow trajectory should NOT cross block interiors
            if crosses_interior:
                return False, "Street crosses existing block interior"
        
        elif action_type == ACTION_SUBDIVIDE_BLOCK:
            # Subdivide REQUIRES crossing the block
            if crosses_interior:
                # This is what we want for subdivision
                continue
    
    # For subdivide_block, verify it actually crosses target block
    if action_type == ACTION_SUBDIVIDE_BLOCK:
        crosses_any = any(
            proposed.crosses(block_row.geometry) or proposed.within(block_row.geometry)
            for _, block_row in blocks.iterrows()
        )
        if not crosses_any:
            return False, "Subdivide action must cross block interior"
    
    return True, "Valid"


def validate_minimum_spacing(
    proposed: LineString,
    streets: gpd.GeoDataFrame,
    min_distance: float = MIN_SPACING
) -> Tuple[bool, str]:
    """Ensure new street maintains minimum spacing from parallel streets.
    
    Prevents streets from being too close together.
    
    Args:
        proposed: Proposed street geometry
        streets: GeoDataFrame of existing streets
        min_distance: Minimum allowed distance in meters
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    if streets.empty:
        return True, "Valid"
    
    for idx, street_row in streets.iterrows():
        existing = street_row.geometry
        
        # Skip if they connect at endpoints (valid intersections)
        if is_endpoint_connection(proposed, existing, ENDPOINT_TOLERANCE):
            continue
        
        # Check minimum distance
        distance = proposed.distance(existing)
        if distance < min_distance and distance > ENDPOINT_TOLERANCE:
            return False, f"Too close to existing street: {distance:.1f}m < {min_distance}m"
    
    return True, "Valid"


def validate_angle_constraint(
    proposed: LineString,
    connecting_edge: LineString,
    max_angle_deg: float = MAX_EXTENSION_ANGLE
) -> Tuple[bool, str]:
    """Validate that extension angle isn't too sharp (prevents hairpin turns).
    
    Args:
        proposed: Proposed street geometry
        connecting_edge: Frontier edge being extended from
        max_angle_deg: Maximum allowed angle in degrees
        
    Returns:
        Tuple of (is_valid, reason_message)
    """
    # Calculate angle in radians
    angle_rad = calculate_angle_between_lines(connecting_edge, proposed)
    angle_deg = math.degrees(angle_rad)
    
    if angle_deg > max_angle_deg:
        return False, f"Extension angle too sharp: {angle_deg:.1f}° > {max_angle_deg}°"
    
    return True, "Valid"


def is_endpoint_connection(
    proposed: LineString,
    existing: LineString,
    tolerance: float = ENDPOINT_TOLERANCE
) -> bool:
    """Check if proposed street connects to existing at endpoints only.
    
    Used to distinguish valid intersections from illegal crossings.
    
    Args:
        proposed: Proposed street geometry
        existing: Existing street geometry
        tolerance: Distance tolerance for endpoint matching
        
    Returns:
        True if connection is at endpoints only, False otherwise
    """
    # Get endpoints
    proposed_start = Point(proposed.coords[0])
    proposed_end = Point(proposed.coords[-1])
    existing_start = Point(existing.coords[0])
    existing_end = Point(existing.coords[-1])
    
    # Check all endpoint combinations
    connections = [
        proposed_start.distance(existing_start) < tolerance,
        proposed_start.distance(existing_end) < tolerance,
        proposed_end.distance(existing_start) < tolerance,
        proposed_end.distance(existing_end) < tolerance,
    ]
    
    return any(connections)
