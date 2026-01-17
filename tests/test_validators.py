#!/usr/bin/env python3
"""
Validators Testing Script
Tests the validation functions for urban growth actions.
"""

import sys
import os
sys.path.append('.')

from src.core.growth.new.validators import *
from src.core.growth.new.actions import GrowthAction, ACTION_GROW_TRAJECTORY
from src.core.contracts import GrowthState, FrontierEdge
from shapely.geometry import LineString, Polygon
import geopandas as gpd

def test_validators():
    """Run all validator tests"""

    # Setup test state
    streets = gpd.GeoDataFrame({
        'geometry': [LineString([(0, 0), (10, 0)])]
    })
    blocks = gpd.GeoDataFrame({
        'geometry': [Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])]
    })
    city_bounds = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])

    state = GrowthState(
        streets=streets,
        blocks=blocks,
        frontiers=[],
        graph=None,
        iteration=0,
        city_bounds=city_bounds
    )

    print("Running validator tests...")

    # Test 1: Valid geometry
    print("Test 1: Valid geometry")
    proposed = LineString([(10, 0), (20, 0)])  # 10m long
    valid, reason = validate_geometry(proposed, state)
    assert valid == True, f"Expected valid geometry, got {valid}: {reason}"
    print("✓ Valid geometry test passed")

    # Test 2: Too short
    print("Test 2: Too short geometry")
    proposed = LineString([(0, 0), (5, 0)])  # 5m long
    valid, reason = validate_geometry(proposed, state)
    assert valid == False, f"Expected invalid geometry, got {valid}: {reason}"
    assert "short" in reason.lower(), f"Expected 'short' in reason, got: {reason}"
    print("✓ Too short geometry test passed")

    # Test 3: Valid endpoint connection
    print("Test 3: Valid endpoint connection")
    proposed = LineString([(10, 0), (20, 0)])  # Connects at (10,0)
    valid, reason = validate_street_intersections(proposed, streets)
    assert valid == True, f"Expected valid intersection, got {valid}: {reason}"
    print("✓ Valid endpoint connection test passed")

    # Test 4: Invalid crossing
    print("Test 4: Invalid crossing")
    proposed = LineString([(5, -5), (5, 5)])  # Crosses existing street
    valid, reason = validate_street_intersections(proposed, streets)
    assert valid == False, f"Expected invalid intersection, got {valid}: {reason}"
    assert "intersection" in reason.lower() or "crossing" in reason.lower(), f"Expected 'intersection' or 'crossing' in reason, got: {reason}"
    print("✓ Invalid crossing test passed")

    # Test 5: Full action validation
    print("Test 5: Full action validation")
    frontier = FrontierEdge(
        frontier_id="test",
        edge_id=("n1", "n2"),
        block_id=None,
        geometry=LineString([(0, 0), (10, 0)]),
        frontier_type="dead_end",
        expansion_weight=0.8,
        spatial_hash=""
    )

    action = GrowthAction(
        action_type=ACTION_GROW_TRAJECTORY,
        frontier_edge=frontier,
        proposed_geometry=LineString([(10, 0), (20, 0)]),
        parameters={}
    )

    valid, reason = validate_growth_action(action, state)
    assert valid == True, f"Expected valid action, got {valid}: {reason}"
    print("✓ Full action validation test passed")

    print("\n🎉 All validator tests passed!")

if __name__ == "__main__":
    test_validators()
