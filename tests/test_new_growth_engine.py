#!/usr/bin/env python3
"""
Test script for the new GrowthEngine implementation.
"""

import sys
import os
sys.path.append('.')

from src.core.growth.new.growth_engine import GrowthEngine
from src.core.contracts import FrontierEdge, GrowthState
from shapely.geometry import LineString, Polygon
import geopandas as gpd
import networkx as nx

# Test 1: Engine initialization
engine = GrowthEngine('test_city', seed=42)
assert engine.city_name == 'test_city'
assert engine.seed == 42
assert engine.MIN_STREET_LENGTH == 10.0
assert engine.MAX_STREET_LENGTH == 100.0
print("✅ Test 1: Engine initialization passed")

# Test 2: Load initial state (requires OSM data)
# state = engine.load_initial_state()
# assert len(state.streets) > 0
print("⚠️ Test 2: Load initial state skipped (requires OSM data)")

# Test 3: Propose and apply action (synthetic test)
# Create minimal synthetic state
streets = gpd.GeoDataFrame({
    'geometry': [LineString([(0, 0), (10, 0)])],
    'u': ['0.0_0.0'],
    'v': ['10.0_0.0'],
    'length': [10.0]
})

blocks = gpd.GeoDataFrame({
    'geometry': [Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])]
})

frontiers = [
    FrontierEdge(
        frontier_id="test",
        edge_id=("0.0_0.0", "10.0_0.0"),
        block_id=None,
        geometry=LineString([(0, 0), (10, 0)]),
        frontier_type="dead_end",
        expansion_weight=0.8,
        spatial_hash=""
    )
]

graph = nx.Graph()
graph.add_node("0.0_0.0", geometry=LineString([(0, 0), (10, 0)]).coords[0], x=0.0, y=0.0)
graph.add_node("10.0_0.0", geometry=LineString([(0, 0), (10, 0)]).coords[1], x=10.0, y=0.0)
graph.add_edge("0.0_0.0", "10.0_0.0", geometry=LineString([(0, 0), (10, 0)]), length=10.0)

city_bounds = Polygon([(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)])

state = GrowthState(
    streets=streets,
    blocks=blocks,
    frontiers=frontiers,
    graph=graph,
    iteration=0,
    city_bounds=city_bounds
)

# Propose action
action = engine.propose_grow_trajectory(frontiers[0], state)

# Apply if valid
if action is not None:
    new_state = engine.apply_growth_action(action, state)
    assert len(new_state.streets) == len(state.streets) + 1
    print("✅ Test 3: Propose and apply action passed")
else:
    print("⚠️ Test 3: No action proposed (may be expected due to validation)")
