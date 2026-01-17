import sys
import os
sys.path.append('.')

from src.core.growth.new.state_updater import *
from src.core.growth.new.actions import GrowthAction, ACTION_GROW_TRAJECTORY
from src.core.contracts import GrowthState, FrontierEdge
from shapely.geometry import LineString, Point, Polygon
import geopandas as gpd
import networkx as nx

# Setup initial state
streets = gpd.GeoDataFrame({
    'u': ['0.0_0.0'],
    'v': ['10.0_0.0'],
    'geometry': [LineString([(0, 0), (10, 0)])],
    'osmid': [-1],
    'highway': ['residential'],
    'length': [10.0]
}, crs='EPSG:32633')

graph = nx.Graph()
graph.add_node('0.0_0.0', geometry=Point(0, 0))
graph.add_node('10.0_0.0', geometry=Point(10, 0))
graph.add_edge('0.0_0.0', '10.0_0.0',
               geometry=LineString([(0, 0), (10, 0)]),
               length=10.0)

blocks = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:32633')
frontiers = []
city_bounds = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])

state = GrowthState(
    streets=streets,
    blocks=blocks,
    frontiers=frontiers,
    graph=graph,
    iteration=0,
    city_bounds=city_bounds
)

# Test: Apply grow_trajectory action
frontier = FrontierEdge(
    frontier_id="test",
    edge_id=('0.0_0.0', '10.0_0.0'),
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

# Apply action
new_state = apply_growth_action(action, state)

# Verify synchronization
assert len(new_state.streets) == len(state.streets) + 1  # +1 street
assert new_state.graph.number_of_edges() == state.graph.number_of_edges() + 1  # +1 edge
assert '20.0_0.0' in new_state.graph.nodes()  # New node added
assert new_state.streets.iloc[-1]['u'] == '10.0_0.0'  # Correct start
assert new_state.streets.iloc[-1]['v'] == '20.0_0.0'  # Correct end

# Verify immutability
assert len(state.streets) == 1  # Original unchanged
assert state.graph.number_of_edges() == 1  # Original unchanged

print("✅ All state synchronization tests passed!")
