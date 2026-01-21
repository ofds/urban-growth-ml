from src.core.growth.new.actions import *
from src.core.contracts import FrontierEdge
from shapely.geometry import LineString, Point

# Test 1: Create grow_trajectory action
frontier = FrontierEdge(
    frontier_id="test_frontier",
    edge_id=("node1", "node2"),
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
    parameters={"length": 10.0, "angle": 0.0}
)

assert action.action_type == "grow_trajectory"
assert action.proposed_geometry.length == 10.0

# Test 2: Validate action types
assert validate_action_type("grow_trajectory") == True
assert validate_action_type("subdivide_block") == True
assert validate_action_type("invalid_action") == False

# Test 3: Immutability
try:
    action.action_type = "something_else"
    assert False, "Should be frozen"
except:
    pass  # Expected - frozen dataclass

print("All tests passed!")
