import math
from src.core.growth.new.geometry_utils import *
import networkx as nx
from shapely.geometry import Point, LineString

# Test 1: Canonical ID generation
assert generate_canonical_node_id(100.123, 200.456) == "100.1_200.5"
assert generate_canonical_node_id(100.06, 200.04) == "100.1_200.0"

# Test 2: Node finding with snapping
graph = nx.Graph()
graph.add_node("100.1_200.0", geometry=Point(100.1, 200.0))
node_id = find_or_create_node(Point(100.12, 200.03), graph, snap_tolerance=0.5)
assert node_id == "100.1_200.0"  # Snapped to existing

# Test 3: Length validation
line = LineString([(0, 0), (50, 0)])
valid, reason = validate_line_length(line, min_length=10, max_length=100)
assert valid == True

# Test 4: Angle calculation
line1 = LineString([(0, 0), (1, 0)])  # Horizontal
line2 = LineString([(0, 0), (0, 1)])  # Vertical
angle = calculate_angle_between_lines(line1, line2)
assert abs(angle - math.pi/2) < 0.01  # 90 degrees

print("All tests passed!")
