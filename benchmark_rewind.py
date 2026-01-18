#!/usr/bin/env python3
"""
Performance benchmark for RewindEngine optimizations.
"""

import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from core.contracts import GrowthState, FrontierEdge
from inverse.rewind import RewindEngine
from inverse.data_structures import InverseGrowthAction, ActionType


def create_benchmark_city(num_streets=1000):
    """Create a larger city for benchmarking."""
    print(f'Creating benchmark city with {num_streets} streets...')

    streets_data = {'geometry': [], 'highway': [], 'osmid': [], 'u': [], 'v': []}
    graph = nx.Graph()

    # Create a grid-like structure
    for i in range(num_streets):
        x1, y1 = i * 10, 0
        x2, y2 = (i + 1) * 10, 0

        streets_data['geometry'].append(LineString([(x1, y1), (x2, y2)]))
        streets_data['highway'].append('primary')
        streets_data['osmid'].append(f'street_{i}')
        streets_data['u'].append(str(i))
        streets_data['v'].append(str(i + 1))

        graph.add_node(str(i), geometry=Point(x1, y1), x=x1, y=y1)
        graph.add_node(str(i + 1), geometry=Point(x2, y2), x=x2, y=y2)
        graph.add_edge(str(i), str(i + 1), geometry=streets_data['geometry'][-1])

    streets = gpd.GeoDataFrame(streets_data, crs='EPSG:4326')
    blocks = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

    # Create frontiers for the last few edges
    frontiers = []
    for i in range(max(1, num_streets - 10), num_streets):
        frontier = FrontierEdge(
            frontier_id=f'frontier_{i}',
            edge_id=(str(i), str(i + 1)),
            block_id=None,
            geometry=LineString([(i * 10, 0), ((i + 1) * 10, 0)]),
            frontier_type='dead_end' if i == num_streets - 1 else 'block_edge',
            expansion_weight=0.8 if i == num_streets - 1 else 0.5,
            spatial_hash='hash_0'
        )
        frontiers.append(frontier)

    city_bounds = Polygon([(-10, -10), (num_streets * 10 + 50, -10),
                          (num_streets * 10 + 50, 50), (-10, 50)])

    return GrowthState(
        streets=streets,
        blocks=blocks,
        frontiers=frontiers,
        graph=graph,
        iteration=0,
        city_bounds=city_bounds
    )


def benchmark_rewind_operations(sizes=[100, 500, 1000]):
    """Benchmark rewind operations at different scales."""
    print("=" * 60)
    print("REwindEngine Performance Benchmark")
    print("=" * 60)

    results = []

    for size in sizes:
        print(f"\nTesting with {size} streets...")

        # Create test city
        city = create_benchmark_city(size)
        rewind_engine = RewindEngine()

        # Test multiple rewind operations
        num_operations = min(10, size // 10)  # Test 10 operations or size/10, whichever is smaller
        total_time = 0
        successful_operations = 0

        for i in range(num_operations):
            # Create action for a random edge that exists
            edge_idx = size - i - 1  # Start from the end
            if edge_idx < 0:
                break

            action = InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=f'street_{edge_idx}',
                intent_params={},
                realized_geometry={'edgeid': (str(edge_idx), str(edge_idx + 1))}
            )

            start_time = time.perf_counter()
            result = rewind_engine.rewind_action(action, city)
            end_time = time.perf_counter()

            if len(result.streets) < len(city.streets):
                total_time += (end_time - start_time)
                successful_operations += 1
                city = result  # Update city for next operation

        if successful_operations > 0:
            avg_time = total_time / successful_operations
            print(".4f")
            results.append((size, avg_time, successful_operations))
        else:
            print(f"  No successful operations")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("<10")
    print("-" * 40)
    for size, avg_time, ops in results:
        print("<10")

    # Calculate scaling factor
    if len(results) >= 2:
        size1, time1, _ = results[0]
        size2, time2, _ = results[-1]
        scaling = (time2 / time1) / (size2 / size1)
        print(".2f")

    return results


if __name__ == '__main__':
    benchmark_rewind_operations()
