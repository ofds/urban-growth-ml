#!/usr/bin/env python3
"""
Comprehensive profiling script for RewindEngine optimizations.
Generates detailed performance reports and identifies bottlenecks.
"""

import cProfile
import pstats
import time
import sys
import os
from pathlib import Path
from memory_profiler import profile, memory_usage
from line_profiler import LineProfiler
import io
import tracemalloc

sys.path.insert(0, str(Path.cwd() / 'src'))

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from core.contracts import GrowthState, FrontierEdge
from inverse.rewind import RewindEngine
from inverse.data_structures import InverseGrowthAction, ActionType


def create_realistic_city(num_streets=1000, branching_factor=0.3):
    """
    Create a more realistic city with branching streets and varied topology.

    Args:
        num_streets: Target number of streets
        branching_factor: Probability of creating branches (0.0-1.0)
    """
    print(f'Creating realistic city with ~{num_streets} streets...')

    streets_data = {'geometry': [], 'highway': [], 'osmid': [], 'u': [], 'v': []}
    graph = nx.Graph()

    # Start with a main artery
    current_id = 0
    streets_data['geometry'].append(LineString([(0, 0), (100, 0)]))
    streets_data['highway'].append('primary')
    streets_data['osmid'].append(f'street_{current_id}')
    streets_data['u'].append(str(current_id))
    streets_data['v'].append(str(current_id + 1))
    graph.add_node(str(current_id), geometry=Point(0, 0), x=0, y=0)
    graph.add_node(str(current_id + 1), geometry=Point(100, 0), x=100, y=0)
    graph.add_edge(str(current_id), str(current_id + 1), geometry=streets_data['geometry'][-1])
    current_id += 2

    # Grow the network
    import random
    random.seed(42)  # For reproducible results

    active_frontiers = [(100, 0)]  # (x, y) coordinates of active frontier points

    while len(streets_data['geometry']) < num_streets and active_frontiers:
        # Pick a random frontier point
        frontier_idx = random.randint(0, len(active_frontiers) - 1)
        fx, fy = active_frontiers[frontier_idx]

        # Create 1-3 streets from this point
        num_branches = random.randint(1, 3)
        for i in range(num_branches):
            if len(streets_data['geometry']) >= num_streets:
                break

            # Random direction and length
            angle = random.uniform(-3.14159/2, 3.14159/2)  # -90 to +90 degrees
            length = random.uniform(50, 150)

            tx = fx + length * 0.9 * random.uniform(0.8, 1.2) * (1 if random.random() > 0.5 else -1)
            ty = fy + length * 0.9 * random.uniform(0.8, 1.2) * (1 if random.random() > 0.5 else -1)

            # Avoid going backwards or too close to existing nodes
            if abs(tx - fx) < 10 and abs(ty - fy) < 10:
                continue

            streets_data['geometry'].append(LineString([(fx, fy), (tx, ty)]))
            streets_data['highway'].append('residential')
            streets_data['osmid'].append(f'street_{current_id}')
            streets_data['u'].append(str(current_id))
            streets_data['v'].append(str(current_id + 1))

            graph.add_node(str(current_id), geometry=Point(fx, fy), x=fx, y=fy)
            graph.add_node(str(current_id + 1), geometry=Point(tx, ty), x=tx, y=ty)
            graph.add_edge(str(current_id), str(current_id + 1), geometry=streets_data['geometry'][-1])

            # Add new frontier point
            if random.random() < branching_factor:
                active_frontiers.append((tx, ty))

            current_id += 2

        # Remove used frontier
        active_frontiers.pop(frontier_idx)

    streets = gpd.GeoDataFrame(streets_data, crs='EPSG:4326')
    blocks = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

    # Create frontiers for recent edges
    frontiers = []
    recent_edges = list(graph.edges())[-min(20, len(graph.edges())):]

    for i, (u, v) in enumerate(recent_edges):
        edge_data = graph.get_edge_data(u, v)
        geometry = edge_data.get('geometry')
        if geometry:
            frontier = FrontierEdge(
                frontier_id=f'frontier_{i}',
                edge_id=(u, v),
                block_id=None,
                geometry=geometry,
                frontier_type='dead_end' if graph.degree[u] == 1 or graph.degree[v] == 1 else 'block_edge',
                expansion_weight=0.8 if graph.degree[u] == 1 or graph.degree[v] == 1 else 0.5,
                spatial_hash=""
            )
            frontiers.append(frontier)

    # Calculate bounds
    all_coords = []
    for geom in streets.geometry:
        all_coords.extend(list(geom.coords))

    if all_coords:
        min_x = min(x for x, y in all_coords) - 50
        max_x = max(x for x, y in all_coords) + 50
        min_y = min(y for x, y in all_coords) - 50
        max_y = max(y for x, y in all_coords) + 50
        city_bounds = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    else:
        city_bounds = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])

    return GrowthState(
        streets=streets,
        blocks=blocks,
        frontiers=frontiers,
        graph=graph,
        iteration=0,
        city_bounds=city_bounds
    )


def profile_memory_usage(func, *args, **kwargs):
    """Profile memory usage of a function."""
    def wrapper():
        return func(*args, **kwargs)

    mem_usage = memory_usage(wrapper, interval=0.1, timeout=None, max_usage=True, retval=True)
    max_mem, result = mem_usage
    return max_mem, result


def profile_with_cprofile(func, *args, **kwargs):
    """Profile function execution with cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Get stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    # Capture output
    output = io.StringIO()
    stats.print_stats(20)  # Top 20 functions
    stats_str = output.getvalue()

    return result, stats_str


def profile_with_line_profiler(func, *args, **kwargs):
    """Profile function execution with line-by-line profiling."""
    profiler = LineProfiler()
    profiler.add_function(func)

    # Add key methods from RewindEngine
    from inverse.rewind import RewindEngine
    profiler.add_function(RewindEngine.rewind_action)
    profiler.add_function(RewindEngine._build_edge_index)
    profiler.add_function(RewindEngine._update_frontiers_delta)
    profiler.add_function(RewindEngine._rebuild_frontiers_simple)

    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    # Capture output
    output = io.StringIO()
    profiler.print_stats(stream=output)
    stats_str = output.getvalue()

    return result, stats_str


def benchmark_rewind_operations_detailed(sizes=[100, 500, 1000, 2000]):
    """Comprehensive benchmarking with multiple profiling methods."""

    print("=" * 80)
    print("COMPREHENSIVE REWINDENGINE PROFILING REPORT")
    print("=" * 80)

    results = []

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"PROFILING CITY WITH {size} STREETS")
        print(f"{'='*60}")

        # Create test city
        city = create_realistic_city(size)
        rewind_engine = RewindEngine()

        # Test multiple rewind operations
        num_operations = min(20, max(5, size // 100))  # Scale with city size
        print(f"Performing {num_operations} rewind operations...")

        # Prepare actions (use the most recent edges)
        actions = []
        recent_edges = list(city.graph.edges())[-num_operations:]
        for i, (u, v) in enumerate(recent_edges):
            action = InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id=f'street_{i}',
                intent_params={},
                realized_geometry={'edgeid': (u, v)}
            )
            actions.append(action)

        # Profile 1: Time-based profiling
        print("\n1. TIME PROFILING:")
        start_time = time.perf_counter()

        successful_operations = 0
        for action in actions:
            result = rewind_engine.rewind_action(action, city)
            if len(result.streets) < len(city.streets):
                successful_operations += 1
                city = result  # Update for next operation

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / successful_operations if successful_operations > 0 else 0

        print(".6f")
        print(".6f")
        print(f"  Success rate: {successful_operations}/{len(actions)} ({100*successful_operations/len(actions):.1f}%)")

        # Profile 2: Memory usage
        print("\n2. MEMORY PROFILING:")
        tracemalloc.start()
        city = create_realistic_city(size)  # Fresh city

        peak_memory = 0
        for action in actions[:5]:  # Test first 5 for memory
            snapshot1 = tracemalloc.get_traced_memory()[1]
            result = rewind_engine.rewind_action(action, city)
            snapshot2 = tracemalloc.get_traced_memory()[1]
            peak_memory = max(peak_memory, snapshot2 - snapshot1)
            if len(result.streets) < len(city.streets):
                city = result

        tracemalloc.stop()
        print(f"  Peak memory per operation: {peak_memory / 1024:.1f} KB")

        # Profile 3: Function-level profiling (sample run)
        if size <= 1000:  # Only for smaller sizes to avoid too much output
            print("\n3. FUNCTION-LEVEL PROFILING (sample run):")
            city = create_realistic_city(size)
            action = actions[0]

            def single_rewind():
                return rewind_engine.rewind_action(action, city)

            _, cprofile_stats = profile_with_cprofile(single_rewind)
            print("  Top 10 functions by cumulative time:")
            lines = cprofile_stats.split('\n')
            for line in lines[1:12]:  # Skip header, show top 10
                if line.strip():
                    print(f"    {line}")

        # Store results
        results.append({
            'size': size,
            'total_time': total_time,
            'avg_time': avg_time,
            'successful_ops': successful_operations,
            'peak_memory_kb': peak_memory / 1024
        })

    # Generate summary report
    print(f"\n{'='*80}")
    print("PROFILING SUMMARY REPORT")
    print(f"{'='*80}")

    print("<10")
    print("-" * 70)
    for result in results:
        print("<10")

    # Calculate scaling analysis
    if len(results) >= 2:
        print(f"\nSCALING ANALYSIS:")
        size1, time1 = results[0]['size'], results[0]['avg_time']
        size2, time2 = results[-1]['size'], results[-1]['avg_time']
        scaling_factor = (time2 / time1) / (size2 / size1)
        print(".2f")
        if scaling_factor > 1.5:
            print("  ⚠️  POTENTIAL O(n²) BEHAVIOR DETECTED")
        elif scaling_factor > 1.1:
            print("  ⚠️  Potential O(n log n) or worse behavior")
        else:
            print("  ✅  Appears to scale linearly or better")

    return results


def identify_hotspots():
    """Identify specific code hotspots using line profiling."""
    print(f"\n{'='*60}")
    print("HOTSPOT ANALYSIS")
    print(f"{'='*60}")

    # Create medium-sized city for detailed analysis
    city = create_realistic_city(500)
    rewind_engine = RewindEngine()

    # Get one action
    recent_edge = list(city.graph.edges())[-1]
    action = InverseGrowthAction(
        action_type=ActionType.EXTEND_FRONTIER,
        target_id='test',
        intent_params={},
        realized_geometry={'edgeid': recent_edge}
    )

    def profile_rewind():
        return rewind_engine.rewind_action(action, city)

    # Line profiling
    _, line_stats = profile_with_line_profiler(profile_rewind)

    print("LINE-BY-LINE PROFILING RESULTS:")
    print("(Only showing functions with significant time)")

    # Parse and display key insights
    lines = line_stats.split('\n')
    current_function = None
    for line in lines:
        if 'Function:' in line:
            current_function = line.split('Function:')[1].strip()
        elif line.strip() and not line.startswith(' ') and '%' in line:
            # This is a function timing line
            parts = line.split()
            if len(parts) >= 6:
                time_pct = parts[0]
                try:
                    if float(time_pct.rstrip('%')) > 1.0:  # Show functions > 1% time
                        print(f"  {time_pct} {current_function}")
                except ValueError:
                    pass

    return line_stats


if __name__ == '__main__':
    # Run comprehensive profiling
    results = benchmark_rewind_operations_detailed()

    # Run hotspot analysis
    hotspot_stats = identify_hotspots()

    # Save detailed results
    output_dir = Path('outputs/profiling')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'profiling_report.txt', 'w') as f:
        f.write("COMPREHENSIVE PROFILING REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Scaling Results:\n")
        for result in results:
            f.write(f"Size {result['size']}: {result['avg_time']:.6f}s avg, {result['peak_memory_kb']:.1f}KB peak\n")
        f.write("\nHotspot Analysis:\n")
        f.write(hotspot_stats)

    print(f"\nDetailed report saved to: {output_dir / 'profiling_report.txt'}")
    print("\nProfiling complete! Analyze the results above to identify optimization targets.")
