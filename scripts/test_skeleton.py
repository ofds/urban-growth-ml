#!/usr/bin/env python3

import geopandas as gpd
import networkx as nx
from src.inverse.skeleton import ArterialSkeletonExtractor
from shapely.geometry import LineString

# Load data
streets = gpd.read_file('scripts/data/processed/piedmont_ca_streets.gpkg')
graph = nx.read_graphml('scripts/data/processed/piedmont_ca_street_graph_cleaned.graphml')

# Calculate betweenness
betweenness = nx.betweenness_centrality(graph, normalized=True)

print('=== DEBUGGING SKELETON EXTRACTION ===')
print(f'Total streets: {len(streets)}')

# Check some sample streets
print('Sample streets analysis:')
sample_count = 0
for idx, street in streets.iterrows():
    if sample_count >= 20:
        break

    geometry = street.geometry
    if not isinstance(geometry, LineString):
        continue

    length = geometry.length
    u, v = street.get('u'), street.get('v')
    if u is None or v is None:
        continue

    edge_betweenness = betweenness.get(u, 0) + betweenness.get(v, 0)
    highway = street.get('highway', 'residential')

    # Show all streets that pass basic checks
    if length >= 5.0:
        curvature = ArterialSkeletonExtractor()._calculate_curvature(geometry)
        print(f'Street {idx}: {highway}, length={length:.1f}m, betweenness={edge_betweenness:.8f}, curvature={curvature:.3f}')
        sample_count += 1

print(f'\nGraph analysis:')
print(f'Nodes: {graph.number_of_nodes()}')
print(f'Edges: {graph.number_of_edges()}')
print(f'Is directed: {nx.is_directed(graph)}')

# For directed graphs, check weak connectivity
if nx.is_directed(graph):
    print(f'Is weakly connected: {nx.is_weakly_connected(graph)}')
    if not nx.is_weakly_connected(graph):
        components = list(nx.weakly_connected_components(graph))
        print(f'Weakly connected components: {len(components)}')
        component_sizes = [len(c) for c in components]
        print(f'Component sizes: {sorted(component_sizes, reverse=True)[:10]}')
else:
    print(f'Is connected: {nx.is_connected(graph)}')

# Check betweenness distribution
betweenness_values = list(betweenness.values())
non_zero_betweenness = [v for v in betweenness_values if v > 0]
print(f'Nodes with betweenness > 0: {len(non_zero_betweenness)}')
if non_zero_betweenness:
    print(f'Max betweenness: {max(non_zero_betweenness):.6f}')
    print(f'Mean betweenness: {sum(non_zero_betweenness)/len(non_zero_betweenness):.8f}')

# Try undirected betweenness
print(f'\n=== TESTING UNDIRECTED GRAPH ===')
undirected_graph = graph.to_undirected()
undirected_betweenness = nx.betweenness_centrality(undirected_graph, normalized=True)
undirected_values = list(undirected_betweenness.values())
undirected_non_zero = [v for v in undirected_values if v > 0]
print(f'Undirected - Nodes with betweenness > 0: {len(undirected_non_zero)}')
if undirected_non_zero:
    print(f'Undirected - Max betweenness: {max(undirected_non_zero):.6f}')
    print(f'Undirected - Mean betweenness: {sum(undirected_non_zero)/len(undirected_non_zero):.8f}')

# Debug node ID matching
print('\n=== DEBUGGING NODE ID MATCHING ===')
street_nodes = set()
for idx, street in streets.iterrows():
    u, v = street.get('u'), street.get('v')
    if u is not None and v is not None:
        street_nodes.add(u)
        street_nodes.add(v)

graph_nodes = set(graph.nodes())
print(f'Street nodes: {len(street_nodes)}')
print(f'Graph nodes: {len(graph_nodes)}')
print(f'Nodes in both: {len(street_nodes & graph_nodes)}')
print(f'Nodes only in streets: {len(street_nodes - graph_nodes)}')
print(f'Nodes only in graph: {len(graph_nodes - street_nodes)}')

# Check a few examples
print('\nSample node ID checks:')
for idx, street in streets.head(5).iterrows():
    u, v = street.get('u'), street.get('v')
    if u is not None and v is not None:
        u_in_graph = u in graph.nodes()
        v_in_graph = v in graph.nodes()
        u_betweenness = betweenness.get(u, 'NOT_FOUND')
        v_betweenness = betweenness.get(v, 'NOT_FOUND')
        print(f'Street {idx}: u={u} (in_graph={u_in_graph}, betweenness={u_betweenness}), v={v} (in_graph={v_in_graph}, betweenness={v_betweenness})')

# Check what the actual graph node IDs look like
print(f'\nActual graph node IDs (first 10): {list(graph.nodes())[:10]}')
print(f'Graph node types: {type(list(graph.nodes())[0])}')
print(f'Street u/v types: {type(streets.iloc[0]["u"])}')

# Check graph edges
print(f'\nGraph edges (first 5): {list(graph.edges())[:5]}')

# Check if there's a mapping in the graph
print(f'\nChecking for node mapping in graph...')
sample_graph_node = list(graph.nodes())[0]
if 'geometry' in graph.nodes[sample_graph_node]:
    print(f'Graph node {sample_graph_node} has geometry')
else:
    print(f'Graph node {sample_graph_node} has no geometry')

# Check if streets have osmid
if 'osmid' in streets.columns:
    print(f'Streets have osmid column')
    print(f'Sample osmids: {streets["osmid"].head().tolist()}')

# Debug the mapping function
print('\n=== DEBUGGING MAPPING FUNCTION ===')
extractor_debug = ArterialSkeletonExtractor()
mapping = extractor_debug._map_streets_to_graph_edges(streets, graph)
print(f'Mapping results: {len(mapping)} streets mapped')

# Check a few graph edges for geometry
print('\nChecking graph edge geometry:')
for i, (u, v, edge_data) in enumerate(graph.edges(data=True)):
    if i >= 5:
        break
    has_geom = 'geometry' in edge_data
    geom_type = type(edge_data.get('geometry')) if has_geom else 'None'
    print(f'Edge {u}-{v}: has_geometry={has_geom}, type={geom_type}')

# Test with very relaxed parameters
print('\n=== TESTING WITH VERY RELAXED PARAMETERS ===')
extractor_relaxed = ArterialSkeletonExtractor(betweenness_threshold=0.0, max_curvature=2.0)
skeleton_edges, skeleton_streets = extractor_relaxed.extract_skeleton(streets, graph)

print(f'Skeleton streets found (relaxed): {len(skeleton_streets)}')
if skeleton_streets:
    highway_types = [s['highway'] for s in skeleton_streets]
    print(f'Highway types in skeleton: {highway_types}')
    print('Top 5 skeleton streets by score:')
    for i, street in enumerate(skeleton_streets[:5]):
        print(f'  {i+1}. {street["highway"]}: length={street["length"]:.1f}m, betweenness={street["betweenness"]:.6f}, score={street["score"]:.1f}')

# Test with default parameters
print('\n=== TESTING WITH DEFAULT PARAMETERS ===')
extractor_default = ArterialSkeletonExtractor()
skeleton_edges, skeleton_streets = extractor_default.extract_skeleton(streets, graph)

print(f'Skeleton streets found: {len(skeleton_streets)}')
if skeleton_streets:
    highway_types = [s['highway'] for s in skeleton_streets]
    print(f'Highway types in skeleton: {highway_types}')
    print('Top 5 skeleton streets by score:')
    for i, street in enumerate(skeleton_streets[:5]):
        print(f'  {i+1}. {street["highway"]}: length={street["length"]:.1f}m, betweenness={street["betweenness"]:.6f}, score={street["score"]:.1f}')
