#!/usr/bin/env python3

from src.inverse.skeleton import ArterialSkeletonExtractor
import geopandas as gpd
import networkx as nx

# Load data
streets = gpd.read_file('scripts/data/processed/piedmont_ca_streets.gpkg')
graph = nx.read_graphml('scripts/data/processed/piedmont_ca_street_graph_cleaned.graphml')

# Test with default parameters
extractor = ArterialSkeletonExtractor()
skeleton_edges, skeleton_streets = extractor.extract_skeleton(streets, graph)

print(f'Default parameters - Skeleton streets found: {len(skeleton_streets)}')
if skeleton_streets:
    highway_types = [s['highway'] for s in skeleton_streets]
    print(f'Highway types in skeleton: {highway_types}')
    print('Top 5 skeleton streets by score:')
    for i, street in enumerate(skeleton_streets[:5]):
        print(f'  {i+1}. {street["highway"]}: length={street["length"]:.1f}m, betweenness={street["betweenness"]:.6f}, score={street["score"]:.1f}')
