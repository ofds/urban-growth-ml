#!/usr/bin/env python3
"""
OSM Data Processor for Urban Growth ML

Extracts and processes street networks from raw OSM PBF files.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union, polygonize
import osmnx as ox

logger = logging.getLogger(__name__)


class OSMProcessor:
    """Process raw OSM data into ML-ready format."""
    
    def __init__(self, output_dir: Path = Path("data/processed")):
        """
        Initialize OSM processor.
        
        Args:
            output_dir: Directory to save processed files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_city(
        self,
        osm_file: Path,
        city_name: str,
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> bool:
        """
        Process a city from OSM PBF file.
        
        Args:
            osm_file: Path to OSM PBF file
            city_name: Name for output files
            bbox: Optional bounding box (west, south, east, north)
        
        Returns:
            True if successful
        """
        logger.info(f"Processing {city_name} from {osm_file}")
        
        try:
            # Step 1: Extract street network
            logger.info("[1/5] Extracting street network...")
            streets_gdf, graph = self._extract_street_network(osm_file, bbox)
            
            if streets_gdf is None or streets_gdf.empty:
                logger.error("No streets extracted")
                return False
            
            logger.info(f"  Extracted {len(streets_gdf)} streets")
            
            # Step 2: Clean and simplify
            logger.info("[2/5] Cleaning street network...")
            streets_clean = self._clean_streets(streets_gdf)
            logger.info(f"  Cleaned to {len(streets_clean)} streets")
            
            # Step 3: Generate blocks
            logger.info("[3/5] Generating blocks...")
            blocks_gdf = self._generate_blocks(streets_clean)
            logger.info(f"  Generated {len(blocks_gdf)} blocks")
            
            # Step 4: Identify frontiers
            logger.info("[4/5] Identifying frontier edges...")
            frontiers_gdf = self._identify_frontiers(streets_clean, blocks_gdf, graph)
            logger.info(f"  Found {len(frontiers_gdf)} frontiers")
            
            # Step 5: Save all outputs
            logger.info("[5/5] Saving processed data...")
            self._save_outputs(city_name, streets_clean, blocks_gdf, frontiers_gdf, graph)
            
            logger.info(f"✅ Successfully processed {city_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to process {city_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_street_network(
    self,
    osm_file: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[gpd.GeoDataFrame, nx.MultiDiGraph]:
        """
        Extract street network from OSM file.
        
        Args:
            osm_file: Path to OSM PBF file
            bbox: Optional bounding box filter
        
        Returns:
            Tuple of (streets GeoDataFrame, NetworkX graph)
        """
        # Use OSMnx to load street network
        
        if bbox:
            # Load from bounding box
            graph = ox.graph_from_bbox(
                bbox=bbox,
                network_type='drive',
                simplify=True,
                retain_all=False
            )
        else:
            # Load from PBF file (requires custom polygon or place name)
            city_name = osm_file.stem.split('-')[0]
            
            try:
                graph = ox.graph_from_place(
                    city_name,
                    network_type='drive',
                    simplify=True
                )
            except Exception as e:
                logger.warning(f"Could not load by place name: {e}")
                raise NotImplementedError("Direct PBF loading requires bbox")
        
        # Convert to GeoDataFrame
        nodes_gdf, streets_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=True)
        
        # Ensure proper CRS
        if streets_gdf.crs is None:
            streets_gdf.set_crs('EPSG:4326', inplace=True)
        
        # Project to UTM for metric calculations (NEW METHOD)
        # Option 1: Project to appropriate UTM zone automatically
        streets_gdf = streets_gdf.to_crs(streets_gdf.estimate_utm_crs())
        
        # Also project the graph for consistency
        graph = ox.project_graph(graph, to_crs=streets_gdf.crs)
        
        return streets_gdf, graph

    def _clean_streets(self, streets_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Clean and simplify street network.
        
        Args:
            streets_gdf: Raw streets GeoDataFrame
        
        Returns:
            Cleaned streets GeoDataFrame
        """
        cleaned = streets_gdf.copy()
        
        # Filter to drivable streets only
        highway_types = ['motorway', 'trunk', 'primary', 'secondary', 
                        'tertiary', 'residential', 'living_street', 'unclassified']
        
        if 'highway' in cleaned.columns:
            # Handle lists of highway types
            def is_valid_highway(hw):
                if isinstance(hw, list):
                    return any(h in highway_types for h in hw)
                return hw in highway_types
            
            cleaned = cleaned[cleaned['highway'].apply(is_valid_highway)]
        
        # Remove very short segments (< 5m)
        cleaned = cleaned[cleaned.geometry.length > 5.0]
        
        # Simplify geometry (reduce vertices)
        cleaned['geometry'] = cleaned.geometry.simplify(tolerance=0.5, preserve_topology=True)
        
        # Reset index
        cleaned = cleaned.reset_index(drop=True)
        
        # Add edge IDs (u, v node identifiers)
        if 'u' not in cleaned.columns:
            cleaned['u'] = range(len(cleaned))
        if 'v' not in cleaned.columns:
            cleaned['v'] = range(1, len(cleaned) + 1)
        
        # Ensure string IDs
        cleaned['u'] = cleaned['u'].astype(str)
        cleaned['v'] = cleaned['v'].astype(str)
        
        return cleaned
    
    def _generate_blocks(self, streets_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Generate city blocks from street network.
        
        Args:
            streets_gdf: Cleaned streets GeoDataFrame
        
        Returns:
            Blocks GeoDataFrame
        """
        # Combine all street geometries
        all_streets = streets_gdf.geometry.unary_union
        
        # Polygonize to create blocks
        # This finds all closed areas bounded by streets
        blocks = list(polygonize(all_streets))
        
        if not blocks:
            logger.warning("No blocks generated - network might not be connected")
            return gpd.GeoDataFrame(columns=['geometry'], crs=streets_gdf.crs)
        
        # Filter tiny blocks (< 100 m²)
        blocks = [b for b in blocks if b.area > 100]
        
        # Create GeoDataFrame
        blocks_gdf = gpd.GeoDataFrame(
            {'geometry': blocks, 'block_id': range(len(blocks))},
            crs=streets_gdf.crs
        )
        
        return blocks_gdf
    
    def _identify_frontiers(
        self,
        streets_gdf: gpd.GeoDataFrame,
        blocks_gdf: gpd.GeoDataFrame,
        graph: nx.MultiDiGraph
    ) -> gpd.GeoDataFrame:
        """
        Identify frontier edges for growth simulation.
        
        Frontiers are:
        - Dead-end streets (degree 1 nodes)
        - Block-edge streets (adjacent to large blocks)
        
        Args:
            streets_gdf: Streets GeoDataFrame
            blocks_gdf: Blocks GeoDataFrame
            graph: Street network graph
        
        Returns:
            Frontiers GeoDataFrame
        """
        frontiers = []
        
        # Convert to undirected for degree calculation
        undirected = graph.to_undirected()
        
        # Find dead-end nodes (degree 1)
        dead_end_nodes = [n for n, d in undirected.degree() if d == 1]
        
        logger.debug(f"Found {len(dead_end_nodes)} dead-end nodes")
        
        # Identify dead-end streets
        for idx, street in streets_gdf.iterrows():
            u, v = str(street['u']), str(street['v'])
            
            # Check if either endpoint is a dead-end
            if u in dead_end_nodes or v in dead_end_nodes:
                frontiers.append({
                    'frontier_id': f'frontier_{len(frontiers)}',
                    'edge_id_u': u,
                    'edge_id_v': v,
                    'block_id': None,
                    'geometry': street.geometry,
                    'frontier_type': 'dead_end',
                    'expansion_weight': 1.0
                })
        
        # Identify block-edge frontiers (streets adjacent to large blocks)
        if not blocks_gdf.empty:
            large_blocks = blocks_gdf[blocks_gdf.geometry.area > 5000]  # > 5000 m²
            
            for idx, street in streets_gdf.iterrows():
                # Check if street borders a large block
                for block_idx, block in large_blocks.iterrows():
                    if street.geometry.intersects(block.geometry.boundary):
                        frontiers.append({
                            'frontier_id': f'frontier_{len(frontiers)}',
                            'edge_id_u': str(street['u']),
                            'edge_id_v': str(street['v']),
                            'block_id': block_idx,
                            'geometry': street.geometry,
                            'frontier_type': 'block_edge',
                            'expansion_weight': 0.8
                        })
                        break  # Only add once per street
        
        # Create GeoDataFrame
        if frontiers:
            frontiers_gdf = gpd.GeoDataFrame(frontiers, crs=streets_gdf.crs)
        else:
            frontiers_gdf = gpd.GeoDataFrame(
                columns=['frontier_id', 'edge_id_u', 'edge_id_v', 'block_id', 
                        'geometry', 'frontier_type', 'expansion_weight'],
                crs=streets_gdf.crs
            )
        
        return frontiers_gdf
    
    def _save_outputs(
        self,
        city_name: str,
        streets: gpd.GeoDataFrame,
        blocks: gpd.GeoDataFrame,
        frontiers: gpd.GeoDataFrame,
        graph: nx.MultiDiGraph
    ):
        """Save all processed outputs."""
        
        # Save streets
        streets_path = self.output_dir / f"{city_name}_streets.gpkg"
        streets.to_file(streets_path, driver='GPKG')
        logger.info(f"  Saved streets: {streets_path}")
        
        # Save blocks
        blocks_path = self.output_dir / f"{city_name}_blocks_cleaned.gpkg"
        blocks.to_file(blocks_path, driver='GPKG')
        logger.info(f"  Saved blocks: {blocks_path}")
        
        # Save frontiers
        frontiers_path = self.output_dir / f"{city_name}_frontier_edges.gpkg"
        frontiers.to_file(frontiers_path, driver='GPKG', layer='frontier_edges')
        logger.info(f"  Saved frontiers: {frontiers_path}")
        
        # Save graph (already projected in _extract_street_network)
        graph_path = self.output_dir / f"{city_name}_street_graph_cleaned.graphml"
        ox.save_graphml(graph, graph_path)
        logger.info(f"  Saved graph: {graph_path}")


def main():
    """Example usage."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = OSMProcessor()
    
    # Example: Process Piedmont, CA
    # Piedmont is a small city, perfect for testing
    success = processor.process_city(
        osm_file=Path("data/raw/piedmont.osm.pbf"),
        city_name="piedmont_ca",
        bbox=None  # Will try to auto-detect
    )
    
    if success:
        print("\n✅ City processed successfully!")
        print("\nGenerated files:")
        print("  - data/processed/piedmont_ca_streets.gpkg")
        print("  - data/processed/piedmont_ca_blocks_cleaned.gpkg")
        print("  - data/processed/piedmont_ca_frontier_edges.gpkg")
        print("  - data/processed/piedmont_ca_street_graph_cleaned.graphml")
    else:
        print("\n❌ Processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
