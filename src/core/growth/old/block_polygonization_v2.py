#!/usr/bin/env python3
"""
Block Polygonization Module v2 - Optimized Geometric Approach
Phase 4 Fix: Mathematical Validation and Geometric Operations

CRITICAL FIX APPLIED:
- unary_union preprocessing to fix geometry normalization issues
- Exterior boundary removal for proper block detection
- Mode-based filtering (OSM vs procedural)

Optimized for performance using:
- Spatial indexing (STRtree) for O(log n) queries
- Vectorized operations where possible
- Removed dead code from abandoned cycle-based approach
- Efficient memory management
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

import geopandas as gpd
import pandas as pd
import numpy as np
import logging
import json
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree
from typing import Optional, List, Tuple
from pathlib import Path


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class GeometricBlockPolygonizer:
    """
    Optimized block polygonization using shapely.ops.polygonize.
    
    Key optimizations:
    - Spatial indexing for O(log n) proximity queries
    - Vectorized filtering operations
    - Minimal memory copies
    - Removed unused cycle-based methods
    
    CRITICAL FIXES:
    - unary_union preprocessing for geometry normalization
    - Exterior boundary removal
    - Mode-based filtering
    """
    
    # Class-level constants
    MIN_BLOCK_AREA = 100  # m²
    MAX_BLOCK_AREA = 100_000  # m²
    MIN_COMPACTNESS = 0.1
    OVERLAP_TOLERANCE = 1.0  # m²
    
    # Urban filter constants (will be overridden by mode)
    URBAN_MIN_AREA = 500  # m²
    URBAN_MAX_AREA = 50_000  # m²
    URBAN_MIN_COMPACTNESS = 0.3
    MAX_ASPECT_RATIO = 10
    
    def __init__(self, city_name: str, streets_gdf: Optional[gpd.GeoDataFrame] = None,
                 skip_street_overlap_filter: bool = False,
                 mode: str = "osm",
                 preprocess_network: bool = True):
        """
        Initialize block polygonizer.
        
        Args:
            city_name: Name of the city
            streets_gdf: Optional pre-loaded streets GeoDataFrame
            skip_street_overlap_filter: Skip street overlap check (for procedural growth)
            mode: "osm" for real data (strict filters) or "procedural" for generated cities (relaxed)
            preprocess_network: Apply unary_union preprocessing (CRITICAL FIX - recommended True)
        """
        self.city_name = city_name
        self.streets_path = Path(f'data/processed/{city_name}_streets.gpkg')
        self.output_path = Path(f'data/processed/{city_name}_blocks_v2.gpkg')
        self.streets_gdf = streets_gdf
        self.stats = {}
        self.skip_street_overlap_filter = skip_street_overlap_filter
        self.mode = mode
        self.preprocess_network = preprocess_network
        self.streets_path = DATA_DIR / f"{city_name}_streets.gpkg"
        self.output_path = DATA_DIR / f"{city_name}_blocks_v2.gpkg"

        # Spatial index cache
        self._spatial_index = None
        
        # Adjust filters based on mode
        if mode == "procedural":
            # Relaxed filters for procedural growth
            self.URBAN_MIN_AREA = 100  # Allow smaller blocks
            self.URBAN_MAX_AREA = 200_000  # Allow larger blocks
            self.URBAN_MIN_COMPACTNESS = 0.1  # Allow irregular shapes
            self.MAX_ASPECT_RATIO = 20  # Allow elongated blocks
            logger.info("Using PROCEDURAL mode filters (relaxed)")
        else:
            # Strict filters for OSM data
            self.URBAN_MIN_AREA = 500
            self.URBAN_MAX_AREA = 50_000
            self.URBAN_MIN_COMPACTNESS = 0.3
            self.MAX_ASPECT_RATIO = 10
            logger.info("Using OSM mode filters (strict)")
    
    def _build_spatial_index(self, geometries: gpd.GeoSeries) -> STRtree:
        """Build spatial index for fast proximity queries."""
        return STRtree(geometries)
    
    def _preprocess_street_network(self, streets: gpd.GeoDataFrame) -> List[LineString]:
        """
        Preprocess street network using unary_union to fix geometry issues.
        
        CRITICAL FIX: OSM data may have precision issues or tiny gaps that
        prevent polygonize() from detecting closed loops. unary_union normalizes
        geometries and ensures proper topology.
        
        This single operation fixes the "5,873 streets → 3 blocks" problem.
        
        Returns:
            List of cleaned/noded LineStrings ready for polygonize()
        """
        logger.info("Preprocessing street network with unary_union...")
        
        # Explode any MultiLineStrings
        streets_exploded = streets.explode(index_parts=False)
        
        # Extract LineStrings
        linestrings = [
            g for g in streets_exploded.geometry 
            if isinstance(g, LineString)
        ]
        
        logger.info(f"  Input: {len(linestrings)} LineStrings")
        
        # Apply unary_union - THE KEY FIX
        # It normalizes geometries and fixes precision/topology issues
        merged = unary_union(linestrings)
        
        # Extract noded LineStrings from result
        if hasattr(merged, 'geoms'):
            noded_lines = list(merged.geoms)
        else:
            noded_lines = [merged]
        
        additional_segments = len(noded_lines) - len(linestrings)
        logger.info(f"  Output: {len(noded_lines)} cleaned LineStrings "
                   f"(+{additional_segments} from normalization)")
        
        return noded_lines
    
    def _remove_exterior_boundary(self, polygons: List[Polygon]) -> List[Polygon]:
        """
        Remove the exterior boundary polygon from polygonize results.
        
        polygonize() creates ALL polygons including the unbounded exterior.
        For cities, this is typically the largest polygon by far.
        
        Detection heuristics:
        - If largest polygon is >5x the second-largest, it's the exterior
        - If largest polygon exceeds threshold, it's too large
        
        For procedural growth vs OSM, use different thresholds.
        """
        if len(polygons) <= 1:
            logger.warning("Only 1 polygon detected - likely exterior boundary with no interior blocks")
            return []
        
        # Sort by area descending
        sorted_polys = sorted(polygons, key=lambda p: p.area, reverse=True)
        
        largest_area = sorted_polys[0].area
        second_largest = sorted_polys[1].area if len(sorted_polys) > 1 else 0
        
        # Heuristic 1: Size ratio check
        if second_largest > 0:
            ratio = largest_area / second_largest
            
            if ratio > 5.0:
                logger.info(f"Removing exterior boundary (ratio test): "
                           f"{largest_area:.0f} m² vs {second_largest:.0f} m² "
                           f"(ratio: {ratio:.1f}x)")
                return sorted_polys[1:]
        
        # Heuristic 2: Absolute size check
        # Different thresholds for OSM vs procedural
        if self.mode == "procedural":
            threshold = 50_000  # 5 hectares for growing cities
        else:
            threshold = 100_000  # 10 hectares for real cities
        
        if largest_area > threshold:
            logger.info(f"Removing oversized polygon: {largest_area:.0f} m² (>{threshold} m²)")
            return sorted_polys[1:]
        
        # No obvious exterior detected
        logger.info(f"Keeping all {len(polygons)} polygons "
                   f"(largest: {largest_area:.0f} m²)")
        return polygons
    
    def _calculate_block_attributes(self, polygon: Polygon) -> dict:
        """Calculate attributes for a single block (vectorizable helper)."""
        area = polygon.area
        perimeter = polygon.length
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        centroid = polygon.centroid
        
        return {
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'centroid_x': centroid.x,
            'centroid_y': centroid.y,
            'geometry': polygon
        }
    
    def _filter_valid_blocks(self, polygons: List[Polygon]) -> gpd.GeoDataFrame:
        """
        Vectorized filtering of valid urban blocks.
        
        Performance: O(n) instead of O(n²) using vectorized operations.
        """
        if not polygons:
            logger.warning("No polygons to filter")
            return gpd.GeoDataFrame()
        
        # Calculate attributes for all polygons at once
        attributes = [self._calculate_block_attributes(p) for p in polygons]
        blocks_gdf = gpd.GeoDataFrame(attributes, geometry='geometry')
        
        # Vectorized filtering (much faster than iterrows)
        mask = (
            (blocks_gdf['area'] >= self.MIN_BLOCK_AREA) &
            (blocks_gdf['area'] <= self.MAX_BLOCK_AREA) &
            (blocks_gdf['compactness'] >= self.MIN_COMPACTNESS)
        )
        
        filtered = blocks_gdf[mask].copy()
        logger.info(f"Filtered blocks: {len(polygons)} → {len(filtered)} "
                   f"({len(filtered)/len(polygons)*100:.1f}% kept)")
        
        return filtered
    
    def _apply_urban_filters_optimized(self, blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Apply urban filters using spatial index for performance.
        
        Performance improvement: O(n log n) vs O(n×m) for street overlap checks.
        """
        if blocks_gdf.empty:
            return blocks_gdf
        
        logger.info("Applying urban filters with spatial indexing...")
        
        # 1. Size and compactness filters (vectorized)
        mask = (
            (blocks_gdf['area'] >= self.URBAN_MIN_AREA) &
            (blocks_gdf['area'] <= self.URBAN_MAX_AREA) &
            (blocks_gdf['compactness'] >= self.URBAN_MIN_COMPACTNESS)
        )
        filtered = blocks_gdf[mask].copy()
        logger.info(f"  Size/compactness filter: {len(blocks_gdf)} → {len(filtered)}")
        
        # 2. Aspect ratio filter (vectorized)
        bounds = filtered.bounds
        width = bounds['maxx'] - bounds['minx']
        height = bounds['maxy'] - bounds['miny']
        aspect_ratio = np.maximum(width, height) / np.maximum(np.minimum(width, height), 0.1)
        
        filtered = filtered[aspect_ratio <= self.MAX_ASPECT_RATIO].copy()
        logger.info(f"  Aspect ratio filter: {len(filtered)} blocks remaining")
        
        # 3. Water exclusion (if available)
        try:
            water = gpd.read_file(self.streets_path, layer='water')
            if not water.empty:
                # Use spatial index for fast water intersection checks
                water_index = self._build_spatial_index(water.geometry)
                
                def intersects_water(geom):
                    nearby = water_index.query(geom, predicate='intersects')
                    return len(nearby) > 0
                
                water_mask = ~filtered.geometry.apply(intersects_water)
                filtered = filtered[water_mask].copy()
                logger.info(f"  Water exclusion: {len(filtered)} blocks remaining")
        except Exception as e:
            logger.debug(f"  No water layer or error: {e}")
        
        # 4. Street overlap exclusion (optimized with spatial index)
        streets = self.streets_gdf if self.streets_gdf is not None else None
        
        if streets is None:
            try:
                streets = gpd.read_file(self.streets_path, layer='highways')
            except Exception as e:
                logger.debug(f"  No streets available for overlap check: {e}")
        
        # Skip street overlap filter for procedural growth (blocks are formed BY streets)
        if not self.skip_street_overlap_filter and streets is not None and not streets.empty:
            # Build spatial index for streets
            streets_index = self._build_spatial_index(streets.geometry)
            
            def has_significant_street_overlap(block_geom):
                """Check if block significantly overlaps streets using spatial index."""
                # Query only nearby streets (O(log n) instead of O(n))
                nearby_indices = streets_index.query(block_geom, predicate='intersects')
                
                if len(nearby_indices) == 0:
                    return False
                
                # For procedural growth, we expect blocks to be formed BY streets
                # So we check for interior overlaps (not just boundary touches)
                for idx in nearby_indices:
                    street_geom = streets.iloc[idx].geometry
                    intersection = block_geom.intersection(street_geom)
                    
                    # Only reject if street crosses through block interior (not just boundary)
                    if hasattr(intersection, 'length'):
                        # Check if intersection is mostly interior (not on boundary)
                        boundary_buff = block_geom.boundary.buffer(1.0)  # 1m tolerance
                        interior_intersection = intersection.difference(boundary_buff)
                        
                        if hasattr(interior_intersection, 'length') and interior_intersection.length > 5.0:
                            # Street crosses through interior, not just boundary
                            return True
                
                return False
            
            overlap_mask = ~filtered.geometry.apply(has_significant_street_overlap)
            filtered = filtered[overlap_mask].copy()
            logger.info(f"  Street overlap filter: {len(filtered)} blocks remaining")
        elif self.skip_street_overlap_filter:
            logger.info(f"  Street overlap filter: SKIPPED (procedural growth mode)")
        
        return filtered.reset_index(drop=True)
    
    def _calculate_adjacency_optimized(self, blocks_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate adjacency using spatial index for O(n log n) performance.
        
        Original: O(n²) - checks all pairs
        Optimized: O(n log n) - uses spatial index to check only nearby blocks
        """
        if blocks_gdf.empty or len(blocks_gdf) < 2:
            blocks_gdf['adjacent_blocks'] = [[] for _ in range(len(blocks_gdf))]
            blocks_gdf['shared_boundary_length'] = 0.0
            return blocks_gdf
        
        logger.info("Calculating adjacency with spatial index...")
        
        # Build spatial index once
        spatial_index = blocks_gdf.sindex
        
        adjacent_blocks = []
        shared_lengths = []
        
        for idx, block in blocks_gdf.iterrows():
            geom = block.geometry
            
            # Query only potentially intersecting blocks (O(log n))
            possible_neighbors_idx = list(spatial_index.intersection(geom.bounds))
            
            adjacent_ids = []
            shared_length = 0.0
            
            for neighbor_idx in possible_neighbors_idx:
                if neighbor_idx == idx:
                    continue
                
                neighbor_geom = blocks_gdf.iloc[neighbor_idx].geometry
                
                # Check if they actually touch (not just bounding boxes)
                if geom.touches(neighbor_geom):
                    intersection = geom.intersection(neighbor_geom)
                    
                    # Only count linear intersections (shared boundaries)
                    if hasattr(intersection, 'length'):
                        shared_length += intersection.length
                        adjacent_ids.append(neighbor_idx)
            
            adjacent_blocks.append(adjacent_ids)
            shared_lengths.append(shared_length)
        
        blocks_gdf = blocks_gdf.copy()
        blocks_gdf['adjacent_blocks'] = adjacent_blocks
        blocks_gdf['shared_boundary_length'] = shared_lengths
        
        avg_adjacent = np.mean([len(adj) for adj in adjacent_blocks])
        logger.info(f"Adjacency calculated: avg {avg_adjacent:.1f} neighbors per block")
        
        return blocks_gdf
    
    def _validate_results(self, blocks_gdf: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> bool:
        """
        Fast validation using spatial index for overlap checks.
        """
        if blocks_gdf.empty:
            logger.error("Validation failed: no blocks generated")
            return False
        
        logger.info("Running validation checks...")
        
        # 1. Check polygon validity (vectorized)
        invalid_count = (~blocks_gdf.is_valid).sum()
        if invalid_count > 0:
            logger.error(f"Validation failed: {invalid_count} invalid polygons")
            return False
        
        # 2. Check for overlaps using spatial index (O(n log n))
        if len(blocks_gdf) >= 2:
            spatial_index = self._build_spatial_index(blocks_gdf.geometry)
            
            overlap_found = False
            for idx, block in blocks_gdf.iterrows():
                # Query nearby blocks
                nearby_idx = list(spatial_index.query(block.geometry, predicate='intersects'))
                
                for other_idx in nearby_idx:
                    if other_idx <= idx:  # Avoid duplicate checks
                        continue
                    
                    intersection = block.geometry.intersection(
                        blocks_gdf.iloc[other_idx].geometry
                    )
                    
                    if hasattr(intersection, 'area') and intersection.area > self.OVERLAP_TOLERANCE:
                        logger.error(f"Overlap detected: blocks {idx} and {other_idx} "
                                   f"({intersection.area:.1f} m²)")
                        overlap_found = True
                        break
                
                if overlap_found:
                    break
            
            if overlap_found:
                return False
        
        # 3. Area coverage check
        street_network_area = streets.unary_union.convex_hull.area
        block_total_area = blocks_gdf['area'].sum()
        coverage_ratio = block_total_area / street_network_area
        
        # Reasonable coverage: 0.01% to 50%
        area_ok = 0.0001 <= coverage_ratio <= 0.5
        
        logger.info(f"Validation results:")
        logger.info(f"  ✓ All polygons valid")
        logger.info(f"  ✓ No overlaps detected")
        logger.info(f"  {'✓' if area_ok else '✗'} Coverage ratio: {coverage_ratio:.4f}")
        
        return area_ok
    
    def polygonize(self) -> gpd.GeoDataFrame:
        """
        Main polygonization method with preprocessing and exterior removal.
        
        CRITICAL FIXES APPLIED:
        1. unary_union preprocessing (if enabled)
        2. Exterior boundary removal
        3. Mode-based filtering
        """
        try:
            logger.info(f"Starting block polygonization for {self.city_name} (mode: {self.mode})")
            
            # 1. Load streets
            if self.streets_gdf is not None:
                streets = self.streets_gdf
                logger.info(f"Using provided streets: {len(streets)} features")
            else:
                streets = gpd.read_file(self.streets_path, layer='highways')
                logger.info(f"Loaded streets from file: {len(streets)} features")
                self.streets_gdf = streets
            
            if streets.empty:
                logger.error("No streets found")
                return gpd.GeoDataFrame()
            
            # 2. Extract/preprocess LineStrings
            if self.preprocess_network:
                line_strings = self._preprocess_street_network(streets)
            else:
                line_strings = [
                    geom for geom in streets.geometry 
                    if isinstance(geom, LineString)
                ]
                logger.info(f"Using raw streets: {len(line_strings)} LineStrings")
            
            if not line_strings:
                logger.error("No LineString geometries found")
                return gpd.GeoDataFrame()
            
            # 3. Polygonize
            logger.info(f"Polygonizing {len(line_strings)} LineStrings...")
            raw_polygons = list(polygonize(line_strings))
            
            if not raw_polygons:
                logger.error("⚠️  Polygonize created ZERO polygons!")
                logger.error("   → Street network has no closed loops")
                logger.error("   → Growth engine needs to create intersections/crossings")
                return gpd.GeoDataFrame()
            
            logger.info(f"Created {len(raw_polygons)} raw polygons")
            
            # DIAGNOSTIC: Log polygon distribution
            areas = sorted([p.area for p in raw_polygons], reverse=True)
            logger.info(f"  Area range: {areas[-1]:.0f} - {areas[0]:.0f} m²")
            if len(areas) >= 2:
                ratio = areas[0] / areas[1]
                logger.info(f"  Largest/2nd ratio: {ratio:.1f}x")
            
            # 4. Remove exterior boundary
            interior_polygons = self._remove_exterior_boundary(raw_polygons)
            
            if not interior_polygons:
                logger.warning("No interior blocks after removing exterior boundary")
                logger.warning("  → Street network might need more closed loops")
                return gpd.GeoDataFrame()
            
            logger.info(f"Kept {len(interior_polygons)} interior blocks")
            
            # 5. Filter to valid blocks
            blocks_gdf = self._filter_valid_blocks(interior_polygons)
            
            if blocks_gdf.empty:
                logger.error("No valid blocks after initial filtering")
                return gpd.GeoDataFrame()
            
            # Set CRS
            blocks_gdf.set_crs(streets.crs, inplace=True)
            
            # 6. Apply urban filters
            filtered = self._apply_urban_filters_optimized(blocks_gdf)
            
            if filtered.empty:
                logger.warning("No blocks remaining after urban filters")
                logger.warning(f"  Try mode='procedural' for relaxed filters")
                return gpd.GeoDataFrame()
            
            # 7. Calculate adjacency
            with_adjacency = self._calculate_adjacency_optimized(filtered)
            
            # 8. Add block_id
            with_adjacency.insert(0, 'block_id', range(len(with_adjacency)))
            
            # 9. Validate
            validation_passed = self._validate_results(with_adjacency, streets)
            
            if not validation_passed:
                logger.warning("Validation checks failed - results may be unreliable")
            
            # 10. Calculate statistics
            self._calculate_statistics(with_adjacency)
            
            self.blocks = with_adjacency
            logger.info(f"✓ Polygonization complete: {len(with_adjacency)} blocks")
            
            return with_adjacency
        
        except Exception as e:
            logger.error(f"Polygonization failed: {str(e)}", exc_info=True)
            raise
    
    def _calculate_statistics(self, blocks_gdf: gpd.GeoDataFrame):
        """Calculate comprehensive statistics (vectorized where possible)."""
        if blocks_gdf.empty:
            return
        
        adjacency_counts = [len(adj) for adj in blocks_gdf['adjacent_blocks']]
        
        self.stats = {
            'total_blocks': len(blocks_gdf),
            'total_area': float(blocks_gdf['area'].sum()),
            'average_block_area': float(blocks_gdf['area'].mean()),
            'median_block_area': float(blocks_gdf['area'].median()),
            'std_block_area': float(blocks_gdf['area'].std()),
            'min_block_area': float(blocks_gdf['area'].min()),
            'max_block_area': float(blocks_gdf['area'].max()),
            'average_compactness': float(blocks_gdf['compactness'].mean()),
            'total_perimeter': float(blocks_gdf['perimeter'].sum()),
            'average_perimeter': float(blocks_gdf['perimeter'].mean()),
            'adjacency_count': sum(adjacency_counts),
            'average_adjacent_blocks': float(np.mean(adjacency_counts)) if adjacency_counts else 0,
        }
        
        # Size distribution (vectorized)
        area_bins = [0, 500, 1000, 2000, 5000, 10000, float('inf')]
        area_labels = ['<500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '>10k']
        blocks_gdf['area_category'] = pd.cut(
            blocks_gdf['area'], 
            bins=area_bins, 
            labels=area_labels, 
            right=False
        )
        
        self.stats['size_distribution'] = blocks_gdf['area_category'].value_counts().to_dict()
        
        logger.info(f"Statistics: {self.stats['total_blocks']} blocks, "
                   f"avg area: {self.stats['average_block_area']:.1f} m², "
                   f"avg compactness: {self.stats['average_compactness']:.3f}")
    
    def save_blocks(self, blocks_gdf: gpd.GeoDataFrame):
        """Save blocks to GeoPackage."""
        try:
            logger.info(f"Saving {len(blocks_gdf)} blocks to {self.output_path}")
            
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare for saving
            save_gdf = blocks_gdf.copy()
            
            # Convert adjacency list to string
            save_gdf['adjacent_blocks_str'] = save_gdf['adjacent_blocks'].apply(
                lambda x: ','.join(map(str, x)) if x else ''
            )
            save_gdf = save_gdf.drop(columns=['adjacent_blocks'])
            
            # Drop area_category if it exists (already captured in stats)
            if 'area_category' in save_gdf.columns:
                save_gdf = save_gdf.drop(columns=['area_category'])
            
            # Save to GeoPackage
            save_gdf.to_file(self.output_path, driver='GPKG', layer='blocks')
            
            # Save statistics
            stats_path = self.output_path.parent / f'{self.city_name}_blocks_v2_stats.json'
            
            # Convert numpy types to native Python for JSON serialization
            stats_serializable = {
                k: (v.tolist() if isinstance(v, np.ndarray) else 
                    {str(k2): int(v2) for k2, v2 in v.items()} if isinstance(v, dict) else v)
                for k, v in self.stats.items()
            }
            
            with open(stats_path, 'w') as f:
                json.dump(stats_serializable, f, indent=2)
            
            logger.info(f"✓ Saved blocks and statistics")
        
        except Exception as e:
            logger.error(f"Error saving blocks: {str(e)}", exc_info=True)
            raise



def polygonize_all_cities_v2():
    """Run optimized geometric polygonization for all cities."""
    cities = ['piedmont_ca', 'brookline_ma', 'osasco_sp']
    
    logger.info("Starting optimized block polygonization for all cities...")
    
    results = {}
    for city in cities:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {city} with optimized approach")
            logger.info('='*60)
            
            polygonizer = GeometricBlockPolygonizer(
                city,
                mode="osm",
                preprocess_network=True  # ENABLE THE FIX
            )
            blocks = polygonizer.polygonize()
            
            if not blocks.empty:
                polygonizer.save_blocks(blocks)
                results[city] = {
                    'blocks': blocks,
                    'stats': polygonizer.stats,
                    'success': True
                }
                logger.info(f"✅ {city}: {len(blocks)} blocks generated")
            else:
                logger.error(f"❌ {city}: No blocks generated")
                results[city] = {'error': 'No blocks generated', 'success': False}
        
        except Exception as e:
            logger.error(f"❌ {city}: {str(e)}")
            results[city] = {'error': str(e), 'success': False}
    
    return results



if __name__ == "__main__":
    results = polygonize_all_cities_v2()
    
    print("\n" + "="*60)
    print("OPTIMIZED POLYGONIZATION SUMMARY")
    print("="*60)
    
    for city, result in results.items():
        if result.get('success'):
            stats = result['stats']
            print(f"✓ {city}: {stats['total_blocks']} blocks, "
                  f"avg area {stats['average_block_area']:.0f} m²")
        else:
            print(f"✗ {city}: {result.get('error', 'Unknown error')}")
