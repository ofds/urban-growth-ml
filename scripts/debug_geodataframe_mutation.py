#!/usr/bin/env python3
"""
Test GeoDataFrame mutation behavior to understand the replay bug.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_geodataframe_mutation():
    """Test how GeoDataFrame mutation works with .copy() and .loc assignment."""

    # Create a sample GeoDataFrame similar to streets
    data = {
        'u': ['node1', 'node2'],
        'v': ['node2', 'node3'],
        'geometry': [LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        'highway': ['residential', 'residential'],
        'length': [1.414, 1.414]
    }

    original_gdf = gpd.GeoDataFrame(data)
    logger.info(f"Original GDF length: {len(original_gdf)}")
    logger.info(f"Original GDF object ID: {id(original_gdf)}")

    # Test 1: Shallow copy behavior
    logger.info("\n=== TEST 1: Shallow copy behavior ===")
    copied_gdf = original_gdf.copy()
    logger.info(f"Copied GDF length: {len(copied_gdf)}")
    logger.info(f"Copied GDF object ID: {id(copied_gdf)}")
    logger.info(f"Are they the same object? {original_gdf is copied_gdf}")

    # Try to add a row using .loc
    new_index = len(copied_gdf)
    new_row = {
        'u': 'node3',
        'v': 'node4',
        'geometry': LineString([(2, 2), (3, 3)]),
        'highway': 'residential',
        'length': 1.414
    }

    logger.info(f"Adding row at index {new_index}")
    copied_gdf.loc[new_index] = new_row
    logger.info(f"After adding - copied GDF length: {len(copied_gdf)}")
    logger.info(f"Original GDF still has length: {len(original_gdf)}")

    # Test 2: Deep copy behavior
    logger.info("\n=== TEST 2: Deep copy behavior ===")
    deep_copied_gdf = original_gdf.copy(deep=True)
    logger.info(f"Deep copied GDF length: {len(deep_copied_gdf)}")
    logger.info(f"Deep copied GDF object ID: {id(deep_copied_gdf)}")

    new_index2 = len(deep_copied_gdf)
    deep_copied_gdf.loc[new_index2] = new_row
    logger.info(f"After adding to deep copy - length: {len(deep_copied_gdf)}")
    logger.info(f"Original GDF still has length: {len(original_gdf)}")

    # Test 3: Using concat instead of loc
    logger.info("\n=== TEST 3: Using concat instead of loc ===")
    concat_gdf = original_gdf.copy()
    new_row_gdf = gpd.GeoDataFrame([new_row])
    concat_result = gpd.GeoDataFrame(pd.concat([concat_gdf, new_row_gdf], ignore_index=True))
    concat_result.crs = concat_gdf.crs  # Preserve CRS
    concat_result.set_geometry('geometry', inplace=True)  # Ensure geometry column

    logger.info(f"After concat - result length: {len(concat_result)}")
    logger.info(f"Original GDF still has length: {len(original_gdf)}")

    # Test 4: Check column compatibility
    logger.info("\n=== TEST 4: Column compatibility check ===")
    logger.info(f"Original columns: {list(original_gdf.columns)}")
    logger.info(f"New row keys: {list(new_row.keys())}")

    # Check if all new_row keys are in original columns
    missing_cols = set(new_row.keys()) - set(original_gdf.columns)
    if missing_cols:
        logger.warning(f"Missing columns in new_row: {missing_cols}")

    extra_cols = set(original_gdf.columns) - set(new_row.keys())
    if extra_cols:
        logger.info(f"Extra columns in GDF not in new_row: {extra_cols}")

if __name__ == "__main__":
    test_geodataframe_mutation()
