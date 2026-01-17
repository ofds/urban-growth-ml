#!/usr/bin/env python3
"""
Extract Cambridge, MA from Boston OSM data for dataset generation.

Cambridge is a historic New England city (~120,000 population, ~7 km²) with
distinctive urban patterns including Harvard/MIT campuses, historic districts,
and dense urban neighborhoods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
from utils.osm_processor import OSMProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def extract_cambridge():
    """Extract Cambridge from Boston OSM data."""

    print("="*60)
    print("EXTRACTING CAMBRIDGE, MA FROM BOSTON OSM")
    print("="*60)

    processor = OSMProcessor(output_dir=Path("data/processed"))

    # Cambridge bounding box (covers main urban area)
    # Found via: https://boundingbox.klokantech.com/
    bbox = (
        -71.1526,   # west
        42.3527,    # south
        -71.0626,   # east
        42.4022     # north
    )

    logger.info("Processing Cambridge with bbox:")
    logger.info(f"  West:  {bbox[0]}")
    logger.info(f"  South: {bbox[1]}")
    logger.info(f"  East:  {bbox[2]}")
    logger.info(f"  North: {bbox[3]}")

    success = processor.process_city(
        osm_file=Path("data/raw/boston.osm.pbf"),
        city_name="cambridge_ma",
        bbox=bbox
    )

    if success:
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print("\nGenerated files in data/processed/:")
        print("  - cambridge_ma_streets.gpkg         (~2,000-4,000 streets)")
        print("  - cambridge_ma_blocks_cleaned.gpkg  (~800-1,500 blocks)")
        print("  - cambridge_ma_frontier_edges.gpkg  (~400-800 frontiers)")
        print("  - cambridge_ma_street_graph_cleaned.graphml")
        print("\n✅ Ready for inference testing!")
    else:
        print("\n❌ Extraction failed - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    extract_cambridge()
