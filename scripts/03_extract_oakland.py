#!/usr/bin/env python3
"""
Extract Oakland, CA from Boston OSM data for dataset generation.

Oakland is a larger city (~420,000 population, ~78 km²) with diverse
urban patterns including port areas, residential neighborhoods,
commercial districts, and industrial zones.
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


def extract_oakland():
    """Extract Oakland from Boston OSM data."""

    print("="*60)
    print("EXTRACTING OAKLAND, CA FROM BOSTON OSM")
    print("="*60)

    processor = OSMProcessor(output_dir=Path("data/processed"))

    # Oakland bounding box (covers main urban area)
    # Found via: https://boundingbox.klokantech.com/
    bbox = (
        -122.3216,  # west
        37.7467,    # south
        -122.1146,  # east
        37.8542     # north
    )

    logger.info("Processing Oakland with bbox:")
    logger.info(f"  West:  {bbox[0]}")
    logger.info(f"  South: {bbox[1]}")
    logger.info(f"  East:  {bbox[2]}")
    logger.info(f"  North: {bbox[3]}")

    success = processor.process_city(
        osm_file=Path("data/raw/boston.osm.pbf"),
        city_name="oakland_ca",
        bbox=bbox
    )

    if success:
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print("\nGenerated files in data/processed/:")
        print("  - oakland_ca_streets.gpkg         (~8,000-12,000 streets)")
        print("  - oakland_ca_blocks_cleaned.gpkg  (~3,000-5,000 blocks)")
        print("  - oakland_ca_frontier_edges.gpkg  (~1,000-2,000 frontiers)")
        print("  - oakland_ca_street_graph_cleaned.graphml")
        print("\n✅ Ready for inference testing!")
    else:
        print("\n❌ Extraction failed - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    extract_oakland()
