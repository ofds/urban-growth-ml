#!/usr/bin/env python3
"""
Extract Savannah, GA from Georgia OSM data for dataset generation.

Savannah is a historic Southern city (~250,000 population, ~108 km²) with
unique urban patterns including its distinctive grid layout, historic district,
and coastal geography.
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


def extract_savannah():
    """Extract Savannah from Georgia OSM data."""

    print("="*60)
    print("EXTRACTING SAVANNAH, GA FROM GEORGIA OSM")
    print("="*60)

    processor = OSMProcessor(output_dir=Path("data/processed"))

    # Savannah bounding box (covers main urban area)
    # Found via: https://boundingbox.klokantech.com/
    bbox = (
        -81.1216,   # west
        31.9847,    # south
        -80.9846,   # east
        32.0922     # north
    )

    logger.info("Processing Savannah with bbox:")
    logger.info(f"  West:  {bbox[0]}")
    logger.info(f"  South: {bbox[1]}")
    logger.info(f"  East:  {bbox[2]}")
    logger.info(f"  North: {bbox[3]}")

    success = processor.process_city(
        osm_file=Path("data/raw/georgia-260109.osm.pbf"),
        city_name="savannah_ga",
        bbox=bbox
    )

    if success:
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print("\nGenerated files in data/processed/:")
        print("  - savannah_ga_streets.gpkg         (~4,000-7,000 streets)")
        print("  - savannah_ga_blocks_cleaned.gpkg  (~1,500-3,000 blocks)")
        print("  - savannah_ga_frontier_edges.gpkg  (~600-1,200 frontiers)")
        print("  - savannah_ga_street_graph_cleaned.graphml")
        print("\n✅ Ready for inference testing!")
    else:
        print("\n❌ Extraction failed - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    extract_savannah()
