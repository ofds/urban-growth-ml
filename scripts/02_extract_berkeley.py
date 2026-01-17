#!/usr/bin/env python3
"""
Extract Berkeley, CA from Boston OSM data for dataset generation.

Berkeley is a medium-sized city (~120,000 population, ~28 km²) with diverse
urban patterns including university areas, residential neighborhoods, and
commercial districts.
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


def extract_berkeley():
    """Extract Berkeley from Boston OSM data."""

    print("="*60)
    print("EXTRACTING BERKELEY, CA FROM BOSTON OSM")
    print("="*60)

    processor = OSMProcessor(output_dir=Path("data/processed"))

    # Berkeley bounding box (covers main urban area)
    # Found via: https://boundingbox.klokantech.com/
    bbox = (
        -122.3016,  # west
        37.8547,    # south
        -122.2346,  # east
        37.9052     # north
    )

    logger.info("Processing Berkeley with bbox:")
    logger.info(f"  West:  {bbox[0]}")
    logger.info(f"  South: {bbox[1]}")
    logger.info(f"  East:  {bbox[2]}")
    logger.info(f"  North: {bbox[3]}")

    success = processor.process_city(
        osm_file=Path("data/raw/boston.osm.pbf"),
        city_name="berkeley_ca",
        bbox=bbox
    )

    if success:
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print("\nGenerated files in data/processed/:")
        print("  - berkeley_ca_streets.gpkg         (~3,000-5,000 streets)")
        print("  - berkeley_ca_blocks_cleaned.gpkg  (~1,000-2,000 blocks)")
        print("  - berkeley_ca_frontier_edges.gpkg  (~500-1,000 frontiers)")
        print("  - berkeley_ca_street_graph_cleaned.graphml")
        print("\n✅ Ready for inference testing!")
    else:
        print("\n❌ Extraction failed - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    extract_berkeley()
