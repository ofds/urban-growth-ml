#!/usr/bin/env python3
"""
Extract Piedmont, CA from OSM data for testing.

Piedmont is a small city (11,000 population, ~5 km²) perfect for:
- Fast processing (~30 seconds)
- Testing inference pipeline
- Debugging ML training
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


def extract_piedmont():
    """Extract Piedmont from OSM data."""
    
    print("="*60)
    print("EXTRACTING PIEDMONT, CA FROM OSM")
    print("="*60)
    
    processor = OSMProcessor(output_dir=Path("data/processed"))
    
    # Piedmont bounding box (tight bounds)
    # Found via: https://boundingbox.klokantech.com/
    bbox = (
        -122.2516,  # west
        37.8097,    # south
        -122.2206,  # east
        37.8312     # north
    )
    
    logger.info("Processing Piedmont with bbox:")
    logger.info(f"  West:  {bbox[0]}")
    logger.info(f"  South: {bbox[1]}")
    logger.info(f"  East:  {bbox[2]}")
    logger.info(f"  North: {bbox[3]}")
    
    # Note: We don't need the PBF file if using OSMnx with bbox!
    # OSMnx downloads directly from Overpass API
    
    success = processor.process_city(
        osm_file=Path("data/raw/piedmont.osm.pbf"),  # Not actually used
        city_name="piedmont_ca",
        bbox=bbox
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ EXTRACTION COMPLETE")
        print("="*60)
        print("\nGenerated files in data/processed/:")
        print("  - piedmont_ca_streets.gpkg         (~500 streets)")
        print("  - piedmont_ca_blocks_cleaned.gpkg  (~300 blocks)")
        print("  - piedmont_ca_frontier_edges.gpkg  (~50 frontiers)")
        print("  - piedmont_ca_street_graph_cleaned.graphml")
        print("\n✅ Ready for inference testing!")
    else:
        print("\n❌ Extraction failed - check logs above")
        sys.exit(1)


if __name__ == '__main__':
    extract_piedmont()
