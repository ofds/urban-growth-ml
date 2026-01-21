#!/usr/bin/env python3
"""
Unit tests for SpatialIndex module.
"""

import unittest
import sys
import os
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inverse.spatial.spatial_index import SpatialIndex


class TestSpatialIndex(unittest.TestCase):
    """Test cases for SpatialIndex class."""

    def setUp(self):
        """Set up test fixtures."""
        self.spatial_index = SpatialIndex()

        # Create test street data
        self.test_streets = pd.DataFrame({
            'geometry': [
                LineString([(0, 0), (10, 0)]),  # Horizontal street
                LineString([(5, 5), (5, 15)]),  # Vertical street
                LineString([(10, 10), (20, 20)]),  # Diagonal street
            ],
            'u': [1, 2, 3],
            'v': [2, 3, 4],
            'highway': ['residential', 'primary', 'secondary']
        })

        # Create test frontier data
        self.test_frontiers = [
            type('MockFrontier', (), {
                'geometry': LineString([(0, 0), (5, 0)]),
                'frontier_id': 'f1',
                'frontier_type': 'dead_end'
            })(),
            type('MockFrontier', (), {
                'geometry': LineString([(5, 5), (10, 5)]),
                'frontier_id': 'f2',
                'frontier_type': 'block_edge'
            })()
        ]

        # Create test block data
        self.test_blocks = pd.DataFrame({
            'geometry': [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # Square block
                Polygon([(15, 15), (25, 15), (25, 25), (15, 25)]),  # Another square block
            ]
        })

    def test_initialization(self):
        """Test SpatialIndex initialization."""
        spatial_index = SpatialIndex()
        self.assertIsNone(spatial_index.street_index)
        self.assertIsNone(spatial_index.frontier_index)
        self.assertIsNone(spatial_index.block_index)
        self.assertEqual(len(spatial_index.street_data), 0)
        self.assertEqual(len(spatial_index.frontier_data), 0)
        self.assertEqual(len(spatial_index.block_data), 0)

    def test_build_street_index(self):
        """Test building street index."""
        self.spatial_index.build_street_index(self.test_streets)

        self.assertIsNotNone(self.spatial_index.street_index)
        self.assertEqual(len(self.spatial_index.street_data), 3)

        # Check that streets are indexed
        for idx in self.test_streets.index:
            self.assertIn(idx, self.spatial_index.street_data)

    def test_build_frontier_index(self):
        """Test building frontier index."""
        self.spatial_index.build_frontier_index(self.test_frontiers)

        self.assertIsNotNone(self.spatial_index.frontier_index)
        self.assertEqual(len(self.spatial_index.frontier_data), 2)

    def test_build_block_index(self):
        """Test building block index."""
        self.spatial_index.build_block_index(self.test_blocks)

        self.assertIsNotNone(self.spatial_index.block_index)
        self.assertEqual(len(self.spatial_index.block_data), 2)

    def test_find_nearby_streets(self):
        """Test finding streets near a point."""
        self.spatial_index.build_street_index(self.test_streets)

        # Test point near first street
        point = Point(2, 1)  # Near (0,0) to (10,0) street
        nearby = self.spatial_index.find_nearby_streets(point, 2.0)

        self.assertGreater(len(nearby), 0)
        # Should find the horizontal street
        found_horizontal = any(
            street.geometry.equals(LineString([(0, 0), (10, 0)]))
            for street in nearby
        )
        self.assertTrue(found_horizontal)

    def test_find_nearby_streets_no_index(self):
        """Test finding streets when no index is built."""
        point = Point(0, 0)
        nearby = self.spatial_index.find_nearby_streets(point, 10.0)
        self.assertEqual(len(nearby), 0)

    def test_find_streets_intersecting_bbox(self):
        """Test finding streets intersecting a bounding box."""
        self.spatial_index.build_street_index(self.test_streets)

        # Bbox that intersects first street
        bbox = (0, -1, 5, 1)  # Covers part of horizontal street
        intersecting = self.spatial_index.find_streets_intersecting_bbox(bbox)

        self.assertGreater(len(intersecting), 0)

    def test_find_streets_intersecting_bbox_no_index(self):
        """Test finding streets intersecting bbox when no index is built."""
        bbox = (0, 0, 10, 10)
        intersecting = self.spatial_index.find_streets_intersecting_bbox(bbox)
        self.assertEqual(len(intersecting), 0)

    def test_find_frontiers_near_point(self):
        """Test finding frontiers near a point."""
        self.spatial_index.build_frontier_index(self.test_frontiers)

        # Point near first frontier
        point = Point(2, 0.5)
        nearby = self.spatial_index.find_frontiers_near_point(point, 2.0)

        self.assertGreater(len(nearby), 0)
        # Should find the first frontier
        found_f1 = any(f.frontier_id == 'f1' for f in nearby)
        self.assertTrue(found_f1)

    def test_find_frontiers_near_point_no_index(self):
        """Test finding frontiers when no index is built."""
        point = Point(0, 0)
        nearby = self.spatial_index.find_frontiers_near_point(point, 10.0)
        self.assertEqual(len(nearby), 0)

    def test_find_blocks_containing_point(self):
        """Test finding blocks containing a point."""
        self.spatial_index.build_block_index(self.test_blocks)

        # Point inside first block
        point = Point(5, 5)
        containing = self.spatial_index.find_blocks_containing_point(point)

        self.assertGreater(len(containing), 0)

        # Point outside all blocks
        point_outside = Point(50, 50)
        containing_outside = self.spatial_index.find_blocks_containing_point(point_outside)
        self.assertEqual(len(containing_outside), 0)

    def test_find_blocks_containing_point_no_index(self):
        """Test finding blocks containing point when no index is built."""
        point = Point(5, 5)
        containing = self.spatial_index.find_blocks_containing_point(point)
        self.assertEqual(len(containing), 0)

    def test_remove_street(self):
        """Test removing a street from index."""
        self.spatial_index.build_street_index(self.test_streets)

        # Verify street exists
        street_id = 0
        self.assertIn(street_id, self.spatial_index.street_data)

        # Remove street
        street_geom = self.test_streets.loc[street_id].geometry
        self.spatial_index.remove_street(street_id, street_geom)

        # Verify street is removed
        self.assertNotIn(street_id, self.spatial_index.street_data)

    def test_remove_street_no_index(self):
        """Test removing street when no index exists."""
        # Should not raise error
        self.spatial_index.remove_street(0, LineString([(0, 0), (1, 1)]))
        # Nothing to verify since no index exists

    def test_update_frontiers_incremental(self):
        """Test incremental frontier updates."""
        self.spatial_index.build_frontier_index(self.test_frontiers)

        # Create new frontier to add
        new_frontier = type('MockFrontier', (), {
            'geometry': LineString([(20, 20), (25, 20)]),
            'frontier_id': 2,  # Use integer ID for R-tree
            'frontier_type': 'dead_end'
        })()

        # Update with empty removals and one addition
        self.spatial_index.update_frontiers_incremental([], [new_frontier])

        # Should contain the new frontier
        self.assertIn(2, self.spatial_index.frontier_data)

    def test_update_blocks_incremental(self):
        """Test incremental block updates."""
        self.spatial_index.build_block_index(self.test_blocks)

        # Create new block to add
        new_block = pd.Series({
            'geometry': Polygon([(30, 30), (40, 30), (40, 40), (30, 40)])
        })

        # Update with empty removals and one addition
        self.spatial_index.update_blocks_incremental([], [new_block])

        # Should contain the new block
        self.assertIn(hash(new_block.geometry.wkt), self.spatial_index.block_data)

    def test_empty_geodataframe_handling(self):
        """Test handling of empty GeoDataFrames."""
        empty_streets = pd.DataFrame(columns=['geometry'])
        empty_blocks = pd.DataFrame(columns=['geometry'])

        # Should not raise errors
        self.spatial_index.build_street_index(empty_streets)
        self.spatial_index.build_block_index(empty_blocks)
        self.spatial_index.build_frontier_index([])

        # Indexes should be created but empty
        self.assertIsNotNone(self.spatial_index.street_index)
        self.assertIsNotNone(self.spatial_index.block_index)
        self.assertIsNotNone(self.spatial_index.frontier_index)

    def test_malformed_geometry_handling(self):
        """Test handling of streets without geometry bounds."""
        # Create street without proper geometry
        bad_streets = pd.DataFrame({
            'geometry': [None, "not_a_geometry"],
            'u': [1, 2],
            'v': [2, 3]
        })

        # Should not raise errors
        self.spatial_index.build_street_index(bad_streets)

        # Index should be created
        self.assertIsNotNone(self.spatial_index.street_index)


if __name__ == '__main__':
    unittest.main()
