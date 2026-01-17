#!/usr/bin/env python3
"""
Curvature Continuity Constraints Module

Implements G¹ (tangent) continuity constraints for street junctions.
Ensures smooth transitions between curved segments at intersections.
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from shapely.geometry import Point, LineString
from .curved_primitives import ParametricCurve, CurvedStreetSegment


class JunctionConstraint:
    """Represents a continuity constraint at a street junction."""

    def __init__(self, junction_point: Point, incoming_segments: List[CurvedStreetSegment],
                 max_tangent_deviation: float = math.pi/6):  # 30 degrees
        """
        Args:
            junction_point: The junction point
            incoming_segments: List of segments meeting at this junction
            max_tangent_deviation: Maximum allowed tangent deviation in radians
        """
        self.junction_point = junction_point
        self.incoming_segments = incoming_segments
        self.max_tangent_deviation = max_tangent_deviation

    def check_tangent_continuity(self) -> Tuple[bool, List[float]]:
        """
        Check if tangent vectors are continuous at the junction.

        Returns:
            (is_continuous, deviation_angles)
        """
        if len(self.incoming_segments) < 2:
            return True, []  # Single segment, no constraint

        # Get tangent vectors at junction for each segment
        tangents = []
        for segment in self.incoming_segments:
            # For incoming segments, tangent at the end point
            # For outgoing segments, tangent at the start point
            try:
                # Assume segments end at junction
                tangent = segment.tangent_at(segment.total_length)
                tangents.append(np.array(tangent))
            except:
                # Fallback for straight segments
                tangents.append(np.array([1.0, 0.0]))  # Default tangent

        # Calculate angle deviations between all pairs
        deviations = []
        for i in range(len(tangents)):
            for j in range(i+1, len(tangents)):
                dot_product = np.dot(tangents[i], tangents[j])
                # Clamp to avoid numerical issues
                dot_product = np.clip(dot_product, -1.0, 1.0)
                angle = math.acos(dot_product)
                deviations.append(angle)

        # Check if all deviations are within tolerance
        is_continuous = all(dev <= self.max_tangent_deviation for dev in deviations)

        return is_continuous, deviations

    def enforce_continuity(self) -> List[CurvedStreetSegment]:
        """
        Attempt to modify segments to enforce continuity.

        Returns:
            Modified segments with enforced continuity
        """
        if len(self.incoming_segments) < 2:
            return self.incoming_segments

        # For now, return unmodified segments
        # Full implementation would adjust curvatures to match tangents
        return self.incoming_segments


class CurvatureContinuityManager:
    """Manages curvature continuity across the entire street network."""

    def __init__(self, max_tangent_deviation: float = math.pi/6):
        self.max_tangent_deviation = max_tangent_deviation
        self.junctions: List[JunctionConstraint] = []

    def add_junction(self, junction_point: Point, segments: List[CurvedStreetSegment]):
        """Add a junction to monitor for continuity."""
        constraint = JunctionConstraint(junction_point, segments, self.max_tangent_deviation)
        self.junctions.append(constraint)

    def check_all_continuity(self) -> Tuple[bool, List[Tuple[Point, List[float]]]]:
        """
        Check continuity at all junctions.

        Returns:
            (all_continuous, [(junction_point, deviations), ...])
        """
        results = []
        all_continuous = True

        for junction in self.junctions:
            is_continuous, deviations = junction.check_tangent_continuity()
            results.append((junction.junction_point, deviations))
            if not is_continuous:
                all_continuous = False

        return all_continuous, results

    def enforce_all_continuity(self) -> List[CurvedStreetSegment]:
        """
        Attempt to enforce continuity at all junctions.

        Returns:
            List of all modified segments
        """
        modified_segments = []

        for junction in self.junctions:
            modified = junction.enforce_continuity()
            modified_segments.extend(modified)

        return modified_segments

    def find_junctions_from_segments(self, segments: List[CurvedStreetSegment],
                                   tolerance: float = 1.0) -> List[JunctionConstraint]:
        """
        Automatically detect junctions from a list of segments.

        Args:
            segments: List of curved street segments
            tolerance: Distance tolerance for junction detection

        Returns:
            List of detected junction constraints
        """
        # Collect all endpoints
        endpoints = []
        for segment in segments:
            start_point = segment.point_at(0)
            end_point = segment.point_at(segment.total_length)
            endpoints.extend([(start_point, segment, 'start'), (end_point, segment, 'end')])

        # Group by proximity
        junctions = []
        processed = set()

        for i, (point1, segment1, end1) in enumerate(endpoints):
            if i in processed:
                continue

            nearby_segments = [segment1]
            processed.add(i)

            for j, (point2, segment2, end2) in enumerate(endpoints):
                if j in processed:
                    continue

                if point1.distance(point2) <= tolerance:
                    nearby_segments.append(segment2)
                    processed.add(j)

            if len(nearby_segments) > 1:
                junctions.append(JunctionConstraint(point1, nearby_segments, self.max_tangent_deviation))

        return junctions


def enforce_street_continuity(streets: List[CurvedStreetSegment],
                            max_tangent_deviation: float = math.pi/6) -> Tuple[bool, List[CurvedStreetSegment]]:
    """
    High-level function to enforce curvature continuity on a street network.

    Args:
        streets: List of curved street segments
        max_tangent_deviation: Maximum allowed tangent deviation in radians

    Returns:
        (success, modified_streets)
    """
    manager = CurvatureContinuityManager(max_tangent_deviation)

    # Detect junctions
    junctions = manager.find_junctions_from_segments(streets)
    manager.junctions = junctions

    # Check continuity
    all_continuous, _ = manager.check_all_continuity()

    if all_continuous:
        return True, streets
    else:
        # Attempt to enforce continuity
        modified_streets = manager.enforce_all_continuity()
        return False, modified_streets  # Continuity not fully enforced yet