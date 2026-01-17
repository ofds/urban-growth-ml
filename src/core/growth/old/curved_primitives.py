#!/usr/bin/env python3
"""
Curved Geometry Primitives Module

Implements parametric curves for urban growth: circular arcs, clothoids, and Bézier curves.
These replace straight LineString segments to enable smooth curvature in streets.
"""

import numpy as np
from shapely.geometry import LineString, Point
from typing import List, Tuple, Optional
import math


class ParametricCurve:
    """Base class for parametric curves defined by arc length s."""

    def __init__(self):
        self.total_length = 0.0

    def point_at(self, s: float) -> Point:
        """Get point at arc length s."""
        raise NotImplementedError

    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get unit tangent vector at arc length s."""
        raise NotImplementedError

    def curvature_at(self, s: float) -> float:
        """Get curvature at arc length s."""
        raise NotImplementedError

    def to_linestring(self, num_points: int = 50) -> LineString:
        """Convert to LineString for compatibility."""
        s_values = np.linspace(0, self.total_length, num_points)
        points = [self.point_at(s) for s in s_values]
        return LineString(points)


class CircularArc(ParametricCurve):
    """Circular arc defined by center, radius, start angle, and arc angle."""

    def __init__(self, center: Point, radius: float, start_angle: float, arc_angle: float):
        """
        Args:
            center: Center point of the circle
            radius: Radius of the arc
            start_angle: Starting angle in radians (0 = positive x-axis)
            arc_angle: Angular span in radians (positive = counterclockwise)
        """
        super().__init__()
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.arc_angle = arc_angle
        self.total_length = abs(radius * arc_angle)

    def point_at(self, s: float) -> Point:
        """Get point at arc length s along the arc."""
        if s < 0 or s > self.total_length:
            raise ValueError(f"Arc length s={s} out of range [0, {self.total_length}]")

        # Convert arc length to angle
        angle_offset = s / self.radius
        if self.arc_angle < 0:  # Clockwise
            angle_offset = -angle_offset

        current_angle = self.start_angle + angle_offset

        x = self.center.x + self.radius * math.cos(current_angle)
        y = self.center.y + self.radius * math.sin(current_angle)

        return Point(x, y)

    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get unit tangent vector at arc length s."""
        angle_offset = s / self.radius
        if self.arc_angle < 0:
            angle_offset = -angle_offset

        current_angle = self.start_angle + angle_offset

        # Tangent is perpendicular to radius vector
        # For counterclockwise: (-sin(θ), cos(θ))
        # For clockwise: (sin(θ), -cos(θ))
        sign = 1 if self.arc_angle >= 0 else -1

        dx = -sign * math.sin(current_angle)
        dy = sign * math.cos(current_angle)

        return (dx, dy)

    def curvature_at(self, s: float) -> float:
        """Get curvature (constant for circular arc)."""
        return 1.0 / self.radius if self.arc_angle >= 0 else -1.0 / self.radius


class Clothoid(ParametricCurve):
    """Euler spiral (clothoid) with linearly varying curvature."""

    def __init__(self, start_point: Point, start_tangent: Tuple[float, float],
                 start_curvature: float, curvature_rate: float, length: float):
        """
        Args:
            start_point: Starting point
            start_tangent: Unit tangent vector at start
            start_curvature: Curvature at start (1/radius)
            curvature_rate: Rate of curvature change (dk/ds)
            length: Total arc length
        """
        super().__init__()
        self.start_point = start_point
        self.start_tangent = np.array(start_tangent)
        self.start_curvature = start_curvature
        self.curvature_rate = curvature_rate
        self.total_length = length

        # Pre-compute Fresnel integrals for efficiency
        self._precompute_fresnel()

    def _precompute_fresnel(self):
        """Pre-compute Fresnel integrals for clothoid evaluation."""
        # Simplified implementation - full clothoid requires numerical integration
        # This is a basic approximation
        pass

    def point_at(self, s: float) -> Point:
        """Approximate clothoid point using numerical integration."""
        if s < 0 or s > self.total_length:
            raise ValueError(f"Arc length s={s} out of range [0, {self.total_length}]")

        # Simplified clothoid approximation
        # In practice, would use Fresnel integrals
        num_steps = 100
        ds = s / num_steps

        x, y = self.start_point.x, self.start_point.y
        theta = math.atan2(self.start_tangent[1], self.start_tangent[0])
        kappa = self.start_curvature

        for i in range(num_steps):
            # Euler integration
            x += ds * math.cos(theta)
            y += ds * math.sin(theta)
            theta += ds * kappa
            kappa += ds * self.curvature_rate

        return Point(x, y)

    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get tangent vector at arc length s."""
        # Simplified - would need proper clothoid tangent calculation
        theta = math.atan2(self.start_tangent[1], self.start_tangent[0])
        kappa = self.start_curvature + s * self.curvature_rate
        theta += s * kappa  # Approximate

        return (math.cos(theta), math.sin(theta))

    def curvature_at(self, s: float) -> float:
        """Get curvature at arc length s."""
        return self.start_curvature + s * self.curvature_rate


class CubicBezier(ParametricCurve):
    """Cubic Bézier curve for smooth interpolation."""

    def __init__(self, p0: Point, p1: Point, p2: Point, p3: Point):
        """
        Args:
            p0, p3: Endpoints
            p1, p2: Control points
        """
        super().__init__()
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

        # Approximate total length using chord length
        self.total_length = self._estimate_length()

    def _estimate_length(self) -> float:
        """Estimate curve length using adaptive sampling."""
        # Simple approximation: sample points and sum distances
        t_values = np.linspace(0, 1, 50)
        points = [self._bezier_point(t) for t in t_values]

        length = 0
        for i in range(1, len(points)):
            length += points[i-1].distance(points[i])

        return length

    def _bezier_point(self, t: float) -> Point:
        """Evaluate Bézier curve at parameter t."""
        # Cubic Bézier formula
        mt = 1 - t
        x = (mt**3 * self.p0.x +
             3 * mt**2 * t * self.p1.x +
             3 * mt * t**2 * self.p2.x +
             t**3 * self.p3.x)
        y = (mt**3 * self.p0.y +
             3 * mt**2 * t * self.p1.y +
             3 * mt * t**2 * self.p2.y +
             t**3 * self.p3.y)
        return Point(x, y)

    def _bezier_tangent(self, t: float) -> Tuple[float, float]:
        """Evaluate tangent vector at parameter t."""
        mt = 1 - t
        dx = (3 * mt**2 * (self.p1.x - self.p0.x) +
              6 * mt * t * (self.p2.x - self.p1.x) +
              3 * t**2 * (self.p3.x - self.p2.x))
        dy = (3 * mt**2 * (self.p1.y - self.p0.y) +
              6 * mt * t * (self.p2.y - self.p1.y) +
              3 * t**2 * (self.p3.y - self.p2.y))

        # Normalize
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            dx /= length
            dy /= length

        return (dx, dy)

    def point_at(self, s: float) -> Point:
        """Get point at arc length s."""
        if s < 0 or s > self.total_length:
            raise ValueError(f"Arc length s={s} out of range [0, {self.total_length}]")

        # Convert arc length to parameter t (approximate)
        t = s / self.total_length
        return self._bezier_point(t)

    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get tangent vector at arc length s."""
        t = s / self.total_length
        return self._bezier_tangent(t)

    def curvature_at(self, s: float) -> float:
        """Approximate curvature using second derivative."""
        # Simplified curvature calculation
        t = s / self.total_length

        # First derivatives
        mt = 1 - t
        dx_dt = (3 * mt**2 * (self.p1.x - self.p0.x) +
                6 * mt * t * (self.p2.x - self.p1.x) +
                3 * t**2 * (self.p3.x - self.p2.x))
        dy_dt = (3 * mt**2 * (self.p1.y - self.p0.y) +
                6 * mt * t * (self.p2.y - self.p1.y) +
                3 * t**2 * (self.p3.y - self.p2.y))

        # Second derivatives
        d2x_dt2 = (6 * mt * (self.p1.x - self.p0.x) +
                  6 * (mt - t) * (self.p2.x - self.p1.x) +
                  6 * t * (self.p3.x - self.p2.x))
        d2y_dt2 = (6 * mt * (self.p1.y - self.p0.y) +
                  6 * (mt - t) * (self.p2.y - self.p1.y) +
                  6 * t * (self.p3.y - self.p2.y))

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = dx_dt * d2y_dt2 - dy_dt * d2x_dt2
        denominator = (dx_dt**2 + dy_dt**2)**(3/2)

        if denominator > 0:
            return numerator / denominator
        return 0.0


class CurvedStreetSegment:
    """A street segment composed of parametric curves with continuity constraints."""

    def __init__(self, curves: List[ParametricCurve]):
        self.curves = curves
        self.total_length = sum(c.total_length for c in curves)

    def point_at(self, s: float) -> Point:
        """Get point at global arc length s."""
        if s < 0 or s > self.total_length:
            raise ValueError(f"Arc length s={s} out of range [0, {self.total_length}]")

        remaining_s = s
        for curve in self.curves:
            if remaining_s <= curve.total_length:
                return curve.point_at(remaining_s)
            remaining_s -= curve.total_length

        # Should not reach here
        return self.curves[-1].point_at(self.curves[-1].total_length)

    def tangent_at(self, s: float) -> Tuple[float, float]:
        """Get tangent at global arc length s."""
        remaining_s = s
        for curve in self.curves:
            if remaining_s <= curve.total_length:
                return curve.tangent_at(remaining_s)
            remaining_s -= curve.total_length

        return self.curves[-1].tangent_at(self.curves[-1].total_length)

    def curvature_at(self, s: float) -> float:
        """Get curvature at global arc length s."""
        remaining_s = s
        for curve in self.curves:
            if remaining_s <= curve.total_length:
                return curve.curvature_at(remaining_s)
            remaining_s -= curve.total_length

        return self.curves[-1].curvature_at(self.curves[-1].total_length)

    def to_linestring(self, points_per_curve: int = 20) -> LineString:
        """Convert to LineString for shapely compatibility."""
        all_points = []
        for curve in self.curves:
            curve_points = [curve.point_at(s) for s in np.linspace(0, curve.total_length, points_per_curve)]
            all_points.extend(curve_points)

        # Remove duplicate points at curve junctions
        unique_points = []
        prev_point = None
        for point in all_points:
            if prev_point is None or point.distance(prev_point) > 1e-6:
                unique_points.append(point)
                prev_point = point

        return LineString(unique_points)