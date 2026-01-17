#!/usr/bin/env python3
"""
Density Field Module

Implements density as a continuous spatial field using kernel density estimation.
Density becomes a causal input signal rather than just a post-hoc metric.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from shapely.geometry import Point, Polygon
from scipy import stats
from sklearn.neighbors import KernelDensity
from scipy.spatial import cKDTree


class DensityField:
    """Continuous spatial density field for urban growth control."""

    def __init__(self, cell_size: float = 10.0, bandwidth: float = 50.0, max_search_radius: float = 200.0):
        """
        Args:
            cell_size: Grid cell size in meters
            bandwidth: KDE bandwidth in meters
            max_search_radius: Maximum radius for local density calculations
        """
        self.cell_size = cell_size
        self.bandwidth = bandwidth
        self.max_search_radius = max_search_radius
        self.kde = None
        self.sample_points = []
        self.spatial_index = None
        self.density_grid = {}
        self.bounds = None

    def update_from_state(self, state: Any):
        """
        Update density field from current growth state.

        Args:
            state: GrowthState containing streets and blocks
        """
        # Collect sample points from streets and blocks
        self.sample_points = []

        # Street endpoints and midpoints
        for idx, street in state.streets.iterrows():
            geom = street.geometry
            if hasattr(geom, 'coords'):
                coords = list(geom.coords)
                # Add endpoints
                for coord in coords:
                    self.sample_points.append(Point(coord))
                # Add midpoint for longer streets
                if len(coords) >= 2 and geom.length > 20:
                    midpoint = geom.interpolate(0.5, normalized=True)
                    self.sample_points.append(midpoint)

        # Block centroids (weighted by area)
        for idx, block in state.blocks.iterrows():
            geom = block.geometry
            if isinstance(geom, Polygon) and not geom.is_empty:
                centroid = geom.centroid
                area = geom.area
                # Add centroid multiple times based on area
                weight = min(int(area / 1000) + 1, 5)  # 1-5 points per block
                for _ in range(weight):
                    self.sample_points.append(centroid)

        # Update bounds
        if self.sample_points:
            x_coords = [p.x for p in self.sample_points]
            y_coords = [p.y for p in self.sample_points]
            self.bounds = (
                min(x_coords), min(y_coords),
                max(x_coords), max(y_coords)
            )

            # Build spatial index for fast proximity queries
            points_array = np.array([[p.x, p.y] for p in self.sample_points])
            if len(points_array) > 0:
                self.spatial_index = cKDTree(points_array)

                # Fit KDE only on a subset for performance
                # Use every 5th point for KDE to reduce computation
                if len(points_array) > 20:
                    kde_points = points_array[::5]  # Subsample for KDE
                else:
                    kde_points = points_array

                if len(kde_points) > 1:
                    self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
                    self.kde.fit(kde_points)
                else:
                    self.kde = None
            else:
                self.spatial_index = None
                self.kde = None

    def density_at(self, point: Point) -> float:
        """
        Get density value at a specific point using local proximity calculation.

        Args:
            point: Location to query

        Returns:
            Density value (normalized 0-1)
        """
        if not self.spatial_index or not self.sample_points:
            return 0.0

        # Find nearby points within search radius
        query_point = np.array([point.x, point.y])
        distances, indices = self.spatial_index.query(
            query_point,
            k=min(50, len(self.sample_points)),  # Limit to 50 nearest neighbors
            distance_upper_bound=self.max_search_radius
        )

        # Filter valid distances (cKDTree returns inf for points beyond distance_upper_bound)
        valid_indices = indices[distances != np.inf]
        valid_distances = distances[distances != np.inf]

        if len(valid_distances) == 0:
            return 0.0

        # Calculate local density using inverse distance weighting
        weights = 1.0 / (valid_distances + 1.0)  # Add 1 to avoid division by zero
        total_weight = np.sum(weights)

        if total_weight == 0:
            return 0.0

        # Normalize to 0-1 scale based on expected density range
        density = min(total_weight / 10.0, 1.0)  # Scale factor based on typical values

        return density

    def _estimate_max_density(self) -> float:
        """Estimate maximum possible density for normalization."""
        if not self.sample_points:
            return 1.0

        # Sample density at all data points
        points_array = np.array([[p.x, p.y] for p in self.sample_points])
        if len(points_array) > 1:
            log_densities = self.kde.score_samples(points_array)
            max_density = np.exp(np.max(log_densities))
            return max_density

        return 1.0

    def gradient_at(self, point: Point, epsilon: float = 5.0) -> Tuple[float, float]:
        """
        Compute density gradient at a point using finite differences.

        Args:
            point: Location to query
            epsilon: Finite difference step size

        Returns:
            (dx, dy) gradient vector
        """
        if not self.spatial_index:
            return (0.0, 0.0)

        # Central difference with larger epsilon for stability
        p_x_plus = Point(point.x + epsilon, point.y)
        p_x_minus = Point(point.x - epsilon, point.y)
        p_y_plus = Point(point.x, point.y + epsilon)
        p_y_minus = Point(point.x, point.y - epsilon)

        d_dx = (self.density_at(p_x_plus) - self.density_at(p_x_minus)) / (2 * epsilon)
        d_dy = (self.density_at(p_y_plus) - self.density_at(p_y_minus)) / (2 * epsilon)

        return (d_dx, d_dy)

    def density_along_line(self, line_geometry, num_samples: int = 10) -> List[float]:
        """
        Sample density along a line geometry.

        Args:
            line_geometry: LineString to sample
            num_samples: Number of sample points

        Returns:
            List of density values
        """
        if not isinstance(line_geometry, LineString):
            return []

        densities = []
        for i in range(num_samples):
            point = line_geometry.interpolate(i / (num_samples - 1), normalized=True)
            density = self.density_at(point)
            densities.append(density)

        return densities

    def find_density_gradient_direction(self, point: Point) -> float:
        """
        Find the direction of maximum density increase.

        Args:
            point: Starting point

        Returns:
            Angle in radians pointing toward higher density
        """
        grad_x, grad_y = self.gradient_at(point)
        if grad_x == 0 and grad_y == 0:
            return 0.0

        return math.atan2(grad_y, grad_x)

    def get_density_zones(self, thresholds: List[float] = [0.2, 0.5, 0.8]) -> Dict[str, Polygon]:
        """
        Create density zones as polygons.

        Args:
            thresholds: Density thresholds for zones

        Returns:
            Dict mapping zone names to polygons
        """
        if not self.bounds or not self.kde:
            return {}

        # Create grid
        min_x, min_y, max_x, max_y = self.bounds
        x_grid = np.arange(min_x, max_x, self.cell_size)
        y_grid = np.arange(min_y, max_y, self.cell_size)

        zones = {}
        thresholds = [0.0] + sorted(thresholds) + [1.0]

        for i in range(len(thresholds) - 1):
            low_thresh = thresholds[i]
            high_thresh = thresholds[i + 1]

            # Sample grid points
            grid_points = []
            for x in x_grid:
                for y in y_grid:
                    density = self.density_at(Point(x, y))
                    if low_thresh <= density < high_thresh:
                        grid_points.append((x, y))

            if grid_points:
                # Create convex hull as zone polygon
                from scipy.spatial import ConvexHull
                if len(grid_points) >= 3:
                    hull = ConvexHull(grid_points)
                    hull_points = [grid_points[i] for i in hull.vertices]
                    zones[f'zone_{i}'] = Polygon(hull_points)

        return zones


class DensityCoupledGrowthRules:
    """Growth rules that respond to density field."""

    def __init__(self, density_field: DensityField):
        self.density_field = density_field

        # Coupling parameters
        self.length_modulation = {
            'low_density_max': 100.0,    # Long exploratory streets
            'high_density_min': 20.0,    # Short infill streets
            'transition_density': 0.6
        }

        self.curvature_modulation = {
            'low_density_max_curvature': 0.005,   # Straighter in low density
            'high_density_max_curvature': 0.02,   # More curved in high density
            'transition_density': 0.5
        }

        self.branch_probability = {
            'base_probability': 0.3,
            'density_gradient_boost': 0.2,  # Boost when following density gradient
            'saturation_penalty': 0.5       # Reduce when local density is high
        }

    def modulate_street_length(self, base_length: float, location: Point) -> float:
        """
        Modulate street length based on local density.

        Args:
            base_length: Base length from action
            location: Location of the proposed street

        Returns:
            Modulated length
        """
        density = self.density_field.density_at(location)

        if density < self.length_modulation['transition_density']:
            # Low density: allow longer streets
            factor = 1.0 + (self.length_modulation['transition_density'] - density)
            max_length = self.length_modulation['low_density_max']
        else:
            # High density: shorter streets
            factor = 1.0 - (density - self.length_modulation['transition_density'])
            max_length = self.length_modulation['high_density_min']

        modulated_length = base_length * factor
        return max(self.length_modulation['high_density_min'],
                  min(modulated_length, max_length))

    def modulate_curvature_bounds(self, location: Point) -> Tuple[float, float]:
        """
        Modulate curvature bounds based on local density.

        Args:
            location: Location to check

        Returns:
            (min_curvature, max_curvature) bounds
        """
        density = self.density_field.density_at(location)

        if density < self.curvature_modulation['transition_density']:
            # Low density: prefer straighter streets
            max_curv = self.curvature_modulation['low_density_max_curvature']
        else:
            # High density: allow more curvature
            max_curv = self.curvature_modulation['high_density_max_curvature']

        return (0.0, max_curv)  # Min curvature always 0

    def modulate_branch_probability(self, frontier_location: Point,
                                  proposed_direction: float) -> float:
        """
        Modulate branching probability based on density field.

        Args:
            frontier_location: Where the branch would start
            proposed_direction: Direction of proposed branch

        Returns:
            Modulated probability (0-1)
        """
        base_prob = self.branch_probability['base_probability']

        # Local density penalty
        local_density = self.density_field.density_at(frontier_location)
        density_penalty = 1.0 - (local_density * self.branch_probability['saturation_penalty'])

        # Density gradient bonus
        gradient_direction = self.density_field.find_density_gradient_direction(frontier_location)
        direction_alignment = abs(proposed_direction - gradient_direction)
        direction_alignment = min(direction_alignment, 2*math.pi - direction_alignment)

        # Boost if aligned with density gradient (within 45 degrees)
        gradient_boost = 0.0
        if direction_alignment < math.pi/4:
            gradient_boost = self.branch_probability['density_gradient_boost']

        probability = base_prob * density_penalty + gradient_boost
        return max(0.0, min(probability, 1.0))

    def should_allow_growth(self, proposed_geometry, state: Any) -> bool:
        """
        Decide whether to allow growth based on density patterns.

        Args:
            proposed_geometry: Proposed LineString
            state: Current GrowthState

        Returns:
            True if growth should be allowed
        """
        if not isinstance(proposed_geometry, LineString):
            return True

        # Check if this would create problematic density patterns
        start_density = self.density_field.density_at(Point(proposed_geometry.coords[0]))
        end_density = self.density_field.density_at(Point(proposed_geometry.coords[-1]))

        # Prevent very long streets in high density areas
        if proposed_geometry.length > 50 and (start_density > 0.8 or end_density > 0.8):
            return False

        # Encourage infill in medium density areas
        if proposed_geometry.length < 30 and (start_density > 0.3 and start_density < 0.7):
            return True

        return True