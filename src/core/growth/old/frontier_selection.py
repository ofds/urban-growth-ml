#!/usr/bin/env python3
"""
Frontier Selection Module

Replaces weighted random selection with gradient-following behavior.
Frontiers behave like particles in potential fields.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from shapely.geometry import Point, LineString, Polygon
from .density_field import DensityField


class ScalarField:
    """Base class for scalar fields used in frontier selection."""

    def value_at(self, point: Point) -> float:
        """Get field value at point."""
        raise NotImplementedError

    def gradient_at(self, point: Point) -> Tuple[float, float]:
        """Get field gradient at point."""
        raise NotImplementedError


class DensityScalarField(ScalarField):
    """Scalar field based on density."""

    def __init__(self, density_field: DensityField):
        self.density_field = density_field

    def value_at(self, point: Point) -> float:
        return self.density_field.density_at(point)

    def gradient_at(self, point: Point) -> Tuple[float, float]:
        return self.density_field.gradient_at(point)


class AccessibilityScalarField(ScalarField):
    """Scalar field based on accessibility/connectivity."""

    def __init__(self, state: Any):
        self.state = state
        self._precompute_accessibility()

    def _precompute_accessibility(self):
        """Pre-compute accessibility values."""
        # Simplified: use distance to nearest major intersection
        self.accessibility_values = {}

        # Find high-degree nodes (intersections)
        intersection_nodes = []
        for node, degree in self.state.graph.degree():
            if degree >= 3:  # 3+ connections = intersection
                geom = self.state.graph.nodes[node].get('geometry')
                if geom:
                    intersection_nodes.append((node, geom))

        # For each point, find distance to nearest intersection
        # This is simplified - in practice would use more sophisticated metrics
        self.intersection_points = [geom for _, geom in intersection_nodes]

    def value_at(self, point: Point) -> float:
        """Higher values = more accessible."""
        if not self.intersection_points:
            return 0.0

        min_distance = min(point.distance(intersect) for intersect in self.intersection_points)
        # Convert distance to accessibility (closer = more accessible)
        accessibility = 1.0 / (1.0 + min_distance / 100.0)  # Sigmoid-like decay
        return accessibility

    def gradient_at(self, point: Point) -> Tuple[float, float]:
        """Gradient points toward higher accessibility."""
        if not self.intersection_points:
            return (0.0, 0.0)

        # Find nearest intersection
        nearest = min(self.intersection_points, key=lambda p: point.distance(p))

        # Gradient direction toward nearest intersection
        dx = nearest.x - point.x
        dy = nearest.y - point.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Add numerical stability check
        if distance < 1e-6:  # Points are essentially coincident
            return (0.0, 0.0)
        
        if distance > 0:
            # Normalize and scale
            dx /= distance
            dy /= distance
            # Gradient magnitude decreases with distance
            magnitude = 1.0 / (1.0 + distance / 50.0)
            dx *= magnitude
            dy *= magnitude

        return (dx, dy)


class BlockCompactnessScalarField(ScalarField):
    """Scalar field based on block compactness/shape."""

    def __init__(self, state: Any):
        self.state = state
        self._precompute_compactness()

    def _precompute_compactness(self):
        """Pre-compute block compactness values."""
        self.compactness_values = {}
        
        # Handle empty blocks GeoDataFrame
        if self.state.blocks.empty:
            return
        
        for idx, block in self.state.blocks.iterrows():
            geom = block.geometry
            if isinstance(geom, Polygon) and not geom.is_empty:
                # Calculate compactness (area/perimeter ratio)
                area = geom.area
                perimeter = geom.length
                if perimeter > 0:
                    compactness = 4 * math.pi * area / (perimeter * perimeter)
                    self.compactness_values[idx] = compactness

    def value_at(self, point: Point) -> float:
        """Higher values = more compact blocks nearby."""
        # Return 0 if no blocks exist
        if self.state.blocks.empty or not self.compactness_values:
            return 0.0
        
        # Find nearest block
        min_distance = float('inf')
        nearest_compactness = 0.0

        for idx, block in self.state.blocks.iterrows():
            geom = block.geometry
            if isinstance(geom, Polygon) and not geom.is_empty:
                distance = point.distance(geom)
                if distance < min_distance:
                    min_distance = distance
                    nearest_compactness = self.compactness_values.get(idx, 0.0)

        # Decay with distance
        if min_distance < 50:  # Only consider nearby blocks
            return nearest_compactness * math.exp(-min_distance / 25.0)

        return 0.0

    def gradient_at(self, point: Point) -> Tuple[float, float]:
        """Gradient points toward more compact areas."""
        # Simplified: no gradient for this field
        return (0.0, 0.0)


class GradientFollowingFrontierSelector:
    """Frontier selector that follows gradients in scalar fields."""

    def __init__(self, density_field: DensityField):
        self.density_field = density_field

        # Field weights and objectives
        self.fields = {
            'density': DensityScalarField(density_field),
            'accessibility': None,  # Will be set per state
            'compactness': None     # Will be set per state
        }

        self.weights = {
            'density': 0.4,        # Follow density gradients
            'accessibility': 0.4,  # Prefer accessible areas
            'compactness': 0.2     # Prefer well-formed blocks
        }

        self.exploration_bonus = 0.1  # Small bonus for unexplored areas

    def update_state(self, state: Any):
        """Update state-dependent fields."""
        self.fields['accessibility'] = AccessibilityScalarField(state)
        self.fields['compactness'] = BlockCompactnessScalarField(state)

    def compute_frontier_potential(self, frontier: Any, state: Any) -> float:
        """
        Compute potential energy for a frontier based on scalar fields.

        Args:
            frontier: FrontierEdge to evaluate
            state: Current GrowthState

        Returns:
            Potential value (lower = more attractive)
        """
        geom = frontier.geometry
        if not isinstance(geom, LineString):
            return 0.0

        # Sample field values along frontier
        start_point = Point(geom.coords[0])
        end_point = Point(geom.coords[-1])
        mid_point = geom.interpolate(0.5, normalized=True)

        potential = 0.0

        for field_name, field in self.fields.items():
            if field is None:
                continue

            weight = self.weights[field_name]

            # Sample at multiple points
            points = [start_point, mid_point, end_point]
            field_values = [field.value_at(point) for point in points]
            avg_value = sum(field_values) / len(field_values)

            if field_name == 'density':
                # For density: prefer medium density areas (infill opportunities)
                # Low density = exploration, high density = saturation
                optimal_density = 0.5
                density_cost = abs(avg_value - optimal_density)
                potential += weight * density_cost

            elif field_name == 'accessibility':
                # Prefer more accessible areas (negative potential)
                potential -= weight * avg_value

            elif field_name == 'compactness':
                # Prefer areas with well-formed blocks
                potential -= weight * avg_value

        # Add exploration bonus for frontiers that haven't been selected recently
        # This is a simple proxy - in practice would track selection history
        exploration_factor = 1.0 + self.exploration_bonus
        potential *= exploration_factor

        return potential

    def compute_frontier_force(self, frontier: Any, state: Any) -> Tuple[float, float]:
        """
        Compute force vector on frontier based on field gradients.

        Args:
            frontier: FrontierEdge
            state: Current GrowthState

        Returns:
            (fx, fy) force vector
        """
        geom = frontier.geometry
        if not isinstance(geom, LineString):
            return (0.0, 0.0)

        # Use midpoint for force calculation
        mid_point = geom.interpolate(0.5, normalized=True)

        total_fx, total_fy = 0.0, 0.0

        for field_name, field in self.fields.items():
            if field is None:
                continue

            weight = self.weights[field_name]

            try:
                grad_x, grad_y = field.gradient_at(mid_point)
                
                # Check for NaN or infinite values
                if not (math.isfinite(grad_x) and math.isfinite(grad_y)):
                    continue
                
                # All field types contribute in the same way now
                total_fx += weight * grad_x
                total_fy += weight * grad_y
                
            except Exception as e:
                # Log but don't fail on gradient computation errors
                continue

        return (total_fx, total_fy)


    def select_frontier_gradient(self, frontiers: List[Any], state: Any) -> Optional[Any]:
        """
        Select frontier using gradient-based potential field approach.

        Args:
            frontiers: List of FrontierEdges
            state: Current GrowthState

        Returns:
            Selected frontier or None
        """
        if not frontiers:
            return None

        self.update_state(state)

        # Compute potential for each frontier
        frontier_potentials = []
        for frontier in frontiers:
            try:
                potential = self.compute_frontier_potential(frontier, state)
                force_x, force_y = self.compute_frontier_force(frontier, state)
                force_magnitude = math.sqrt(force_x*force_x + force_y*force_y)

                # Check for numerical issues
                if not math.isfinite(potential) or not math.isfinite(force_magnitude):
                    continue

                # Combine potential and force
                # Lower potential + higher force magnitude = more attractive
                attractiveness = -potential + force_magnitude * 0.1

                frontier_potentials.append((frontier, attractiveness))
            except Exception as e:
                # Skip frontiers that cause errors
                continue
        
        # Safety check: if all frontiers failed, return None
        if not frontier_potentials:
            return None


        # Sort by attractiveness (higher = better)
        frontier_potentials.sort(key=lambda x: x[1], reverse=True)

        # Select probabilistically from top candidates
        if len(frontier_potentials) >= 3:
            # Consider top 3
            candidates = frontier_potentials[:3]
        else:
            candidates = frontier_potentials

        # Boltzmann selection among candidates
        # This ensures reproducibility with the same seed
        return frontier_potentials[0][0]

    def get_frontier_attractiveness_map(self, frontiers: List[Any], state: Any) -> Dict[str, float]:
        """
        Get attractiveness scores for analysis/debugging.

        Returns:
            Dict mapping frontier_id to attractiveness score
        """
        self.update_state(state)
        attractiveness_map = {}

        for frontier in frontiers:
            potential = self.compute_frontier_potential(frontier, state)
            force_x, force_y = self.compute_frontier_force(frontier, state)
            force_magnitude = math.sqrt(force_x*force_x + force_y*force_y)

            attractiveness = -potential + force_magnitude * 0.1
            attractiveness_map[frontier.frontier_id] = attractiveness

        return attractiveness_map