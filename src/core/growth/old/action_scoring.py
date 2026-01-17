#!/usr/bin/env python3
"""
Multi-Objective Action Scoring Module

Replaces binary validation with soft scoring using penalties and bonuses.
Actions compete probabilistically based on multiple objectives.
"""

import numpy as np
import math
from typing import Dict, List, Optional, Any, Tuple
from shapely.geometry import LineString, Point, Polygon
from ..geometry.curved_primitives import CurvedStreetSegment


class ActionScorer:
    """Multi-objective scorer for growth actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default scoring configuration."""
        return {
            'intersection_penalty': {
                'weight': 1.0,
                'max_penalty': 10.0,
                'distance_threshold': 1.0
            },
            'curvature_penalty': {
                'weight': 0.5,
                'max_curvature': 0.01,  # 1/radius = 100m radius
                'penalty_scale': 5.0
            },
            'density_bonus': {
                'weight': 0.3,
                'optimal_density': 0.5,
                'bonus_scale': 2.0
            },
            'alignment_bonus': {
                'weight': 0.2,
                'angle_threshold': math.pi/6,  # 30 degrees
                'bonus_scale': 1.5
            },
            'length_bonus': {
                'weight': 0.1,
                'optimal_length': 50.0,
                'bonus_scale': 1.0
            },
            'temperature': 0.1  # Boltzmann temperature for selection
        }

    def score_action(self, action: Any, state: Any) -> float:
        """
        Score an action based on multiple objectives.

        Args:
            action: GrowthAction to score
            state: Current GrowthState

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Intersection penalty
        intersection_penalty = self._intersection_penalty(action, state)
        score -= self.config['intersection_penalty']['weight'] * intersection_penalty

        # Curvature penalty
        curvature_penalty = self._curvature_penalty(action)
        score -= self.config['curvature_penalty']['weight'] * curvature_penalty

        # Density bonus
        density_bonus = self._density_bonus(action, state)
        score += self.config['density_bonus']['weight'] * density_bonus

        # Alignment bonus
        alignment_bonus = self._alignment_bonus(action, state)
        score += self.config['alignment_bonus']['weight'] * alignment_bonus

        # Length bonus
        length_bonus = self._length_bonus(action)
        score += self.config['length_bonus']['weight'] * length_bonus

        return score

    def _intersection_penalty(self, action: Any, state: Any) -> float:
        """Calculate penalty for intersections with existing streets."""
        proposed_geom = action.proposed_geometry
        penalty = 0.0

        if not isinstance(proposed_geom, LineString):
            return penalty

        threshold = self.config['intersection_penalty']['distance_threshold']

        for idx, existing_street in state.streets.iterrows():
            existing_geom = existing_street.geometry

            if proposed_geom.intersects(existing_geom):
                intersection = proposed_geom.intersection(existing_geom)

                if intersection.length > threshold:
                    # Real intersection - apply penalty
                    penalty += min(intersection.length, self.config['intersection_penalty']['max_penalty'])
                elif isinstance(intersection, Point):
                    # Endpoint touch - smaller penalty
                    penalty += 0.1

        return penalty

    def _curvature_penalty(self, action: Any) -> float:
        """Calculate penalty for excessive curvature."""
        penalty = 0.0

        # Extract curvature information from action parameters
        curve_segment = action.parameters.get('curve_segment')
        if curve_segment and isinstance(curve_segment, CurvedStreetSegment):
            # Sample curvature along the curve
            s_values = np.linspace(0, curve_segment.total_length, 10)
            curvatures = [abs(curve_segment.curvature_at(s)) for s in s_values]

            max_curvature = max(curvatures) if curvatures else 0.0
            max_allowed = self.config['curvature_penalty']['max_curvature']

            if max_curvature > max_allowed:
                excess = max_curvature - max_allowed
                penalty = excess * self.config['curvature_penalty']['penalty_scale']

        return penalty

    def _density_bonus(self, action: Any, state: Any) -> float:
        """Calculate bonus based on local density using proximity queries."""
        proposed_geom = action.proposed_geometry

        if not isinstance(proposed_geom, LineString):
            return 0.0

        # Sample density at multiple points along the proposed geometry
        points = [
            Point(proposed_geom.coords[0]),  # Start
            proposed_geom.centroid,          # Center
            Point(proposed_geom.coords[-1])  # End
        ]

        total_density = 0.0
        for point in points:
            # Use simplified local density calculation
            search_radius = 100.0
            nearby_count = 0

            # Quick proximity check - only streets within bounding box first
            for idx, existing_street in state.streets.iterrows():
                existing_geom = existing_street.geometry
                # Quick bounding box check
                if point.distance(existing_geom) < search_radius:
                    nearby_count += 1
                    if nearby_count > 20:  # Cap at reasonable number
                        break

            # Normalize density (0-1 scale)
            density = min(nearby_count / 15.0, 1.0)  # Assume 15 is high density
            total_density += density

        avg_density = total_density / len(points)

        # Bonus peaks at optimal density
        optimal = self.config['density_bonus']['optimal_density']
        bonus = 1.0 - abs(avg_density - optimal)
        bonus *= self.config['density_bonus']['bonus_scale']

        return bonus

    def _alignment_bonus(self, action: Any, state: Any) -> float:
        """Calculate bonus for alignment with existing streets."""
        proposed_geom = action.proposed_geometry
        bonus = 0.0

        if not isinstance(proposed_geom, LineString):
            return bonus

        # Get proposed street direction
        coords = list(proposed_geom.coords)
        if len(coords) < 2:
            return bonus

        dx = coords[-1][0] - coords[0][0]
        dy = coords[-1][1] - coords[0][1]
        proposed_angle = math.atan2(dy, dx)

        threshold = self.config['alignment_bonus']['angle_threshold']

        # Check alignment with nearby streets
        center = proposed_geom.centroid
        search_radius = 50.0

        alignment_count = 0
        for idx, existing_street in state.streets.iterrows():
            existing_geom = existing_street.geometry
            if center.distance(existing_geom) < search_radius:
                # Get existing street direction
                existing_coords = list(existing_geom.coords)
                if len(existing_coords) >= 2:
                    ex_dx = existing_coords[-1][0] - existing_coords[0][0]
                    ex_dy = existing_coords[-1][1] - existing_coords[0][1]
                    existing_angle = math.atan2(ex_dy, ex_dx)

                    # Check angle difference
                    angle_diff = abs(proposed_angle - existing_angle)
                    angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Handle wraparound

                    if angle_diff < threshold:
                        alignment_count += 1

        if alignment_count > 0:
            bonus = min(alignment_count, 3) * self.config['alignment_bonus']['bonus_scale']

        return bonus

    def _length_bonus(self, action: Any) -> float:
        """Calculate bonus based on street length."""
        proposed_geom = action.proposed_geometry

        if not isinstance(proposed_geom, LineString):
            return 0.0

        length = proposed_geom.length
        optimal = self.config['length_bonus']['optimal_length']

        # Bonus peaks at optimal length
        bonus = 1.0 - abs(length - optimal) / optimal
        bonus = max(0, bonus)  # No penalty for suboptimal length
        bonus *= self.config['length_bonus']['bonus_scale']

        return bonus

    def select_action_probabilistically(self, actions: List[Any], state: Any) -> Optional[Any]:
        """
        Select an action probabilistically based on scores.

        Args:
            actions: List of GrowthActions to choose from
            state: Current GrowthState

        Returns:
            Selected action or None if no actions
        """
        if not actions:
            return None

        # Score all actions
        scores = [self.score_action(action, state) for action in actions]

        # Apply Boltzmann selection
        temperature = self.config['temperature']
        if temperature > 0:
            # Convert to probabilities
            exp_scores = np.exp(np.array(scores) / temperature)
            probabilities = exp_scores / np.sum(exp_scores)
        else:
            # Greedy selection
            max_score = max(scores)
            probabilities = [1.0 if s == max_score else 0.0 for s in scores]

        # Sample from distribution
        r = np.random.random()
        cumulative = 0.0
        for action, prob in zip(actions, probabilities):
            cumulative += prob
            if r <= cumulative:
                return action

        return actions[-1]  # Fallback


class MultiObjectiveActionSelector:
    """Manages action selection with multi-objective scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.scorer = ActionScorer(config)

    def select_best_action(self, candidate_actions: List[Any], state: Any) -> Optional[Any]:
        """
        Select the best action from candidates using multi-objective scoring.

        Args:
            candidate_actions: List of proposed GrowthActions
            state: Current GrowthState

        Returns:
            Best action or None
        """
        if not candidate_actions:
            return None

        # Use probabilistic selection instead of deterministic
        return self.scorer.select_action_probabilistically(candidate_actions, state)

    def get_action_scores(self, actions: List[Any], state: Any) -> List[float]:
        """Get scores for debugging/analysis."""
        return [self.scorer.score_action(action, state) for action in actions]