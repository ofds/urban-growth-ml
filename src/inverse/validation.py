#!/usr/bin/env python3
"""
Replay Validation with Morphological Tolerances
Phase A: Validate trace replay against morphological equivalence criteria.
"""

from typing import Dict, Any, Tuple, Optional, List
import logging
from shapely.geometry import Point
from shapely.strtree import STRtree  # OPTIMIZATION: Spatial indexing
from shapely.ops import unary_union

from src.core.contracts import GrowthState
from .data_structures import GrowthTrace

logger = logging.getLogger(__name__)

class MorphologicalValidator:
    """
    Validates trace replay using morphological equivalence rather than geometric identity.
    
    Focuses on structural preservation over exact coordinate matching.
    """
    
    def __init__(self,
                 endpoint_tolerance: float = 1.0,      # meters
                 frechet_tolerance: float = 2.0,       # meters
                 block_count_tolerance: float = 0.05,  # 5% relative
                 connectivity_preservation: bool = True):
        self.endpoint_tolerance = endpoint_tolerance
        self.frechet_tolerance = frechet_tolerance
        self.block_count_tolerance = block_count_tolerance
        self.connectivity_preservation = connectivity_preservation
    
    def validate_replay(self, original_state: GrowthState, replayed_state: GrowthState) -> Dict[str, Any]:
        """
        Validate replayed state against original using morphological criteria.

        Args:
            original_state: Original final state
            replayed_state: State produced by replaying inferred trace

        Returns:
            Validation results dictionary
        """
        results = {
            'geometric_valid': True,
            'topological_valid': True,
            'morphological_valid': True,
            'endpoint_errors': [],
            'frechet_errors': [],
            'block_count_error': 0.0,
            'connectivity_preserved': True,
            'overall_score': 1.0,
            'issues': [],
            # Coverage metrics
            'street_coverage': 0.0,
            'matched_streets': 0,
            'total_original_streets': len(original_state.streets),
            'total_replayed_streets': len(replayed_state.streets)
        }

        # Geometric validation (now includes coverage metrics)
        geom_results = self._validate_geometric(original_state, replayed_state)
        results.update(geom_results)

        # Topological validation
        topo_results = self._validate_topological(original_state, replayed_state)
        results.update(topo_results)

        # Morphological validation
        morph_results = self._validate_morphological(original_state, replayed_state)
        results.update(morph_results)

        # Overall assessment
        results['overall_score'] = self._calculate_overall_score(results)
        results['morphological_valid'] = results['overall_score'] >= 0.7  # 70% threshold

        if not results['morphological_valid']:
            results['issues'].append(f"Low morphological fidelity: {results['overall_score']:.2f}")

        # Add coverage-based issues
        if results['street_coverage'] < 0.5:
            results['issues'].append(f"Low street coverage: {results['street_coverage']:.1%}")

        return results
    
    def _validate_geometric(self, original: GrowthState, replayed: GrowthState) -> Dict[str, Any]:
        """Validate geometric properties with relaxed tolerances."""
        results = {'endpoint_errors': [], 'frechet_errors': []}

        # Get matched pairs and coverage statistics
        matched_pairs, coverage_stats = self._match_streets_geometrically(original.streets, replayed.streets)

        # Store coverage metrics
        results.update(coverage_stats)

        total_endpoint_error = 0.0
        total_frechet_error = 0.0
        matched_count = 0

        for orig_geom, replay_geom in matched_pairs:
            # Endpoint distance
            orig_start = Point(orig_geom.coords[0])
            orig_end = Point(orig_geom.coords[-1])
            replay_start = Point(replay_geom.coords[0])
            replay_end = Point(replay_geom.coords[-1])

            start_dist = orig_start.distance(replay_start)
            end_dist = orig_end.distance(replay_end)
            endpoint_error = (start_dist + end_dist) / 2

            if endpoint_error > self.endpoint_tolerance:
                results['endpoint_errors'].append(endpoint_error)

            # Simplified Fréchet distance
            frechet_error = self._approximate_frechet_distance(orig_geom, replay_geom)

            if frechet_error > self.frechet_tolerance:
                results['frechet_errors'].append(frechet_error)

            total_endpoint_error += endpoint_error
            total_frechet_error += frechet_error
            matched_count += 1

        results['avg_endpoint_error'] = total_endpoint_error / max(matched_count, 1)
        results['avg_frechet_error'] = total_frechet_error / max(matched_count, 1)
        results['geometric_valid'] = (
            results['avg_endpoint_error'] <= self.endpoint_tolerance and
            results['avg_frechet_error'] <= self.frechet_tolerance and
            results.get('street_coverage', 0) > 0.1  # At least 10% coverage required
        )

        return results
    
    def _validate_topological(self, original: GrowthState, replayed: GrowthState) -> Dict[str, Any]:
        """Validate topological structure preservation."""
        results = {}
        
        # Block count comparison
        orig_blocks = len(original.blocks)
        replay_blocks = len(replayed.blocks)
        
        if orig_blocks > 0:
            block_error = abs(replay_blocks - orig_blocks) / orig_blocks
            results['block_count_error'] = block_error
        else:
            results['block_count_error'] = 0.0
        
        # Connectivity preservation
        orig_nodes = len(original.graph.nodes())
        replay_nodes = len(replayed.graph.nodes())
        orig_edges = len(original.graph.edges())
        replay_edges = len(replayed.graph.edges())
        
        connectivity_preserved = (
            abs(orig_nodes - replay_nodes) <= max(2, orig_nodes * 0.1) and
            abs(orig_edges - replay_edges) <= max(2, orig_edges * 0.1)
        )
        
        results['connectivity_preserved'] = connectivity_preserved
        results['topological_valid'] = (
            results['block_count_error'] <= self.block_count_tolerance and
            connectivity_preserved
        )
        
        return results
    
    def _validate_morphological(self, original: GrowthState, replayed: GrowthState) -> Dict[str, Any]:
        """Validate morphological properties like density and shape."""
        results = {}
        results['morphological_valid'] = True  # Placeholder
        return results
    
    def _match_streets_geometrically(self, orig_streets, replay_streets) -> Tuple[List[Tuple], Dict[str, Any]]:
        """
        FIXED: Proper geometric matching that penalizes non-matches and uses strict criteria.

        Key fixes:
        - No sampling: Compare ALL streets to get accurate coverage metrics
        - Strict geometric matching: Use Hausdorff distance, not just centroids
        - Coverage tracking: Return match statistics, not just matched pairs

        Returns:
            Tuple of (matched_pairs, coverage_stats)
        """
        logger.info(f"Starting comprehensive geometric matching: {len(orig_streets)} original, {len(replay_streets)} replayed")

        if orig_streets.empty:
            logger.warning("No original streets to match against")
            coverage_stats = {
                'street_coverage': 0.0,
                'matched_streets': 0,
                'total_original_streets': 0,
                'total_replayed_streets': len(replay_streets)
            }
            return [], coverage_stats

        if replay_streets.empty:
            logger.warning("No replayed streets to match with")
            coverage_stats = {
                'street_coverage': 0.0,
                'matched_streets': 0,
                'total_original_streets': len(orig_streets),
                'total_replayed_streets': 0
            }
            return [], coverage_stats

        # Build spatial index for replay streets - O(n log n)
        replay_geoms = [(geom, idx) for idx, geom in enumerate(replay_streets.geometry)
                       if geom and not geom.is_empty]

        if not replay_geoms:
            logger.warning("No valid replay geometries found")
            coverage_stats = {
                'street_coverage': 0.0,
                'matched_streets': 0,
                'total_original_streets': len(orig_streets),
                'total_replayed_streets': len(replay_streets)
            }
            return [], coverage_stats

        spatial_index = STRtree([geom for geom, _ in replay_geoms])
        logger.info(f"Built STRtree index with {len(replay_geoms)} geometries")

        matches = []
        strict_match_threshold = 1.0  # 1m for strict geometric matching
        loose_match_threshold = 5.0   # 5m for loose matching

        matched_replay_indices = set()

        # Match each original street against replay streets
        for idx, orig_street in orig_streets.iterrows():
            orig_geom = orig_street.geometry
            if not orig_geom or orig_geom.is_empty:
                continue

            # Query spatial index for nearby candidates
            candidate_indices = spatial_index.query(orig_geom.buffer(loose_match_threshold))

            if len(candidate_indices) == 0:
                continue

            # Find best match using Hausdorff distance (strict geometric similarity)
            best_match = None
            best_hausdorff = loose_match_threshold
            best_candidate_idx = None

            for candidate_idx in candidate_indices:
                candidate_geom, replay_idx = replay_geoms[candidate_idx]

                # Calculate Hausdorff distance (measures maximum deviation between geometries)
                hausdorff_dist = orig_geom.hausdorff_distance(candidate_geom)

                if hausdorff_dist < best_hausdorff:
                    best_hausdorff = hausdorff_dist
                    best_match = candidate_geom
                    best_candidate_idx = replay_idx

            # Only accept matches within strict threshold
            if best_match is not None and best_hausdorff <= strict_match_threshold:
                matches.append((orig_geom, best_match))
                matched_replay_indices.add(best_candidate_idx)

        # Calculate coverage statistics
        original_coverage = len(matches) / len(orig_streets) if len(orig_streets) > 0 else 0
        replay_coverage = len(matched_replay_indices) / len(replay_streets) if len(replay_streets) > 0 else 0

        coverage_stats = {
            'street_coverage': original_coverage,
            'matched_streets': len(matches),
            'total_original_streets': len(orig_streets),
            'total_replayed_streets': len(replay_streets),
            'replay_coverage': replay_coverage
        }

        logger.info(f"Geometric matching results:")
        logger.info(f"  Original streets matched: {len(matches)}/{len(orig_streets)} ({original_coverage:.1%})")
        logger.info(f"  Replay streets used: {len(matched_replay_indices)}/{len(replay_streets)} ({replay_coverage:.1%})")
        logger.info(f"  Average Hausdorff distance: {sum(orig_geom.hausdorff_distance(replay_geom) for orig_geom, replay_geom in matches[:10])/min(10, len(matches)):.2f}m (sample)")

        return matches, coverage_stats
    
    def _approximate_frechet_distance(self, geom1, geom2) -> float:
        """Approximate Fréchet distance between two geometries."""
        if len(geom1.coords) == 0 or len(geom2.coords) == 0:
            return 0.0
        
        # Sample points along both geometries
        points1 = [Point(c) for c in geom1.coords]
        points2 = [Point(c) for c in geom2.coords]
        
        max_min_dist = 0.0
        for p1 in points1:
            min_dist = min(p1.distance(p2) for p2 in points2)
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        FIXED: Calculate overall morphological fidelity score with proper coverage penalties.

        Now uses actual coverage metrics to penalize poor street reproduction.
        """
        scores = []

        # Geometric score (only for matched streets)
        if results.get('geometric_valid', False):
            geom_score = 1.0 - min(1.0, results.get('avg_endpoint_error', 0) / self.endpoint_tolerance)
            scores.append(geom_score)
        else:
            # No valid geometric matches found
            scores.append(0.0)

        # Topological score
        if results.get('topological_valid', False):
            topo_score = 1.0 - results.get('block_count_error', 0) / self.block_count_tolerance
            scores.append(topo_score)
        else:
            scores.append(0.0)

        # Coverage penalty: Use actual street coverage from geometric matching
        street_coverage = results.get('street_coverage', 0.0)

        if street_coverage < 0.1:  # Less than 10% coverage - critical failure
            coverage_penalty = 0.0  # Maximum penalty
        elif street_coverage < 0.3:  # Less than 30% coverage
            coverage_penalty = 0.2  # Heavy penalty
        elif street_coverage < 0.5:  # Less than 50% coverage
            coverage_penalty = 0.5  # Moderate penalty
        elif street_coverage < 0.7:  # Less than 70% coverage
            coverage_penalty = 0.8  # Light penalty
        else:
            coverage_penalty = 1.0  # No penalty

        scores.append(coverage_penalty)

        # Morphological score (placeholder for now)
        scores.append(1.0 if results.get('morphological_valid', True) else 0.5)

        final_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"Overall morphological score: {final_score:.2f} (components: {scores})")
        logger.info(f"Street coverage: {street_coverage:.1%} -> Coverage penalty: {coverage_penalty:.2f}")

        return final_score


def validate_trace_quality(
    original_state: GrowthState,
    replayed_state: GrowthState,
    trace: 'GrowthTrace'
) -> Dict[str, Any]:
    """
    Convenience function to validate trace quality using MorphologicalValidator.

    This is a wrapper around MorphologicalValidator.validate_replay() that provides
    a simpler interface for the dataset generator.

    Args:
        original_state: Original final city state
        replayed_state: State produced by replaying the trace
        trace: The growth trace (for additional metrics)

    Returns:
        Dict with validation metrics:
        - replay_fidelity: Overall morphological fidelity [0.0-1.0]
        - connectivity_preserved: Whether graph connectivity matches
        - morphological_valid: Whether validation passed thresholds
        - geometric_valid: Whether geometric properties match
        - topological_valid: Whether topological properties match
    """
    validator = MorphologicalValidator()
    results = validator.validate_replay(original_state, replayed_state)

    # Add trace-specific metrics
    results['num_actions_inferred'] = len(trace.actions)
    results['average_action_confidence'] = trace.average_confidence
    results['trace_length'] = len(trace.actions)

    return results
