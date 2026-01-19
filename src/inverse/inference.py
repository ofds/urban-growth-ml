#!/usr/bin/env python3
"""
Multi-Strategy Inference Engine
Advanced backward inference with mathematical strategies for urban growth patterns.
"""

from typing import List, Optional, Any, Dict, Tuple
import logging
import time
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.mixture import GaussianMixture
import networkx as nx
from rtree import index

from src.core.contracts import GrowthState, FrontierEdge
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType, compute_frontier_signature
from .skeleton import ArterialSkeletonExtractor
from .rewind import RewindEngine

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Comprehensive performance tracking for inference operations.

    Tracks timing, memory usage, and operation counts to identify bottlenecks.
    Includes bounds checking to prevent unlimited memory growth.
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize performance tracker with configurable bounds.

        Args:
            max_history_size: Maximum number of entries to keep in history lists
        """
        self.max_history_size = max_history_size
        self.reset()

    def reset(self):
        """Reset all performance metrics."""
        self.start_time = time.perf_counter()
        self.step_times = []
        self.operation_times = {}
        self.strategy_times = {}
        self.spatial_index_times = {}
        self.memory_usage = []
        self.step_metrics = []

    def start_operation(self, operation_name: str):
        """Start timing an operation."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        # Store start time
        self._current_ops = getattr(self, '_current_ops', {})
        self._current_ops[operation_name] = time.perf_counter()

    def end_operation(self, operation_name: str):
        """End timing an operation."""
        if operation_name in getattr(self, '_current_ops', {}):
            start_time = self._current_ops[operation_name]
            duration = time.perf_counter() - start_time
            self.operation_times[operation_name].append(duration)
            del self._current_ops[operation_name]

    def record_strategy_time(self, strategy_name: str, duration: float):
        """Record time spent in a strategy with bounds checking."""
        if strategy_name not in self.strategy_times:
            self.strategy_times[strategy_name] = []
        self.strategy_times[strategy_name].append(duration)

        # Bounds checking: keep only the most recent entries
        if len(self.strategy_times[strategy_name]) > self.max_history_size:
            self.strategy_times[strategy_name] = self.strategy_times[strategy_name][-self.max_history_size:]

    def record_spatial_index_time(self, operation: str, duration: float):
        """Record time spent in spatial index operations with bounds checking."""
        if operation not in self.spatial_index_times:
            self.spatial_index_times[operation] = []
        self.spatial_index_times[operation].append(duration)

        # Bounds checking: keep only the most recent entries
        if len(self.spatial_index_times[operation]) > self.max_history_size:
            self.spatial_index_times[operation] = self.spatial_index_times[operation][-self.max_history_size:]

    def record_step_metrics(self, step: int, streets_before: int, streets_after: int,
                          candidates_count: int, strategy_stats: Dict[str, int]):
        """Record metrics for a completed inference step with bounds checking."""
        step_duration = time.perf_counter() - self.start_time
        self.step_times.append(step_duration)

        self.step_metrics.append({
            'step': step,
            'duration': step_duration,
            'streets_before': streets_before,
            'streets_after': streets_after,
            'candidates_count': candidates_count,
            'strategy_stats': strategy_stats.copy()
        })

        # Bounds checking: keep only the most recent entries
        if len(self.step_times) > self.max_history_size:
            self.step_times = self.step_times[-self.max_history_size:]
        if len(self.step_metrics) > self.max_history_size:
            self.step_metrics = self.step_metrics[-self.max_history_size:]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_time = time.perf_counter() - self.start_time

        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                operation_stats[op_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'pct_total': sum(times) / total_time * 100
                }

        # Calculate strategy statistics
        strategy_stats = {}
        for strategy_name, times in self.strategy_times.items():
            if times:
                strategy_stats[strategy_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'pct_total': sum(times) / total_time * 100
                }

        # Calculate spatial index statistics
        spatial_stats = {}
        for op_name, times in self.spatial_index_times.items():
            if times:
                spatial_stats[op_name] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times)
                }

        # Calculate throughput metrics
        total_steps = len(self.step_times)
        total_streets_processed = sum(m['streets_before'] - m['streets_after'] for m in self.step_metrics)

        return {
            'total_time': total_time,
            'total_steps': total_steps,
            'total_streets_processed': total_streets_processed,
            'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            'streets_per_second': total_streets_processed / total_time if total_time > 0 else 0,
            'steps_per_second': total_steps / total_time if total_time > 0 else 0,
            'operation_stats': operation_stats,
            'strategy_stats': strategy_stats,
            'spatial_index_stats': spatial_stats,
            'step_metrics': self.step_metrics
        }

    def log_performance_summary(self):
        """Log a human-readable performance summary."""
        stats = self.get_summary_stats()

        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(f"Total time: {stats['total_time']:.2f}s")
        logger.info(f"Total steps: {stats['total_steps']}")
        logger.info(f"Streets processed: {stats['total_streets_processed']}")
        logger.info(f"Avg step time: {stats['avg_step_time']:.2f}s")
        logger.info(f"Streets per second: {stats['streets_per_second']:.2f}")
        logger.info(f"Steps per second: {stats['steps_per_second']:.2f}")

        # Log top time-consuming operations
        if stats['operation_stats']:
            logger.info("Top operations by time:")
            sorted_ops = sorted(stats['operation_stats'].items(),
                              key=lambda x: x[1]['total_time'], reverse=True)
            for op_name, op_stats in sorted_ops[:5]:
                logger.info(f"  {op_name}: {op_stats['total_time']:.3f}s ({op_stats['pct_total']:.1f}%)")

        # Log strategy performance
        if stats['strategy_stats']:
            logger.info("Strategy performance:")
            sorted_strategies = sorted(stats['strategy_stats'].items(),
                                     key=lambda x: x[1]['total_time'], reverse=True)
            for strategy_name, strategy_stats in sorted_strategies:
                logger.info(f"  {strategy_name}: {strategy_stats['total_time']:.3f}s ({strategy_stats['pct_total']:.1f}%)")


class SpatialIndex:
    """
    R-tree based spatial indexing for fast geometric queries.

    Provides O(log n) lookups for:
    - Street proximity searches
    - Frontier geometric queries
    - Block connectivity analysis
    """

    def __init__(self):
        self.street_index = None
        self.frontier_index = None
        self.block_index = None
        self.street_data = {}  # Use dict for O(1) lookup
        self.frontier_data = {}  # Use dict for O(1) lookup
        self.block_data = {}  # Use dict for O(1) lookup

    def build_street_index(self, streets_gdf):
        """Build R-tree index for streets."""
        logger.debug("Building spatial index for streets...")
        self.street_index = index.Index()
        self.street_data = {}

        for idx, street in streets_gdf.iterrows():
            if hasattr(street.geometry, 'bounds'):
                bounds = street.geometry.bounds  # (minx, miny, maxx, maxy)
                self.street_index.insert(idx, bounds)
                self.street_data[idx] = street

        logger.debug(f"Built street index with {len(self.street_data)} entries")

    def build_frontier_index(self, frontiers):
        """Build R-tree index for frontiers."""
        logger.debug("Building spatial index for frontiers...")
        self.frontier_index = index.Index()
        self.frontier_data = {}

        for i, frontier in enumerate(frontiers):
            if hasattr(frontier.geometry, 'bounds'):
                bounds = frontier.geometry.bounds
                self.frontier_index.insert(i, bounds)
                self.frontier_data[i] = frontier

        logger.debug(f"Built frontier index with {len(self.frontier_data)} entries")

    def build_block_index(self, blocks_gdf):
        """Build R-tree index for blocks."""
        logger.debug("Building spatial index for blocks...")
        self.block_index = index.Index()
        self.block_data = {}

        for idx, block in blocks_gdf.iterrows():
            if hasattr(block.geometry, 'bounds'):
                bounds = block.geometry.bounds
                self.block_index.insert(idx, bounds)
                self.block_data[idx] = block

        logger.debug(f"Built block index with {len(self.block_data)} entries")

    def find_nearby_streets(self, point: Point, radius: float) -> List:
        """Find streets within radius of a point."""
        if self.street_index is None:
            return []

        # Create bounding box for query
        x, y = point.x, point.y
        bbox = (x - radius, y - radius, x + radius, y + radius)

        # Get candidate streets from R-tree
        candidates = list(self.street_index.intersection(bbox))

        # Filter by actual distance
        nearby = []
        for idx in candidates:
            if idx in self.street_data:
                street = self.street_data[idx]
                if street.geometry.distance(point) <= radius:
                    nearby.append(street)

        return nearby

    def find_streets_intersecting_bbox(self, bbox) -> List:
        """Find streets intersecting a bounding box."""
        if self.street_index is None:
            return []

        candidates = list(self.street_index.intersection(bbox))
        streets = []

        for idx in candidates:
            if idx in self.street_data:
                streets.append(self.street_data[idx])

        return streets

    def find_frontiers_near_point(self, point: Point, radius: float) -> List:
        """Find frontiers within radius of a point."""
        if self.frontier_index is None:
            return []

        x, y = point.x, point.y
        bbox = (x - radius, y - radius, x + radius, y + radius)

        candidates = list(self.frontier_index.intersection(bbox))
        nearby = []

        for idx in candidates:
            if idx in self.frontier_data:
                frontier = self.frontier_data[idx]
                if frontier.geometry.distance(point) <= radius:
                    nearby.append(frontier)

        return nearby

    def find_blocks_containing_point(self, point: Point) -> List:
        """Find blocks containing a point."""
        if self.block_index is None:
            return []

        # Use point as a degenerate bbox
        bbox = (point.x, point.y, point.x, point.y)
        candidates = list(self.block_index.intersection(bbox))

        containing_blocks = []
        for idx in candidates:
            if idx in self.block_data:
                block = self.block_data[idx]
                if block.geometry.contains(point):
                    containing_blocks.append(block)

        return containing_blocks

    def remove_street(self, street_id, geometry):
        """Remove single street from index - O(log n)"""
        if self.street_index and street_id in self.street_data:
            bounds = geometry.bounds
            self.street_index.delete(street_id, bounds)
            del self.street_data[street_id]

    def update_frontiers_incremental(self, removed_frontiers, added_frontiers):
        """Update only changed frontiers - O(k log n) where k = changes"""
        # Remove old frontiers
        for frontier in removed_frontiers:
            if hasattr(frontier, 'frontier_id') and frontier.frontier_id in self.frontier_data:
                if hasattr(frontier.geometry, 'bounds'):
                    bounds = frontier.geometry.bounds
                    # Try to find the index - this is a limitation of R-tree
                    # For now, we'll rebuild the entire frontier index
                    # TODO: Consider using a different spatial index that supports individual removals
                    pass

        # Add new frontiers
        for frontier in added_frontiers:
            if hasattr(frontier.geometry, 'bounds'):
                bounds = frontier.geometry.bounds
                # Use frontier_id as index if available, otherwise use hash
                idx = getattr(frontier, 'frontier_id', hash(frontier.geometry.wkt))
                self.frontier_index.insert(idx, bounds)
                self.frontier_data[idx] = frontier

    def update_blocks_incremental(self, removed_blocks, added_blocks):
        """Update only changed blocks - O(k log n) where k = changes"""
        # Remove old blocks
        for block in removed_blocks:
            if hasattr(block, 'block_id') and block.block_id in self.block_data:
                if hasattr(block.geometry, 'bounds'):
                    bounds = block.geometry.bounds
                    # Similar limitation as frontiers
                    pass

        # Add new blocks
        for block in added_blocks:
            if hasattr(block.geometry, 'bounds'):
                bounds = block.geometry.bounds
                idx = getattr(block, 'block_id', hash(block.geometry.wkt))
                self.block_index.insert(idx, bounds)
                self.block_data[idx] = block


class BasicInferenceEngine:
    """
    Basic backward inference engine for synthetic cities.
    
    Implements simplified "peeling" heuristics for proof-of-concept.
    """
    
    def __init__(self):
        self.skeleton_extractor = ArterialSkeletonExtractor()
        self.rewind_engine = RewindEngine()
    
    def infer_trace(self, final_state: GrowthState, max_steps: int = 10000000,
                    initial_state: Optional[GrowthState] = None,
                    progress_callback: Optional[callable] = None) -> GrowthTrace:
        """
        Infer growth trace from final city state.
        
        Args:
            final_state: Final grown city state
            max_steps: Maximum inference steps  
            initial_state: Optional known initial state for testing
        
        Returns:
            Inferred GrowthTrace
        """
        logger.info("Starting basic inference...")
        
        if initial_state is None:
            # Extract arterial skeleton
            skeleton_edges, skeleton_streets = self.skeleton_extractor.extract_skeleton(
                final_state.streets, final_state.graph
            )
            initial_state = self.skeleton_extractor.create_skeleton_state(skeleton_streets, final_state)
            skeleton_edges_set = skeleton_edges
        else:
            logger.info(f"Using provided initial state with {len(initial_state.streets)} streets")
            skeleton_edges_set = set()
            for idx, street in initial_state.streets.iterrows():
                u, v = street.get('u'), street.get('v')
                if u and v:
                    skeleton_edges_set.add((min(u, v), max(u, v)))
        
        # Initialize inference
        current_state = final_state
        actions = []
        step = 0

        # BATCH COMPUTATION CACHING: Cache expensive computations
        cache_iteration = 0
        CACHE_INTERVAL = 50  # Recalculate every N steps
        cached_city_center = None
        cached_candidate_streets = None
        cached_frontier_distances = None

        logger.info(f"Inference setup: final={len(final_state.streets)} streets, initial={len(initial_state.streets)} streets, skeleton={len(skeleton_edges_set)} edges")

        # Main inference loop - rewind until we reach initial state
        while step < max_steps:
            # Check if we've reached the initial state
            if len(current_state.streets) <= len(initial_state.streets):
                logger.info(f"Reached initial state size at step {step}")
                break

            # BATCH COMPUTATION CACHING: Update cached values periodically
            if step % CACHE_INTERVAL == 0 or cached_city_center is None:
                cached_city_center = self._get_city_center(current_state)
                # Precompute frontier distances for dead-end frontiers
                dead_end_frontiers = [f for f in current_state.frontiers if f.frontier_type == "dead_end"]
                cached_frontier_distances = {}
                for frontier in dead_end_frontiers:
                    cached_frontier_distances[frontier] = self.distance_from_center(frontier.geometry, cached_city_center)
                cache_iteration = step
                logger.debug(f"Updated cached computations at step {step}")

            # Try to infer next action
            action = self.infer_most_recent_action(current_state, skeleton_edges_set)
            if action is None:
                logger.info(f"No more actions to infer at step {step}")
                break

            # PHASE 2: Capture complete state diff before rewind
            state_diff = self._compute_state_diff(current_state, action)

            # Try to rewind
            prev_state = self.rewind_engine.rewind_action(action, current_state)

            # Check if rewind actually worked
            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed at step {step} - state unchanged ({len(prev_state.streets)} >= {len(current_state.streets)})")
                break

            # PHASE 2: Update action with complete state diff
            action_with_diff = self._add_state_diff_to_action(action, state_diff)

            # Only record action if rewind succeeded
            actions.insert(0, action_with_diff)
            if step % 100 == 0 or step < 10:
                logger.info(f"Inference step {step}: streets {len(current_state.streets)} -> {len(prev_state.streets)}")
            current_state = prev_state
            step += 1
        
        # Create trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                "inference_method": "basic_peeling",
                "max_steps": max_steps,
                "steps_taken": step,
                "skeleton_streets": len(skeleton_edges_set)
            }
        )
        
        logger.info(f"Inference complete: {len(actions)} actions inferred")
        return trace


    def _is_minimal_state(self, state: GrowthState, skeleton_edges: set) -> bool:
        """Check if state is minimal (just skeleton or seed)."""
        # Simplified: check if we have very few streets
        return len(state.streets) <= max(5, len(skeleton_edges) * 2)
    

    def infer_most_recent_action(self, state: GrowthState, skeleton_edges: set):
        """Infer the most recently added action using simple heuristics."""
        center = self._get_city_center(state)

        # FRONTIER LOOKUP OPTIMIZATION: Build dictionary mapping edge tuples to frontiers
        frontier_lookup = {}
        for frontier in state.frontiers:
            edge_key = (min(frontier.edge_id[0], frontier.edge_id[1]),
                       max(frontier.edge_id[0], frontier.edge_id[1]))
            frontier_lookup[edge_key] = frontier

        # Find dead-end frontiers
        dead_end_frontiers = [f for f in state.frontiers if f.frontier_type == "dead_end"]

        if dead_end_frontiers:
            peripheral_frontier = max(
                dead_end_frontiers,
                key=lambda f: self.distance_from_center(f.geometry, center)
            )


            from shapely import wkt

            # CRITICAL FIX: Compute and store stable geometric ID
            stable_id = self._compute_stable_frontier_id(peripheral_frontier)

            # Compute geometric signature for stable frontier matching
            geometric_signature = compute_frontier_signature(peripheral_frontier)

            return InverseGrowthAction(
                action_type=ActionType.REMOVE_STREET,
                street_id=peripheral_frontier.frontier_id,  # Keep for debugging
                intent_params={
                    "direction": "peripheral_expansion",
                    "edge_u": str(peripheral_frontier.edge_id[0]),
                    "edge_v": str(peripheral_frontier.edge_id[1]),
                    "stable_id": stable_id  # ← ADD THIS
                },
                confidence=0.8,
                timestamp=len(state.streets),
                state_diff={
                    "geometry_wkt": wkt.dumps(peripheral_frontier.geometry),
                    "edgeid": peripheral_frontier.edge_id,
                    "frontier_type": peripheral_frontier.frontier_type,
                    "stable_id": stable_id  # ← ADD THIS TOO
                },
                action_metadata={
                    "geometric_signature": geometric_signature  # ← ADD GEOMETRIC SIGNATURE
                }
            )

        # Fallback for short streets
        candidate_streets = []
        for idx, street in state.streets.iterrows():
            geometry = street.geometry
            if not isinstance(geometry, LineString):
                continue

            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue

            edge_key = (min(u, v), max(u, v))
            if edge_key in skeleton_edges:
                continue

            # OPTIMIZED FRONTIER LOOKUP: O(1) dictionary lookup instead of O(n) linear search
            matching_frontier = frontier_lookup.get(edge_key)

            if matching_frontier:
                length = geometry.length
                candidate_streets.append((idx, length, street, matching_frontier))


        if candidate_streets:
            shortest_idx, length, street, frontier = min(candidate_streets, key=lambda x: x[1])

            from shapely import wkt

            # CRITICAL FIX: Compute and store stable geometric ID
            stable_id = self._compute_stable_frontier_id(frontier)

            # Compute geometric signature for stable frontier matching
            geometric_signature = compute_frontier_signature(frontier)

            return InverseGrowthAction(
                action_type=ActionType.REMOVE_STREET,
                street_id=frontier.frontier_id,
                intent_params={
                    'strategy': 'short_segment',
                    'edge_u': str(frontier.edge_id[0]),
                    'edge_v': str(frontier.edge_id[1]),
                    'stable_id': stable_id  # ← ADD THIS
                },
                realized_geometry={
                    'geometry_wkt': wkt.dumps(frontier.geometry),
                    'edgeid': frontier.edge_id,
                    'frontier_type': frontier.frontier_type,
                    'stable_id': stable_id  # ← ADD THIS TOO
                },
                confidence=0.6,
                timestamp=len(state.streets),
                action_metadata={
                    'geometric_signature': geometric_signature  # ← ADD GEOMETRIC SIGNATURE
                }
            )

        logger.warning(f"DEBUG: No actions found - cannot infer further")
        return None


    def _get_city_center(self, state: GrowthState):
        """Get approximate city center."""
        if state.city_bounds:
            return state.city_bounds.centroid
        # Fallback to mean of all street coordinates
        all_coords = []
        for idx, street in state.streets.iterrows():
            if hasattr(street.geometry, 'coords'):
                all_coords.extend(street.geometry.coords)
        if all_coords:
            x_coords = [c[0] for c in all_coords]
            y_coords = [c[1] for c in all_coords]
            return type('Point', (), {'x': sum(x_coords)/len(x_coords), 'y': sum(y_coords)/len(y_coords)})()
        return type('Point', (), {'x': 0, 'y': 0})()
    
    def distance_from_center(self, geometry: LineString, center) -> float:
        """Calculate distance from geometry to city center."""
        if hasattr(geometry, 'centroid'):
            geom_center = geometry.centroid
            return ((geom_center.x - center.x)**2 + (geom_center.y - center.y)**2)**0.5
        return 0.0

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry coordinates with consistent rounding."""
        import hashlib
        if not hasattr(frontier, 'geometry') or frontier.geometry is None:
            return "invalid_frontier"

        geom = frontier.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            # FIX: Use consistent 2-decimal precision for both coordinates
            start = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
            end = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
            frontier_type = getattr(frontier, 'frontier_type', 'unknown')

            hash_input = f"{start}_{end}_{frontier_type}".encode('utf-8')
            return hashlib.md5(hash_input).hexdigest()[:16]

        return "invalid_geometry"

    def _compute_state_diff(self, current_state: GrowthState, action: InverseGrowthAction) -> Dict[str, Any]:
        """
        Compute the complete state diff that this action represents.

        This captures exactly what streets are added/removed and how the graph changes,
        eliminating the need for frontier matching during replay.

        Args:
            current_state: State before rewind (contains the street to be removed)
            action: Action being rewound

        Returns:
            Dict containing complete state changes
        """
        state_diff = {
            'added_streets': [],    # Streets that will be added during replay
            'removed_streets': [],  # Streets that are removed during rewind
            'graph_changes': {},    # Node/edge changes
            'frontier_changes': {}  # Frontier state changes
        }

        # For EXTEND_FRONTIER actions, the street being removed during rewind
        # is the one that was added during forward growth
        if action.action_type == ActionType.EXTEND_FRONTIER:
            # Find the street that matches this action's edge
            edge_u = action.realized_geometry.get('edgeid', (None, None))[0] if action.realized_geometry else None
            edge_v = action.realized_geometry.get('edgeid', (None, None))[1] if action.realized_geometry else None

            if edge_u is not None and edge_v is not None:
                # Find the street with this edge
                for idx, street in current_state.streets.iterrows():
                    u, v = street.get('u'), street.get('v')
                    if (u == edge_u and v == edge_v) or (u == edge_v and v == edge_u):
                        # Store complete street data including geometry for replay
                        street_data = {
                            'edge_id': (min(u, v), max(u, v)),  # Normalized edge tuple for lookup
                            'u': u,
                            'v': v,
                            'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                            'osmid': street.get('osmid'),
                            'highway': street.get('highway'),
                            'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                        }
                        state_diff['added_streets'].append(street_data)
                        state_diff['removed_streets'].append(idx)
                        break

        # Store graph state information
        state_diff['graph_changes'] = {
            'nodes_before': current_state.graph.number_of_nodes(),
            'edges_before': current_state.graph.number_of_edges(),
            'nodes_after': None,  # Will be filled after rewind
            'edges_after': None   # Will be filled after rewind
        }

        # Store frontier state information
        state_diff['frontier_changes'] = {
            'frontiers_before': len(current_state.frontiers),
            'frontiers_after': None  # Will be filled after rewind
        }

        return state_diff

    def _add_state_diff_to_action(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
        """
        Create a new action with the state diff included.

        Since InverseGrowthAction is frozen, we need to create a new instance.

        Args:
            action: Original action
            state_diff: Computed state diff

        Returns:
            New action with state_diff populated
        """
        return InverseGrowthAction(
            action_type=action.action_type,
            street_id=action.street_id,
            intent_params=action.intent_params,
            realized_geometry=action.realized_geometry,
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=state_diff,  # Add the state diff
            action_metadata=action.action_metadata
        )


# =============================================================================
# ADVANCED MATHEMATICAL STRATEGIES
# =============================================================================

class InferenceStrategy:
    """
    Base class for inference strategies using mathematical approaches.

    Strategies score existing streets for removal, never invent geometry.
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """
        Generate candidate actions with confidence scores.

        Returns:
            List of (action, confidence) tuples for actions to rewind
        """
        raise NotImplementedError

    def _create_action_from_frontier(self, frontier, strategy_name: str, confidence: float,
                                   state: GrowthState, intent_params: Dict = None) -> Optional[InverseGrowthAction]:
        """Helper to create action from frontier with freshness validation."""
        from shapely import wkt

        # FRONTIER FRESHNESS GATE: Validate that frontier's edge exists in current graph
        if hasattr(frontier, 'edge_id') and frontier.edge_id:
            if not self._validate_frontier_freshness(frontier.edge_id, state):
                logger.debug(f"Frontier freshness validation failed for edge {frontier.edge_id} - skipping")
                return None

        # GEOMETRY-BASED RESOLUTION: Resolve frontier geometry to current graph edge
        current_edge_id = None
        if hasattr(frontier, 'geometry') and frontier.geometry:
            current_edge_id = self._resolve_frontier_to_current_edge(frontier.geometry, state)

        # If we can't resolve to a current edge, skip this frontier
        if current_edge_id is None:
            logger.debug(f"Could not resolve frontier geometry to current graph edge - skipping")
            return None

        stable_id = self._compute_stable_frontier_id(frontier)
        geometric_signature = compute_frontier_signature(frontier)

        if intent_params is None:
            intent_params = {}

        # Use resolved current edge IDs instead of stale frontier.edge_id
        intent_params.update({
            'strategy': strategy_name,
            'edge_u': current_edge_id[0],
            'edge_v': current_edge_id[1],
            'stable_id': stable_id
        })

        # For REMOVE_STREET actions, populate state_diff with the street that will be removed
        state_diff = {
            'geometry_wkt': wkt.dumps(frontier.geometry),
            'edgeid': current_edge_id,  # Use resolved current edge
            'frontier_type': getattr(frontier, 'frontier_type', 'unknown'),
            'stable_id': stable_id
        }

        # Find the street that corresponds to the resolved edge and add it to state_diff
        street_id = None
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                # Try both orderings since edges are undirected
                if (u == current_edge_id[0] and v == current_edge_id[1]) or \
                   (u == current_edge_id[1] and v == current_edge_id[0]):
                    street_id = idx
                    # Add the street data to state_diff
                    state_diff['added_streets'] = [{
                        'edge_id': current_edge_id,
                        'u': u,
                        'v': v,
                        'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                        'osmid': street.get('osmid'),
                        'highway': street.get('highway'),
                        'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                    }]
                    state_diff['removed_streets'] = [street_id]
                    break

        if street_id is None:
            logger.debug(f"No street found for resolved edge {current_edge_id} - skipping")
            # DEBUG: Show some sample streets from DataFrame
            sample_streets = []
            for i, (idx, street) in enumerate(state.streets.iterrows()):
                if i >= 5:  # Show first 5 streets
                    break
                u, v = street.get('u'), street.get('v')
                sample_streets.append(f"{idx}: u='{u}', v='{v}'")
            logger.debug(f"Sample streets from DataFrame: {sample_streets}")
            return None

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=street_id,
            intent_params=intent_params,
            confidence=confidence,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={
                'geometric_signature': geometric_signature
            }
        )

    def _resolve_frontier_to_current_edge(self, frontier_geometry, state: GrowthState) -> Optional[Tuple[int, int]]:
        """
        Resolve frontier geometry to current graph edge using morphological similarity.

        Uses geometric similarity (length, angle, position) rather than exact matching.
        Returns (u, v) tuple of the most similar valid street.
        """
        if not isinstance(frontier_geometry, LineString):
            return None

        logger.debug(f"Resolving frontier geometry: {frontier_geometry.wkt[:100]}...")

        # Calculate frontier properties
        frontier_length = frontier_geometry.length
        frontier_centroid = frontier_geometry.centroid
        frontier_angle = self._get_street_angle(frontier_geometry)

        best_score = 0
        best_street_id = None

        # Find most morphologically similar street
        for idx, street in state.streets.iterrows():
            if not isinstance(street.geometry, LineString):
                continue

            u, v = street.get('u'), street.get('v')
            if u is None or v is None:
                continue

            # Verify this edge exists in current graph
            if not (state.graph.has_edge(u, v) or state.graph.has_edge(v, u)):
                continue

            # Calculate morphological similarity score
            similarity_score = self._calculate_morphological_similarity(
                frontier_geometry, frontier_length, frontier_centroid, frontier_angle,
                street.geometry
            )

            if similarity_score > best_score:
                best_score = similarity_score
                best_street_id = idx

        # Return the best matching street's edge
        if best_street_id is not None and best_score > 0.5:  # Lowered similarity threshold
            street = state.streets.loc[best_street_id]
            u, v = street.get('u'), street.get('v')
            actual_edge = (min(u, v), max(u, v))
            logger.debug(f"Frontier resolved to street {best_street_id} with edge {actual_edge} (similarity {best_score:.3f})")
            return actual_edge

        logger.debug(f"Frontier resolution failed - best similarity {best_score:.3f} (threshold 0.5)")
        return None

    def _calculate_morphological_similarity(self, frontier_geom, frontier_length, frontier_centroid, frontier_angle, street_geom):
        """Calculate morphological similarity between frontier and street geometries."""
        # Length similarity (0-1 scale)
        street_length = street_geom.length
        length_diff = abs(frontier_length - street_length)
        max_length = max(frontier_length, street_length)
        length_score = 1.0 - (length_diff / max_length) if max_length > 0 else 1.0

        # Position similarity (distance-based)
        street_centroid = street_geom.centroid
        distance = frontier_centroid.distance(street_centroid)
        position_score = max(0, 1.0 - distance / 50.0)  # 50m radius

        # Angle similarity (0-1 scale)
        street_angle = self._get_street_angle(street_geom)
        angle_diff = abs(frontier_angle - street_angle) % 180
        min_angle_diff = min(angle_diff, 180 - angle_diff)
        angle_score = max(0, 1.0 - min_angle_diff / 45.0)  # 45 degree tolerance

        # Combined score (weighted average)
        return 0.4 * length_score + 0.4 * position_score + 0.2 * angle_score

    def _validate_frontier_freshness(self, edge_id: Tuple[int, int], state: GrowthState) -> bool:
        """
        Validate that a frontier's edge ID is still valid in current state.

        This is the freshness gate that prevents stale references.
        """
        if edge_id is None:
            return False
        u, v = edge_id
        return state.graph.has_edge(u, v) or state.graph.has_edge(v, u)

    def _create_action_from_frontier(self, frontier, strategy_name: str, confidence: float,
                                   state: GrowthState, intent_params: Dict = None) -> Optional[InverseGrowthAction]:
        """Helper to create action from frontier with freshness validation."""
        from shapely import wkt

        # FRONTIER FRESHNESS GATE: Validate that frontier's edge exists in current graph
        if hasattr(frontier, 'edge_id') and frontier.edge_id:
            if not self._validate_frontier_freshness(frontier.edge_id, state):
                logger.debug(f"Frontier freshness validation failed for edge {frontier.edge_id} - skipping")
                return None

        # GEOMETRY-BASED RESOLUTION: Resolve frontier geometry to current graph edge
        current_edge_id = None
        if hasattr(frontier, 'geometry') and frontier.geometry:
            current_edge_id = self._resolve_frontier_to_current_edge(frontier.geometry, state)

        # If we can't resolve to a current edge, skip this frontier
        if current_edge_id is None:
            logger.debug(f"Could not resolve frontier geometry to current graph edge - skipping")
            return None

        stable_id = self._compute_stable_frontier_id(frontier)
        geometric_signature = compute_frontier_signature(frontier)

        if intent_params is None:
            intent_params = {}

        # Use resolved current edge IDs instead of stale frontier.edge_id
        intent_params.update({
            'strategy': strategy_name,
            'edge_u': current_edge_id[0],
            'edge_v': current_edge_id[1],
            'stable_id': stable_id
        })

        # For REMOVE_STREET actions, populate state_diff with the street that will be removed
        state_diff = {
            'geometry_wkt': wkt.dumps(frontier.geometry),
            'edgeid': current_edge_id,  # Use resolved current edge
            'frontier_type': getattr(frontier, 'frontier_type', 'unknown'),
            'stable_id': stable_id
        }

        # Find the street that corresponds to the resolved edge and add it to state_diff
        street_id = None
        logger.debug(f"Looking for street with edge {current_edge_id}")
        for idx, street in state.streets.iterrows():
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                street_edge = (min(u, v), max(u, v))
                #logger.debug(f"Checking street {idx}: u='{u}', v='{v}' -> edge {street_edge}")
                if street_edge == current_edge_id:
                    street_id = str(idx)  # Convert to string as required by InverseGrowthAction
                    logger.debug(f"Found matching street {street_id}")
                    # Add the street data to state_diff
                    state_diff['added_streets'] = [{
                        'edge_id': current_edge_id,
                        'u': u,
                        'v': v,
                        'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                        'osmid': street.get('osmid'),
                        'highway': street.get('highway'),
                        'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                    }]
                    state_diff['removed_streets'] = [street_id]
                    break

        if street_id is None:
            logger.debug(f"No street found for resolved edge {current_edge_id} - skipping")
            # DEBUG: Show some sample streets from DataFrame
            sample_streets = []
            for i, (idx, street) in enumerate(state.streets.iterrows()):
                if i >= 3:  # Show first 3 streets
                    break
                u, v = street.get('u'), street.get('v')
                sample_streets.append(f"{idx}: u='{u}', v='{v}'")
            logger.debug(f"Sample streets from DataFrame: {sample_streets}")
            return None

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=street_id,
            intent_params=intent_params,
            confidence=confidence,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={
                'geometric_signature': geometric_signature
            }
        )

    def _create_action_from_street(self, street_id, street, strategy_name: str, confidence: float, state: GrowthState) -> Optional[InverseGrowthAction]:
        """Create action directly from street data."""
        from shapely import wkt

        # Ensure street_id is a string
        street_id_str = str(street_id)

        # Get edge information
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            logger.debug(f"Street {street_id} missing edge information (u={u}, v={v})")
            return None

        # CRITICAL VALIDATION: Ensure the street's edge exists in the current graph
        edge_exists = state.graph.has_edge(u, v) or state.graph.has_edge(v, u)
        if not edge_exists:
            logger.debug(f"Street {street_id} edge ({u}, {v}) does not exist in current graph - skipping")
            return None

        # Create stable ID from street geometry
        stable_id = self._compute_stable_frontier_id(type('MockFrontier', (), {'geometry': street.geometry})())

        # Create intent params
        intent_params = {
            'strategy': strategy_name,
            'edge_u': str(u),
            'edge_v': str(v),
            'stable_id': stable_id
        }

        # Create state diff with street data
        state_diff = {
            'geometry_wkt': wkt.dumps(street.geometry),
            'edgeid': (min(u, v), max(u, v)),
            'frontier_type': 'street_removal',
            'stable_id': stable_id,
            'added_streets': [{
                'edge_id': (min(u, v), max(u, v)),
                'u': u,
                'v': v,
                'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                'osmid': street.get('osmid'),
                'highway': street.get('highway'),
                'length': street.geometry.length if hasattr(street.geometry, 'length') else None
            }],
            'removed_streets': [street_id_str]
        }

        # Compute geometric signature
        geometric_signature = compute_frontier_signature(type('MockFrontier', (), {'geometry': street.geometry})())

        return InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=street_id_str,
            intent_params=intent_params,
            confidence=confidence,
            timestamp=len(state.streets),
            state_diff=state_diff,
            action_metadata={
                'geometric_signature': geometric_signature
            }
        )

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry coordinates."""
        import hashlib
        if not hasattr(frontier, 'geometry') or frontier.geometry is None:
            return "invalid_frontier"

        geom = frontier.geometry
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            start = (round(geom.coords[0][0], 2), round(geom.coords[0][1], 2))
            end = (round(geom.coords[-1][0], 2), round(geom.coords[-1][1], 2))
            frontier_type = getattr(frontier, 'frontier_type', 'unknown')

            hash_input = f"{start}_{end}_{frontier_type}".encode('utf-8')
            return hashlib.md5(hash_input).hexdigest()[:16]
        return "invalid_geometry"


class FractalPatternStrategy(InferenceStrategy):
    """
    Fractal Pattern Strategy: Detect self-similar patterns and continue them.

    Mathematical Approach:
    - Analyzes scaling properties of street networks
    - Uses fractal dimension estimation
    - Detects self-similar growth patterns
    """

    def __init__(self):
        super().__init__("fractal_pattern", weight=1.2)

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        candidates = []

        # Debug: Log skeleton edges info
        logger.debug(f"FractalPatternStrategy: Received {len(skeleton_edges)} skeleton edges")
        if len(skeleton_edges) > 0:
            sample_edges = list(skeleton_edges)[:3]
            logger.debug(f"FractalPatternStrategy: Sample skeleton edges: {sample_edges}")

        # Score existing streets for removal based on fractal pattern disruption
        for street_id in state.streets.index:
            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Check if this street's edge is part of the skeleton - don't remove skeleton streets
            u, v = street.get('u'), street.get('v')
            if u is not None and v is not None:
                edge_key = (min(u, v), max(u, v))  # Normalize edge order
                if edge_key in skeleton_edges:
                    logger.debug(f"FractalPatternStrategy: Skipping skeleton street {street_id} with edge {edge_key}")
                    continue  # Don't remove skeleton streets

            # Score based on how much removing this street would disrupt fractal patterns
            confidence = self._score_fractal_disruption(street, state.streets, skeleton_edges)

            if confidence > 0.1:
                # Debug: Log street selection details
                logger.debug(f"FractalPatternStrategy: Selecting street {street_id} with edge ({u}, {v}), confidence {confidence:.3f}")

                # Create action directly from street
                action = self._create_action_from_street(street_id, street, self.name, confidence, state)
                if action is not None:
                    candidates.append((action, confidence))
                    logger.debug(f"FractalPatternStrategy: Created action for street {street_id}")
                else:
                    logger.debug(f"FractalPatternStrategy: Failed to create action for street {street_id}")

        logger.debug(f"FractalPatternStrategy: Generated {len(candidates)} candidates")
        return candidates

    def _score_fractal_disruption(self, street, streets_gdf, skeleton_edges) -> float:
        """Score how disruptive removing this street would be to fractal patterns."""
        # Simplified: score based on street length and connectivity
        length_score = min(1.0, street.geometry.length / 100.0)  # Prefer longer streets
        connectivity_score = 1.0  # Could be based on degree in graph

        return 0.6 * length_score + 0.4 * connectivity_score

    def _compute_fractal_dimension(self, streets_gdf) -> float:
        """Estimate fractal dimension using box-counting method."""
        if len(streets_gdf) < 10:
            return 1.5  # Default for small networks

        # Get all street coordinates
        all_coords = []
        for idx, street in streets_gdf.iterrows():
            if hasattr(street.geometry, 'coords'):
                all_coords.extend(street.geometry.coords)

        if len(all_coords) < 20:
            return 1.5

        points = np.array(all_coords)

        # Simple box-counting with different box sizes
        min_coord = points.min(axis=0)
        max_coord = points.max(axis=0)
        diagonal = np.linalg.norm(max_coord - min_coord)

        box_sizes = [diagonal / (2 ** i) for i in range(3, 8)]  # 8 to 128 boxes
        box_counts = []

        for box_size in box_sizes:
            # Count occupied boxes
            x_bins = np.arange(min_coord[0], max_coord[0] + box_size, box_size)
            y_bins = np.arange(min_coord[1], max_coord[1] + box_size, box_size)

            occupied = set()
            for point in points:
                x_idx = np.digitize(point[0], x_bins) - 1
                y_idx = np.digitize(point[1], y_bins) - 1
                occupied.add((x_idx, y_idx))

            box_counts.append(len(occupied))

        # Linear regression for fractal dimension
        if len(box_counts) >= 3:
            log_sizes = np.log(1 / np.array(box_sizes))
            log_counts = np.log(box_counts)

            # Simple linear regression
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            return max(1.0, min(2.0, slope))  # Clamp to reasonable range

        return 1.5

    def _is_fractal_continuation(self, frontier, streets_gdf, fractal_dim: float) -> bool:
        """Check if extending this frontier would continue fractal pattern."""
        if not isinstance(frontier.geometry, LineString):
            return False

        # Check if frontier direction aligns with existing fractal patterns
        frontier_angle = self._get_street_angle(frontier.geometry)

        # Count streets with similar angles
        similar_angle_count = 0
        total_streets = 0

        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                total_streets += 1
                street_angle = self._get_street_angle(street.geometry)
                angle_diff = abs(street_angle - frontier_angle) % 180
                if min(angle_diff, 180 - angle_diff) < 15:  # 15 degree tolerance
                    similar_angle_count += 1

        # Fractal continuation if angle is common in network
        return similar_angle_count / max(total_streets, 1) > 0.1

    def _score_fractal_fit(self, frontier, streets_gdf, fractal_dim: float, cached_angles=None) -> float:
        """Score how well frontier fits fractal continuation using vectorized operations."""
        if not self._is_fractal_continuation_vectorized(frontier, streets_gdf, fractal_dim, cached_angles):
            return 0.0

        # VECTORIZED ANGLE SCORING: Pre-compute all angles if not cached
        if cached_angles is None:
            frontier_angle = self._get_street_angle(frontier.geometry)
            street_angles = self._extract_street_angles_vectorized(streets_gdf)
        else:
            frontier_angle = cached_angles['frontier_angle']
            street_angles = cached_angles['street_angles']

        # Vectorized angle difference calculation
        angle_diffs = np.abs(street_angles - frontier_angle) % 180
        min_diffs = np.minimum(angle_diffs, 180 - angle_diffs)

        # Count angles within tolerance (15 degrees)
        angle_matches = np.sum(min_diffs < 15)
        angle_score = angle_matches / max(len(street_angles), 1)

        # VECTORIZED LENGTH SCORING
        frontier_length = frontier.geometry.length
        if cached_angles and 'street_lengths' in cached_angles:
            street_lengths = cached_angles['street_lengths']
        else:
            street_lengths = self._extract_street_lengths_vectorized(streets_gdf)

        if len(street_lengths) > 0:
            mean_length = np.mean(street_lengths)
            std_length = np.std(street_lengths)
            if std_length == 0:
                std_length = 1.0
            length_score = 1 / (1 + abs(frontier_length - mean_length) / std_length)
        else:
            length_score = 0.5

        return 0.7 * angle_score + 0.3 * length_score

    def _extract_street_angles_vectorized(self, streets_gdf) -> np.ndarray:
        """Extract all street angles in a vectorized way."""
        angles = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                angles.append(self._get_street_angle(street.geometry))
        return np.array(angles)

    def _extract_street_lengths_vectorized(self, streets_gdf) -> np.ndarray:
        """Extract all street lengths in a vectorized way."""
        lengths = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                lengths.append(street.geometry.length)
        return np.array(lengths)

    def _is_fractal_continuation_vectorized(self, frontier, streets_gdf, fractal_dim: float, cached_angles=None) -> bool:
        """Check if extending this frontier would continue fractal pattern (vectorized)."""
        if not isinstance(frontier.geometry, LineString):
            return False

        # VECTORIZED: Pre-compute all angles once
        if cached_angles is None:
            frontier_angle = self._get_street_angle(frontier.geometry)
            street_angles = self._extract_street_angles_vectorized(streets_gdf)
        else:
            frontier_angle = cached_angles['frontier_angle']
            street_angles = cached_angles['street_angles']

        # Vectorized angle comparison
        angle_diffs = np.abs(street_angles - frontier_angle) % 180
        min_diffs = np.minimum(angle_diffs, 180 - angle_diffs)

        # Count similar angles (within 15 degrees)
        similar_count = np.sum(min_diffs < 15)

        # Fractal continuation if angle is common in network (>10% of streets)
        return similar_count / max(len(street_angles), 1) > 0.1

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180  # Normalize to 0-180


class AngleHarmonizationStrategy(InferenceStrategy):
    """
    Angle Harmonization Strategy: Add streets matching dominant angle distributions.

    Mathematical Approach:
    - Fits mixture models to street angle distributions
    - Identifies harmonic angle patterns
    - Proposes streets that fit the statistical model
    """

    def __init__(self):
        super().__init__("angle_harmonization", weight=1.1)

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        candidates = []

        # PRE-COMPUTE: Extract all angles once instead of per street
        all_angles = self._extract_street_angles_vectorized(state.streets)
        angle_model = self._fit_angle_distribution(all_angles)

        # Score existing streets for removal based on angle harmonization disruption
        for street_id in state.streets.index:
            if street_id in skeleton_edges:
                continue  # Don't remove skeleton streets

            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Score based on how much removing this street would disrupt angle harmonization
            # Use pre-computed model instead of recomputing
            confidence = self._score_angle_disruption_cached(street, all_angles, angle_model)

            if confidence > 0.1:
                # Find corresponding frontier
                frontier = None
                u, v = street.get('u'), street.get('v')
                if u is not None and v is not None:
                    for f in state.frontiers:
                        if hasattr(f, 'edge_id') and f.edge_id == (u, v) or f.edge_id == (v, u):
                            frontier = f
                            break

                if frontier is not None:
                    action = self._create_action_from_frontier(frontier, self.name, confidence, state)
                    candidates.append((action, confidence))

        logger.debug(f"AngleHarmonizationStrategy: Generated {len(candidates)} candidates")
        return candidates

    def _score_angle_disruption(self, street, streets_gdf) -> float:
        """Score how disruptive removing this street would be to angle harmonization."""
        # Simplified: score based on street length (shorter streets are more likely recent additions)
        length = street.geometry.length
        return min(1.0, length / 100.0)  # Prefer shorter streets

    def _score_angle_disruption_cached(self, street, all_angles: np.ndarray, angle_model) -> float:
        """Score angle disruption using pre-computed angles and model."""
        if not isinstance(street.geometry, LineString):
            return 0.1

        # Get street angle
        street_angle = self._get_street_angle(street.geometry)

        # Score how well this street fits the overall angle distribution
        fit_score = self._score_angle_fit(street_angle, angle_model)

        # Prefer removing streets that are outliers (don't fit well)
        # Also consider length as secondary factor
        length_score = min(1.0, street.geometry.length / 100.0)

        # Combine: prefer short streets that are angle outliers
        return 0.7 * (1.0 - fit_score) + 0.3 * length_score

    def _extract_street_angles(self, streets_gdf) -> np.ndarray:
        """Extract angles from all streets."""
        angles = []

        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                angle = self._get_street_angle(street.geometry)
                angles.append(angle)

        return np.array(angles)

    def _extract_street_angles_vectorized(self, streets_gdf) -> np.ndarray:
        """Extract all street angles in a vectorized way."""
        angles = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                angles.append(self._get_street_angle(street.geometry))
        return np.array(angles)

    def _fit_angle_distribution(self, angles: np.ndarray):
        """Fit Gaussian mixture model to angle distribution."""
        if len(angles) < 3:
            return None

        # Normalize angles to 0-180 range
        angles_norm = angles % 180

        # Fit GMM with 2-4 components
        best_model = None
        best_bic = float('inf')

        for n_components in range(2, min(5, len(angles) // 3 + 1)):
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(angles_norm.reshape(-1, 1))
                bic = gmm.bic(angles_norm.reshape(-1, 1))

                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
            except:
                continue

        return best_model

    def _score_angle_fit(self, angle: float, model) -> float:
        """Score how well angle fits the distribution model."""
        if model is None:
            return 0.0

        angle_norm = angle % 180
        log_likelihood = model.score_samples([[angle_norm]])[0]

        # Convert to probability-like score
        score = 1 / (1 + np.exp(-log_likelihood))

        # Boost score if angle is near a component mean
        means = model.means_.flatten()
        min_distance = min(abs(angle_norm - mean) for mean in means)
        min_distance = min(min_distance, 180 - min_distance)  # Handle circular nature

        proximity_bonus = max(0, 1 - min_distance / 10)  # Bonus within 10 degrees

        return min(1.0, score + 0.2 * proximity_bonus)

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180


class ConnectivityDrivenStrategy(InferenceStrategy):
    """
    Connectivity-Driven Strategy: Fill gaps in street connectivity using graph theory.

    Mathematical Approach:
    - Analyzes graph connectivity and articulation points
    - Identifies critical missing connections
    - Uses betweenness centrality to prioritize connections
    """

    def __init__(self):
        super().__init__("connectivity_driven", weight=1.4)

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        candidates = []

        # Score existing streets for removal based on connectivity impact
        for street_id in state.streets.index:
            if street_id in skeleton_edges:
                continue  # Don't remove skeleton streets

            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Score based on how removing this street affects connectivity
            confidence = self._score_connectivity_impact(street, state.streets, state.graph)

            if confidence > 0.1:
                # Find corresponding frontier
                frontier = None
                u, v = street.get('u'), street.get('v')
                if u is not None and v is not None:
                    for f in state.frontiers:
                        if hasattr(f, 'edge_id') and f.edge_id == (u, v) or f.edge_id == (v, u):
                            frontier = f
                            break

                if frontier is not None:
                    action = self._create_action_from_frontier(frontier, self.name, confidence, state)
                    candidates.append((action, confidence))

        return candidates[:10]  # Limit candidates

    def _score_connectivity_impact(self, street, streets_gdf, graph) -> float:
        """Score how removing this street would affect connectivity."""
        # Simplified: score based on degree (higher degree = more important for connectivity)
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return 0.1

        degree_u = graph.degree[u] if graph.has_node(u) else 0
        degree_v = graph.degree[v] if graph.has_node(v) else 0

        # Higher degree means more important for connectivity
        avg_degree = (degree_u + degree_v) / 2.0
        connectivity_score = min(1.0, avg_degree / 4.0)  # Normalize by typical degree

        return connectivity_score

    def _build_street_graph(self, streets_gdf) -> nx.Graph:
        """Build NetworkX graph from street geometries."""
        graph = nx.Graph()

        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                coords = list(street.geometry.coords)
                if len(coords) >= 2:
                    start = coords[0]
                    end = coords[-1]

                    # Add nodes with positions
                    graph.add_node(start, pos=start)
                    graph.add_node(end, pos=end)

                    # Add edge
                    graph.add_edge(start, end, street_idx=idx)

        return graph

    def _create_synthetic_frontier_from_node(self, node, state: GrowthState):
        """Create frontier from articulation point."""
        # Create a small extension from the articulation point
        pos = node  # node is already (x, y) coordinates
        extended_pos = (pos[0] + 50, pos[1] + 50)  # 50m extension

        line = LineString([pos, extended_pos])

        class MockFrontier:
            def __init__(self, geometry, edge_id):
                self.geometry = geometry
                self.edge_id = edge_id
                self.frontier_id = f"synthetic_articulation_{hash(edge_id)}"
                self.frontier_type = "synthetic_articulation"

        edge_id = (hash(pos), hash(extended_pos))
        return MockFrontier(line, edge_id)

    def _create_synthetic_frontier_from_nodes(self, node1, node2, state: GrowthState):
        """Create frontier connecting two nodes."""
        line = LineString([node1, node2])

        class MockFrontier:
            def __init__(self, geometry, edge_id):
                self.geometry = geometry
                self.edge_id = edge_id
                self.frontier_id = f"synthetic_connection_{hash(edge_id)}"
                self.frontier_type = "synthetic_connection"

        edge_id = (hash(node1), hash(node2))
        return MockFrontier(line, edge_id)


class MorphologicalPatternStrategy(InferenceStrategy):
    """
    Morphological Pattern Strategy: Detect and extend urban morphological patterns.

    Mathematical Approach:
    - Analyzes street patterns using morphological operations
    - Detects grid, radial, and organic patterns
    - Extends patterns using morphological dilation
    """

    def __init__(self):
        super().__init__("morphological_pattern", weight=1.5)

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        candidates = []

        # Score existing streets for removal based on morphological pattern disruption
        for street_id in state.streets.index:
            if street_id in skeleton_edges:
                continue  # Don't remove skeleton streets

            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Score based on how much removing this street would disrupt morphological patterns
            confidence = self._score_morphological_disruption(street, state.streets)

            if confidence > 0.1:
                # Find corresponding frontier
                frontier = None
                u, v = street.get('u'), street.get('v')
                if u is not None and v is not None:
                    for f in state.frontiers:
                        if hasattr(f, 'edge_id') and f.edge_id == (u, v) or f.edge_id == (v, u):
                            frontier = f
                            break

                if frontier is not None:
                    action = self._create_action_from_frontier(frontier, self.name, confidence, state)
                    candidates.append((action, confidence))

        return candidates[:50]  # Limit candidates for performance

    def _score_morphological_disruption(self, street, streets_gdf) -> float:
        """Score how disruptive removing this street would be to morphological patterns."""
        # Simplified: score based on pattern type detection
        pattern_type = self._detect_pattern_type(streets_gdf)

        if pattern_type == "grid":
            # For grid patterns, score based on orthogonality
            angle = self._get_street_angle(street.geometry)
            normalized_angle = angle % 90
            orthogonality = min(normalized_angle, 90 - normalized_angle) / 45.0  # 0-1 scale
            return 1.0 - orthogonality  # Prefer removing non-orthogonal streets
        else:
            # For organic patterns, prefer removing shorter streets
            length_score = min(1.0, street.geometry.length / 50.0)
            return 1.0 - length_score  # Prefer removing shorter streets

    def _detect_pattern_type(self, streets_gdf) -> str:
        """Detect the dominant pattern type in the street network."""
        angles = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                angle = self._get_street_angle(street.geometry)
                angles.append(angle)

        if len(angles) < 10:
            return "organic"

        angles = np.array(angles)

        # Check for grid patterns (orthogonal angles)
        orthogonal_count = 0
        for angle in angles:
            # Check if angle is close to 0, 90, 180 degrees
            normalized_angle = angle % 90
            if min(normalized_angle, 90 - normalized_angle) < 15:
                orthogonal_count += 1

        if orthogonal_count / len(angles) > 0.6:
            return "grid"

        # Check for radial patterns (angles converging to center)
        # This would require more complex analysis
        return "organic"

    def _generate_grid_extensions(self, state: GrowthState, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate extensions for grid patterns."""
        candidates = []

        # Find frontiers that would complete grid intersections
        frontiers = list(state.frontiers)[:50]  # Limit for performance

        for frontier in frontiers:
            if isinstance(frontier.geometry, LineString):
                # Check if this frontier would complete a grid intersection
                confidence = self._score_grid_completion(frontier, state.streets, spatial_index)

                if confidence > 0.4:
                    action = self._create_action_from_frontier(
                        frontier, "morphological_pattern", confidence, state,
                        intent_params={'pattern_type': 'grid', 'completion_score': confidence}
                    )
                    candidates.append((action, confidence))

        return candidates

    def _generate_radial_extensions(self, state: GrowthState, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate extensions for radial patterns."""
        # Simplified radial pattern extension
        return []

    def _generate_organic_extensions(self, state: GrowthState, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate extensions for organic patterns."""
        candidates = []

        # Use fractal pattern strategy for organic patterns
        fractal_strategy = FractalPatternStrategy()
        return fractal_strategy.generate_candidates(state, set(), spatial_index)

    def _score_grid_completion(self, frontier, streets_gdf, spatial_index: Optional[SpatialIndex] = None) -> float:
        """Score how well frontier completes a grid pattern."""
        if not isinstance(frontier.geometry, LineString):
            return 0.0

        frontier_angle = self._get_street_angle(frontier.geometry)

        # Count nearby streets with similar angles (parallel)
        parallel_count = 0
        perpendicular_count = 0

        if spatial_index:
            nearby_streets = spatial_index.find_nearby_streets(frontier.geometry.centroid, 100.0)
        else:
            nearby_streets = []
            for idx, street in streets_gdf.iterrows():
                if isinstance(street.geometry, LineString):
                    if street.geometry.distance(frontier.geometry.centroid) < 100:
                        nearby_streets.append(street)

        for street in nearby_streets:
            if isinstance(street.geometry, LineString):
                street_angle = self._get_street_angle(street.geometry)

                # Check parallel (similar angle)
                angle_diff = abs(street_angle - frontier_angle) % 180
                if min(angle_diff, 180 - angle_diff) < 15:
                    parallel_count += 1

                # Check perpendicular (90 degree difference)
                angle_diff = abs(street_angle - frontier_angle) % 180
                perp_diff = abs(angle_diff - 90)
                if perp_diff < 15:
                    perpendicular_count += 1

        # Grid completion score: bonus for both parallel and perpendicular streets
        parallel_score = min(1.0, parallel_count / 3.0)  # Max score with 3+ parallel
        perpendicular_score = min(1.0, perpendicular_count / 2.0)  # Max score with 2+ perpendicular

        return 0.6 * parallel_score + 0.4 * perpendicular_score

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180


class BlockCentroidStrategy(InferenceStrategy):
    """
    Block Centroid Strategy: Close incomplete blocks by connecting to centroids.

    Mathematical Approach:
    - Identifies incomplete blocks (planar faces, graph cycles, block polygons)
    - Generates candidate edges that close block perimeters
    - Scores candidates by centroid alignment and geometric constraints
    - Uses centroids as scoring signals, not direct connection targets
    """

    def __init__(self):
        super().__init__("block_centroid", weight=1.3)
        self.max_block_area = 50000  # m²
        self.max_missing_edges = 2

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        candidates = []

        # BLOCK-FIRST REFACTOR: Iterate over blocks first, then frontiers within each block
        # This prevents O(N²) complexity by making each block independent

        # Step 1: Identify top incomplete blocks (prioritize by incompleteness)
        incomplete_blocks = self._identify_incomplete_blocks(state)

        # Limit to MAX_BLOCKS to prevent O(N²) scaling
        max_blocks = getattr(self, 'MAX_BLOCKS', MultiStrategyInferenceEngine.MAX_BLOCKS)
        top_blocks = self._prioritize_blocks_by_incompleteness(incomplete_blocks)[:max_blocks]

        # Step 2: For each block, generate candidates from its relevant streets
        max_candidates = getattr(self, 'MAX_CANDIDATES', MultiStrategyInferenceEngine.MAX_CANDIDATES)
        for block_info in top_blocks:
            # Check remaining capacity before processing block
            remaining_capacity = max_candidates - len(candidates)
            if remaining_capacity <= 0:
                break

            block_candidates = self._generate_candidates_for_block(block_info, state, skeleton_edges, spatial_index, remaining_capacity)
            candidates.extend(block_candidates)

            # Early exit if we hit MAX_CANDIDATES
            if len(candidates) >= max_candidates:
                break

        # Limit final candidates
        candidates = candidates[:max_candidates]

        logger.debug(f"BlockCentroidStrategy: Generated {len(candidates)} candidates from {len(top_blocks)} blocks")
        return candidates

    def _prioritize_blocks_by_incompleteness(self, blocks: List[Dict]) -> List[Dict]:
        """Prioritize blocks by how incomplete they are (most incomplete first)."""
        def incompleteness_score(block_info: Dict) -> float:
            """Calculate incompleteness score for a block."""
            if block_info['type'] == 'planar_face':
                # Score based on missing edges and area
                missing_penalty = block_info.get('missing_edges', 0) / self.max_missing_edges
                area_penalty = min(1.0, block_info.get('polygon', type('obj', (), {'area': 0})()).area / self.max_block_area)
                return missing_penalty + area_penalty
            elif block_info['type'] == 'cycle_chord':
                # Score based on number of missing chords
                return len(block_info.get('missing_chords', [])) / 3.0  # Normalize by typical max
            elif block_info['type'] == 'polygon_open':
                # Score based on number of open frontiers
                return len(block_info.get('open_frontiers', [])) / 4.0  # Normalize by typical max
            return 0.0

        # Sort by incompleteness score (highest first)
        return sorted(blocks, key=incompleteness_score, reverse=True)

    def _generate_candidates_for_block(self, block_info: Dict, state: GrowthState,
                                     skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None,
                                     remaining_capacity: int = 50) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate candidates for a specific block."""
        candidates = []

        # Get streets that are relevant to this block
        relevant_streets = self._find_streets_relevant_to_block(block_info, state)

        # Limit to prevent O(N²) scaling within a block
        max_streets_per_block = min(20, remaining_capacity)
        relevant_streets = relevant_streets[:max_streets_per_block]

        # Generate candidates from relevant streets
        for street_id, street in relevant_streets:
            # **FIX 1: Check remaining capacity BEFORE processing**
            if len(candidates) >= remaining_capacity:
                logger.debug(f"BlockCentroidStrategy: Reached capacity limit ({remaining_capacity})")
                break

            if street_id in skeleton_edges:
                continue  # Don't remove skeleton streets

            if not isinstance(street.geometry, LineString):
                continue

            # Score based on how removing this street affects this specific block
            confidence = self._score_block_closure_impact_for_block(street, block_info, state)

            if confidence > 0.1:
                # Create action directly from street
                action = self._create_action_from_street(street_id, street, self.name, confidence, state)
                if action is not None:
                    candidates.append((action, confidence))

                    # **FIX 2: Early exit on high confidence**
                    if confidence >= 0.95:
                        logger.debug(f"BlockCentroidStrategy: Early exit on high confidence ({confidence:.3f})")
                        break

        return candidates

    def _find_streets_relevant_to_block(self, block_info: Dict, state: GrowthState) -> List[Tuple[str, any]]:
        """Find streets that are relevant to a specific block."""
        relevant_streets = []

        # Get block geometry
        polygon = block_info.get('polygon')
        if polygon is None:
            return relevant_streets

        # Find streets that intersect or are near the block
        for street_id in state.streets.index:
            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Check if street intersects block or is very close
            if street.geometry.intersects(polygon) or street.geometry.distance(polygon) < 10.0:
                relevant_streets.append((street_id, street))

        return relevant_streets

    def _score_block_closure_impact_for_block(self, street, block_info: Dict, state: GrowthState) -> float:
        """Score how removing this street affects a specific block."""
        polygon = block_info.get('polygon')
        if polygon is None:
            return 0.1

        # Check if street is on block boundary
        if street.geometry.intersects(polygon.boundary):
            # Higher score for boundary streets that help close the block
            if block_info['type'] == 'planar_face':
                return 0.8  # High confidence for closing planar faces
            elif block_info['type'] == 'cycle_chord':
                return 0.7  # Good confidence for adding chords
            elif block_info['type'] == 'polygon_open':
                return 0.9  # Very high confidence for closing open frontiers

        # Lower score for internal streets
        return 0.2

    def _score_block_closure_impact(self, street, state) -> float:
        """Score how removing this street affects block closure."""
        # Simplified: score based on whether street is on block boundary
        # Higher score for streets that help close blocks
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return 0.1

        # Check if this street is part of any block boundary
        for idx, block in state.blocks.iterrows():
            if hasattr(block.geometry, 'exterior'):
                # Simplified check: if street endpoints are near block boundary
                block_coords = list(block.geometry.exterior.coords)
                street_start = street.geometry.coords[0]
                street_end = street.geometry.coords[-1]

                # Check if either endpoint is near block boundary
                for block_coord in block_coords:
                    if (self._points_close(street_start, block_coord) or
                        self._points_close(street_end, block_coord)):
                        return 0.8  # High score for boundary streets

        return 0.2  # Low score for internal streets

    def _identify_incomplete_blocks(self, state: GrowthState) -> List[Dict]:
        """Identify incomplete blocks using multiple criteria with memory bounds."""
        incomplete_blocks = []

        # Handle empty blocks gracefully
        if state.blocks.empty:
            logger.debug("BlockCentroidStrategy: No blocks in state")
            return incomplete_blocks

        # MEMORY BOUNDS: Limit total blocks to prevent O(N²) scaling
        MAX_TOTAL_BLOCKS = 20  # Hard cap on total incomplete blocks

        # Criterion 1: Planar faces with small area and few missing edges
        planar_blocks = self._find_planar_incomplete_blocks(state)
        incomplete_blocks.extend(planar_blocks)

        if len(incomplete_blocks) >= MAX_TOTAL_BLOCKS:
            logger.debug(f"BlockCentroidStrategy: Hit block limit ({MAX_TOTAL_BLOCKS}) after planar faces")
            return incomplete_blocks[:MAX_TOTAL_BLOCKS]

        # Criterion 2: Graph cycles with missing chords
        cycle_blocks = self._find_cycle_incomplete_blocks(state)
        incomplete_blocks.extend(cycle_blocks)

        if len(incomplete_blocks) >= MAX_TOTAL_BLOCKS:
            logger.debug(f"BlockCentroidStrategy: Hit block limit ({MAX_TOTAL_BLOCKS}) after cycle chords")
            return incomplete_blocks[:MAX_TOTAL_BLOCKS]

        # Criterion 3: Block polygons with open frontiers
        polygon_blocks = self._find_polygon_incomplete_blocks(state)
        incomplete_blocks.extend(polygon_blocks)

        # Final limit
        return incomplete_blocks[:MAX_TOTAL_BLOCKS]

    def _find_planar_incomplete_blocks(self, state: GrowthState) -> List[Dict]:
        """Find incomplete blocks from planar faces with memory bounds."""
        blocks = []

        # MEMORY BOUNDS: Limit face processing to prevent O(N²) scaling
        MAX_FACES_TO_PROCESS = 50  # Hard cap on faces to analyze

        # Use networkx to find faces (planar regions)
        try:
            # Get undirected graph for face detection
            undirected_graph = state.graph.to_undirected()

            # Find all faces (cycles) - limit to prevent excessive computation
            faces = list(nx.cycle_basis(undirected_graph))[:MAX_FACES_TO_PROCESS]

            for face in faces:
                if len(face) < 3:  # Need at least triangle
                    continue

                # Get face geometry
                face_coords = []
                for node in face:
                    if node in state.graph.nodes and 'geometry' in state.graph.nodes[node]:
                        geom = state.graph.nodes[node]['geometry']
                        if hasattr(geom, 'coords'):
                            face_coords.append(geom.coords[0])

                if len(face_coords) < 3:
                    continue

                # Create polygon
                try:
                    from shapely.geometry import Polygon
                    polygon = Polygon(face_coords)
                    area = polygon.area

                    if area > self.max_block_area:
                        continue

                    # Count missing edges (gaps in the cycle)
                    missing_edges = 0
                    for i in range(len(face)):
                        u, v = face[i], face[(i + 1) % len(face)]
                        if not state.graph.has_edge(u, v):
                            missing_edges += 1

                    if missing_edges <= self.max_missing_edges and missing_edges > 0:
                        centroid = polygon.centroid
                        blocks.append({
                            'type': 'planar_face',
                            'polygon': polygon,
                            'centroid': centroid,
                            'missing_edges': missing_edges,
                            'face_nodes': face
                        })

                except Exception as e:
                    logger.debug(f"BlockCentroidStrategy: Error creating polygon for face: {e}")
                    continue

        except Exception as e:
            logger.debug(f"BlockCentroidStrategy: Error in planar face detection: {e}")

        return blocks

    def _find_cycle_incomplete_blocks(self, state: GrowthState) -> List[Dict]:
        """Find incomplete blocks from graph cycles with missing chords and memory bounds."""
        blocks = []

        # MEMORY BOUNDS: Limit cycle processing to prevent O(N²) scaling
        MAX_CYCLES_TO_PROCESS = 20  # Hard cap on cycles to analyze

        try:
            # Find cycles in the graph - limit to prevent excessive computation
            cycles = list(nx.simple_cycles(state.graph.to_undirected()))[:MAX_CYCLES_TO_PROCESS]

            for cycle in cycles:
                if len(cycle) < 4:  # Need at least quadrilateral
                    continue

                # Check for missing chords (diagonals)
                missing_chords = []
                for i in range(len(cycle)):
                    for j in range(i + 2, len(cycle) if i > 0 else len(cycle) - 1):
                        u, v = cycle[i], cycle[j]
                        if not state.graph.has_edge(u, v):
                            # Check if this chord would create a valid block
                            if self._is_valid_block_chord(u, v, cycle, state):
                                missing_chords.append((u, v))

                if missing_chords:
                    # Create block polygon from cycle
                    cycle_coords = []
                    for node in cycle:
                        if node in state.graph.nodes and 'geometry' in state.graph.nodes[node]:
                            geom = state.graph.nodes[node]['geometry']
                            if hasattr(geom, 'coords'):
                                cycle_coords.append(geom.coords[0])

                    if len(cycle_coords) >= 4:
                        try:
                            from shapely.geometry import Polygon
                            polygon = Polygon(cycle_coords)
                            centroid = polygon.centroid
                            blocks.append({
                                'type': 'cycle_chord',
                                'polygon': polygon,
                                'centroid': centroid,
                                'missing_chords': missing_chords,
                                'cycle_nodes': cycle
                            })
                        except Exception as e:
                            logger.debug(f"BlockCentroidStrategy: Error creating cycle polygon: {e}")

        except Exception as e:
            logger.debug(f"BlockCentroidStrategy: Error in cycle detection: {e}")

        return blocks

    def _find_polygon_incomplete_blocks(self, state: GrowthState) -> List[Dict]:
        """Find incomplete blocks from block polygons with open frontiers."""
        blocks = []

        for idx, block in state.blocks.iterrows():
            if not hasattr(block.geometry, 'exterior'):
                continue

            polygon = block.geometry
            if polygon.area > self.max_block_area:
                continue

            # Count bounding street segments
            bounding_segments = 0
            open_frontiers = []

            # Check each edge of the block polygon
            exterior_coords = list(polygon.exterior.coords)
            for i in range(len(exterior_coords) - 1):
                edge_start = exterior_coords[i]
                edge_end = exterior_coords[i + 1]

                # Check if this edge has a corresponding street
                has_street = False
                for s_idx, street in state.streets.iterrows():
                    if isinstance(street.geometry, LineString):
                        street_coords = list(street.geometry.coords)
                        if len(street_coords) >= 2:
                            # Check if street covers this edge (simplified)
                            if (self._points_close(edge_start, street_coords[0]) and
                                self._points_close(edge_end, street_coords[-1])) or \
                               (self._points_close(edge_start, street_coords[-1]) and
                                self._points_close(edge_end, street_coords[0])):
                                has_street = True
                                bounding_segments += 1
                                break

                if not has_street:
                    # This is an open frontier
                    open_frontiers.append((edge_start, edge_end))

            # Check if block has enough bounding segments and open frontiers pointing inward
            if bounding_segments >= 3 and open_frontiers:
                # Check if open frontiers point inward
                inward_frontiers = []
                centroid = polygon.centroid

                for start, end in open_frontiers:
                    # Check if frontier points toward centroid
                    if self._frontier_points_inward(start, end, centroid, polygon):
                        inward_frontiers.append((start, end))

                if inward_frontiers:
                    blocks.append({
                        'type': 'polygon_open',
                        'polygon': polygon,
                        'centroid': centroid,
                        'open_frontiers': inward_frontiers,
                        'bounding_segments': bounding_segments
                    })

        return blocks

    def _generate_block_closing_candidates(self, block_info: Dict, state: GrowthState,
                                         spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate candidates that close the incomplete block."""
        candidates = []
        centroid = block_info['centroid']

        # Get potential start/end points based on block type
        if block_info['type'] == 'planar_face':
            # Use face nodes as potential connection points
            connection_points = block_info['face_nodes']
        elif block_info['type'] == 'cycle_chord':
            # Use missing chord endpoints
            connection_points = []
            for u, v in block_info['missing_chords']:
                connection_points.extend([u, v])
        elif block_info['type'] == 'polygon_open':
            # Use open frontier endpoints
            connection_points = []
            for start, end in block_info['open_frontiers']:
                connection_points.extend([start, end])
        else:
            return candidates

        # Generate candidate edges between connection points
        candidate_edges = self._generate_candidate_edges(connection_points, state, spatial_index)

        # Score each candidate
        for edge_geom, start_frontier, end_point in candidate_edges:
            confidence = self._score_block_closing_candidate(edge_geom, block_info, state)

            if confidence > 0:  # Only create actions for valid candidates
                # Create synthetic frontier for the candidate edge
                synthetic_frontier = self._create_synthetic_frontier(edge_geom, start_frontier)

                action = self._create_action_from_frontier(
                    synthetic_frontier, "block_centroid", confidence, state,
                    intent_params={
                        'block_type': block_info['type'],
                        'centroid_alignment': True,
                        'closure_score': self._calculate_closure_score(edge_geom, block_info)
                    }
                )
                candidates.append((action, confidence))

        return candidates

    def _generate_candidate_edges(self, connection_points, state: GrowthState,
                                spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[LineString, FrontierEdge, Tuple]]:
        """Generate geometrically plausible candidate edges."""
        candidates = []

        # Get frontiers that could serve as starting points
        start_frontiers = [f for f in state.frontiers if f.frontier_type in ['dead_end', 'block_edge']]

        # Calculate median edge length for constraints
        edge_lengths = []
        for idx, street in state.streets.iterrows():
            if isinstance(street.geometry, LineString):
                edge_lengths.append(street.geometry.length)

        median_length = np.median(edge_lengths) if edge_lengths else 50.0
        min_length = median_length * 0.5
        max_length = median_length * 1.5

        for frontier in start_frontiers:
            if not isinstance(frontier.geometry, LineString):
                continue

            frontier_end = frontier.geometry.coords[-1]  # Use endpoint of frontier

            # Try connecting to various endpoints
            potential_ends = []

            # 1. Other frontiers
            for other_frontier in start_frontiers:
                if other_frontier != frontier and isinstance(other_frontier.geometry, LineString):
                    other_end = other_frontier.geometry.coords[-1]
                    potential_ends.append(('frontier', other_end, other_frontier))

            # 2. Nearby street vertices
            if spatial_index:
                nearby_streets = spatial_index.find_nearby_streets(
                    type('Point', (), {'x': frontier_end[0], 'y': frontier_end[1]})(), 100.0
                )
                for street in nearby_streets:
                    if isinstance(street.geometry, LineString):
                        for coord in [street.geometry.coords[0], street.geometry.coords[-1]]:
                            potential_ends.append(('vertex', coord, None))

            # 3. Projected intersections on existing edges
            potential_ends.extend(self._find_projected_intersections(frontier_end, state, spatial_index))

            # Generate candidates
            for end_type, end_point, end_frontier in potential_ends:
                try:
                    candidate_line = LineString([frontier_end, end_point])
                    length = candidate_line.length

                    # Length constraint
                    if not (min_length <= length <= max_length):
                        continue

                    # Angle constraint
                    if not self._check_angle_constraints(candidate_line, state):
                        continue

                    # No self-intersection
                    if self._check_self_intersection(candidate_line, state):
                        continue

                    # No illegal crossings
                    if self._check_illegal_crossings(candidate_line, state):
                        continue

                    # No dangling dead-ends inside blocks
                    if self._creates_dangling_dead_end(candidate_line, state):
                        continue

                    candidates.append((candidate_line, frontier, end_point))

                except Exception as e:
                    logger.debug(f"BlockCentroidStrategy: Error generating candidate: {e}")
                    continue

        return candidates

    def _score_block_closing_candidate(self, edge_geom: LineString, block_info: Dict, state: GrowthState) -> float:
        """Calculate multiplicative score for block closing candidate."""
        # Calculate individual scores
        closure_score = self._calculate_closure_score(edge_geom, block_info)
        centroid_alignment_score = self._calculate_centroid_alignment_score(edge_geom, block_info, state)
        angle_score = self._calculate_angle_score(edge_geom, state)
        length_score = self._calculate_length_score(edge_geom, state)
        intersection_penalty = self._calculate_intersection_penalty(edge_geom, state)
        planarity_penalty = self._calculate_planarity_penalty(edge_geom, state)

        # Multiplicative scoring
        confidence = (
            closure_score *
            centroid_alignment_score *
            angle_score *
            length_score *
            intersection_penalty *
            planarity_penalty
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))

    def _calculate_closure_score(self, edge_geom: LineString, block_info: Dict) -> float:
        """Score how well the edge closes the block perimeter."""
        if block_info['type'] == 'planar_face':
            # Reduces missing edges
            return min(1.0, block_info['missing_edges'] / self.max_missing_edges)
        elif block_info['type'] == 'cycle_chord':
            # Adds missing chord
            return 0.8
        elif block_info['type'] == 'polygon_open':
            # Closes open frontier
            return 0.9
        return 0.5

    def _calculate_centroid_alignment_score(self, edge_geom: LineString, block_info: Dict, state: GrowthState) -> float:
        """Score centroid alignment - does edge reduce avg perimeter-centroid distance."""
        centroid = block_info['centroid']
        polygon = block_info['polygon']

        # Calculate current avg distance from perimeter to centroid
        current_distances = []
        exterior_coords = list(polygon.exterior.coords)
        for coord in exterior_coords:
            dist = ((coord[0] - centroid.x)**2 + (coord[1] - centroid.y)**2)**0.5
            current_distances.append(dist)

        current_avg_dist = np.mean(current_distances)

        # Calculate new avg distance if edge is added
        # Approximate by adding edge endpoints
        new_coords = exterior_coords + [edge_geom.coords[0], edge_geom.coords[-1]]
        new_distances = []
        for coord in new_coords:
            dist = ((coord[0] - centroid.x)**2 + (coord[1] - centroid.y)**2)**0.5
            new_distances.append(dist)

        new_avg_dist = np.mean(new_distances)

        # Score improvement (lower distance is better)
        if current_avg_dist > 0:
            improvement = (current_avg_dist - new_avg_dist) / current_avg_dist
            return max(0.1, 0.5 + improvement)  # Base 0.5, bonus for improvement
        return 0.5

    def _calculate_angle_score(self, edge_geom: LineString, state: GrowthState) -> float:
        """Score angle fit with local dominant angles."""
        edge_angle = self._get_street_angle(edge_geom)

        # Get nearby street angles
        nearby_angles = []
        buffer_geom = edge_geom.buffer(50.0)  # 50m buffer

        for idx, street in state.streets.iterrows():
            if isinstance(street.geometry, LineString) and buffer_geom.intersects(street.geometry):
                nearby_angles.append(self._get_street_angle(street.geometry))

        if not nearby_angles:
            return 0.5

        # Check if edge angle matches or complements dominant angles
        dominant_angle = np.median(nearby_angles) % 180

        angle_diff = abs(edge_angle - dominant_angle) % 180
        min_diff = min(angle_diff, 180 - angle_diff)

        complementary_diff = abs(angle_diff - 90) % 180
        min_complementary_diff = min(complementary_diff, 180 - complementary_diff)

        # Score based on how well it fits parallel or perpendicular
        fit_score = max(0, 1 - min_diff / 30.0)  # Within 30 degrees
        complementary_score = max(0, 1 - min_complementary_diff / 30.0)

        return max(fit_score, complementary_score * 0.8)  # Prefer parallel but accept perpendicular

    def _calculate_length_score(self, edge_geom: LineString, state: GrowthState) -> float:
        """Score length fit with existing streets."""
        length = edge_geom.length

        lengths = []
        for idx, street in state.streets.iterrows():
            if isinstance(street.geometry, LineString):
                lengths.append(street.geometry.length)

        if not lengths:
            return 0.5

        median_length = np.median(lengths)
        std_length = np.std(lengths) or median_length * 0.1

        # Gaussian score centered on median
        z_score = abs(length - median_length) / std_length
        return np.exp(-0.5 * z_score**2)

    def _calculate_intersection_penalty(self, edge_geom: LineString, state: GrowthState) -> float:
        """Penalty for creating too many intersections."""
        # Count intersections with existing streets
        intersection_count = 0

        for idx, street in state.streets.iterrows():
            if isinstance(street.geometry, LineString):
                if edge_geom.crosses(street.geometry):
                    intersection_count += 1

        # Penalty decreases with more intersections
        if intersection_count == 0:
            return 1.0  # No penalty
        elif intersection_count == 1:
            return 0.8  # Slight penalty
        else:
            return 0.1  # Heavy penalty

    def _calculate_planarity_penalty(self, edge_geom: LineString, state: GrowthState) -> float:
        """Penalty for illegal crossings or planarity violations."""
        # Check for crossings with blocks
        for idx, block in state.blocks.iterrows():
            if edge_geom.crosses(block.geometry) or edge_geom.within(block.geometry):
                return 0.0  # Illegal crossing

        return 1.0  # No penalty

    # Helper methods
    def _is_valid_block_chord(self, u, v, cycle, state: GrowthState) -> bool:
        """Check if a chord between u,v would create a valid block."""
        # Simplified: check if chord doesn't cross other edges
        try:
            u_geom = state.graph.nodes[u]['geometry']
            v_geom = state.graph.nodes[v]['geometry']
            chord = LineString([u_geom.coords[0], v_geom.coords[0]])

            # Check for crossings with cycle edges
            for i in range(len(cycle)):
                edge_u, edge_v = cycle[i], cycle[(i + 1) % len(cycle)]
                if {edge_u, edge_v} != {u, v}:  # Not the chord itself
                    if edge_u in state.graph.nodes and edge_v in state.graph.nodes:
                        edge_geom = LineString([
                            state.graph.nodes[edge_u]['geometry'].coords[0],
                            state.graph.nodes[edge_v]['geometry'].coords[0]
                        ])
                        if chord.crosses(edge_geom):
                            return False
            return True
        except:
            return False

    def _points_close(self, p1, p2, tolerance=1.0) -> bool:
        """Check if two points are close."""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 < tolerance

    def _frontier_points_inward(self, start, end, centroid, polygon) -> bool:
        """Check if frontier points toward centroid."""
        # Simplified: check if frontier is on polygon boundary and points inward
        from shapely.geometry import Point
        mid_point = Point((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        return polygon.contains(mid_point) or polygon.touches(mid_point)

    def _check_angle_constraints(self, line: LineString, state: GrowthState) -> bool:
        """Check if line angle fits local constraints."""
        # Already checked in scoring, but add basic constraint
        return True

    def _check_self_intersection(self, line: LineString, state: GrowthState) -> bool:
        """Check if line self-intersects."""
        return not line.is_simple

    def _check_illegal_crossings(self, line: LineString, state: GrowthState) -> bool:
        """Check for illegal crossings with blocks."""
        for idx, block in state.blocks.iterrows():
            if line.crosses(block.geometry):
                return True
        return False

    def _creates_dangling_dead_end(self, line: LineString, state: GrowthState) -> bool:
        """Check if line creates dangling dead-end inside block."""
        # Simplified check
        return False

    def _find_projected_intersections(self, point, state: GrowthState, spatial_index: Optional[SpatialIndex] = None):
        """Find projected intersection points on existing edges."""
        # Simplified implementation
        return []

    def _create_synthetic_frontier(self, geometry: LineString, base_frontier: FrontierEdge) -> FrontierEdge:
        """Create synthetic frontier for candidate edge."""
        from shapely import wkt

        # Create synthetic frontier based on the candidate geometry
        stable_id = self._compute_stable_frontier_id(type('MockFrontier', (), {'geometry': geometry})())

        return type('SyntheticFrontier', (), {
            'geometry': geometry,
            'edge_id': (f"synthetic_{stable_id}", f"synthetic_{stable_id}_end"),
            'frontier_id': f"synthetic_block_centroid_{stable_id}",
            'frontier_type': 'synthetic_block_closure',
            'expansion_weight': 0.8
        })()

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180


class AdaptiveLearningStrategy(InferenceStrategy):
    """
    Adaptive Learning Strategy: Learn from successful patterns and adapt.

    Mathematical Approach:
    - Uses reinforcement learning concepts
    - Tracks success rates of different strategies
    - Adapts weights based on performance
    """

    def __init__(self):
        super().__init__("adaptive_learning", weight=1.0)
        self.success_history = {}

    def generate_candidates(self, state: GrowthState, skeleton_edges: set, spatial_index: Optional[SpatialIndex] = None) -> List[Tuple[InverseGrowthAction, float]]:
        # Score existing streets for removal based on multi-criteria analysis
        candidates = []

        for street_id in state.streets.index:
            if street_id in skeleton_edges:
                continue  # Don't remove skeleton streets

            street = state.streets.loc[street_id]
            if not isinstance(street.geometry, LineString):
                continue

            # Multi-criteria scoring for removal
            angle_score = self._score_angle_disruption(street, state.streets)
            length_score = self._score_length_disruption(street, state.streets)
            connectivity_score = self._score_connectivity_disruption(street, state.graph)

            confidence = 0.4 * angle_score + 0.3 * length_score + 0.3 * connectivity_score

            if confidence > 0.1:
                # Find corresponding frontier
                frontier = None
                u, v = street.get('u'), street.get('v')
                if u is not None and v is not None:
                    for f in state.frontiers:
                        if hasattr(f, 'edge_id') and f.edge_id == (u, v) or f.edge_id == (v, u):
                            frontier = f
                            break

                if frontier is not None:
                    action = self._create_action_from_frontier(frontier, self.name, confidence, state)
                    candidates.append((action, confidence))

        return candidates

    def _score_angle_disruption(self, street, streets_gdf) -> float:
        """Score how disruptive removing this street would be to angle patterns."""
        street_angle = self._get_street_angle(street.geometry)

        angles = []
        for idx, other_street in streets_gdf.iterrows():
            if isinstance(other_street.geometry, LineString):
                angles.append(self._get_street_angle(other_street.geometry))

        if not angles:
            return 0.5

        # Count streets with similar angles
        similar_count = sum(1 for angle in angles if abs(angle - street_angle) % 180 < 20)
        commonality_score = similar_count / len(angles)

        # Prefer removing streets with uncommon angles
        return 1.0 - commonality_score

    def _score_length_disruption(self, street, streets_gdf) -> float:
        """Score how disruptive removing this street would be to length patterns."""
        street_length = street.geometry.length

        lengths = []
        for idx, other_street in streets_gdf.iterrows():
            if isinstance(other_street.geometry, LineString):
                lengths.append(other_street.geometry.length)

        if not lengths:
            return 0.5

        mean_length = np.mean(lengths)
        std_length = np.std(lengths) or 1.0

        # Prefer removing streets that are outliers in length
        z_score = abs(street_length - mean_length) / std_length
        return min(1.0, z_score / 2.0)  # Higher score for more unusual lengths

    def _score_connectivity_disruption(self, street, graph) -> float:
        """Score how disruptive removing this street would be to connectivity."""
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            return 0.1

        degree_u = graph.degree[u] if graph.has_node(u) else 0
        degree_v = graph.degree[v] if graph.has_node(v) else 0

        # Higher degree means more important for connectivity
        avg_degree = (degree_u + degree_v) / 2.0
        return min(1.0, avg_degree / 4.0)

    def _score_angle_fit(self, frontier, streets_gdf) -> float:
        """Score angle fit (simplified)."""
        frontier_angle = self._get_street_angle(frontier.geometry)

        angles = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                angles.append(self._get_street_angle(street.geometry))

        if not angles:
            return 0.5

        # Count similar angles
        similar = sum(1 for angle in angles if abs(angle - frontier_angle) % 180 < 20)
        return similar / len(angles)

    def _score_length_fit(self, frontier, streets_gdf) -> float:
        """Score length fit (simplified)."""
        frontier_length = frontier.geometry.length

        lengths = []
        for idx, street in streets_gdf.iterrows():
            if isinstance(street.geometry, LineString):
                lengths.append(street.geometry.length)

        if not lengths:
            return 0.5

        mean_length = np.mean(lengths)
        std_length = np.std(lengths) or 1.0

        return 1 / (1 + abs(frontier_length - mean_length) / std_length)

    def _score_connectivity_impact(self, frontier, state: GrowthState) -> float:
        """Score connectivity impact (simplified)."""
        # Simplified: just return a neutral score
        return 0.6

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180


# =============================================================================
# MULTI-STRATEGY INFERENCE ENGINE
# =============================================================================

class MultiStrategyInferenceEngine:
    """
    Advanced inference engine using multiple mathematical strategies with spatial indexing.
    """

    # HARD CAPS: Prevent O(N²–N³) complexity
    MAX_BLOCKS = 50
    MAX_FRONTIERS = 100
    MAX_CANDIDATES = 50

    def __init__(self, strategy_config=None):
        """
        Initialize with configurable strategy selection.

        Args:
            strategy_config: Dict controlling which strategies to enable
                Default enables all Phase 1 strategies
        """
        # Default strategy configuration - Phase 1 strategies enabled by default
        default_config = {
            'fractal_pattern': True,
            'angle_harmonization': True,
            'block_centroid': True,
            'ml_augmented': False,      # Phase 2A - ML Augmented Action Selection
            'multi_resolution': False,  # Phase 2A - Multi-Resolution Action Spaces
            'advanced_search': False    # Phase 2A - Advanced Search Algorithms
        }

        # Merge with provided config
        self.strategy_config = {**default_config, **(strategy_config or {})}

        # Initialize strategies based on configuration
        self.strategies = []
        self._initialize_strategies()

        self.skeleton_extractor = ArterialSkeletonExtractor()
        self.rewind_engine = RewindEngine()
        self.spatial_index = SpatialIndex()  # R-tree spatial indexing

        # STEP 5: Silence Incremental Graph Warnings
        # Track warnings per step and fallback to full rebuild every N steps
        self.warning_count_this_step = 0
        self.full_rebuild_interval = 25  # Full rebuild every 25 steps
        self.step_count = 0

    def _initialize_strategies(self):
        """Initialize strategies based on configuration."""
        if self.strategy_config.get('fractal_pattern', True):
            self.strategies.append(FractalPatternStrategy())

        if self.strategy_config.get('angle_harmonization', True):
            self.strategies.append(AngleHarmonizationStrategy())

        if self.strategy_config.get('block_centroid', True):
            self.strategies.append(BlockCentroidStrategy())

        if self.strategy_config.get('ml_augmented', False):
            # Will be implemented in Phase 2A
            logger.warning("ML Augmented strategy not yet implemented - skipping")

        if self.strategy_config.get('multi_resolution', False):
            # Will be implemented in Phase 2A
            logger.warning("Multi-Resolution strategy not yet implemented - skipping")

        if self.strategy_config.get('advanced_search', False):
            # Will be implemented in Phase 2A
            logger.warning("Advanced Search strategy not yet implemented - skipping")

        logger.info(f"Initialized {len(self.strategies)} strategies: {[s.name for s in self.strategies]}")

    def infer_trace(self, final_state: GrowthState, max_steps: int = 10000000,
                    initial_state: Optional[GrowthState] = None,
                    progress_callback: Optional[callable] = None) -> GrowthTrace:
        """
        Infer growth trace using multiple mathematical strategies with comprehensive performance tracking.
        """
        logger.info("Starting multi-strategy inference...")

        # Initialize performance tracker
        perf_tracker = PerformanceTracker()

        perf_tracker.start_operation("skeleton_extraction")
        if initial_state is None:
            skeleton_edges, skeleton_streets = self.skeleton_extractor.extract_skeleton(
                final_state.streets, final_state.graph
            )
            initial_state = self.skeleton_extractor.create_skeleton_state(skeleton_streets, final_state)
            skeleton_edges_set = skeleton_edges
        else:
            logger.info(f"Using provided initial state with {len(initial_state.streets)} streets")
            skeleton_edges_set = set()
            for idx, street in initial_state.streets.iterrows():
                u, v = street.get('u'), street.get('v')
                if u and v:
                    skeleton_edges_set.add((min(u, v), max(u, v)))
        perf_tracker.end_operation("skeleton_extraction")

        current_state = final_state
        actions = []
        step = 0

        # Build spatial indexes for fast geometric queries
        logger.info("Building spatial indexes for performance optimization...")
        perf_tracker.start_operation("spatial_index_building")

        # Time street index building
        start_time = time.perf_counter()
        self.spatial_index.build_street_index(final_state.streets)
        street_index_time = time.perf_counter() - start_time
        perf_tracker.record_spatial_index_time("build_street_index", street_index_time)
        logger.debug(f"Built street index in {street_index_time:.2f}s")

        # Time frontier index building
        start_time = time.perf_counter()
        self.spatial_index.build_frontier_index(final_state.frontiers)
        frontier_index_time = time.perf_counter() - start_time
        perf_tracker.record_spatial_index_time("build_frontier_index", frontier_index_time)
        logger.debug(f"Built frontier index in {frontier_index_time:.2f}s")

        # Time block index building
        start_time = time.perf_counter()
        self.spatial_index.build_block_index(final_state.blocks)
        block_index_time = time.perf_counter() - start_time
        perf_tracker.record_spatial_index_time("build_block_index", block_index_time)
        logger.debug(f"Built block index in {block_index_time:.2f}s")

        perf_tracker.end_operation("spatial_index_building")
        logger.info(f"Spatial index building complete: streets={street_index_time:.2f}s, frontiers={frontier_index_time:.2f}s, blocks={block_index_time:.2f}s")

        logger.info(f"Multi-strategy setup: {len(self.strategies)} strategies, final={len(final_state.streets)} streets, frontiers={len(final_state.frontiers)}")
        logger.info("Starting inference loop...")

        # Setup progress tracking
        total_streets_to_remove = len(final_state.streets) - len(initial_state.streets)
        logger.info(f"Inference target: remove {total_streets_to_remove} streets to reach {len(initial_state.streets)} streets")
        logger.info("Progress tracking initialized - monitoring street count and strategy performance")

        # Progress detection: track consecutive steps with no progress
        last_street_count = len(final_state.streets)
        no_progress_count = 0
        MAX_NO_PROGRESS = 10

        while step < max_steps:
            if len(current_state.streets) <= len(initial_state.streets):
                logger.info(f"Reached initial state size at step {step}")
                break

            step_start_time = time.perf_counter()
            logger.debug(f"Step {step}: Processing state with {len(current_state.streets)} streets, {len(current_state.frontiers)} frontiers")

            # FRONTIER LOOKUP OPTIMIZATION: Build O(1) frontier lookup dictionary
            frontier_lookup = {}
            for frontier in current_state.frontiers:
                if hasattr(frontier, 'edge_id') and frontier.edge_id:
                    edge_key = (min(frontier.edge_id[0], frontier.edge_id[1]),
                               max(frontier.edge_id[0], frontier.edge_id[1]))
                    frontier_lookup[edge_key] = frontier

            # Generate candidates from all strategies with error handling and safeguards
            perf_tracker.start_operation("candidate_generation")
            all_candidates = []
            strategy_stats = {}
            active_strategies = []

            # STRATEGY SAFEGUARDS: Prevent infinite loops and memory leaks
            STRATEGY_TIMEOUT = 30.0  # 30 second timeout per strategy
            MAX_CANDIDATES_PER_STRATEGY = 20  # Limit candidates per strategy

            for strategy in self.strategies:
                strategy_start = time.perf_counter()
                candidates = []
                strategy_stats[strategy.name] = 0

                try:
                    # TIMEOUT PROTECTION: Use signal-based timeout if available
                    import signal
                    from contextlib import contextmanager

                    @contextmanager
                    def timeout_context(seconds):
                        def timeout_handler(signum, frame):
                            raise TimeoutError(f"Strategy {strategy.name} timed out after {seconds}s")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(seconds)
                        try:
                            yield
                        finally:
                            signal.alarm(0)

                    try:
                        with timeout_context(STRATEGY_TIMEOUT):
                            candidates = strategy.generate_candidates(current_state, skeleton_edges_set, self.spatial_index)
                    except TimeoutError:
                        logger.warning(f"Strategy {strategy.name} timed out after {STRATEGY_TIMEOUT}s - skipping")
                        candidates = []
                    except AttributeError:
                        # Signal not available on Windows, fallback to basic timeout
                        candidates = strategy.generate_candidates(current_state, skeleton_edges_set, self.spatial_index)

                    # CANDIDATE LIMIT: Prevent memory explosion
                    if len(candidates) > MAX_CANDIDATES_PER_STRATEGY:
                        logger.warning(f"Strategy {strategy.name} generated {len(candidates)} candidates, limiting to {MAX_CANDIDATES_PER_STRATEGY}")
                        candidates = candidates[:MAX_CANDIDATES_PER_STRATEGY]

                    strategy_time = time.perf_counter() - strategy_start
                    perf_tracker.record_strategy_time(strategy.name, strategy_time)

                    strategy_stats[strategy.name] = len(candidates)

                    # Apply post-normalized weighting: confidence × strategy.weight
                    weighted_candidates = []
                    for action, confidence in candidates:
                        weighted_confidence = confidence * strategy.weight
                        weighted_candidates.append((action, weighted_confidence))

                    all_candidates.extend(weighted_candidates)
                    active_strategies.append(strategy)

                except Exception as e:
                    strategy_time = time.perf_counter() - strategy_start
                    perf_tracker.record_strategy_time(strategy.name, strategy_time)
                    logger.warning(f"Strategy {strategy.name} failed at step {step}: {e}")
                    strategy_stats[strategy.name] = 0
                    # Strategy is automatically excluded from this step
                    continue

            perf_tracker.end_operation("candidate_generation")

            logger.debug(f"Step {step}: Strategy candidates - {strategy_stats}")
            logger.debug(f"Step {step}: Total candidates = {len(all_candidates)}")

            if not all_candidates:
                logger.info(f"No more actions to infer at step {step} (all strategies failed or no candidates)")
                break

            # Renormalize confidences across active strategies
            perf_tracker.start_operation("confidence_normalization")
            if all_candidates:
                max_confidence = max(conf for _, conf in all_candidates)
                if max_confidence > 0:
                    all_candidates = [(action, conf / max_confidence) for action, conf in all_candidates]
            perf_tracker.end_operation("confidence_normalization")

            # Select best candidate
            best_action, confidence = max(all_candidates, key=lambda x: x[1])
            strategy_name = best_action.intent_params.get('strategy', 'unknown')

            logger.debug(f"Step {step}: Selected {strategy_name} with confidence {confidence:.3f}")
            logger.debug(f"Step {step}: Best action - street_id: {best_action.street_id}, edge: {best_action.intent_params.get('edge_u', '?')}->{best_action.intent_params.get('edge_v', '?')}")
            logger.debug(f"Step {step}: Action state_diff removed_streets: {best_action.state_diff.get('removed_streets', []) if best_action.state_diff else 'None'}")

            # Compute state diff and rewind
            perf_tracker.start_operation("state_diff_computation")
            state_diff = self._compute_state_diff(current_state, best_action)
            perf_tracker.end_operation("state_diff_computation")

            perf_tracker.start_operation("rewind_operation")
            prev_state = self.rewind_engine.rewind_action(best_action, current_state)
            perf_tracker.end_operation("rewind_operation")

            # Validate that graph edges changed appropriately for the action type
            current_edges = current_state.graph.number_of_edges()
            prev_edges = prev_state.graph.number_of_edges()
            if best_action.action_type == ActionType.REMOVE_STREET:
                # REMOVE_STREET rewind should add edges back
                if prev_edges <= current_edges:
                    logger.error(f"Rewind failed: graph edges unchanged for REMOVE_STREET ({prev_edges} <= {current_edges})")
                    break
            else:
                # Other rewinds should remove edges
                if prev_edges >= current_edges:
                    logger.error(f"Rewind failed: graph edges unchanged ({prev_edges} >= {current_edges})")
                    break

            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind failed at step {step} - streets: {len(prev_state.streets)} >= {len(current_state.streets)}")
                break

            perf_tracker.start_operation("action_finalization")
            action_with_diff = self._add_state_diff_to_action(best_action, state_diff)
            actions.insert(0, action_with_diff)
            perf_tracker.end_operation("action_finalization")

            # Call progress callback with current trace state
            if progress_callback:
                current_trace = GrowthTrace(
                    actions=actions,
                    initial_state=initial_state,
                    final_state=current_state,
                    metadata={
                        "inference_method": "multi_strategy_mathematical",
                        "strategies": [s.name for s in self.strategies],
                        "max_steps": max_steps,
                        "steps_taken": step + 1,
                        "skeleton_streets": len(skeleton_edges_set),
                        "performance_stats": perf_tracker.get_summary_stats(),
                        "interrupted": True
                    }
                )
                progress_callback(current_trace)

            # STEP 5: Incremental Spatial Index Updates
            # Track changes and update indexes incrementally instead of full rebuild
            self.step_count += 1
            self.warning_count_this_step = 0

            perf_tracker.start_operation("spatial_index_update")

            # Check if we should do a full rebuild (every N steps for consistency)
            if self.step_count % self.full_rebuild_interval == 0:
                logger.debug(f"Step {step}: Performing full spatial index rebuild (every {self.full_rebuild_interval} steps)")
                # EXPLICIT CLEANUP: Clear old index data before rebuild to prevent memory leaks
                self.spatial_index.street_index = None
                self.spatial_index.frontier_index = None
                self.spatial_index.block_index = None
                self.spatial_index.street_data.clear()
                self.spatial_index.frontier_data.clear()
                self.spatial_index.block_data.clear()
                self.spatial_index.build_street_index(prev_state.streets)
                self.spatial_index.build_frontier_index(prev_state.frontiers)
                self.spatial_index.build_block_index(prev_state.blocks)
            else:
                # INCREMENTAL UPDATE: Track what changed and update only affected items
                try:
                    # Track removed streets from this step
                    removed_street_ids = state_diff.get('removed_streets', [])
                    removed_streets = []
                    added_frontiers = []
                    removed_frontiers = []
                    added_blocks = []
                    removed_blocks = []

                    # Find removed streets and their geometries
                    for street_id in removed_street_ids:
                        if street_id in current_state.streets.index:
                            street = current_state.streets.loc[street_id]
                            if hasattr(street, 'geometry'):
                                removed_streets.append((street_id, street.geometry))

                    # For now, we rebuild frontiers and blocks since tracking their changes
                    # is more complex. Focus incremental updates on streets first.
                    self.spatial_index.build_frontier_index(prev_state.frontiers)
                    self.spatial_index.build_block_index(prev_state.blocks)

                    # Incremental street updates - remove old streets
                    for street_id, geometry in removed_streets:
                        self.spatial_index.remove_street(street_id, geometry)

                    # Rebuild street index with remaining streets (simplified incremental approach)
                    # TODO: Implement true incremental addition of new streets
                    self.spatial_index.build_street_index(prev_state.streets)

                    logger.debug(f"Step {step}: Incremental update - removed {len(removed_streets)} streets")

                except Exception as e:
                    # Log once per step maximum, then fall back to lazy full rebuild
                    if self.warning_count_this_step == 0:
                        logger.debug(f"Step {step}: Incremental spatial index update failed, will retry with full rebuild: {e}")
                        self.warning_count_this_step += 1

                    # Force full rebuild on next step
                    self.step_count = self.full_rebuild_interval - 1

                    # Fallback to full rebuild
                    self.spatial_index.build_street_index(prev_state.streets)
                    self.spatial_index.build_frontier_index(prev_state.frontiers)
                    self.spatial_index.build_block_index(prev_state.blocks)

            perf_tracker.end_operation("spatial_index_update")

            # Record step metrics
            streets_before = len(current_state.streets)
            streets_after = len(prev_state.streets)
            perf_tracker.record_step_metrics(step, streets_before, streets_after, len(all_candidates), strategy_stats)

            # Check for progress: if street count hasn't decreased, increment no_progress counter
            if streets_after >= last_street_count:
                no_progress_count += 1
                if no_progress_count >= MAX_NO_PROGRESS:
                    logger.error(f"STUCK: No progress for {MAX_NO_PROGRESS} consecutive steps (street count: {last_street_count})")
                    break
            else:
                no_progress_count = 0
                last_street_count = streets_after

            # Improved logging: less repetitive but informative
            should_log = (
                step < 10 or  # First 10 steps
                step % 100 == 0 or  # Every 100 steps
                streets_after <= len(initial_state.streets) + 50  # Near completion
            )

            if should_log:
                remaining = max(0, streets_before - len(initial_state.streets))
                progress_pct = (1 - remaining / max(1, len(final_state.streets) - len(initial_state.streets))) * 100
                logger.info(f"Step {step}: {strategy_name} (conf={confidence:.2f}), streets {streets_before} -> {streets_after} ({progress_pct:.1f}% complete)")

            current_state = prev_state
            step += 1

        # Log comprehensive performance summary
        perf_tracker.log_performance_summary()

        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                "inference_method": "multi_strategy_mathematical",
                "strategies": [s.name for s in self.strategies],
                "max_steps": max_steps,
                "steps_taken": step,
                "skeleton_streets": len(skeleton_edges_set),
                "performance_stats": perf_tracker.get_summary_stats()
            }
        )

        logger.info(f"Multi-strategy inference complete: {len(actions)} actions")
        return trace

    def _create_action_from_street_id(self, street_id: str, strategy_name: str, confidence: float, state: GrowthState) -> Optional[InverseGrowthAction]:
        """Create an InverseGrowthAction from a street ID."""
        if street_id not in state.streets.index:
            logger.warning(f"Street ID {street_id} not found in current state")
            return None

        street = state.streets.loc[street_id]
        if not isinstance(street.geometry, LineString):
            logger.warning(f"Street {street_id} has invalid geometry")
            return None

        # Get edge information
        u, v = street.get('u'), street.get('v')
        if u is None or v is None:
            logger.warning(f"Street {street_id} missing edge information")
            return None

        # Find corresponding frontier
        frontier = None
        for f in state.frontiers:
            if hasattr(f, 'edge_id') and f.edge_id == (u, v) or f.edge_id == (v, u):
                frontier = f
                break

        if frontier is None:
            logger.warning(f"No frontier found for street {street_id} with edge ({u}, {v})")
            return None

        # Create action using the base class helper
        action = self._create_action_from_frontier(frontier, strategy_name, confidence, state)
        # Override the street_id to match the input
        return InverseGrowthAction(
            action_type=action.action_type,
            street_id=street_id,
            intent_params=action.intent_params,
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=action.state_diff,
            action_metadata=action.action_metadata
        )

    def _compute_state_diff(self, current_state: GrowthState, action: InverseGrowthAction) -> Dict[str, Any]:
        """Compute state diff (same as BasicInferenceEngine)."""
        state_diff = {
            'added_streets': [],
            'removed_streets': [],
            'graph_changes': {},
            'frontier_changes': {}
        }

        if action.action_type == ActionType.REMOVE_STREET:
            # For REMOVE_STREET actions, the street being removed during rewind
            # is the one that was added during forward growth
            street_id = action.street_id
            if street_id in current_state.streets.index:
                street = current_state.streets.loc[street_id]
                street_data = {
                    'edge_id': (min(street.get('u'), street.get('v')), max(street.get('u'), street.get('v'))),
                    'u': street.get('u'),
                    'v': street.get('v'),
                    'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                    'osmid': street.get('osmid'),
                    'highway': street.get('highway'),
                    'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                }
                state_diff['added_streets'].append(street_data)
                state_diff['removed_streets'].append(street_id)

        state_diff['graph_changes'] = {
            'nodes_before': current_state.graph.number_of_nodes(),
            'edges_before': current_state.graph.number_of_edges(),
            'nodes_after': None,
            'edges_after': None
        }

        state_diff['frontier_changes'] = {
            'frontiers_before': len(current_state.frontiers),
            'frontiers_after': None
        }

        return state_diff

    def _add_state_diff_to_action(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
        """Add state diff to action (same as BasicInferenceEngine)."""
        return InverseGrowthAction(
            action_type=action.action_type,
            street_id=action.street_id,
            intent_params=action.intent_params,
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=state_diff,
            action_metadata=action.action_metadata
        )

    def _build_geometry_cache(self, state: GrowthState) -> Dict[str, Any]:
        """
        Build geometry cache for the current step to avoid recomputing expensive geometric calculations.

        Precomputes:
        - Block centroids
        - Block dominant angles
        - Frontier orientations
        - Street angles and lengths
        """
        cache = {
            'block_centroids': {},
            'block_angles': {},
            'frontier_orientations': {},
            'street_angles': {},
            'street_lengths': {},
            'dominant_angles': []
        }

        # Cache block centroids and angles
        for idx, block in state.blocks.iterrows():
            if hasattr(block.geometry, 'centroid'):
                cache['block_centroids'][idx] = block.geometry.centroid

                # Calculate dominant angle for block (simplified)
                if hasattr(block.geometry, 'exterior'):
                    coords = list(block.geometry.exterior.coords)
                    if len(coords) >= 4:  # At least a triangle
                        # Calculate angles of block edges
                        block_angles = []
                        for i in range(len(coords) - 1):
                            dx = coords[i+1][0] - coords[i][0]
                            dy = coords[i+1][1] - coords[i][1]
                            angle = np.degrees(np.arctan2(dy, dx)) % 180
                            block_angles.append(angle)
                        cache['block_angles'][idx] = np.median(block_angles) if block_angles else 0.0

        # Cache frontier orientations
        for frontier in state.frontiers:
            if hasattr(frontier, 'geometry') and isinstance(frontier.geometry, LineString):
                frontier_id = getattr(frontier, 'frontier_id', id(frontier))
                angle = self._get_street_angle(frontier.geometry)
                cache['frontier_orientations'][frontier_id] = angle

        # Cache street angles and lengths
        street_angles = []
        street_lengths = []
        for idx, street in state.streets.iterrows():
            if isinstance(street.geometry, LineString):
                angle = self._get_street_angle(street.geometry)
                length = street.geometry.length
                cache['street_angles'][idx] = angle
                cache['street_lengths'][idx] = length
                street_angles.append(angle)
                street_lengths.append(length)

        # Calculate dominant angles for the network
        if street_angles:
            cache['dominant_angles'] = [
                np.median(street_angles),
                np.mean(street_angles),
                np.std(street_angles)
            ]

        return cache

    def _get_street_angle(self, geometry: LineString) -> float:
        """Get street angle in degrees (utility method)."""
        if len(geometry.coords) < 2:
            return 0.0

        dx = geometry.coords[1][0] - geometry.coords[0][0]
        dy = geometry.coords[1][1] - geometry.coords[0][1]

        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 180
