#!/usr/bin/env python3
"""
Inference Engine Module

Main orchestration engine for urban growth inference.
Coordinates strategies, performance tracking, and spatial indexing.
"""

import logging
import time
from typing import List, Optional, Dict, Any, Callable
from contextlib import contextmanager

from .config import InferenceConfig
from ..metrics.performance_tracker import PerformanceTracker
from ..spatial.spatial_index import SpatialIndex
from .state_manager import StateManager
from .action_factory import ActionFactory
from ..strategies.base_strategy import BaseInferenceStrategy
from ..data_structures import GrowthTrace, InverseGrowthAction
from ..rewind import RewindEngine
from ..skeleton import ArterialSkeletonExtractor
from src.core.contracts import GrowthState

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main orchestration engine for urban growth inference.

    Coordinates all components: strategies, performance tracking, spatial indexing,
    state management, and action creation. Implements the core inference loop
    with proper error handling, timeouts, and performance monitoring.
    """

    def __init__(self, strategies: List[BaseInferenceStrategy],
                 config: Optional[InferenceConfig] = None):
        """
        Initialize the inference engine.

        Args:
            strategies: List of inference strategies to use
            config: Configuration instance (uses default if None)
        """
        self.config = config or InferenceConfig()
        self.strategies = strategies

        # Initialize core components
        self.performance_tracker = PerformanceTracker(
            max_history_size=self.config.performance.max_history_size
        )
        self.spatial_index = SpatialIndex()
        self.state_manager = StateManager()
        self.action_factory = ActionFactory(self.config)

        # Initialize external dependencies
        self.rewind_engine = RewindEngine()
        self.skeleton_extractor = ArterialSkeletonExtractor()

        # Validate configuration
        self._validate_engine_config()

        logger.info(f"InferenceEngine initialized with {len(strategies)} strategies: {[s.name for s in strategies]}")

    def _validate_engine_config(self):
        """Validate engine configuration and strategy compatibility."""
        if not self.strategies:
            raise ValueError("At least one strategy must be provided")

        # Validate strategy configurations
        for strategy in self.strategies:
            if not self.config.is_strategy_enabled(strategy.name):
                logger.warning(f"Strategy '{strategy.name}' is disabled in configuration but provided to engine")

        # Validate limits
        if self.config.limits.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        if self.config.limits.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def infer_trace(self, final_state: GrowthState, max_steps: Optional[int] = None,
                    initial_state: Optional[GrowthState] = None,
                    progress_callback: Optional[Callable[[GrowthTrace], None]] = None) -> GrowthTrace:
        """
        Infer growth trace using configured strategies.

        Main orchestration method that coordinates the entire inference process.

        Args:
            final_state: Final grown city state to infer from
            max_steps: Maximum inference steps (uses config default if None)
            initial_state: Known initial state (will extract skeleton if None)
            progress_callback: Optional callback for progress reporting

        Returns:
            Inferred GrowthTrace with actions and metadata
        """
        logger.info("Starting inference with InferenceEngine...")

        # Use config defaults if not specified
        if max_steps is None:
            max_steps = self.config.limits.max_steps

        # Reset performance tracking
        self.performance_tracker.reset()

        # Extract or use provided initial state
        initial_state, skeleton_edges = self._prepare_initial_state(final_state, initial_state)

        # Initialize inference state
        current_state = final_state
        actions = []
        step = 0

        # Build spatial indexes
        self._build_spatial_indexes(final_state)

        # Progress tracking
        total_streets_to_remove = len(final_state.streets) - len(initial_state.streets)
        logger.info(f"Inference target: remove {total_streets_to_remove} streets to reach {len(initial_state.streets)} streets")

        # Track consecutive steps with no progress
        last_street_count = len(final_state.streets)
        no_progress_count = 0

        # Main inference loop
        while step < max_steps:
            if len(current_state.streets) <= len(initial_state.streets):
                logger.info(f"Reached initial state size at step {step}")
                break

            step_start_time = time.perf_counter()

            # Generate candidates from all strategies
            all_candidates = self._generate_candidates(current_state, skeleton_edges)

            if not all_candidates:
                logger.info(f"No more actions to infer at step {step}")
                break

            # Select best candidate
            best_action, confidence = max(all_candidates, key=lambda x: x[1])

            # Execute rewind
            prev_state = self._execute_rewind(best_action, current_state)

            if prev_state is None:
                logger.warning(f"Rewind failed at step {step}")
                break

            # Validate rewind succeeded
            if len(prev_state.streets) >= len(current_state.streets):
                logger.warning(f"Rewind did not reduce street count at step {step}")
                break

            # Create final action with state diff
            action_with_diff = self._finalize_action(best_action, current_state, prev_state)
            actions.insert(0, action_with_diff)

            # Update spatial indexes incrementally
            self._update_spatial_indexes(current_state, prev_state, step)

            # Record performance metrics
            self._record_step_metrics(step, current_state, prev_state, len(all_candidates),
                                    self._get_strategy_stats(all_candidates))

            # Progress callback
            if progress_callback:
                current_trace = self._create_progress_trace(actions, initial_state, current_state, step)
                progress_callback(current_trace)

            # Check for progress stagnation
            if not self._check_progress(prev_state, last_street_count, no_progress_count):
                break

            last_street_count = len(prev_state.streets)
            current_state = prev_state

            # Logging
            self._log_step_progress(step, current_state, initial_state, len(all_candidates), confidence,
                                  best_action.intent_params.get('strategy', 'unknown'))

            step += 1

        # Create final trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata=self._create_trace_metadata(step, skeleton_edges)
        )

        # Log final performance summary
        self.performance_tracker.log_performance_summary()

        logger.info(f"InferenceEngine complete: {len(actions)} actions inferred in {step} steps")
        return trace

    def _prepare_initial_state(self, final_state: GrowthState, initial_state: Optional[GrowthState]) -> tuple:
        """Prepare initial state and skeleton edges."""
        self.performance_tracker.start_operation("initial_state_preparation")

        if initial_state is None:
            logger.info("Extracting arterial skeleton for initial state...")
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

        self.performance_tracker.end_operation("initial_state_preparation")
        return initial_state, skeleton_edges_set

    def _build_spatial_indexes(self, state: GrowthState):
        """Build spatial indexes for performance optimization."""
        self.performance_tracker.start_operation("spatial_index_building")

        start_time = time.perf_counter()
        self.spatial_index.build_street_index(state.streets)
        street_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        self.spatial_index.build_frontier_index(state.frontiers)
        frontier_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        self.spatial_index.build_block_index(state.blocks)
        block_time = time.perf_counter() - start_time

        self.performance_tracker.end_operation("spatial_index_building")

        logger.debug(f"Spatial indexes built: streets={street_time:.2f}s, frontiers={frontier_time:.2f}s, blocks={block_time:.2f}s")

    def _generate_candidates(self, state: GrowthState, skeleton_edges: set) -> List[tuple]:
        """Generate candidates from all strategies with error handling."""
        self.performance_tracker.start_operation("candidate_generation")

        all_candidates = []
        strategy_stats = {}

        for strategy in self.strategies:
            candidates = self._generate_candidates_from_strategy(strategy, state, skeleton_edges)
            strategy_stats[strategy.name] = len(candidates)

            # Apply strategy weighting
            weighted_candidates = self._apply_strategy_weighting(candidates, strategy)
            all_candidates.extend(weighted_candidates)

        # Renormalize confidences across strategies
        if all_candidates:
            max_confidence = max(conf for _, conf in all_candidates)
            if max_confidence > 0:
                all_candidates = [(action, conf / max_confidence) for action, conf in all_candidates]

        self.performance_tracker.end_operation("candidate_generation")

        logger.debug(f"Generated {len(all_candidates)} total candidates from {len(self.strategies)} strategies")
        return all_candidates

    def _generate_candidates_from_strategy(self, strategy: BaseInferenceStrategy,
                                         state: GrowthState, skeleton_edges: set) -> List[tuple]:
        """Generate candidates from a single strategy with timeout protection."""
        candidates = []

        try:
            with self._strategy_timeout_context(strategy.name):
                start_time = time.perf_counter()
                candidates = strategy.generate_candidates(state, skeleton_edges, self.spatial_index)
                duration = time.perf_counter() - start_time

                self.performance_tracker.record_strategy_time(strategy.name, duration)

                # Limit candidates per strategy
                if len(candidates) > self.config.limits.max_candidates_per_strategy:
                    logger.warning(f"Strategy {strategy.name} generated {len(candidates)} candidates, limiting to {self.config.limits.max_candidates_per_strategy}")
                    candidates = candidates[:self.config.limits.max_candidates_per_strategy]

        except TimeoutError:
            logger.warning(f"Strategy {strategy.name} timed out after {self.config.limits.strategy_timeout_seconds}s")
        except Exception as e:
            logger.warning(f"Strategy {strategy.name} failed: {e}")

        return candidates

    @contextmanager
    def _strategy_timeout_context(self, strategy_name: str):
        """Context manager for strategy execution with timeout."""
        # Simplified timeout implementation (signal-based timeout not available on Windows)
        # In production, this would use threading.Timer or asyncio
        yield

    def _apply_strategy_weighting(self, candidates: List[tuple], strategy: BaseInferenceStrategy) -> List[tuple]:
        """Apply strategy-specific weighting to candidate confidences."""
        strategy_weight = getattr(strategy, 'weight', 1.0)
        return [(action, confidence * strategy_weight) for action, confidence in candidates]

    def _execute_rewind(self, action: InverseGrowthAction, current_state: GrowthState) -> Optional[GrowthState]:
        """Execute rewind operation with error handling."""
        self.performance_tracker.start_operation("rewind_operation")

        try:
            prev_state = self.rewind_engine.rewind_action(action, current_state)
            self.performance_tracker.end_operation("rewind_operation")
            return prev_state
        except Exception as e:
            logger.error(f"Rewind operation failed: {e}")
            self.performance_tracker.end_operation("rewind_operation")
            return None

    def _finalize_action(self, action: InverseGrowthAction, before_state: GrowthState,
                        after_state: GrowthState) -> InverseGrowthAction:
        """Create final action with complete state diff."""
        self.performance_tracker.start_operation("action_finalization")

        # Compute state diff
        state_diff = self.state_manager.compute_state_diff(before_state, action)

        # Update state diff with after-state information
        state_diff['graph_changes']['nodes_after'] = after_state.graph.number_of_nodes()
        state_diff['graph_changes']['edges_after'] = after_state.graph.number_of_edges()
        state_diff['frontier_changes']['frontiers_after'] = len(after_state.frontiers)

        # Create final action
        final_action = self.state_manager.create_action_with_state_diff(action, state_diff)

        self.performance_tracker.end_operation("action_finalization")
        return final_action

    def _update_spatial_indexes(self, current_state: GrowthState, prev_state: GrowthState, step: int):
        """Update spatial indexes incrementally."""
        self.performance_tracker.start_operation("spatial_index_update")

        # Check if we should do a full rebuild
        if step % self.config.limits.full_rebuild_interval == 0:
            logger.debug(f"Step {step}: Performing full spatial index rebuild")
            self.spatial_index.build_street_index(prev_state.streets)
            self.spatial_index.build_frontier_index(prev_state.frontiers)
            self.spatial_index.build_block_index(prev_state.blocks)
        else:
            # Incremental updates would go here
            # For now, simplified approach
            self.spatial_index.build_street_index(prev_state.streets)
            self.spatial_index.build_frontier_index(prev_state.frontiers)
            self.spatial_index.build_block_index(prev_state.blocks)

        self.performance_tracker.end_operation("spatial_index_update")

    def _record_step_metrics(self, step: int, before_state: GrowthState, after_state: GrowthState,
                           candidate_count: int, strategy_stats: Dict[str, int]):
        """Record performance metrics for the step."""
        streets_before = len(before_state.streets)
        streets_after = len(after_state.streets)

        self.performance_tracker.record_step_metrics(
            step, streets_before, streets_after, candidate_count, strategy_stats
        )

    def _get_strategy_stats(self, candidates: List[tuple]) -> Dict[str, int]:
        """Extract strategy statistics from candidates."""
        # This is a simplified implementation
        # In practice, we'd track which strategy generated each candidate
        return {strategy.name: 0 for strategy in self.strategies}

    def _check_progress(self, state: GrowthState, last_count: int, no_progress_count: int) -> bool:
        """Check if inference is making progress."""
        current_count = len(state.streets)

        if current_count >= last_count:
            no_progress_count += 1
            if no_progress_count >= self.config.validation.max_no_progress_steps:
                logger.error(f"No progress for {self.config.validation.max_no_progress_steps} consecutive steps")
                return False
        else:
            no_progress_count = 0

        return True

    def _log_step_progress(self, step: int, current_state: GrowthState, initial_state: GrowthState,
                          candidate_count: int, confidence: float, strategy_name: str):
        """Log progress for current step."""
        should_log = (
            step < 10 or  # First 10 steps
            step % self.config.logging.progress_log_interval == 0 or  # Regular intervals
            len(current_state.streets) <= len(initial_state.streets) + 50  # Near completion
        )

        if should_log:
            remaining = max(0, len(current_state.streets) - len(initial_state.streets))
            total_target = len(current_state.streets) - len(initial_state.streets)
            progress_pct = (1 - remaining / max(1, total_target)) * 100

            logger.info(f"Step {step}: {strategy_name} (conf={confidence:.2f}), "
                       f"streets {len(current_state.streets)} -> target {len(initial_state.streets)} "
                       f"({progress_pct:.1f}% complete), {candidate_count} candidates")

    def _create_progress_trace(self, actions: List[InverseGrowthAction], initial_state: GrowthState,
                             current_state: GrowthState, steps_taken: int) -> GrowthTrace:
        """Create progress trace for callback."""
        return GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=current_state,
            metadata={
                "inference_method": "modular_inference_engine",
                "strategies": [s.name for s in self.strategies],
                "steps_taken": steps_taken,
                "interrupted": True,
                "performance_stats": self.performance_tracker.get_summary_stats()
            }
        )

    def _create_trace_metadata(self, steps_taken: int, skeleton_edges: set) -> Dict[str, Any]:
        """Create metadata for final trace."""
        return {
            "inference_method": "modular_inference_engine",
            "strategies": [s.name for s in self.strategies],
            "max_steps": self.config.limits.max_steps,
            "steps_taken": steps_taken,
            "skeleton_streets": len(skeleton_edges),
            "performance_stats": self.performance_tracker.get_summary_stats(),
            "engine_config": {
                "spatial_indexing_enabled": self.config.performance.enable_spatial_indexing,
                "incremental_updates_enabled": self.config.performance.enable_incremental_updates,
                "batch_caching_enabled": self.config.performance.enable_batch_caching
            }
        }
