#!/usr/bin/env python3
"""
Inverse Growth Data Structures
Phase A: Core abstractions for inverse urban growth inference.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import numpy as np
import hashlib
from shapely.geometry import LineString


def compute_frontier_signature(frontier) -> Optional[str]:
    """
    Create stable geometric signature for frontier matching across state changes.

    This signature should remain consistent even when:
    - Graph operations shift coordinates slightly
    - Frontier IDs change during rewind/replay
    - Node IDs get renumbered

    Returns a 16-character hash based on normalized geometry.
    """
    if not hasattr(frontier, 'geometry') or frontier.geometry is None:
        return None

    geom = frontier.geometry
    if isinstance(geom, LineString) and len(geom.coords) >= 2:
        # Normalize coordinates: round to 1 decimal for stability, sort start/end
        start = (round(geom.coords[0][0], 1), round(geom.coords[0][1], 1))
        end = (round(geom.coords[-1][0], 1), round(geom.coords[-1][1], 1))

        # Sort to ensure order independence
        coords_sorted = tuple(sorted([start, end]))

        # Include frontier type for additional stability
        frontier_type = getattr(frontier, 'frontier_type', 'unknown')

        # Create signature string
        signature = f"{coords_sorted[0]}_{coords_sorted[1]}_{frontier_type}"

        # Return 16-character hash
        return hashlib.md5(signature.encode()).hexdigest()[:16]

    return None


class ActionType(Enum):
    """Types of growth actions that can be inferred."""
    EXTEND_FRONTIER = "extend_frontier"      # Add street by extending frontier
    SUBDIVIDE_BLOCK = "subdivide_block"      # Split block with new street
    REALIGN_STREET = "realign_street"        # Modify existing street geometry
    REMOVE_STREET = "remove_street"          # Delete street (non-monotonic)


@dataclass(frozen=True)
class InverseGrowthAction:
    """
    Represents an inferred growth action, separating decision intent from geometric realization.

    PHASE 2: Now stores complete state diffs to eliminate frontier matching dependency.
    """
    action_type: ActionType
    target_id: str                           # frontier/block/street identifier (legacy)
    intent_params: Dict[str, Any]            # decision parameters (e.g., {'direction': 'follow_density'})
    realized_geometry: Optional[Dict[str, Any]] = None  # geometric params for replay (legacy)
    confidence: float = 1.0                  # inference certainty 0-1
    timestamp: Optional[float] = None         # temporal ordering
    geometric_signature: Optional[str] = None  # stable geometric signature for replay matching (legacy)

    # PHASE 2: Complete state diff storage
    state_diff: Optional[Dict[str, Any]] = None  # Complete geometric changes
    # Structure:
    # {
    #   'added_streets': List[Dict],     # Complete street geometries to add
    #   'removed_streets': List[str],    # Street IDs to remove
    #   'graph_changes': Dict,           # Node/edge additions/removals
    #   'frontier_changes': Dict,        # Frontier state before/after
    # }

    action_metadata: Dict[str, Any] = field(default_factory=dict)  # inference artifacts


@dataclass(frozen=True)
class GrowthTrace:
    """
    Complete sequence of growth actions representing a city's development history.
    """
    actions: List[InverseGrowthAction]
    initial_state: Any  # GrowthState before any actions
    final_state: Any    # GrowthState after all actions
    metadata: Dict[str, Any] = field(default_factory=dict)  # inference metadata

    @property
    def total_actions(self) -> int:
        """Total number of actions in the trace."""
        return len(self.actions)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all actions."""
        if not self.actions:
            return 0.0
        return sum(action.confidence for action in self.actions) / len(self.actions)

    @property
    def high_confidence_actions(self) -> List[InverseGrowthAction]:
        """Actions with confidence >= 0.8."""
        return [action for action in self.actions if action.confidence >= 0.8]

    def filter_by_confidence(self, min_confidence: float) -> 'GrowthTrace':
        """Return a new trace with only actions meeting minimum confidence."""
        filtered_actions = [action for action in self.actions if action.confidence >= min_confidence]
        return GrowthTrace(
            actions=filtered_actions,
            initial_state=self.initial_state,
            final_state=self.final_state,
            metadata={**self.metadata, 'filtered_confidence': min_confidence}
        )


@dataclass
class StateActionSample:
    """ML training sample from validated trace."""

    # State features (frozen schema)
    state_features: np.ndarray          # Shape: (128,) - normalized per city

    # Action targets
    action_type: int                    # 0=EXTEND_FRONTIER, 1=SUBDIVIDE_BLOCK, etc.
    action_params: np.ndarray           # Continuous parameters (angle, length, etc.)

    # Metadata
    confidence: float                   # From inverse inference [0.0-1.0]
    city_id: str                        # Source city identifier
    step_index: int                     # Position in growth sequence

    # Context (not trained on, but useful for debugging)
    frontier_geometry: Optional[str] = None  # WKT for visualization
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLDataset:
    """Complete ML training dataset."""
    samples: List[StateActionSample]
    train_indices: List[int]            # Indices for training
    val_indices: List[int]              # Indices for validation
    test_indices: List[int]             # Indices for testing

    # Dataset metadata
    city_ids: Set[str]                  # All cities in dataset
    feature_stats: Dict[str, Any]       # Normalization statistics
    action_distribution: Dict[int, int] # Action type counts
