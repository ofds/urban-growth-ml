# Urban Growth ML: SOLID Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to extract the "good parts" from the current inference system and restructure it following SOLID principles. The refactoring will improve maintainability, testability, and extensibility while preserving proven functionality.

## Current System Analysis

### Good Parts to Extract

Based on analysis of `src/inverse/inference.py`, the following components are well-designed and should be preserved:

#### 1. **ArterialSkeletonExtractor** ✅
- **Location**: `src/inverse/skeleton.py`
- **Strengths**: Sophisticated multi-criteria skeleton detection
- **Keep**: Length ≥200m, betweenness ≥0.1, curvature ≤0.01 logic
- **Status**: Already well-separated, minor cleanup needed

#### 2. **PerformanceTracker** ✅
- **Location**: Currently embedded in `inference.py`
- **Strengths**: Memory-bounded tracking, comprehensive metrics
- **Keep**: Operation timing, strategy performance, bounds checking
- **Action**: Extract to `metrics/performance_tracker.py`

#### 3. **SpatialIndex** ✅
- **Location**: Currently embedded in `inference.py`
- **Strengths**: R-tree based O(log n) queries, incremental updates
- **Keep**: Spatial indexing logic, memory management
- **Action**: Extract to `spatial/spatial_index.py`

#### 4. **Simple Heuristics** ✅
- **Location**: `BasicInferenceEngine.infer_most_recent_action()`
- **Strengths**: Computationally efficient two-tier priority system
- **Keep**: Peripheral dead-end → short street logic
- **Action**: Extract to `strategies/peripheral_strategy.py`

#### 5. **State Diff Computation** ✅
- **Location**: `BasicInferenceEngine._compute_state_diff()`
- **Strengths**: Comprehensive state change tracking
- **Keep**: Complete diff calculation for replay validation
- **Action**: Extract to `core/state_manager.py`

## SOLID Violations in Current System

### Single Responsibility Principle (SRP)
**Violation**: `inference.py` (2000+ lines) handles:
- Strategy orchestration
- Performance tracking
- Spatial indexing
- State management
- Multiple algorithm implementations

### Open/Closed Principle (OCP)
**Violation**: Strategies are hardcoded, not easily extensible without modifying core classes.

### Liskov Substitution Principle (LSP)
**Violation**: Strategy inheritance is inconsistent, some strategies override methods improperly.

### Interface Segregation Principle (ISP)
**Violation**: Large interfaces with many methods that clients don't need.

### Dependency Inversion Principle (DIP)
**Violation**: High-level modules depend on low-level strategy implementations rather than abstractions.

## Proposed Architecture

### Directory Structure

```
src/inverse/
├── core/                          # Core inference logic
│   ├── __init__.py
│   ├── inference_engine.py       # Main orchestration (SRP)
│   ├── state_manager.py          # State transitions (SRP)
│   └── action_factory.py         # Action creation (SRP)
├── strategies/                   # Strategy implementations
│   ├── __init__.py
│   ├── base_strategy.py          # Abstract strategy (ISP)
│   ├── peripheral_strategy.py    # Extracted heuristics
│   ├── fractal_strategy.py       # Existing (refactored)
│   ├── angle_strategy.py         # Existing (refactored)
│   └── block_strategy.py         # Existing (refactored)
├── metrics/                      # Performance & quality
│   ├── __init__.py
│   ├── performance_tracker.py    # Extracted (SRP)
│   ├── quality_assessor.py       # New functionality
│   └── confidence_scorer.py      # New functionality
├── spatial/                      # Geometric operations
│   ├── __init__.py
│   ├── spatial_index.py          # Extracted (SRP)
│   └── geometry_utils.py         # Geometric utilities
└── __init__.py
```

### Class Responsibilities

#### Core Module
- **InferenceEngine**: Orchestrates strategies, manages execution flow
- **StateManager**: Handles state transitions and diff computation
- **ActionFactory**: Creates actions from strategy candidates

#### Strategies Module
- **BaseStrategy**: Abstract interface for all strategies (ISP)
- **PeripheralStrategy**: Fast heuristics from BasicInferenceEngine
- **FractalStrategy**: Mathematical pattern detection
- **AngleStrategy**: Statistical angle analysis
- **BlockStrategy**: Geometric block completion

#### Metrics Module
- **PerformanceTracker**: Operation timing and resource tracking
- **QualityAssessor**: Data quality evaluation
- **ConfidenceScorer**: Confidence calculation algorithms

#### Spatial Module
- **SpatialIndex**: R-tree based geometric queries
- **GeometryUtils**: Common geometric operations

## Implementation Plan

### Phase 1: Infrastructure Setup (3 days)

#### Day 1: Directory Structure & Interfaces
```bash
# Create new directory structure
mkdir -p src/inverse/{core,strategies,metrics,spatial}

# Create base strategy interface
cat > src/inverse/strategies/base_strategy.py << 'EOF'
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from ..data_structures import InverseGrowthAction
from ..core.contracts import GrowthState

class InferenceStrategy(ABC):
    """Abstract base class for inference strategies."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def generate_candidates(
        self,
        state: GrowthState,
        skeleton_edges: set,
        spatial_index: Optional['SpatialIndex'] = None
    ) -> List[Tuple[InverseGrowthAction, float]]:
        """
        Generate candidate actions with confidence scores.

        Args:
            state: Current city state
            skeleton_edges: Set of skeleton edge tuples
            spatial_index: Optional spatial index for queries

        Returns:
            List of (action, confidence) tuples
        """
        pass

    def _validate_inputs(self, state: GrowthState, skeleton_edges: set) -> bool:
        """Common input validation for all strategies."""
        if state is None or state.streets is None:
            return False
        if not isinstance(skeleton_edges, set):
            return False
        return True
EOF
```

#### Day 2: Extract Core Components
```bash
# Extract PerformanceTracker
sed -n '/class PerformanceTracker/,/^class /p' src/inverse/inference.py | head -n -1 > src/inverse/metrics/performance_tracker.py

# Extract SpatialIndex
sed -n '/class SpatialIndex/,/^class /p' src/inverse/inference.py | head -n -1 > src/inverse/spatial/spatial_index.py

# Add necessary imports
echo "from typing import Dict, Any, List, Tuple
import time
import logging

logger = logging.getLogger(__name__)" >> src/inverse/metrics/performance_tracker.py

echo "from typing import Dict, Any, List, Optional
import logging
from rtree import index

logger = logging.getLogger(__name__)" >> src/inverse/spatial/spatial_index.py
```

#### Day 3: Create State Manager
```bash
# Extract state management logic
cat > src/inverse/core/state_manager.py << 'EOF'
from typing import Dict, Any
from ..core.contracts import GrowthState
from ..data_structures import InverseGrowthAction, ActionType

class StateManager:
    """Manages state transitions and diff computation."""

    def compute_state_diff(
        self,
        current_state: GrowthState,
        action: InverseGrowthAction
    ) -> Dict[str, Any]:
        """Compute complete state diff for replay validation."""
        state_diff = {
            'added_streets': [],
            'removed_streets': [],
            'graph_changes': {},
            'frontier_changes': {}
        }

        if action.action_type == ActionType.REMOVE_STREET:
            street_id = action.street_id
            if street_id in current_state.streets.index:
                street = current_state.streets.loc[street_id]
                street_data = {
                    'edge_id': (min(street.get('u'), street.get('v')),
                               max(street.get('u'), street.get('v'))),
                    'u': street.get('u'),
                    'v': street.get('v'),
                    'geometry_wkt': street.geometry.wkt if hasattr(street.geometry, 'wkt') else None,
                    'osmid': street.get('osmid'),
                    'highway': street.get('highway'),
                    'length': street.geometry.length if hasattr(street.geometry, 'length') else None
                }
                state_diff['added_streets'].append(street_data)
                state_diff['removed_streets'].append(street_id)

        # Record graph state
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

    def validate_state_transition(
        self,
        before_state: GrowthState,
        after_state: GrowthState,
        action: InverseGrowthAction
    ) -> bool:
        """Validate that state transition is consistent with action."""
        if action.action_type == ActionType.REMOVE_STREET:
            # Rewind should add streets back
            if len(after_state.streets) <= len(before_state.streets):
                return False

        # Graph edges should change appropriately
        before_edges = before_state.graph.number_of_edges()
        after_edges = after_state.graph.number_of_edges()

        if action.action_type == ActionType.REMOVE_STREET:
            if after_edges <= before_edges:
                return False

        return True
EOF
```

### Phase 2: Strategy Extraction (4 days)

#### Day 4-5: Extract Peripheral Strategy
```python
# Extract the good heuristic logic from BasicInferenceEngine
cat > src/inverse/strategies/peripheral_strategy.py << 'EOF'
from typing import List, Tuple, Optional
import logging
from shapely.geometry import LineString
from .base_strategy import InferenceStrategy
from ..data_structures import InverseGrowthAction, ActionType
from ..core.contracts import GrowthState

logger = logging.getLogger(__name__)

class PeripheralStrategy(InferenceStrategy):
    """
    Fast peripheral removal strategy extracted from BasicInferenceEngine.

    Uses efficient two-tier priority system:
    1. Most peripheral dead-end frontiers
    2. Shortest non-skeleton streets
    """

    def __init__(self):
        super().__init__("peripheral", weight=1.0)

    def generate_candidates(
        self,
        state: GrowthState,
        skeleton_edges: set,
        spatial_index: Optional['SpatialIndex'] = None
    ) -> List[Tuple[InverseGrowthAction, float]]:
        """Generate candidates using fast peripheral heuristics."""

        if not self._validate_inputs(state, skeleton_edges):
            return []

        candidates = []

        # Priority 1: Peripheral dead-end frontiers
        peripheral_candidates = self._find_peripheral_frontiers(state, skeleton_edges)
        candidates.extend(peripheral_candidates)

        # Priority 2: Short streets (if no peripheral frontiers)
        if not candidates:
            short_candidates = self._find_short_streets(state, skeleton_edges)
            candidates.extend(short_candidates)

        return candidates

    def _find_peripheral_frontiers(
        self,
        state: GrowthState,
        skeleton_edges: set
    ) -> List[Tuple[InverseGrowthAction, float]]:
        """Find most peripheral dead-end frontiers."""

        center = self._calculate_city_center(state)
        dead_end_frontiers = [
            f for f in state.frontiers
            if f.frontier_type == "dead_end"
        ]

        if not dead_end_frontiers:
            return []

        # Find most distant frontier
        peripheral_frontier = max(
            dead_end_frontiers,
            key=lambda f: self._distance_from_center(f.geometry, center)
        )

        # Create action (extracted from original logic)
        from shapely import wkt
        stable_id = self._compute_stable_frontier_id(peripheral_frontier)
        geometric_signature = self._compute_frontier_signature(peripheral_frontier)

        action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=peripheral_frontier.frontier_id,
            intent_params={
                "direction": "peripheral_expansion",
                "edge_u": str(peripheral_frontier.edge_id[0]),
                "edge_v": str(peripheral_frontier.edge_id[1]),
                "stable_id": stable_id
            },
            confidence=0.8,
            timestamp=len(state.streets),
            state_diff={
                "geometry_wkt": wkt.dumps(peripheral_frontier.geometry),
                "edgeid": peripheral_frontier.edge_id,
                "frontier_type": peripheral_frontier.frontier_type,
                "stable_id": stable_id
            },
            action_metadata={
                "geometric_signature": geometric_signature
            }
        )

        return [(action, 0.8)]

    def _find_short_streets(
        self,
        state: GrowthState,
        skeleton_edges: set
    ) -> List[Tuple[InverseGrowthAction, float]]:
        """Find shortest non-skeleton streets."""

        candidates = []
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

            length = geometry.length
            candidates.append((idx, length, street))

        if not candidates:
            return []

        # Get shortest street
        shortest_idx, length, street = min(candidates, key=lambda x: x[1])

        # Create action (extracted from original logic)
        from shapely import wkt
        stable_id = self._compute_stable_frontier_id(street)
        geometric_signature = self._compute_frontier_signature(street)

        action = InverseGrowthAction(
            action_type=ActionType.REMOVE_STREET,
            street_id=str(shortest_idx),
            intent_params={
                'strategy': 'short_segment',
                'edge_u': str(street.get('u')),
                'edge_v': str(street.get('v')),
                'stable_id': stable_id
            },
            confidence=0.6,
            timestamp=len(state.streets),
            state_diff={
                'geometry_wkt': wkt.dumps(street.geometry),
                'edgeid': (min(street.get('u'), street.get('v')),
                          max(street.get('u'), street.get('v'))),
                'frontier_type': 'street_removal',
                'stable_id': stable_id
            },
            action_metadata={
                'geometric_signature': geometric_signature
            }
        )

        return [(action, 0.6)]

    def _calculate_city_center(self, state: GrowthState):
        """Calculate city center (extracted from BasicInferenceEngine)."""
        if state.city_bounds:
            return state.city_bounds.centroid

        # Fallback: mean of all street coordinates
        all_coords = []
        for idx, street in state.streets.iterrows():
            if hasattr(street.geometry, 'coords'):
                all_coords.extend(street.geometry.coords)

        if all_coords:
            x_coords = [c[0] for c in all_coords]
            y_coords = [c[1] for c in all_coords]
            return type('Point', (), {
                'x': sum(x_coords)/len(x_coords),
                'y': sum(y_coords)/len(y_coords)
            })()

        return type('Point', (), {'x': 0, 'y': 0})()

    def _distance_from_center(self, geometry, center) -> float:
        """Calculate distance from geometry to city center."""
        if hasattr(geometry, 'centroid'):
            geom_center = geometry.centroid
            return ((geom_center.x - center.x)**2 + (geom_center.y - center.y)**2)**0.5
        return 0.0

    def _compute_stable_frontier_id(self, frontier) -> str:
        """Generate stable ID from geometry (extracted logic)."""
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

    def _compute_frontier_signature(self, frontier):
        """Compute geometric signature (placeholder for extracted logic)."""
        # Extracted from data_structures.compute_frontier_signature
        return {"placeholder": "signature"}
EOF
```

#### Day 6-7: Refactor Existing Strategies
- Update `FractalPatternStrategy`, `AngleHarmonizationStrategy`, `BlockCentroidStrategy` to inherit from `BaseStrategy`
- Ensure consistent interface implementation
- Add proper error handling and logging

### Phase 3: Core Engine Implementation (3 days)

#### Day 8: Create Inference Engine
```python
# Main orchestration engine
cat > src/inverse/core/inference_engine.py << 'EOF'
from typing import List, Optional, Dict, Any
import logging
from ..data_structures import GrowthTrace, InverseGrowthAction
from ..core.contracts import GrowthState
from ..strategies.base_strategy import InferenceStrategy
from ..metrics.performance_tracker import PerformanceTracker
from ..spatial.spatial_index import SpatialIndex
from ..core.state_manager import StateManager
from .rewind import RewindEngine

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Main inference orchestration engine following SOLID principles.

    Responsibilities:
    - Coordinate strategies (SRP)
    - Manage execution flow
    - Track performance
    - Ensure data consistency
    """

    def __init__(
        self,
        strategies: List[InferenceStrategy],
        performance_tracker: Optional[PerformanceTracker] = None,
        spatial_index: Optional[SpatialIndex] = None,
        state_manager: Optional[StateManager] = None
    ):
        # Dependency injection for DIP compliance
        self.strategies = strategies
        self.performance_tracker = performance_tracker or PerformanceTracker()
        self.spatial_index = spatial_index or SpatialIndex()
        self.state_manager = state_manager or StateManager()
        self.rewind_engine = RewindEngine()

    def infer_trace(
        self,
        final_state: GrowthState,
        max_steps: int = 1000,
        initial_state: Optional[GrowthState] = None,
        progress_callback: Optional[callable] = None
    ) -> GrowthTrace:
        """
        Execute inference using configured strategies.

        Args:
            final_state: Final city state to analyze
            max_steps: Maximum inference steps
            initial_state: Optional known initial state
            progress_callback: Optional progress reporting

        Returns:
            Complete growth trace
        """
        logger.info(f"Starting inference with {len(self.strategies)} strategies")

        # Initialize tracking
        self.performance_tracker.reset()
        self.spatial_index = SpatialIndex()  # Fresh index

        # Setup initial state
        current_state = final_state
        actions = []
        step = 0

        # Build spatial indexes for performance
        self.performance_tracker.start_operation("spatial_index_building")
        self.spatial_index.build_street_index(final_state.streets)
        self.spatial_index.build_frontier_index(final_state.frontiers)
        self.spatial_index.build_block_index(final_state.blocks)
        self.performance_tracker.end_operation("spatial_index_building")

        # Main inference loop
        while step < max_steps:
            if len(current_state.streets) <= len(initial_state.streets if initial_state else 5):
                logger.info(f"Reached minimal state at step {step}")
                break

            # Generate candidates from all strategies
            self.performance_tracker.start_operation("candidate_generation")
            all_candidates = []

            for strategy in self.strategies:
                strategy_start = self.performance_tracker.operation_times.get('strategy_timing', {}).get('start', time.time())
                try:
                    candidates = strategy.generate_candidates(
                        current_state,
                        set(),  # skeleton_edges - would be computed
                        self.spatial_index
                    )
                    all_candidates.extend(candidates)
                except Exception as e:
                    logger.warning(f"Strategy {strategy.name} failed: {e}")
                finally:
                    strategy_time = time.time() - strategy_start
                    self.performance_tracker.record_strategy_time(strategy.name, strategy_time)

            self.performance_tracker.end_operation("candidate_generation")

            if not all_candidates:
                logger.info(f"No candidates found at step {step}")
                break

            # Select best candidate
            best_action, confidence = max(all_candidates, key=lambda x: x[1])

            # Execute rewind
            self.performance_tracker.start_operation("rewind_operation")
            state_diff = self.state_manager.compute_state_diff(current_state, best_action)
            prev_state = self.rewind_engine.rewind_action(best_action, current_state)
            self.performance_tracker.end_operation("rewind_operation")

            # Validate transition
            if not self.state_manager.validate_state_transition(current_state, prev_state, best_action):
                logger.error(f"Invalid state transition at step {step}")
                break

            # Record action
            action_with_diff = self._add_state_diff_to_action(best_action, state_diff)
            actions.insert(0, action_with_diff)

            # Progress tracking
            if progress_callback and step % 10 == 0:
                progress_callback(GrowthTrace(actions, None, current_state, {}))

            current_state = prev_state
            step += 1

        # Create final trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=initial_state,
            final_state=final_state,
            metadata={
                "strategies_used": [s.name for s in self.strategies],
                "performance_stats": self.performance_tracker.get_summary_stats()
            }
        )

        logger.info(f"Inference complete: {len(actions)} actions")
        return trace

    def _add_state_diff_to_action(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
        """Add state diff to action (immutable update)."""
        # Create new action instance with state_diff
        return InverseGrowthAction(
            action_type=action.action_type,
            street_id=action.street_id,
            intent_params=action.intent_params,
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=state_diff,
            action_metadata=action.action_metadata
        )
EOF
```

#### Day 9: Integration & Backward Compatibility
- Create facade class maintaining existing API
- Update imports and dependencies
- Add comprehensive error handling

### Phase 4: Testing & Validation (2 days)

#### Day 10-11: Unit Tests & Integration
- Test each extracted component independently
- Validate strategy interfaces
- Performance regression testing
- Backward compatibility verification

## Benefits of Refactoring

### Maintainability
- **Single Responsibility**: Each class has one clear purpose
- **Smaller Files**: Easier to navigate and understand
- **Clear Dependencies**: Explicit relationships between components

### Testability
- **Isolated Units**: Each component can be tested independently
- **Mock Injection**: Dependencies can be easily mocked
- **Focused Tests**: Tests target specific functionality

### Extensibility
- **Strategy Pattern**: New strategies don't require core changes
- **Plugin Architecture**: Components can be swapped or extended
- **Interface Compliance**: New implementations must meet contracts

### Reusability
- **Shared Components**: PerformanceTracker and SpatialIndex usable elsewhere
- **Modular Design**: Components can be combined differently
- **Clean APIs**: Well-defined interfaces for integration

## Risk Mitigation

### Backward Compatibility
- **Facade Pattern**: Maintain existing `MultiStrategyInferenceEngine` API
- **Gradual Migration**: Old system remains functional during transition
- **Comprehensive Testing**: Ensure no regressions in functionality

### Performance Impact
- **Benchmarking**: Compare performance before/after refactoring
- **Optimization**: Profile and optimize critical paths
- **Memory Management**: Ensure no memory leaks in new structure

### Development Overhead
- **Incremental Approach**: Extract components one at a time
- **Continuous Integration**: Regular testing during refactoring
- **Documentation**: Update docs as components are extracted

## Success Metrics

### Code Quality
- **Cyclomatic Complexity**: Reduce average from 15 to <8 per method
- **File Size**: Keep files under 500 lines
- **Test Coverage**: Maintain >90% coverage

### Performance
- **No Regression**: Inference speed within 5% of original
- **Memory Usage**: No significant increase
- **Scalability**: Maintain ability to handle large cities

### Maintainability
- **Bug Fixes**: Easier to locate and fix issues
- **Feature Addition**: New strategies added in <2 days
- **Code Reviews**: Faster and more focused reviews

## Conclusion

This SOLID refactoring plan extracts the proven good parts of the current system while restructuring it for better maintainability and extensibility. The phased approach minimizes risk while delivering significant long-term benefits for the urban growth ML project.

The refactoring preserves the sophisticated skeleton extraction, efficient heuristics, and robust state management while making the system more modular, testable, and ready for future enhancements like center-out inference and advanced quality assessment.
