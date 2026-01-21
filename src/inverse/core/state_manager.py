#!/usr/bin/env python3
"""
State Manager Module

Manages state transitions and computes state differences for urban growth inference.
Provides utilities for tracking what changes when actions are applied or rewound.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from shapely.geometry import LineString

from src.core.contracts import GrowthState
from ..data_structures import InverseGrowthAction, ActionType

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages state transitions and computes state differences.

    Responsible for:
    - Computing complete state diffs for actions
    - Validating state transitions
    - Tracking graph and frontier changes
    """

    def __init__(self):
        """Initialize the state manager."""
        pass

    def compute_state_diff(self, current_state: GrowthState, action: InverseGrowthAction) -> Dict[str, Any]:
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
            edge_u = action.intent_params.get('edge_u')
            edge_v = action.intent_params.get('edge_v')

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

        # For REMOVE_STREET actions, the street being removed during rewind
        # is the one that was added during forward growth
        elif action.action_type == ActionType.REMOVE_STREET:
            # Find the street that matches this action's street_id
            street_id = action.street_id

            # Handle type conversion for DataFrame index lookup
            try:
                # Try to convert string index to appropriate type
                if isinstance(street_id, str) and street_id.isdigit():
                    numeric_id = int(street_id)
                    if numeric_id in current_state.streets.index:
                        street_id = numeric_id
                elif isinstance(street_id, int):
                    # Try string version if int not found
                    if street_id not in current_state.streets.index and str(street_id) in current_state.streets.index:
                        street_id = str(street_id)
            except (ValueError, TypeError):
                pass  # Keep original street_id

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

    def validate_state_transition(self, before_state: GrowthState, after_state: GrowthState,
                                expected_changes: Dict[str, Any]) -> bool:
        """
        Validate that a state transition matches expected changes.

        Args:
            before_state: State before transition
            after_state: State after transition
            expected_changes: Expected changes from state diff

        Returns:
            True if transition is valid
        """
        try:
            # Check street count changes
            expected_removed = len(expected_changes.get('removed_streets', []))
            expected_added = len(expected_changes.get('added_streets', []))
            actual_street_change = len(before_state.streets) - len(after_state.streets)

            if actual_street_change != expected_removed - expected_added:
                logger.warning(f"Street count change mismatch: expected {expected_removed - expected_added}, got {actual_street_change}")
                return False

            # Check graph changes
            graph_changes = expected_changes.get('graph_changes', {})
            expected_edge_change = (graph_changes.get('edges_before', 0) -
                                  graph_changes.get('edges_after', 0))
            actual_edge_change = (before_state.graph.number_of_edges() -
                                after_state.graph.number_of_edges())

            if expected_edge_change != actual_edge_change:
                logger.warning(f"Graph edge change mismatch: expected {expected_edge_change}, got {actual_edge_change}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating state transition: {e}")
            return False

    def get_state_summary(self, state: GrowthState) -> Dict[str, Any]:
        """
        Get a summary of the current state for debugging and monitoring.

        Args:
            state: Current growth state

        Returns:
            Dict with state summary information
        """
        summary = {
            'streets_count': len(state.streets),
            'frontiers_count': len(state.frontiers),
            'blocks_count': len(state.blocks) if hasattr(state, 'blocks') and state.blocks is not None else 0,
            'graph_nodes': state.graph.number_of_nodes(),
            'graph_edges': state.graph.number_of_edges(),
            'city_bounds': None,
            'street_types': {},
            'frontier_types': {}
        }

        # City bounds
        if hasattr(state, 'city_bounds') and state.city_bounds:
            summary['city_bounds'] = {
                'area': state.city_bounds.area,
                'centroid': (state.city_bounds.centroid.x, state.city_bounds.centroid.y)
            }

        # Street type distribution
        if hasattr(state.streets, 'iterrows'):
            for idx, street in state.streets.iterrows():
                highway_type = street.get('highway', 'unknown')
                summary['street_types'][highway_type] = summary['street_types'].get(highway_type, 0) + 1

        # Handle blocks safely
        if state.blocks is not None and hasattr(state.blocks, 'iterrows'):
            summary['blocks_count'] = len(state.blocks)
        else:
            summary['blocks_count'] = 0

        # Frontier type distribution
        for frontier in state.frontiers:
            frontier_type = getattr(frontier, 'frontier_type', 'unknown')
            summary['frontier_types'][frontier_type] = summary['frontier_types'].get(frontier_type, 0) + 1

        return summary

    def compute_transition_metrics(self, before_state: GrowthState, after_state: GrowthState) -> Dict[str, Any]:
        """
        Compute metrics for a state transition.

        Args:
            before_state: State before transition
            after_state: State after transition

        Returns:
            Dict with transition metrics
        """
        metrics = {
            'streets_removed': len(before_state.streets) - len(after_state.streets),
            'streets_added': len(after_state.streets) - len(before_state.streets),
            'graph_edges_removed': before_state.graph.number_of_edges() - after_state.graph.number_of_edges(),
            'graph_edges_added': after_state.graph.number_of_edges() - before_state.graph.number_of_edges(),
            'frontiers_removed': len(before_state.frontiers) - len(after_state.frontiers),
            'frontiers_added': len(after_state.frontiers) - len(before_state.frontiers),
        }

        # Net changes (same as the 'added' values since they represent net change)
        metrics['net_streets'] = metrics['streets_added']
        metrics['net_edges'] = metrics['graph_edges_added']
        metrics['net_frontiers'] = metrics['frontiers_added']

        return metrics

    def create_action_with_state_diff(self, action: InverseGrowthAction, state_diff: Dict[str, Any]) -> InverseGrowthAction:
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
            confidence=action.confidence,
            timestamp=action.timestamp,
            state_diff=state_diff,  # Add the state diff
            action_metadata=action.action_metadata
        )
