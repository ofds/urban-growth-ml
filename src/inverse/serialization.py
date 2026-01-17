#!/usr/bin/env python3
"""
Trace Serialization Utilities
Phase A: Save and load growth traces for persistence and analysis.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any
from .data_structures import GrowthTrace, InverseGrowthAction, ActionType


class TraceEncoder(json.JSONEncoder):
    """Custom JSON encoder for GrowthTrace objects."""

    def default(self, obj):
        if isinstance(obj, GrowthTrace):
            return {
                '_type': 'GrowthTrace',
                'actions': obj.actions,
                'initial_state': self._serialize_growth_state(obj.initial_state),
                'final_state': self._serialize_growth_state(obj.final_state),
                'metadata': obj.metadata
            }
        elif isinstance(obj, InverseGrowthAction):
            return {
                '_type': 'InverseGrowthAction',
                'action_type': obj.action_type.value,
                'target_id': obj.target_id,
                'intent_params': obj.intent_params,
                'realized_geometry': obj.realized_geometry,
                'confidence': obj.confidence,
                'timestamp': obj.timestamp,
                'action_metadata': obj.action_metadata
            }
        elif isinstance(obj, ActionType):
            return obj.value
        else:
            return super().default(obj)

    def _serialize_growth_state(self, state) -> Dict[str, Any]:
        """Serialize GrowthState for JSON (simplified representation)."""
        if state is None:
            return None

        # Create a simplified serializable representation
        return {
            'iteration': getattr(state, 'iteration', 0),
            'streets_count': len(getattr(state, 'streets', [])),
            'blocks_count': len(getattr(state, 'blocks', [])),
            'frontiers_count': len(getattr(state, 'frontiers', [])),
            # Note: Full geometry serialization would be complex and is omitted for now
            'city_bounds': str(getattr(state, 'city_bounds', None))
        }


def trace_decoder(obj: Dict[str, Any]):
    """Custom JSON decoder for GrowthTrace objects."""
    if '_type' in obj:
        if obj['_type'] == 'GrowthTrace':
            return GrowthTrace(
                actions=obj['actions'],
                initial_state=obj['initial_state'],
                final_state=obj['final_state'],
                metadata=obj['metadata']
            )
        elif obj['_type'] == 'InverseGrowthAction':
            return InverseGrowthAction(
                action_type=ActionType(obj['action_type']),
                target_id=obj['target_id'],
                intent_params=obj['intent_params'],
                realized_geometry=obj['realized_geometry'],
                confidence=obj['confidence'],
                timestamp=obj['timestamp'],
                action_metadata=obj['action_metadata']
            )
    return obj


def save_trace(trace: GrowthTrace, filepath: str, format: str = 'json') -> None:
    """
    Save a GrowthTrace to file.

    Args:
        trace: The trace to save
        filepath: Path to save to
        format: 'json' or 'pickle'
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(path, 'w') as f:
            json.dump(trace, f, cls=TraceEncoder, indent=2)
    elif format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(trace, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_trace(filepath: str, format: str = 'json') -> GrowthTrace:
    """
    Load a GrowthTrace from file.

    Args:
        filepath: Path to load from
        format: 'json' or 'pickle'

    Returns:
        The loaded GrowthTrace
    """
    path = Path(filepath)

    if format == 'json':
        with open(path, 'r') as f:
            return json.load(f, object_hook=trace_decoder)
    elif format == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def save_trace_summary(trace: GrowthTrace, filepath: str) -> None:
    """
    Save a human-readable summary of the trace.

    Args:
        trace: The trace to summarize
        filepath: Path to save summary to
    """
    summary = f"""Growth Trace Summary
==================

Total Actions: {trace.total_actions}
Average Confidence: {trace.average_confidence:.3f}
High Confidence Actions: {len(trace.high_confidence_actions)}

Action Breakdown:
"""

    action_counts = {}
    for action in trace.actions:
        action_counts[action.action_type.value] = action_counts.get(action.action_type.value, 0) + 1

    for action_type, count in action_counts.items():
        summary += f"- {action_type}: {count}\n"

    summary += f"\nMetadata: {trace.metadata}\n"

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(summary)