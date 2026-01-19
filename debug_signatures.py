#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inverse.inference import BasicInferenceEngine
from inverse.replay import TraceReplayEngine
from core.growth.new.growth_engine import GrowthEngine
from inverse.data_structures import compute_frontier_signature

# Load a small city state for testing
engine = GrowthEngine('piedmont_ca', seed=42)
city = engine.load_initial_state()

# Run inference with very limited steps
inference = BasicInferenceEngine()
trace = inference.infer_trace(city, max_steps=3)  # Just 3 steps

print(f'Inferred {len(trace.actions)} actions')
for i, action in enumerate(trace.actions[:3]):
    print(f'Action {i}: type={action.action_type}, street_id={action.street_id}')
    print(f'  geometric_signature: {action.geometric_signature}')
    print(f'  state_diff: {action.state_diff is not None}')
    if action.state_diff:
        print(f'    added_streets: {len(action.state_diff.get("added_streets", []))}')
        print(f'    removed_streets: {len(action.state_diff.get("removed_streets", []))}')
    if hasattr(action, 'intent_params') and action.intent_params:
        print(f'  intent stable_id: {action.intent_params.get("stable_id")}')
    if hasattr(action, 'realized_geometry') and action.realized_geometry:
        print(f'  realized stable_id: {action.realized_geometry.get("stable_id")}')

# Now test replay
replay_engine = TraceReplayEngine()
replay_actions = replay_engine._convert_trace_to_replayable_actions(trace)

print(f'\nConverted to {len(replay_actions)} replay actions')
for i, action in enumerate(replay_actions[:3]):
    print(f'Replay Action {i}:')
    print(f'  target_id: {action.get("target_id")}')
    print(f'  stable_frontier_id: {action.get("stable_frontier_id")}')
    print(f'  geometric_signature: {action.get("geometric_signature")}')
    print(f'  has intent: {action.get("intent") is not None}')

# Test signature computation on initial frontiers
initial_state = trace.initial_state
print(f'\nInitial state has {len(initial_state.frontiers)} frontiers')
print('Computing signatures for initial frontiers:')
for i, f in enumerate(initial_state.frontiers):
    signature = compute_frontier_signature(f)
    print(f'  Frontier {i}: {f.frontier_id} -> signature: {signature}')
