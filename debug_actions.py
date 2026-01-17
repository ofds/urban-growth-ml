#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inverse.inference import BasicInferenceEngine
from core.growth.new.growth_engine import GrowthEngine

# Load a small city state for testing
engine = GrowthEngine('piedmont_ca', seed=42)
city = engine.load_initial_state()

# Run inference with very limited steps
inference = BasicInferenceEngine()
trace = inference.infer_trace(city, max_steps=5)  # Just 5 steps

print(f'Inferred {len(trace.actions)} actions')
for i, action in enumerate(trace.actions[:3]):
    print(f'Action {i}: type={action.action_type}, target_id={action.target_id}')
    if hasattr(action, 'intent_params') and action.intent_params:
        print(f'  intent stable_id: {action.intent_params.get("stable_id")}')
    if hasattr(action, 'realized_geometry') and action.realized_geometry:
        print(f'  realized stable_id: {action.realized_geometry.get("stable_id")}')

# Now test replay conversion
from inverse.replay import TraceReplayEngine
replay_engine = TraceReplayEngine()
replay_actions = replay_engine._convert_trace_to_replayable_actions(trace)

print(f'\nConverted to {len(replay_actions)} replay actions')
for i, action in enumerate(replay_actions[:3]):
    print(f'Replay Action {i}:')
    print(f'  target_id: {action.get("target_id")}')
    print(f'  stable_frontier_id: {action.get("stable_frontier_id")}')
    print(f'  has intent: {action.get("intent") is not None}')
    if action.get("intent"):
        print(f'  intent stable_id: {action["intent"].get("stable_id")}')
    print(f'  has geometry: {action.get("geometry_for_matching") is not None}')
