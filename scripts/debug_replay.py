#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from inverse.inference import BasicInferenceEngine
from inverse.replay import TraceReplayEngine
from core.growth.new.growth_engine import GrowthEngine
import logging

logging.basicConfig(level=logging.DEBUG)

# Load a small city state for testing
engine = GrowthEngine('piedmont_ca', seed=42)
city = engine.load_initial_state()

# Run inference with very limited steps
inference = BasicInferenceEngine()
trace = inference.infer_trace(city, max_steps=3)  # Just 3 steps

print(f'Inferred {len(trace.actions)} actions')

# Now test replay
replay_engine = TraceReplayEngine()
replay_actions = replay_engine._convert_trace_to_replayable_actions(trace)

print(f'Converted to {len(replay_actions)} replay actions')

# Try to replay just the first action
initial_state = trace.initial_state
print(f'Initial state has {len(initial_state.frontiers)} frontiers')

# Show some frontier stable_ids from initial state
print('Sample initial frontiers:')
for i, f in enumerate(initial_state.frontiers[:5]):
    stable_id = replay_engine._compute_stable_frontier_id(f)
    print(f'  Frontier {i}: {f.frontier_id} -> stable_id: {stable_id}')

# Try the multi-stage matching for the first action
action = replay_actions[0]
print(f'\nTrying to match action with stable_id: {action.get("stable_frontier_id")}')
print(f'Action target_id: {action.get("target_id")}')

# Check what node IDs are stored in the action
if action.get('intent') and 'edge_u' in action['intent'] and 'edge_v' in action['intent']:
    print(f'Action has node IDs: {action["intent"]["edge_u"]} -> {action["intent"]["edge_v"]}')

# Show what frontiers exist in initial state
print('Initial state frontiers:')
for i, f in enumerate(initial_state.frontiers):
    stable_id = replay_engine._compute_stable_frontier_id(f)
    print(f'  Frontier {i}: {f.frontier_id} -> stable_id: {stable_id}')
    if hasattr(f, 'edge_id'):
        print(f'    edge_id: {f.edge_id}')
    if hasattr(f, 'nodes'):
        node_ids = [str(n.id) if hasattr(n, 'id') else str(n) for n in f.nodes[:2]]
        print(f'    nodes: {node_ids}')

match = replay_engine._find_frontier_by_multi_stage_matching(action, initial_state.frontiers)
if match:
    print(f'✓ Found match: {match.frontier_id}')
else:
    print('✗ No match found')

    # Debug: check if stable_id exists in current frontiers
    target_stable_id = action.get('stable_frontier_id')
    print(f'Looking for stable_id: {target_stable_id}')
    found_ids = []
    for f in initial_state.frontiers:
        fid = replay_engine._compute_stable_frontier_id(f)
        found_ids.append(fid)
        if fid == target_stable_id:
            print(f'  Found matching frontier: {f.frontier_id}')
            break
    else:
        print(f'  Stable_id {target_stable_id} not found in current frontiers')
        print(f'  Available stable_ids: {found_ids[:10]}...')
