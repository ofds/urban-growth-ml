#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.growth.new.growth_engine import GrowthEngine
from src.inverse.inference import BasicInferenceEngine
from src.inverse.replay import TraceReplayEngine

# Test inference with fixed skeleton
print("Testing Piedmont inference with fixed skeleton extraction...")

try:
    # Load Piedmont data
    engine = GrowthEngine('piedmont_ca', seed=42)
    city = engine.load_initial_state()

    print(f"Loaded Piedmont: {len(city.streets)} streets, {len(city.blocks)} blocks")

    # Run inference
    inference = BasicInferenceEngine()
    trace = inference.infer_trace(city, max_steps=1000)

    print("Inference Results:")
    print(f"- Actions inferred: {len(trace.actions)}")
    print(f"- Initial streets: {trace.initial_state.streets.shape[0]}")
    print(f"- Final streets: {trace.final_state.streets.shape[0]}")
    print(".1f")

    # Test replay
    replay_engine = TraceReplayEngine()
    validation = replay_engine.validate_trace_replay(
        trace=trace,
        original_state=city,
        city_name='piedmont_ca'
    )

    print("Replay Results:")
    print(f"- Actions replayed: {validation.get('replay_actions', 0)}/{validation.get('trace_actions', 0)}")
    print(f"- Streets reproduced: {validation.get('replayed_streets', 0)}/{validation.get('original_streets', 0)}")
    print(".2f")
    print(f"- Morphological valid: {validation.get('morphological_valid', False)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
