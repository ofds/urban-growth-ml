#!/usr/bin/env python3
"""Test complete pipeline on real Piedmont data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import logging
from core.growth.new.growth_engine import GrowthEngine
from inverse.inference import BasicInferenceEngine
from inverse.replay import TraceReplayEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_piedmont_pipeline():
    """Test complete inverse inference + replay on Piedmont."""
    
    print("="*60)
    print("PIEDMONT FULL PIPELINE TEST")
    print("="*60)
    
    try:
        # Step 1: Load processed city
        print("\n[1/4] Loading Piedmont city state...")
        engine = GrowthEngine('piedmont_ca', seed=42)
        city = engine.load_initial_state()
        
        print(f"  ✅ Loaded city:")
        print(f"     Streets: {len(city.streets)}")
        print(f"     Blocks: {len(city.blocks)}")
        print(f"     Frontiers: {len(city.frontiers)}")
        print(f"     Graph nodes: {city.graph.number_of_nodes()}")
        
        # Check frontier types
        frontier_types = {}
        for f in city.frontiers:
            frontier_types[f.frontier_type] = frontier_types.get(f.frontier_type, 0) + 1
        print(f"     Frontier types: {frontier_types}")
        
    except Exception as e:
        print(f"  ❌ Failed to load city: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Step 2: Run inference (limit to 20 steps for testing)
        print("\n[2/4] Running inverse inference...")
        print("  (Limited to 20 steps for speed)")
        
        inference = BasicInferenceEngine()
        trace = inference.infer_trace(city, max_steps=20)
        
        print(f"  ✅ Inference complete:")
        print(f"     Actions inferred: {len(trace.actions)}")
        print(f"     Average confidence: {trace.average_confidence:.2f}")
        
        if len(trace.actions) > 0:
            print(f"     First 3 actions:")
            for i, action in enumerate(trace.actions[:3]):
                print(f"       {i+1}. {action.action_type.value} (conf={action.confidence:.2f})")
        else:
            print("  ⚠️  No actions inferred")
            return False
            
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Step 3: Replay with growth engine
        print("\n[3/4] Replaying with growth engine...")
        
        replay_engine = TraceReplayEngine()
        validation = replay_engine.validate_trace_replay(
            trace=trace,
            original_state=city,
            city_name='piedmont_ca'
        )
        
        print(f"  ✅ Replay complete:")
        print(f"     Success: {validation.get('success', False)}")
        print(f"     Replay fidelity: {validation.get('replay_fidelity', 0):.2f}")
        print(f"     Morphological valid: {validation.get('morphological_valid', False)}")
        print(f"     Geometric valid: {validation.get('geometric_valid', False)}")
        
    except Exception as e:
        print(f"  ❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Final assessment
    print("\n[4/4] Final Assessment")
    print("-"*60)
    
    fidelity = validation.get('replay_fidelity', 0)
    success = validation.get('success', False)
    
    if success and fidelity > 0.7:
        print("🎉 EXCELLENT: Full pipeline working at high quality!")
        print(f"   Replay fidelity: {fidelity:.2%}")
        print("\n✅ READY FOR ML TRAINING")
        return True
    elif success and fidelity > 0.5:
        print("✅ GOOD: Pipeline working, acceptable quality")
        print(f"   Replay fidelity: {fidelity:.2%}")
        print("\n⚠️  Can proceed to ML, but tune inference first")
        return True
    elif success:
        print("⚠️  LOW QUALITY: Pipeline runs but fidelity is low")
        print(f"   Replay fidelity: {fidelity:.2%}")
        print("\n❌ Needs improvement before ML training")
        return False
    else:
        print("❌ FAILURE: Pipeline has critical errors")
        print(f"   Error: {validation.get('error', 'Unknown')}")
        return False


def main():
    success = test_piedmont_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("NEXT STEPS:")
        print("1. Generate datasets from multiple cities")
        print("2. Train ML model on inferred actions")
        print("3. Generate new cities with trained model")
    else:
        print("FIX ISSUES:")
        print("1. Check inference heuristics")
        print("2. Verify growth engine validators")
        print("3. Tune skeleton extraction")
    print("="*60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
