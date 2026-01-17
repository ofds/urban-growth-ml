#!/usr/bin/env python3
"""
Growth Engine Testing Script
Tests the current functionality of the urban growth simulation system.
"""

import sys
import os
sys.path.append('.')

from src.core.growth_engine import GrowthEngine
from src.core.contracts import GrowthState
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_growth_engine():
    """Test 1: Basic Growth Engine Functionality"""
    print("=" * 60)
    print("TEST 1: Basic Growth Engine Functionality")
    print("=" * 60)

    # Initialize engine
    engine = GrowthEngine('test_city', seed=42)

    # Test bud initialization
    print("1. Testing bud initialization...")
    state = engine.initialize_from_bud((0, 0))
    print(f"   ✓ Created bud with {len(state.streets)} streets, {len(state.blocks)} blocks")

    # Test frontier initialization
    print("2. Testing frontier initialization...")
    frontiers = engine.initialize_frontiers_for_bud(state)
    state = GrowthState(
        streets=state.streets,
        blocks=state.blocks,
        frontiers=frontiers,
        graph=state.graph,
        iteration=state.iteration,
        city_bounds=state.city_bounds
    )
    print(f"   ✓ Created {len(state.frontiers)} frontiers")

    # Test single growth step
    print("3. Testing single growth step...")
    initial_streets = len(state.streets)
    initial_frontiers = len(state.frontiers)

    new_state = engine.grow_one_step(state)

    print(f"   ✓ Grew from {initial_streets} to {len(new_state.streets)} streets")
    print(f"   ✓ Frontiers: {initial_frontiers} → {len(new_state.frontiers)}")

    return new_state

def test_multi_step_growth():
    """Test 2: Multi-Step Growth Evolution"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Step Growth Evolution")
    print("=" * 60)

    # Initialize
    engine = GrowthEngine('evolution_city', seed=123)
    state = engine.initialize_from_bud((-122.4, 37.8))  # San Francisco coordinates
    frontiers = engine.initialize_frontiers_for_bud(state)
    state = GrowthState(
        streets=state.streets,
        blocks=state.blocks,
        frontiers=frontiers,
        graph=state.graph,
        iteration=state.iteration,
        city_bounds=state.city_bounds
    )

    print(f"Initial state: {len(state.streets)} streets, {len(state.frontiers)} frontiers")

    # Run multiple growth steps
    states = [state]
    max_iterations = 10

    for i in range(max_iterations):
        if not state.frontiers:
            print(f"   ✓ Growth stopped at iteration {i} - no more frontiers")
            break

        state = engine.grow_one_step(state)
        states.append(state)

        streets_added = len(state.streets) - len(states[-2].streets)
        print(f"   Iteration {i+1}: {len(state.streets)} streets (+{streets_added}), {len(state.frontiers)} frontiers")

        # Stop if we have a reasonable city
        if len(state.streets) >= 25:
            print(f"   ✓ Reached target size at iteration {i+1}")
            break

    print(f"\nFinal city: {len(state.streets)} streets, {len(state.blocks)} blocks")
    print(f"Growth iterations: {len(states) - 1}")

    return states

def test_inverse_analysis():
    """Test 3: Inverse Analysis Capabilities"""
    print("\n" + "=" * 60)
    print("TEST 3: Inverse Analysis Test")
    print("=" * 60)

    try:
        from src.inverse import InverseGrowthAction, GrowthTrace, ActionType
        print("✓ Inverse data structures imported successfully")

        # Create a simple synthetic trace
        actions = [
            InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id="frontier_001",
                intent_params={'direction': 'radial', 'length': 50.0},
                confidence=0.9
            )
        ]

        trace = GrowthTrace(
            actions=actions,
            initial_state=None,  # Would be a GrowthState
            final_state=None,    # Would be a GrowthState
            metadata={'synthetic': True}
        )

        print(f"✓ Created synthetic trace with {len(trace.actions)} actions")
        print(f"✓ Average confidence: {trace.average_confidence}")
        print(f"✓ High confidence actions: {len(trace.high_confidence_actions)}")

        # Test serialization
        from src.inverse import save_trace, load_trace
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = os.path.join(temp_dir, 'test_trace.json')
            save_trace(trace, trace_path)
            loaded_trace = load_trace(trace_path)

            print(f"✓ Trace serialization works: saved and loaded {len(loaded_trace.actions)} actions")

        return True

    except Exception as e:
        print(f"❌ Inverse analysis test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("URBAN GROWTH ENGINE TESTING SUITE")
    print("Testing current functionality after spaghetti code extraction")
    print()

    # Test 1: Basic functionality
    try:
        final_state = test_basic_growth_engine()
        test1_passed = True
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        test1_passed = False

    # Test 2: Multi-step growth
    try:
        states = test_multi_step_growth()
        test2_passed = True
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        test2_passed = False

    # Test 3: Inverse analysis
    test3_passed = test_inverse_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Basic Growth Engine): {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Multi-Step Growth): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Test 3 (Inverse Analysis): {'✅ PASSED' if test3_passed else '❌ FAILED'}")

    total_passed = sum([test1_passed, test2_passed, test3_passed])
    print(f"\nOverall: {total_passed}/3 tests passed")

    if total_passed == 3:
        print("\n🎉 All systems operational! The growth engine is functional.")
        print("You have a working urban growth simulation system.")
    else:
        print(f"\n⚠️ {3-total_passed} tests failed. Some functionality needs attention.")

if __name__ == "__main__":
    main()
