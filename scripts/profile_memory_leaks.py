#!/usr/bin/env python3
"""
Memory profiling script to identify memory leaks in inference strategies.
"""

import sys
import os
import gc
import tracemalloc
import psutil
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.growth.new.growth_engine import GrowthEngine
from src.inverse.inference import MultiStrategyInferenceEngine

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def profile_strategy_memory(strategy_name: str, max_steps: int = 5):
    """
    Profile memory usage for a specific strategy.

    Args:
        strategy_name: Name of strategy to test ('fractal_pattern', 'angle_harmonization', 'block_centroid')
        max_steps: Maximum inference steps to run
    """
    print(f"\n{'='*60}")
    print(f"PROFILING MEMORY USAGE: {strategy_name}")
    print(f"{'='*60}")

    # Force garbage collection before starting
    gc.collect()
    tracemalloc.start()

    # Get baseline memory
    baseline_memory = get_memory_usage()
    baseline_snapshot = tracemalloc.take_snapshot()

    print(f"Baseline memory: {baseline_memory:.1f} MB")
    try:
        # Load data
        engine = GrowthEngine('piedmont_ca', seed=42)
        city = engine.load_initial_state()

        print(f"Loaded Piedmont: {len(city.streets)} streets, {len(city.blocks)} blocks")

        # Configure strategy
        strategy_config = {
            'fractal_pattern': strategy_name == 'fractal_pattern',
            'angle_harmonization': strategy_name == 'angle_harmonization',
            'block_centroid': strategy_name == 'block_centroid',
            'ml_augmented': False,
            'multi_resolution': False,
            'advanced_search': False
        }

        # Create inference engine
        inference = MultiStrategyInferenceEngine(strategy_config=strategy_config)

        # Track memory during inference
        memory_readings = []
        snapshots = []

        def memory_callback(step_count):
            """Callback to record memory usage during inference."""
            current_memory = get_memory_usage()
            memory_readings.append((step_count, current_memory))
            snapshots.append(tracemalloc.take_snapshot())
            print(f"Step {step_count}: Memory = {current_memory:.1f} MB")

        # Run inference with memory tracking
        print(f"Running inference with {strategy_name} strategy...")

        # We'll manually track memory since the inference engine doesn't have built-in callbacks
        initial_memory = get_memory_usage()
        memory_readings.append((0, initial_memory))
        snapshots.append(tracemalloc.take_snapshot())

        trace = inference.infer_trace(city, max_steps=max_steps)

        final_memory = get_memory_usage()
        memory_readings.append((max_steps, final_memory))
        snapshots.append(tracemalloc.take_snapshot())

        # Analyze results
        print("\nMEMORY ANALYSIS:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
        print(f"Actions inferred: {len(trace.actions)}")

        # Check for memory leaks
        if final_memory - initial_memory > 50:  # More than 50MB increase
            print(f"⚠️  POTENTIAL MEMORY LEAK: {final_memory - initial_memory:.1f} MB increase")

            # Analyze top memory consumers
            if len(snapshots) >= 2:
                stats = snapshots[-1].compare_to(snapshots[0], 'lineno')
                print("\nTop memory consumers:")
                for stat in stats[:10]:
                    print(f"  {stat.traceback.format()[0]}: +{stat.size_diff} bytes")
        else:
            print("✅ Memory usage appears normal")

        # Force cleanup
        del trace, inference, city, engine
        gc.collect()

        final_cleanup_memory = get_memory_usage()
        print(f"After cleanup: {final_cleanup_memory:.1f} MB")
        print(f"Cleanup recovered: {final_memory - final_cleanup_memory:.1f} MB")

        return {
            'strategy': strategy_name,
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_increase': final_memory - initial_memory,
            'cleanup_memory': final_cleanup_memory,
            'actions': len(trace.actions),
            'memory_readings': memory_readings
        }

    except Exception as e:
        print(f"Error profiling {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        tracemalloc.stop()

def main():
    """Profile memory usage for all strategies."""
    print("MEMORY LEAK PROFILING FOR INFERENCE STRATEGIES")
    print("=" * 60)

    strategies = ['fractal_pattern', 'angle_harmonization', 'block_centroid']
    results = []

    for strategy in strategies:
        result = profile_strategy_memory(strategy, max_steps=5)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("MEMORY PROFILING SUMMARY")
    print(f"{'='*60}")

    for result in results:
        status = "⚠️  LEAK" if result['memory_increase'] > 50 else "✅ OK"
        print(f"{result['strategy']}: {status} (+{result['memory_increase']:.1f} MB, {result['actions']} actions)")

if __name__ == "__main__":
    main()
