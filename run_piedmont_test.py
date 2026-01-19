#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import signal

# Enable debug logging (will be overridden by --debug flag)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.growth.new.growth_engine import GrowthEngine
from src.inverse.inference import MultiStrategyInferenceEngine
from src.inverse.replay import TraceReplayEngine

# Global variables for signal handling
current_trace = None
interrupted = False

def signal_handler(signum, frame):
    """Handle interrupt signal and show preliminary results."""
    global current_trace, interrupted
    interrupted = True
    print("\n\n=== INTERRUPTED - SHOWING PRELIMINARY RESULTS ===")

    if current_trace is not None:
        print("Inference Results (interrupted):")
        print(f"- Actions inferred so far: {len(current_trace.actions)}")
        if hasattr(current_trace, 'initial_state') and current_trace.initial_state:
            print(f"- Initial streets: {current_trace.initial_state.streets.shape[0]}")
        if hasattr(current_trace, 'final_state') and current_trace.final_state:
            print(f"- Final streets: {current_trace.final_state.streets.shape[0]}")

        # Show performance stats if available
        if hasattr(current_trace, 'metadata') and current_trace.metadata:
            perf_stats = current_trace.metadata.get('performance_stats')
            if perf_stats:
                print(f"- Total time: {perf_stats.get('total_time', 'N/A'):.2f}s")
                print(f"- Total steps: {perf_stats.get('total_steps', 'N/A')}")
                print(f"- Streets processed: {perf_stats.get('total_streets_processed', 'N/A')}")
                print(f"- Avg step time: {perf_stats.get('avg_step_time', 'N/A'):.2f}s")
    else:
        print("No results available yet - inference may not have started.")

    print("\nExiting gracefully...")
    sys.exit(0)

def parse_strategy_config(strategy_str: str) -> dict:
    """Parse strategy configuration string into dict."""
    if strategy_str.lower() == 'all':
        return {
            'fractal_pattern': True,
            'angle_harmonization': True,
            'block_centroid': True,
            'ml_augmented': True,
            'multi_resolution': True,
            'advanced_search': True
        }
    elif strategy_str.lower() == 'phase1':
        return {
            'fractal_pattern': True,
            'angle_harmonization': True,
            'block_centroid': True,
            'ml_augmented': False,
            'multi_resolution': False,
            'advanced_search': False
        }
    else:
        # Parse comma-separated list
        strategies = [s.strip() for s in strategy_str.split(',')]
        config = {
            'fractal_pattern': False,
            'angle_harmonization': False,
            'block_centroid': False,
            'ml_augmented': False,
            'multi_resolution': False,
            'advanced_search': False
        }
        for strategy in strategies:
            if strategy in config:
                config[strategy] = True
        return config

def main():
    global current_trace

    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Test urban growth inference with configurable strategies')
    parser.add_argument('--max-steps', type=int, default=10, help='Maximum inference steps')
    parser.add_argument('--validate', action='store_true', help='Run validation checks')
    parser.add_argument('--strategies', type=str, default='phase1',
                       help='Comma-separated list of strategies to enable, or "all", or "phase1" (default: phase1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for detailed output')
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Debug mode enabled - showing detailed logging")
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Parse strategy configuration
    strategy_config = parse_strategy_config(args.strategies)
    enabled_strategies = [k for k, v in strategy_config.items() if v]
    print(f"Testing with strategies: {enabled_strategies}")

    # Test inference with fixed skeleton
    print(f"Testing Piedmont inference with {args.max_steps} steps...")
    print("Press Ctrl+C to interrupt and see preliminary results.")

    try:
        # Load Piedmont data
        engine = GrowthEngine('piedmont_ca', seed=42)
        city = engine.load_initial_state()

        print(f"Loaded Piedmont: {len(city.streets)} streets, {len(city.blocks)} blocks")

        # Run inference with limited steps
        inference = MultiStrategyInferenceEngine(strategy_config=strategy_config)

        # Create a partial trace for interruption handling
        from src.inverse.data_structures import GrowthTrace
        current_trace = GrowthTrace(
            actions=[],
            initial_state=None,
            final_state=city,
            metadata={"interrupted": True}
        )

        # Progress callback to update current_trace during inference
        def progress_callback(trace):
            global current_trace
            current_trace = trace

        trace = inference.infer_trace(city, max_steps=args.max_steps, progress_callback=progress_callback)

        # Update current_trace with final results
        current_trace = trace

        print("Inference Results:")
        print(f"- Actions inferred: {len(trace.actions)}")
        print(f"- Initial streets: {trace.initial_state.streets.shape[0]}")
        print(f"- Final streets: {trace.final_state.streets.shape[0]}")
        print(".1f")

        # Basic validation checks
        if args.validate:
            print("\n=== VALIDATION CHECKS ===")

            # Check 1: Monotonic street decrease
            initial_streets = trace.initial_state.streets.shape[0]
            final_streets = trace.final_state.streets.shape[0]
            street_decrease = initial_streets <= final_streets
            print(f"✓ Monotonic street decrease: {street_decrease} ({initial_streets} -> {final_streets})")

            # Check 2: Actions reference existing streets
            validation_failures = 0
            for action in trace.actions:
                if action.street_id not in trace.final_state.streets.index:
                    validation_failures += 1
            print(f"✓ No validation failures: {validation_failures == 0} ({validation_failures} failures)")

            # Check 3: Graph connectivity (basic check)
            try:
                import networkx as nx
                graph_connected = nx.is_connected(trace.final_state.graph.to_undirected())
                print(f"✓ Graph connectivity: {graph_connected}")
            except:
                print("? Graph connectivity: Unable to check")

        # Test replay if requested
        if args.validate:
            print("\n=== REPLAY TEST ===")
            try:
                replay_engine = TraceReplayEngine()
                validation = replay_engine.validate_trace_replay(
                    trace=trace,
                    original_state=city,
                    city_name='piedmont_ca'
                )

                actions_replayed = validation.get('replay_actions', 0)
                actions_attempted = validation.get('trace_actions', 0)
                streets_replayed = validation.get('replayed_streets', 0)
                streets_original = validation.get('original_streets', 0)
                replay_fidelity = validation.get('replay_fidelity', 0)

                replay_success_rate = (actions_replayed / actions_attempted * 100) if actions_attempted > 0 else 0
                street_reproduction_rate = (streets_replayed / streets_original * 100) if streets_original > 0 else 0

                print(f"  📊 Replay Diagnostics:")
                print(f"     Action replay rate: {actions_replayed}/{actions_attempted} ({replay_success_rate:.1f}%)")
                print(f"     Street reproduction: {streets_replayed}/{streets_original} ({street_reproduction_rate:.1f}%)")
                print(f"     Morphological fidelity: {replay_fidelity:.2f}")
            except Exception as replay_error:
                print(f"Replay failed: {replay_error}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
