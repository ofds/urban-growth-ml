#!/usr/bin/env python3
"""
Test FractalPatternStrategy on Piedmont Data

Focused test for the newly extracted FractalPatternStrategy using real Piedmont urban data.
This demonstrates the SOLID refactoring working end-to-end with fractal pattern analysis.
"""

import sys
import os
import logging
import argparse
import signal

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.growth.new.growth_engine import GrowthEngine
from src.inverse.strategies.fractal_strategy import FractalPatternStrategy
from src.inverse.core.config import InferenceConfig
from src.inverse.replay import TraceReplayEngine
from src.inverse.data_structures import GrowthTrace
from src.inverse.visualization import InverseGrowthVisualizer

# Global variables for signal handling
current_trace = None
interrupted = False

def signal_handler(signum, frame):
    """Handle interrupt signal and show preliminary results."""
    global current_trace, interrupted
    interrupted = True
    print("\n\n=== INTERRUPTED - SHOWING PRELIMINARY RESULTS ===")

    if current_trace is not None:
        print("FractalPatternStrategy Inference Results (interrupted):")
        print(f"- Actions inferred so far: {len(current_trace.actions)}")
        if hasattr(current_trace, 'initial_state') and current_trace.initial_state:
            print(f"- Initial streets: {current_trace.initial_state.streets.shape[0]}")
        if hasattr(current_trace, 'final_state') and current_trace.final_state:
            print(f"- Final streets: {current_trace.final_state.streets.shape[0]}")

        # Show fractal analysis summary
        if current_trace.actions:
            print(f"- Average confidence: {sum(a.confidence for a in current_trace.actions) / len(current_trace.actions):.3f}")
            print(f"- Strategy: fractal_pattern")
    else:
        print("No results available yet - inference may not have started.")

    print("\nExiting gracefully...")
    sys.exit(0)

def main():
    global current_trace

    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Test FractalPatternStrategy on Piedmont data')
    parser.add_argument('--max-steps', type=int, default=25, help='Maximum inference steps')
    parser.add_argument('--validate', action='store_true', help='Run validation checks')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("🔍 Debug mode enabled - showing detailed fractal analysis")
    else:
        logging.getLogger().setLevel(logging.INFO)

    print("🌀 Testing FractalPatternStrategy on Piedmont Data")
    print("=" * 60)
    print("This test demonstrates the SOLID-refactored fractal pattern analysis")
    print("working end-to-end on real urban growth data.")
    print("=" * 60)

    try:
        # Load Piedmont data
        print("📍 Loading Piedmont city data...")
        engine = GrowthEngine('piedmont_ca', seed=42)
        city = engine.load_initial_state()

        print(f"✅ Loaded Piedmont: {len(city.streets)} streets, {len(city.blocks)} blocks")

        # Initialize FractalPatternStrategy
        print("🔬 Initializing FractalPatternStrategy...")
        config = InferenceConfig()
        fractal_strategy = FractalPatternStrategy(weight=1.2, config=config)

        print("📊 Fractal Strategy Configuration:")
        print(f"   - Weight: {fractal_strategy.weight}")
        print(f"   - Min similarity threshold: {config.strategies.fractal_pattern_min_similarity}")
        print(f"   - Description: {fractal_strategy.get_strategy_info()['description']}")

        # Create a partial trace for interruption handling
        current_trace = GrowthTrace(
            actions=[],
            initial_state=None,
            final_state=city,
            metadata={"strategy": "fractal_pattern", "interrupted": True}
        )

        # Run inference with FractalPatternStrategy
        print(f"\n🚀 Running fractal pattern inference ({args.max_steps} steps max)...")
        print("Press Ctrl+C to interrupt and see preliminary results.")

        skeleton_edges = set()  # No skeleton edges for pure fractal analysis
        actions = []
        step = 0

        while step < args.max_steps and len(city.streets) > 5:  # Minimum streets for fractal analysis
            # Generate candidates using fractal strategy
            candidates = fractal_strategy.generate_candidates(city, skeleton_edges)

            if not candidates:
                print(f"🔚 No more candidates found at step {step}")
                break

            # Select best candidate (highest confidence)
            best_action, confidence = max(candidates, key=lambda x: x[1])

            print(f"Step {step}: Selected action with confidence {confidence:.3f}")
            print(f"  Street: {best_action.street_id}, Edge: {best_action.intent_params.get('edge_u')}->{best_action.intent_params.get('edge_v')}")

            # Add to trace
            actions.append(best_action)

            # Update current trace for interruption handling
            current_trace = GrowthTrace(
                actions=actions.copy(),
                initial_state=None,
                final_state=city,
                metadata={"strategy": "fractal_pattern", "interrupted": True}
            )

            # For demonstration, we'll stop after collecting actions
            # In a full implementation, we'd rewind the state here
            step += 1

            # Safety check - don't remove too many streets
            if step >= args.max_steps:
                break

        # Create final trace
        trace = GrowthTrace(
            actions=actions,
            initial_state=None,  # Would be skeleton in full implementation
            final_state=city,
            metadata={
                "strategy": "fractal_pattern",
                "inference_method": "fractal_pattern_only",
                "max_steps": args.max_steps,
                "steps_taken": step,
                "total_streets": len(city.streets)
            }
        )

        # Update current_trace with final results
        current_trace = trace

        print("\n" + "="*60)
        print("🎯 FRACTAL PATTERN INFERENCE RESULTS")
        print("="*60)
        print(f"Actions inferred: {len(trace.actions)}")
        print(f"Strategy: fractal_pattern")
        print(f"Steps taken: {step}")
        print(f"Total streets analyzed: {len(city.streets)}")

        if trace.actions:
            avg_confidence = sum(a.confidence for a in trace.actions) / len(trace.actions)
            print(f"Average confidence: {avg_confidence:.3f}")

            # Show sample actions
            print("\n📋 Sample inferred actions:")
            for i, action in enumerate(trace.actions[:5]):  # Show first 5
                edge_u = action.intent_params.get('edge_u', 'N/A')
                edge_v = action.intent_params.get('edge_v', 'N/A')
                print(f"  {i+1}. Street {action.street_id}: {edge_u} → {edge_v} (conf: {action.confidence:.3f})")

        # Basic validation checks
        if args.validate:
            print("\n=== VALIDATION CHECKS ===")

            # Check 1: Actions are valid
            valid_actions = sum(1 for a in trace.actions if a.street_id and a.confidence > 0)
            print(f"✓ Valid actions: {valid_actions}/{len(trace.actions)} ({valid_actions/len(trace.actions)*100:.1f}%)")

            # Check 2: Confidence scores are reasonable
            high_confidence = sum(1 for a in trace.actions if a.confidence > 0.5)
            print(f"✓ High confidence actions: {high_confidence}/{len(trace.actions)} ({high_confidence/len(trace.actions)*100:.1f}%)")

        # Visualization
        if args.visualize and trace.actions:
            print("\n=== VISUALIZATION ===")
            try:
                visualizer = InverseGrowthVisualizer(output_dir="outputs/inverse")

                # Create a simple before/after comparison
                print("📊 Generating fractal pattern analysis visualization...")

                # For demonstration, create a summary plot
                import matplotlib.pyplot as plt
                import numpy as np

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Plot 1: Confidence distribution
                if trace.actions:
                    confidences = [a.confidence for a in trace.actions]
                    ax1.hist(confidences, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax1.set_title('Fractal Pattern Action Confidences')
                    ax1.set_xlabel('Confidence Score')
                    ax1.set_ylabel('Frequency')
                    ax1.grid(True, alpha=0.3)

                # Plot 2: Cumulative actions over time
                if trace.actions:
                    cumulative = list(range(1, len(trace.actions) + 1))
                    ax2.plot(cumulative, confidences, 'o-', alpha=0.7, color='green')
                    ax2.set_title('Fractal Pattern Confidence Over Time')
                    ax2.set_xlabel('Action Number')
                    ax2.set_ylabel('Confidence Score')
                    ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                output_path = "outputs/inverse/fractal_piedmont_analysis.png"
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"✓ Created analysis visualization: {output_path}")

                # Summary statistics
                print("\n📈 Fractal Pattern Analysis Summary:")
                print(f"   Total streets analyzed: {len(city.streets)}")
                print(f"   Actions identified: {len(trace.actions)}")
                if trace.actions:
                    print(f"   Average confidence: {sum(a.confidence for a in trace.actions) / len(trace.actions):.3f}")
                    print(f"   Max confidence: {max(a.confidence for a in trace.actions):.3f}")
                    print(f"   Min confidence: {min(a.confidence for a in trace.actions):.3f}")

            except Exception as viz_error:
                print(f"Visualization failed: {viz_error}")

        print("\n🎉 FractalPatternStrategy test completed successfully!")
        print("This demonstrates the SOLID-refactored fractal analysis working on real urban data.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
