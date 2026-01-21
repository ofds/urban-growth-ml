#!/usr/bin/env python3
"""
Urban Growth Visualization CLI

Command-line tool for visualizing urban growth traces, replays, and validation metrics.
Supports:
- Growth sequence frames and GIF generation
- Replay comparison plots
- Validation summary dashboards
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.growth.new.growth_engine import GrowthEngine
from inverse.inference import BasicInferenceEngine
from inverse.replay import TraceReplayEngine
from inverse.visualization import InverseGrowthVisualizer
from inverse.serialization import save_trace, load_trace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_or_infer_trace(city_name: str, engine: GrowthEngine, force_inference: bool = False):
    """
    Load serialized trace if exists, otherwise infer new trace.

    Args:
        city_name: Name of city
        engine: Growth engine instance
        force_inference: Force new inference even if trace exists

    Returns:
        Tuple of (trace, city_state)
    """
    trace_path = Path(f"outputs/inverse/{city_name}_trace.json")

    # Try loading existing trace
    if not force_inference and trace_path.exists():
        logger.info(f"Loading existing trace from {trace_path}")
        try:
            trace = load_trace(str(trace_path))
            city = engine.load_initial_state()
            logger.info(f"Loaded trace with {len(trace.actions)} actions")
            return trace, city
        except Exception as e:
            logger.warning(f"Failed to load trace: {e}. Running new inference...")

    # Infer new trace
    logger.info(f"Inferring growth trace for {city_name}...")
    city = engine.load_initial_state()
    inference_engine = BasicInferenceEngine()
    trace = inference_engine.infer_trace(city)

    # Save for future use
    try:
        save_trace(trace, str(trace_path))
        logger.info(f"Saved trace to {trace_path}")
    except Exception as e:
        logger.warning(f"Failed to save trace: {e}")

    return trace, city


def visualize_replay_validation(city_name: str, output_dir: str = "outputs/inverse"):
    """
    Generate replay validation visualization for a city.

    Args:
        city_name: Name of city to visualize
        output_dir: Output directory for visualizations
    """
    logger.info(f"=== Replay Validation Visualization: {city_name} ===")

    # Load city and trace
    engine = GrowthEngine(city_name, seed=42)
    trace, original_city = load_or_infer_trace(city_name, engine)

    logger.info(f"Trace: {len(trace.actions)} actions, avg confidence: {trace.average_confidence:.3f}")

    # Run replay validation
    logger.info("Running replay validation...")
    replay_engine = TraceReplayEngine()
    validation = replay_engine.validate_trace_replay(
        trace=trace,
        original_state=original_city,
        city_name=city_name
    )

    # Create visualization
    visualizer = InverseGrowthVisualizer(output_dir=output_dir)

    # Get replayed city from validation
    replayed_city = replay_engine.replay_trace(trace, city_name)

    output_path = visualizer.plot_replay_validation(
        original_city=original_city,
        replayed_city=replayed_city,
        validation_metrics=validation,
        city_name=city_name
    )

    logger.info(f"✓ Created replay validation: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("REPLAY VALIDATION SUMMARY")
    print("="*60)
    print(f"Streets: {validation.get('replayed_streets', 0)}/{validation.get('original_streets', 0)}")
    print(f"Actions: {validation.get('replay_actions', 0)}/{validation.get('trace_actions', 0)}")
    print(f"Fidelity: {validation.get('replay_fidelity', 0):.3f}")
    print(f"Valid: {'✓ PASS' if validation.get('morphological_valid') else '✗ FAIL'}")
    print("="*60)


def visualize_trace_summary(city_name: str, output_dir: str = "outputs/inverse"):
    """
    Generate trace summary visualization for a city.

    Args:
        city_name: Name of city to visualize
        output_dir: Output directory for visualizations
    """
    logger.info(f"=== Trace Summary Visualization: {city_name} ===")

    # Load city and trace
    engine = GrowthEngine(city_name, seed=42)
    trace, city = load_or_infer_trace(city_name, engine)

    # Create visualization
    visualizer = InverseGrowthVisualizer(output_dir=output_dir)
    output_path = visualizer.plot_trace_summary(
        trace=trace,
        city=city,
        city_name=city_name
    )

    logger.info(f"✓ Created trace summary: {output_path}")


def visualize_growth_sequence(
    city_name: str,
    output_dir: str = "outputs/inverse",
    step_stride: int = 10,
    make_gif: bool = False,
    max_frames: int = 100
):
    """
    Generate growth sequence frames and optional GIF for a city.

    Args:
        city_name: Name of city to visualize
        output_dir: Output directory for visualizations
        step_stride: Save frame every N actions
        make_gif: Whether to create animated GIF
        max_frames: Maximum number of frames to generate
    """
    logger.info(f"=== Growth Sequence Visualization: {city_name} ===")

    # Load city and trace
    engine = GrowthEngine(city_name, seed=42)
    trace, final_city = load_or_infer_trace(city_name, engine)

    # Get initial city state (skeleton)
    initial_city = engine.initialize_from_osm()

    # Create visualization
    visualizer = InverseGrowthVisualizer(output_dir=output_dir)

    logger.info(f"Generating frames (stride={step_stride}, max={max_frames})...")
    frame_paths, gif_path = visualizer.plot_growth_sequence(
        trace=trace,
        initial_city=initial_city,
        step_stride=step_stride,
        make_gif=make_gif,
        max_frames=max_frames,
        city_name=city_name
    )

    logger.info(f"✓ Created {len(frame_paths)} frames")
    if gif_path:
        logger.info(f"✓ Created GIF: {gif_path}")

    print("\n" + "="*60)
    print("GROWTH SEQUENCE SUMMARY")
    print("="*60)
    print(f"Frames generated: {len(frame_paths)}")
    print(f"GIF created: {'Yes' if gif_path else 'No'}")
    print(f"Output directory: {Path(frame_paths[0]).parent if frame_paths else 'N/A'}")
    print("="*60)


def visualize_all(city_name: str, output_dir: str = "outputs/inverse", **kwargs):
    """
    Generate all visualization types for a city.

    Args:
        city_name: Name of city to visualize
        output_dir: Output directory for visualizations
        **kwargs: Additional arguments for growth sequence
    """
    logger.info(f"=== Generating All Visualizations: {city_name} ===\n")

    try:
        visualize_replay_validation(city_name, output_dir)
    except Exception as e:
        logger.error(f"Failed to create replay validation: {e}")

    try:
        visualize_trace_summary(city_name, output_dir)
    except Exception as e:
        logger.error(f"Failed to create trace summary: {e}")

    try:
        visualize_growth_sequence(city_name, output_dir, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create growth sequence: {e}")

    logger.info("\n✓ All visualizations complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize urban growth traces and validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations for Piedmont, CA
  python visualize_growth.py piedmont_ca --all

  # Generate only replay validation
  python visualize_growth.py piedmont_ca --replay-validation

  # Generate growth sequence with GIF
  python visualize_growth.py piedmont_ca --growth-sequence --gif --stride 5

  # Generate trace summary
  python visualize_growth.py piedmont_ca --trace-summary
        """
    )

    parser.add_argument('city_name', help='Name of city to visualize (e.g., piedmont_ca)')
    parser.add_argument('-o', '--output-dir', default='outputs/inverse',
                       help='Output directory for visualizations (default: outputs/inverse)')

    # Visualization types
    viz_group = parser.add_argument_group('Visualization types')
    viz_group.add_argument('--all', action='store_true',
                          help='Generate all visualization types')
    viz_group.add_argument('--replay-validation', action='store_true',
                          help='Generate replay validation comparison')
    viz_group.add_argument('--trace-summary', action='store_true',
                          help='Generate trace summary dashboard')
    viz_group.add_argument('--growth-sequence', action='store_true',
                          help='Generate growth sequence frames')

    # Growth sequence options
    seq_group = parser.add_argument_group('Growth sequence options')
    seq_group.add_argument('--stride', type=int, default=10,
                          help='Save frame every N actions (default: 10)')
    seq_group.add_argument('--gif', action='store_true',
                          help='Create animated GIF from frames')
    seq_group.add_argument('--max-frames', type=int, default=100,
                          help='Maximum number of frames (default: 100)')

    # Other options
    parser.add_argument('--force-inference', action='store_true',
                       help='Force new trace inference even if cached')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine which visualizations to generate
    if args.all:
        visualize_all(
            args.city_name,
            args.output_dir,
            step_stride=args.stride,
            make_gif=args.gif,
            max_frames=args.max_frames
        )
    else:
        # Generate requested visualizations
        if not any([args.replay_validation, args.trace_summary, args.growth_sequence]):
            # Default: generate replay validation if no specific type requested
            logger.info("No specific visualization type specified, generating replay validation")
            args.replay_validation = True

        if args.replay_validation:
            visualize_replay_validation(args.city_name, args.output_dir)

        if args.trace_summary:
            visualize_trace_summary(args.city_name, args.output_dir)

        if args.growth_sequence:
            visualize_growth_sequence(
                args.city_name,
                args.output_dir,
                step_stride=args.stride,
                make_gif=args.gif,
                max_frames=args.max_frames
            )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
