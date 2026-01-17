#!/usr/bin/env python3
"""
Visualization for Inverse Growth Results
Phase A: Visual comparison between original and replayed cities.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import geopandas as gpd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class InverseGrowthVisualizer:
    """
    Visualizes inverse growth results and replay validation.
    """

    def __init__(self, output_dir: str = "outputs/inverse"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def create_replay_comparison(self,
                               original_state,
                               replayed_state,
                               validation_results: Dict[str, Any],
                               trace_metadata: Dict[str, Any],
                               filename: str = "replay_comparison.png") -> str:
        """
        Create visual comparison between original and replayed city states.

        Args:
            original_state: Original final state
            replayed_state: State from trace replay
            validation_results: Morphological validation results
            trace_metadata: Trace inference metadata
            filename: Output filename

        Returns:
            Path to created visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Inverse Growth: Original vs Replayed City", fontsize=16)

        # Plot 1: Original city
        self._plot_city_state(axes[0], original_state, "Original City")
        axes[0].set_title("Original City")

        # Plot 2: Replayed city
        self._plot_city_state(axes[1], replayed_state, "Replayed City")
        axes[1].set_title("Replayed City")

        # Plot 3: Difference analysis
        self._plot_difference_analysis(axes[2], original_state, replayed_state, validation_results)
        axes[2].set_title("Difference Analysis")

        # Add metadata annotation
        avg_conf = trace_metadata.get('avg_confidence', 'N/A')
        morph_score = validation_results.get('overall_score', 'N/A')
        metadata_text = f"""Inference Results:
Actions: {trace_metadata.get('steps_taken', 'N/A')}
Avg Confidence: {avg_conf if isinstance(avg_conf, str) else f'{avg_conf:.2f}'}
Morphological Score: {morph_score if isinstance(morph_score, str) else f'{morph_score:.2f}'}

Validation:
Geometric: {'✓' if validation_results.get('geometric_valid') else '✗'}
Topological: {'✓' if validation_results.get('topological_valid') else '✗'}
Morphological: {'✓' if validation_results.get('morphological_valid') else '✗'}
"""

        fig.text(0.02, 0.02, metadata_text, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        plt.tight_layout()

        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Created replay comparison visualization: {filepath}")
        return filepath

    def create_trace_summary_visualization(self,
                                         trace,
                                         filename: str = "trace_summary.png") -> str:
        """
        Create visualization summarizing the inferred growth trace.

        Args:
            trace: GrowthTrace object
            filename: Output filename

        Returns:
            Path to created visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Growth Trace Summary", fontsize=16)

        # Plot 1: Action sequence timeline
        self._plot_action_timeline(axes[0, 0], trace)

        # Plot 2: Confidence distribution
        self._plot_confidence_distribution(axes[0, 1], trace)

        # Plot 3: Action type breakdown
        self._plot_action_breakdown(axes[1, 0], trace)

        # Plot 4: State evolution (simplified)
        self._plot_state_evolution(axes[1, 1], trace)

        plt.tight_layout()

        filepath = f"{self.output_dir}/{filename}"
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Created trace summary visualization: {filepath}")
        return filepath

    def _plot_city_state(self, ax, state, title: str):
        """Plot a single city state."""
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title)

        # Plot streets
        if hasattr(state, 'streets') and not state.streets.empty:
            state.streets.plot(ax=ax, color='blue', linewidth=1, alpha=0.7, label='Streets')

        # Plot blocks
        if hasattr(state, 'blocks') and not state.blocks.empty:
            state.blocks.plot(ax=ax, color='lightgray', alpha=0.5, label='Blocks')

        # Plot city bounds
        if hasattr(state, 'city_bounds') and state.city_bounds:
            x, y = state.city_bounds.exterior.xy
            ax.plot(x, y, color='red', linewidth=2, linestyle='--', label='City Bounds')

        ax.legend()

    def _plot_difference_analysis(self, ax, original, replayed, validation_results):
        """Plot difference analysis between original and replayed."""
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title("Difference Analysis")

        # Plot both cities overlaid
        if hasattr(original, 'streets') and not original.streets.empty:
            original.streets.plot(ax=ax, color='blue', linewidth=1, alpha=0.5, label='Original')

        if hasattr(replayed, 'streets') and not replayed.streets.empty:
            replayed.streets.plot(ax=ax, color='red', linewidth=1, alpha=0.5, label='Replayed')

        # Add validation metrics as text
        metrics_text = f"""Validation Metrics:
Geometric: {'PASS' if validation_results.get('geometric_valid') else 'FAIL'}
Topological: {'PASS' if validation_results.get('topological_valid') else 'FAIL'}
Score: {validation_results.get('overall_score', 0):.2f}
"""
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.legend()

    def _plot_action_timeline(self, ax, trace):
        """Plot action sequence over time."""
        actions = trace.actions
        if not actions:
            ax.text(0.5, 0.5, 'No actions in trace', ha='center', va='center')
            return

        timestamps = [i for i in range(len(actions))]
        confidences = [action.confidence for action in actions]

        ax.plot(timestamps, confidences, 'b-o', linewidth=2, markersize=4)
        ax.set_xlabel('Action Sequence')
        ax.set_ylabel('Confidence')
        ax.set_title('Action Confidence Timeline')
        ax.grid(True, alpha=0.3)

    def _plot_confidence_distribution(self, ax, trace):
        """Plot distribution of action confidences."""
        actions = trace.actions
        if not actions:
            ax.text(0.5, 0.5, 'No actions in trace', ha='center', va='center')
            return

        confidences = [action.confidence for action in actions]
        ax.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution')
        ax.grid(True, alpha=0.3)

    def _plot_action_breakdown(self, ax, trace):
        """Plot breakdown of action types."""
        actions = trace.actions
        if not actions:
            ax.text(0.5, 0.5, 'No actions in trace', ha='center', va='center')
            return

        action_types = {}
        for action in actions:
            action_type = action.action_type.value
            action_types[action_type] = action_types.get(action_type, 0) + 1

        labels = list(action_types.keys())
        sizes = list(action_types.values())

        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Action Type Breakdown')
        ax.axis('equal')

    def _plot_state_evolution(self, ax, trace):
        """Plot simplified state evolution metrics."""
        # Placeholder - would show streets/blocks over time
        ax.text(0.5, 0.5, 'State Evolution\n(Not implemented)', ha='center', va='center')
        ax.set_title('State Evolution')