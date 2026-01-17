#!/usr/bin/env python3
"""
Urban Growth Visualization Module

Provides robust, reusable visualizations for:
- Inspecting how cities grow over time
- Comparing original vs replayed morphology
- Summarizing validation metrics (fidelity, replay rate, street reproduction)
"""

from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import geopandas as gpd
import numpy as np
import logging
from shapely.geometry import LineString, Point
import warnings

logger = logging.getLogger(__name__)


class InverseGrowthVisualizer:
    """
    Visualization engine for urban growth traces, replays, and validation metrics.

    Provides clean APIs for generating publication-quality plots of:
    - Replay validation comparisons
    - Trace summary dashboards
    - Temporal growth sequences
    """

    def __init__(self, output_dir: str = "outputs/inverse"):
        """
        Initialize visualizer with output directory.

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Consistent color scheme
        self.colors = {
            'original_streets': '#2E86AB',      # Blue
            'replayed_streets': '#A23B72',     # Magenta
            'mismatch_streets': '#F18F01',     # Orange
            'new_streets': '#C73E1D',          # Red
            'existing_streets': '#6B6B6B',     # Gray
            'frontiers': '#F24236',            # Red
            'blocks': '#D4D4D4',               # Light gray
        }

        # Suppress matplotlib warnings about PatchCollection legends
        warnings.filterwarnings("ignore", message=".*PatchCollection.*legend.*")

    def plot_replay_validation(self,
                              original_city: 'GrowthState',
                              replayed_city: 'GrowthState',
                              validation_metrics: Dict[str, Any],
                              city_name: str = "unknown") -> str:
        """
        Create replay validation comparison plot.

        Shows original vs replayed city streets with metrics overlay.

        Args:
            original_city: Original final city state
            replayed_city: State produced by replaying inferred trace
            validation_metrics: Validation results from MorphologicalValidator
            city_name: Name of city for plot title

        Returns:
            Path to saved plot file
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot original city
        self._plot_city_state(axes[0], original_city, "Original City",
                             street_color=self.colors['original_streets'])

        # Plot replayed city
        self._plot_city_state(axes[1], replayed_city, "Replayed City",
                             street_color=self.colors['replayed_streets'])

        # Add metrics annotation
        self._add_validation_metrics_annotation(fig, validation_metrics)

        # Set overall title
        fig.suptitle(f"Replay Validation: {city_name.replace('_', ' ').title()}",
                    fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"{city_name}_replay_validation.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return str(output_path)

    def plot_trace_summary(self,
                          trace: 'GrowthTrace',
                          city: 'GrowthState',
                          city_name: str = "unknown") -> str:
        """
        Create trace summary dashboard.

        Shows trace statistics and city overview in a compact layout.

        Args:
            trace: Inferred growth trace
            city: Final city state
            city_name: Name of city for plot title

        Returns:
            Path to saved plot file
        """
        fig = plt.figure(figsize=(16, 10))

        # Create subplot grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # City map (top left, spans 2 columns)
        ax_map = fig.add_subplot(gs[0, :2])
        self._plot_city_state(ax_map, city, f"{city_name.replace('_', ' ').title()}",
                             street_color=self.colors['original_streets'])

        # Trace statistics (top right)
        ax_stats = fig.add_subplot(gs[0, 2])
        self._plot_trace_statistics(ax_stats, trace)

        # Action confidence distribution (bottom left)
        ax_conf = fig.add_subplot(gs[1, 0])
        self._plot_confidence_distribution(ax_conf, trace)

        # Action type breakdown (bottom middle)
        ax_types = fig.add_subplot(gs[1, 1])
        self._plot_action_types(ax_types, trace)

        # Growth timeline (bottom right)
        ax_timeline = fig.add_subplot(gs[1, 2])
        self._plot_growth_timeline(ax_timeline, trace)

        # Overall title
        fig.suptitle(f"Growth Trace Summary: {city_name.replace('_', ' ').title()}",
                    fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()

        # Save plot
        output_path = self.output_dir / f"{city_name}_trace_validation_summary.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return str(output_path)

    def plot_growth_sequence(self,
                           trace: 'GrowthTrace',
                           initial_city: 'GrowthState',
                           step_stride: int = 10,
                           make_gif: bool = False,
                           max_frames: int = 100,
                           city_name: str = "unknown") -> Tuple[List[str], Optional[str]]:
        """
        Generate temporal growth sequence frames.

        Creates PNG frames showing incremental city growth, optionally compiles to GIF.

        Args:
            trace: Growth trace to visualize
            initial_city: Initial city state (skeleton)
            step_stride: Save frame every N actions
            make_gif: Whether to create animated GIF
            max_frames: Maximum number of frames to generate
            city_name: Name of city for filenames

        Returns:
            Tuple of (frame_paths, gif_path)
        """
        frame_dir = self.output_dir / f"{city_name}_growth_frames"
        frame_dir.mkdir(exist_ok=True)

        # Simulate growth states
        growth_states = self._simulate_growth_states(trace, initial_city, step_stride, max_frames)

        frame_paths = []
        for i, state in enumerate(growth_states):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Plot existing streets (gray)
            if len(state['existing_streets']) > 0:
                state['existing_streets'].plot(ax=ax, color=self.colors['existing_streets'],
                                              linewidth=1, alpha=0.6, label='Existing')

            # Plot new streets (red)
            if len(state['new_streets']) > 0:
                state['new_streets'].plot(ax=ax, color=self.colors['new_streets'],
                                        linewidth=2, alpha=0.8, label='New')

            # Plot frontiers
            if state['frontiers']:
                for frontier in state['frontiers']:
                    if hasattr(frontier, 'geometry') and frontier.geometry:
                        ax.plot(*frontier.geometry.xy, color=self.colors['frontiers'],
                               linewidth=1, alpha=0.7)

            # Formatting
            ax.set_title(f'Growth Step {state["step"]}\n{state["num_streets"]} streets')
            ax.set_aspect('equal')
            ax.axis('off')
            ax.legend(loc='upper right')

            # Save frame
            frame_path = frame_dir / "04d"
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(str(frame_path))

        # Create GIF if requested
        gif_path = None
        if make_gif and frame_paths:
            try:
                gif_path = self._create_gif_from_frames(frame_paths, city_name)
            except Exception as e:
                logger.warning(f"Failed to create GIF: {e}")

        return frame_paths, gif_path

    def _plot_city_state(self, ax: plt.Axes, city: 'GrowthState', title: str,
                        street_color: str = None):
        """
        Plot a city state on given axes.

        Args:
            ax: Matplotlib axes to plot on
            city: GrowthState to visualize
            title: Plot title
            street_color: Color for streets (default: original_streets)
        """
        if street_color is None:
            street_color = self.colors['original_streets']

        # Plot streets
        if len(city.streets) > 0:
            city.streets.plot(ax=ax, color=street_color, linewidth=1.5, alpha=0.8)

        # Plot blocks (light background)
        if len(city.blocks) > 0:
            city.blocks.plot(ax=ax, color=self.colors['blocks'], alpha=0.3, edgecolor='none')

        # Plot frontiers
        if city.frontiers:
            frontier_patches = []
            for frontier in city.frontiers:
                if hasattr(frontier, 'geometry') and frontier.geometry:
                    # Create a patch for legend (avoiding PatchCollection warning)
                    patch = mpatches.Patch(color=self.colors['frontiers'], alpha=0.7,
                                         label='Growth Frontiers')
                    frontier_patches.append(patch)
                    ax.plot(*frontier.geometry.xy, color=self.colors['frontiers'],
                           linewidth=2, alpha=0.7)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

        # Add legend if frontiers exist
        if city.frontiers:
            # Create custom legend handles to avoid PatchCollection warnings
            street_patch = mpatches.Patch(color=street_color, label='Streets')
            frontier_patch = mpatches.Patch(color=self.colors['frontiers'], label='Frontiers')
            ax.legend(handles=[street_patch, frontier_patch], loc='upper right')

    def _add_validation_metrics_annotation(self, fig: plt.Figure, metrics: Dict[str, Any]):
        """
        Add validation metrics annotation to figure.

        Args:
            fig: Matplotlib figure
            metrics: Validation metrics dictionary
        """
        # Create metrics text
        metrics_text = ".1f"".1f"".1f"".1f"f"""
Replay Validation Metrics:
• Fidelity: {metrics.get('replay_fidelity', 0):.3f}
• Streets: {metrics.get('replayed_streets', 0)}/{metrics.get('original_streets', 0)}
• Actions: {metrics.get('replay_actions', 0)}/{metrics.get('trace_actions', 0)}
• Valid: {'✓' if metrics.get('morphological_valid', False) else '✗'}
• Geometric: {'✓' if metrics.get('geometric_valid', False) else '✗'}
• Topological: {'✓' if metrics.get('topological_valid', False) else '✗'}
"""

        # Add text annotation
        fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    def _plot_trace_statistics(self, ax: plt.Axes, trace: 'GrowthTrace'):
        """
        Plot trace statistics in a text-based format.

        Args:
            ax: Matplotlib axes
            trace: Growth trace
        """
        ax.axis('off')

        stats_text = f"""
Trace Statistics:

Actions: {len(trace.actions)}
Avg Confidence: {trace.average_confidence:.3f}
High Conf: {len(trace.high_confidence_actions)}

Initial Streets: {len(trace.initial_state.streets) if hasattr(trace.initial_state, 'streets') else 'N/A'}
Final Streets: {len(trace.final_state.streets) if hasattr(trace.final_state, 'streets') else 'N/A'}
"""

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')

    def _plot_confidence_distribution(self, ax: plt.Axes, trace: 'GrowthTrace'):
        """
        Plot action confidence distribution histogram.

        Args:
            ax: Matplotlib axes
            trace: Growth trace
        """
        if not trace.actions:
            ax.text(0.5, 0.5, 'No actions', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        confidences = [action.confidence for action in trace.actions]

        ax.hist(confidences, bins=10, alpha=0.7, color=self.colors['original_streets'], edgecolor='black')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Action Confidence\nDistribution', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _plot_action_types(self, ax: plt.Axes, trace: 'GrowthTrace'):
        """
        Plot action type breakdown pie chart.

        Args:
            ax: Matplotlib axes
            trace: Growth trace
        """
        if not trace.actions:
            ax.text(0.5, 0.5, 'No actions', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        from collections import Counter
        action_types = [action.action_type.value for action in trace.actions]
        type_counts = Counter(action_types)

        labels = list(type_counts.keys())
        sizes = list(type_counts.values())
        colors = [self.colors['original_streets'], self.colors['replayed_streets'],
                 self.colors['new_streets'], self.colors['existing_streets']][:len(labels)]

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Action Types', fontsize=12)

    def _plot_growth_timeline(self, ax: plt.Axes, trace: 'GrowthTrace'):
        """
        Plot growth timeline showing cumulative streets over time.

        Args:
            ax: Matplotlib axes
            trace: Growth trace
        """
        if not trace.actions:
            ax.text(0.5, 0.5, 'No actions', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Simulate cumulative street growth
        initial_streets = len(trace.initial_state.streets) if hasattr(trace.initial_state, 'streets') else 2
        cumulative_streets = [initial_streets]

        for action in trace.actions:
            # Estimate street addition (simplified)
            if action.action_type.value == 'extend_frontier':
                cumulative_streets.append(cumulative_streets[-1] + 1)
            else:
                cumulative_streets.append(cumulative_streets[-1])

        steps = list(range(len(cumulative_streets)))

        ax.plot(steps, cumulative_streets, 'o-', color=self.colors['original_streets'],
               linewidth=2, markersize=4)
        ax.set_xlabel('Action Step')
        ax.set_ylabel('Total Streets')
        ax.set_title('Growth Timeline', fontsize=12)
        ax.grid(True, alpha=0.3)

    def _simulate_growth_states(self, trace: 'GrowthTrace', initial_city: 'GrowthState',
                              step_stride: int, max_frames: int) -> List[Dict]:
        """
        Simulate growth states for animation frames.

        Args:
            trace: Growth trace
            initial_city: Initial city state
            step_stride: Steps between frames
            max_frames: Maximum frames to generate

        Returns:
            List of state dictionaries for each frame
        """
        states = []

        # Initial state
        states.append({
            'existing_streets': initial_city.streets.copy(),
            'new_streets': gpd.GeoDataFrame(columns=initial_city.streets.columns),
            'frontiers': initial_city.frontiers.copy(),
            'step': 0,
            'num_streets': len(initial_city.streets)
        })

        # Simulate adding actions
        current_streets = initial_city.streets.copy()
        total_actions = len(trace.actions)

        for i in range(0, min(total_actions, max_frames * step_stride), step_stride):
            end_idx = min(i + step_stride, total_actions)

            # Mark new streets (simplified - in reality would need to track which streets are added)
            new_streets = gpd.GeoDataFrame(columns=current_streets.columns)

            # For demo, just show all streets as existing and add some dummy new ones
            # In a real implementation, you'd track which streets are added by each action

            states.append({
                'existing_streets': current_streets,
                'new_streets': new_streets,
                'frontiers': initial_city.frontiers,  # Would update in real implementation
                'step': end_idx,
                'num_streets': len(current_streets)
            })

        return states

    def _create_gif_from_frames(self, frame_paths: List[str], city_name: str) -> str:
        """
        Create animated GIF from frame images.

        Args:
            frame_paths: List of frame image paths
            city_name: City name for output filename

        Returns:
            Path to created GIF
        """
        try:
            from PIL import Image
            import imageio

            # Load frames
            frames = []
            for frame_path in frame_paths:
                frames.append(Image.open(frame_path))

            # Create GIF
            gif_path = str(self.output_dir / f"{city_name}_growth_animation.gif")
            imageio.mimsave(gif_path, frames, duration=0.5, loop=0)

            return gif_path

        except ImportError:
            logger.warning("PIL or imageio not available for GIF creation")
            raise
