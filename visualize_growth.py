#!/usr/bin/env python3
"""
Simple visualization of grown cities using matplotlib.
"""

import sys
import os
sys.path.append('.')

from src.core.growth_engine import GrowthEngine
from src.core.contracts import GrowthState
import matplotlib.pyplot as plt
import geopandas as gpd

def visualize_growth_evolution():
    """Create a simple visualization of city growth evolution."""
    print("Creating growth visualization...")

    # Initialize and grow a city
    engine = GrowthEngine('viz_city', seed=42)
    state = engine.initialize_from_bud((0, 0))
    frontiers = engine.initialize_frontiers_for_bud(state)
    state = GrowthState(
        streets=state.streets,
        blocks=state.blocks,
        frontiers=frontiers,
        graph=state.graph,
        iteration=state.iteration,
        city_bounds=state.city_bounds
    )

    # Grow for a few steps
    states = [state]
    for i in range(5):
        if not state.frontiers:
            break
        state = engine.grow_one_step(state)
        states.append(state)

    # Create visualization
    fig, axes = plt.subplots(1, len(states), figsize=(15, 4))
    if len(states) == 1:
        axes = [axes]

    for i, state in enumerate(states):
        ax = axes[i]

        # Plot streets
        if not state.streets.empty:
            state.streets.plot(ax=ax, color='blue', linewidth=2, alpha=0.7)

        # Plot frontiers
        if state.frontiers:
            for frontier in state.frontiers:
                ax.plot(*frontier.geometry.xy, color='red', linewidth=1, alpha=0.5)

        ax.set_title(f'Iteration {i}\n{len(state.streets)} streets')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('growth_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to growth_visualization.png")
    print(f"✓ Final city: {len(states[-1].streets)} streets, {len(states[-1].frontiers)} frontiers")

def show_city_stats():
    """Show statistics about the grown cities."""
    print("\nCity Growth Statistics:")
    print("-" * 40)

    # Test different seeds
    seeds = [42, 123, 456]
    for seed in seeds:
        engine = GrowthEngine(f'stats_city_{seed}', seed=seed)
        state = engine.initialize_from_bud((0, 0))
        frontiers = engine.initialize_frontiers_for_bud(state)
        state = GrowthState(
            streets=state.streets,
            blocks=state.blocks,
            frontiers=frontiers,
            graph=state.graph,
            iteration=state.iteration,
            city_bounds=state.city_bounds
        )

        # Grow for 10 steps
        for _ in range(10):
            if not state.frontiers:
                break
            state = engine.grow_one_step(state)

        print(f"Seed {seed}: {len(state.streets)} streets, {len(state.frontiers)} frontiers")

if __name__ == "__main__":
    visualize_growth_evolution()
    show_city_stats()
    print("\n🎨 Visualization complete!")
