#!/usr/bin/env python3
"""Test inverse inference on synthetic grown cities."""

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

def test_synthetic_growth_pipeline():
    """Test complete pipeline on synthetically grown city."""
    print("="*60)
    print("SYNTHETIC GROWTH PIPELINE TEST")
    print("="*60)
    
    try:
        # Step 1: Load base city and grow it forward
        print("\n[1/5] Loading base Piedmont city...")
        engine = GrowthEngine('piedmont_ca', seed=42)
        base_city = engine.load_initial_state()
        print(f"  ✅ Loaded base city:")
        print(f"     Streets: {len(base_city.streets)}")
        print(f"     Blocks: {len(base_city.blocks)}")
        print(f"     Frontiers: {len(base_city.frontiers)}")
        
        # Step 2: Create minimal seed from base city
        print("\n[2/5] Creating minimal seed state...")
        import geopandas as gpd
        from core.contracts import GrowthState, FrontierEdge
        from shapely.geometry import Point, LineString
        import hashlib
        import networkx as nx
        
        # Take first 5 streets as seed
        seed_streets = base_city.streets.iloc[:5].copy()
        
        # Build graph from seed
        seed_graph = nx.Graph()
        for idx, street in seed_streets.iterrows():
            u, v = street['u'], street['v']
            geom = street.geometry
            
            if u not in seed_graph.nodes:
                seed_graph.add_node(u, geometry=Point(geom.coords[0]))
            if v not in seed_graph.nodes:
                seed_graph.add_node(v, geometry=Point(geom.coords[-1]))
            
            seed_graph.add_edge(u, v, geometry=geom, length=geom.length)
        
        # Create frontiers (dead-ends only)
        seed_frontiers = []
        for node in seed_graph.nodes():
            if seed_graph.degree(node) == 1:
                neighbors = list(seed_graph.neighbors(node))
                if neighbors:
                    neighbor = neighbors[0]
                    edge_data = seed_graph.edges.get((node, neighbor), {})
                    geometry = edge_data.get('geometry')
                    
                    if geometry and isinstance(geometry, LineString) and geometry.is_valid:
                        frontier_id = hashlib.sha256(
                            f"dead_end_{min(node, neighbor)}_{max(node, neighbor)}".encode()
                        ).hexdigest()[:16]
                        
                        frontier = FrontierEdge(
                            frontier_id=frontier_id,
                            edge_id=(node, neighbor),
                            block_id=None,
                            geometry=geometry,
                            frontier_type="dead_end",
                            expansion_weight=0.8,
                            spatial_hash=""
                        )
                        seed_frontiers.append(frontier)
        
        seed_state = GrowthState(
            streets=seed_streets,
            blocks=gpd.GeoDataFrame(columns=['geometry'], crs=base_city.streets.crs),
            frontiers=seed_frontiers,
            graph=seed_graph,
            iteration=0,
            city_bounds=base_city.city_bounds
        )
        
        print(f"  ✅ Created seed state:")
        print(f"     Streets: {len(seed_state.streets)}")
        print(f"     Frontiers: {len(seed_state.frontiers)} dead-ends")
        
    except Exception as e:
        print(f"  ❌ Failed to create seed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Step 3: Grow city forward for 15 steps
        print("\n[3/5] Growing city forward (15 steps)...")
        current_state = seed_state
        growth_steps = 0
        max_growth_steps = 15
        
        while growth_steps < max_growth_steps and current_state.frontiers:
            # Select a frontier
            frontier = engine.select_frontier_edge(current_state.frontiers, current_state)
            if not frontier:
                print(f"  ⚠️  No frontier selected at step {growth_steps}")
                break
            
            # Propose action
            action = engine.propose_grow_trajectory(frontier, current_state)
            if not action:
                print(f"  ⚠️  No valid action at step {growth_steps}")
                break
            
            # Apply action
            current_state = engine.apply_growth_action(action, current_state)
            growth_steps += 1
            
            if growth_steps % 5 == 0:
                print(f"    Step {growth_steps}: {len(current_state.streets)} streets, {len(current_state.frontiers)} frontiers")
        
        grown_city = current_state
        print(f"  ✅ Growth complete:")
        print(f"     Final streets: {len(grown_city.streets)}")
        print(f"     Final frontiers: {len(grown_city.frontiers)}")
        print(f"     Growth steps: {growth_steps}")
        
        # Check frontier types
        frontier_types = {}
        for f in grown_city.frontiers:
            frontier_types[f.frontier_type] = frontier_types.get(f.frontier_type, 0) + 1
        print(f"     Frontier types: {frontier_types}")
        
    except Exception as e:
        print(f"  ❌ Growth failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Step 4: Run inverse inference
        print("\n[4/5] Running inverse inference...")
        inference = BasicInferenceEngine()
        trace = inference.infer_trace(grown_city, max_steps=10000, initial_state=seed_state)
        
        print(f"  ✅ Inference complete:")
        print(f"     Actions inferred: {len(trace.actions)}")
        print(f"     Average confidence: {trace.average_confidence:.2f}")
        print(f"     Initial state streets: {len(trace.initial_state.streets)}")
        
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
        # Step 5: Replay and validate
        print("\n[5/5] Replaying trace...")
        replay_engine = TraceReplayEngine()
        validation = replay_engine.validate_trace_replay(
            trace=trace,
            original_state=grown_city,
            city_name='synthetic_test'
        )
        
        print(f"  ✅ Replay complete:")
        print(f"     Success: {validation.get('success', False)}")
        print(f"     Replay fidelity: {validation.get('replay_fidelity', 0):.2f}")
        print(f"     Actions replayed: {validation.get('replay_actions', 0)}/{len(trace.actions)}")
        print(f"     Morphological valid: {validation.get('morphological_valid', False)}")
        
    except Exception as e:
        print(f"  ❌ Replay failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    fidelity = validation.get('replay_fidelity', 0)
    success = validation.get('success', False)
    
    print(f"Ground truth: {growth_steps} forward growth steps")
    print(f"Inferred: {len(trace.actions)} actions")
    print(f"Replayed: {validation.get('replay_actions', 0)} actions")
    print(f"Replay fidelity: {fidelity:.2%}")
    
    if success and fidelity > 0.7:
        print("\n🎉 EXCELLENT: Pipeline working correctly!")
        return True
    elif success and fidelity > 0.5:
        print("\n✅ GOOD: Pipeline working, needs tuning")
        return True
    else:
        print("\n❌ NEEDS WORK: Low fidelity or failed replay")
        return False

def main():
    success = test_synthetic_growth_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("✅ READY FOR MULTI-CITY TESTING")
        print("Next: Run on multiple cities and collect metrics")
    else:
        print("❌ FIX PIPELINE BEFORE SCALING")
        print("Debug: Check inference heuristics and replay matching")
    print("="*60)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
