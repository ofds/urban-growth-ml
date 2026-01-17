#!/usr/bin/env python3
"""Test complete pipeline on real Piedmont data with detailed diagnostics."""

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


class PipelineMetrics:
    """Track detailed metrics across pipeline stages."""
    
    def __init__(self):
        self.stages = {
            'loading': {'success': False, 'metrics': {}},
            'inference': {'success': False, 'metrics': {}},
            'replay': {'success': False, 'metrics': {}},
        }
    
    def record_stage(self, stage: str, success: bool, **metrics):
        """Record stage results and metrics."""
        self.stages[stage]['success'] = success
        self.stages[stage]['metrics'] = metrics
    
    def get_summary(self) -> dict:
        """Generate comprehensive summary."""
        return {
            'all_stages_passed': all(s['success'] for s in self.stages.values()),
            'stages': self.stages
        }


def test_piedmont_pipeline():
    """Test complete inverse inference + replay on Piedmont."""
    
    metrics = PipelineMetrics()
    
    print("="*60)
    print("PIEDMONT FULL PIPELINE TEST - DETAILED DIAGNOSTICS")
    print("="*60)
    
    # ==================================================================
    # STAGE 1: LOADING
    # ==================================================================
    print("\n[1/4] Loading Piedmont city state...")
    try:
        engine = GrowthEngine('piedmont_ca', seed=42)
        city = engine.load_initial_state()
        
        metrics.record_stage('loading', True,
            streets=len(city.streets),
            blocks=len(city.blocks),
            frontiers=len(city.frontiers),
            graph_nodes=city.graph.number_of_nodes()
        )
        
        print(f"  ✅ Loaded city:")
        print(f"     Streets: {len(city.streets)}")
        print(f"     Blocks: {len(city.blocks)}")
        print(f"     Frontiers: {len(city.frontiers)}")
        print(f"     Graph nodes: {city.graph.number_of_nodes()}")
        
    except Exception as e:
        print(f"  ❌ Failed to load city: {e}")
        metrics.record_stage('loading', False, error=str(e))
        return False, metrics
    
    # ==================================================================
    # STAGE 2: INFERENCE
    # ==================================================================
    print("\n[2/4] Running inverse inference...")
    
    try:
        inference = BasicInferenceEngine()
        trace = inference.infer_trace(city)
        
        # Calculate inference quality metrics
        actions_inferred = len(trace.actions)
        avg_confidence = trace.average_confidence
        expected_actions = len(city.streets) - 2  # Minus initial skeleton
        inference_coverage = (actions_inferred / expected_actions) * 100 if expected_actions > 0 else 0
        
        metrics.record_stage('inference', True,
            actions_inferred=actions_inferred,
            avg_confidence=avg_confidence,
            expected_total=expected_actions,
            coverage_pct=inference_coverage
        )
        
        print(f"  ✅ Inference complete:")
        print(f"     Actions inferred: {actions_inferred}/{expected_actions} ({inference_coverage:.1f}% coverage)")
        print(f"     Average confidence: {avg_confidence:.2f}")
        
        if actions_inferred == 0:
            print("  ❌ CRITICAL: No actions inferred")
            return False, metrics
            
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        metrics.record_stage('inference', False, error=str(e))
        import traceback
        traceback.print_exc()
        return False, metrics
    
    # ==================================================================
    # STAGE 3: REPLAY
    # ==================================================================
    print("\n[3/4] Replaying with growth engine...")
    
    try:
        replay_engine = TraceReplayEngine()
        validation = replay_engine.validate_trace_replay(
            trace=trace,
            original_state=city,
            city_name='piedmont_ca'
        )
        
        # Extract detailed replay metrics
        actions_attempted = validation.get('trace_actions', 0)
        actions_replayed = validation.get('replay_actions', 0)
        replay_success_rate = (actions_replayed / actions_attempted * 100) if actions_attempted > 0 else 0
        
        streets_original = validation.get('original_streets', 0)
        streets_replayed = validation.get('replayed_streets', 0)
        street_reproduction_rate = (streets_replayed / streets_original * 100) if streets_original > 0 else 0
        
        replay_fidelity = validation.get('replay_fidelity', 0)
        morphological_valid = validation.get('morphological_valid', False)
        
        metrics.record_stage('replay', True,
            actions_attempted=actions_attempted,
            actions_successfully_replayed=actions_replayed,
            replay_success_rate=replay_success_rate,
            streets_original=streets_original,
            streets_reproduced=streets_replayed,
            street_reproduction_rate=street_reproduction_rate,
            replay_fidelity=replay_fidelity,
            morphological_valid=morphological_valid
        )
        
        print(f"  📊 Replay Diagnostics:")
        print(f"     Action replay rate: {actions_replayed}/{actions_attempted} ({replay_success_rate:.1f}%)")
        print(f"     Street reproduction: {streets_replayed}/{streets_original} ({street_reproduction_rate:.1f}%)")
        print(f"     Morphological fidelity: {replay_fidelity:.2f}")
        print(f"     Morphological valid: {morphological_valid}")
        
        # CRITICAL: Check for false positive
        if replay_success_rate < 50 and morphological_valid:
            print(f"\n  ⚠️  WARNING: Validation may be misleading!")
            print(f"     Only {replay_success_rate:.1f}% of actions replayed successfully")
            print(f"     But validator reports 'success' - likely comparing only {streets_replayed} streets")
        
    except Exception as e:
        print(f"  ❌ Replay failed: {e}")
        metrics.record_stage('replay', False, error=str(e))
        import traceback
        traceback.print_exc()
        return False, metrics
    
    # ==================================================================
    # STAGE 4: FINAL ASSESSMENT
    # ==================================================================
    print("\n[4/4] Final Assessment")
    print("-"*60)
    
    replay_metrics = metrics.stages['replay']['metrics']
    replay_rate = replay_metrics.get('replay_success_rate', 0)
    street_repro = replay_metrics.get('street_reproduction_rate', 0)
    fidelity = replay_metrics.get('replay_fidelity', 0)
    
    # Determine pipeline quality
    pipeline_quality = None
    ready_for_ml = False
    
    if replay_rate >= 80 and street_repro >= 80:
        pipeline_quality = "EXCELLENT"
        ready_for_ml = True
        print("🎉 EXCELLENT: Full pipeline working at high quality!")
    elif replay_rate >= 50 and street_repro >= 50:
        pipeline_quality = "GOOD"
        ready_for_ml = True
        print("✅ GOOD: Pipeline working, acceptable quality")
    elif replay_rate >= 20:
        pipeline_quality = "POOR"
        ready_for_ml = False
        print("⚠️  POOR: Pipeline runs but quality is low")
    else:
        pipeline_quality = "FAILING"
        ready_for_ml = False
        print("❌ FAILING: Critical pipeline errors")
    
    print(f"   Action replay: {replay_rate:.1f}%")
    print(f"   Street reproduction: {street_repro:.1f}%")
    print(f"   Fidelity score: {fidelity:.2f}")
    
    if ready_for_ml:
        print("\n✅ READY FOR ML TRAINING")
    else:
        print("\n❌ NOT READY - Fix issues before ML training")
    
    return ready_for_ml, metrics


def print_detailed_report(metrics: PipelineMetrics):
    """Print comprehensive pipeline report."""
    print("\n" + "="*60)
    print("DETAILED PIPELINE REPORT")
    print("="*60)
    
    summary = metrics.get_summary()
    
    for stage_name, stage_data in summary['stages'].items():
        print(f"\n{stage_name.upper()}: {'✅ PASS' if stage_data['success'] else '❌ FAIL'}")
        for key, value in stage_data['metrics'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print(f"OVERALL: {'✅ ALL STAGES PASSED' if summary['all_stages_passed'] else '❌ SOME STAGES FAILED'}")
    print("="*60)


def main():
    ready_for_ml, metrics = test_piedmont_pipeline()
    
    print_detailed_report(metrics)
    
    print("\n" + "="*60)
    if ready_for_ml:
        print("NEXT STEPS:")
        print("1. Increase max_steps to infer full trace (1400+ actions)")
        print("2. Verify replay rate stays >80% on full trace")
        print("3. Generate datasets from multiple cities")
        print("4. Train ML model on inferred actions")
    else:
        print("FIX REQUIRED:")
        replay_rate = metrics.stages['replay']['metrics'].get('replay_success_rate', 0)
        
        if replay_rate < 20:
            print("1. CRITICAL: Action ordering bug - verify insert(0) vs append()")
            print("2. Check stable_id computation matches between inference and replay")
            print("3. Verify geometry_for_matching is correctly stored in actions")
        elif replay_rate < 50:
            print("1. Improve stable_id matching algorithm")
            print("2. Tune geometry matching tolerance")
            print("3. Add fallback matching strategies")
        else:
            print("1. Fine-tune inference heuristics")
            print("2. Verify growth engine validators")
            print("3. Check skeleton extraction quality")
    print("="*60)
    
    sys.exit(0 if ready_for_ml else 1)


if __name__ == '__main__':
    main()
