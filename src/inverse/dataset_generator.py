#!/usr/bin/env python3
"""
Dataset Generator
Phase 2: Generate ML training datasets from validated inverse growth traces.
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pickle
import logging
import json

from .data_structures import StateActionSample, MLDataset, ActionType
from .feature_extractor import StateFeatureExtractor
from ..core.contracts import GrowthState

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate ML datasets from validated traces."""

    def __init__(self, output_dir: str, acceptance_criteria: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criteria = acceptance_criteria
        self.feature_extractor = StateFeatureExtractor()

        self.accepted_samples = []
        self.rejected_traces = []
        self.city_ids = set()

    def meets_criteria(self, validation: Dict[str, Any], trace=None) -> bool:
        """Check if trace meets acceptance criteria."""
        meets_fidelity = validation.get('replay_fidelity', 0) >= self.criteria['min_replay_fidelity']
        meets_connectivity = validation.get('connectivity_preserved', False) == self.criteria['connectivity_required']
        meets_min_actions = trace is None or len(trace.actions) >= self.criteria.get('min_actions', 0)
        meets_confidence = trace is None or all(action.confidence >= self.criteria.get('min_action_confidence', 0) for action in trace.actions)

        return meets_fidelity and meets_connectivity and meets_min_actions and meets_confidence

    def add_trace(self, city_id: str, trace, original_state: GrowthState, validation: Dict[str, Any]):
        """Extract samples from validated trace."""

        self.city_ids.add(city_id)

        # Create accepted traces directory
        accepted_dir = self.output_dir / 'accepted_traces' / city_id
        accepted_dir.mkdir(parents=True, exist_ok=True)

        # Save trace and validation
        with open(accepted_dir / 'trace.json', 'w') as f:
            json.dump({
                'actions': [{'action_type': action.action_type.value, 'confidence': action.confidence,
                            'target_id': action.target_id, 'intent_params': action.intent_params}
                           for action in trace.actions],
                'metadata': trace.metadata
            }, f, indent=2)

        with open(accepted_dir / 'validation.json', 'w') as f:
            json.dump(validation, f, indent=2)

        # Simulate state progression (in full impl, use actual replay states)
        current_state = trace.initial_state
        samples_data = []

        for step_idx, action in enumerate(trace.actions):
            # Extract state features
            state_features = self.feature_extractor.extract_features(current_state)

            # Encode action
            action_type_id = self._encode_action_type(action.action_type)
            action_params = self._encode_action_params(action)

            # Create sample
            sample = StateActionSample(
                state_features=state_features,
                action_type=action_type_id,
                action_params=action_params,
                confidence=action.confidence,
                city_id=city_id,
                step_index=step_idx,
                metadata={'validation_fidelity': validation['replay_fidelity']}
            )

            self.accepted_samples.append(sample)
            samples_data.append({
                'state_features': state_features.tolist(),
                'action_type': action_type_id,
                'action_params': action_params.tolist(),
                'confidence': action.confidence,
                'step_index': step_idx
            })

            # Update state (placeholder - use actual replay)
            current_state = self._apply_action_to_state(current_state, action)

        # Save samples as npz
        np.savez_compressed(accepted_dir / 'samples.npz', samples=samples_data)

    def reject_trace(self, city_id: str, trace, validation: Dict[str, Any]):
        """Store rejected trace for analysis."""
        # Create rejected traces directory
        rejected_dir = self.output_dir / 'rejected_traces' / city_id
        rejected_dir.mkdir(parents=True, exist_ok=True)

        # Save trace and validation
        with open(rejected_dir / 'trace.json', 'w') as f:
            json.dump({
                'actions': [{'action_type': action.action_type.value, 'confidence': action.confidence,
                            'target_id': action.target_id, 'intent_params': action.intent_params}
                           for action in trace.actions],
                'metadata': trace.metadata
            }, f, indent=2)

        with open(rejected_dir / 'validation.json', 'w') as f:
            json.dump(validation, f, indent=2)

        self.rejected_traces.append({
            'city_id': city_id,
            'trace': trace,
            'validation': validation
        })

    def save_dataset(self):
        """Save dataset with train/val/test splits."""
        # Split by city (not by sample!)
        city_list = list(self.city_ids)
        np.random.shuffle(city_list)

        n_cities = len(city_list)
        train_cities = set(city_list[:int(0.7 * n_cities)])
        val_cities = set(city_list[int(0.7 * n_cities):int(0.85 * n_cities)])
        test_cities = set(city_list[int(0.85 * n_cities):])

        # Assign samples to splits
        train_indices = [i for i, s in enumerate(self.accepted_samples) if s.city_id in train_cities]
        val_indices = [i for i, s in enumerate(self.accepted_samples) if s.city_id in val_cities]
        test_indices = [i for i, s in enumerate(self.accepted_samples) if s.city_id in test_cities]

        dataset = MLDataset(
            samples=self.accepted_samples,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            city_ids=self.city_ids,
            feature_stats=self._compute_feature_stats(),
            action_distribution=self._compute_action_distribution()
        )

        # Save
        with open(self.output_dir / 'ml_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        logger.info(f"Saved dataset: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    def print_summary(self):
        """Print dataset generation summary."""
        n_accepted = len(self.accepted_samples)
        n_rejected = len(self.rejected_traces)
        n_cities = len(self.city_ids)

        print(f"\nDataset Generation Summary:")
        print(f"Accepted cities: {n_cities}")
        print(f"Accepted samples: {n_accepted}")
        print(f"Rejected traces: {n_rejected}")
        print(f"Acceptance rate: {n_cities / (n_cities + n_rejected) * 100:.1f}%" if n_cities + n_rejected > 0 else "0%")

        if self.accepted_samples:
            action_dist = self._compute_action_distribution()
            print(f"Action distribution: {action_dist}")

    def _encode_action_type(self, action_type: ActionType) -> int:
        """Encode action type as integer."""
        action_mapping = {
            ActionType.EXTEND_FRONTIER: 0,
            ActionType.SUBDIVIDE_BLOCK: 1,
            ActionType.REALIGN_STREET: 2,
            ActionType.REMOVE_STREET: 3,
        }
        return action_mapping.get(action_type, 0)

    def _encode_action_params(self, action) -> np.ndarray:
        """Encode action parameters as fixed-size array."""
        # Simplified: return fixed-size parameter vector
        # In full implementation, extract specific parameters from action.intent_params
        params = np.zeros(16, dtype=np.float32)  # 16-dim parameter space

        # Extract basic parameters from intent
        if hasattr(action, 'intent_params') and action.intent_params:
            intent = action.intent_params

            # Map common parameters
            if 'direction' in intent:
                if intent['direction'] == 'peripheral_expansion':
                    params[0] = 1.0
                elif intent['direction'] == 'central_growth':
                    params[1] = 1.0

            if 'strategy' in intent:
                if intent['strategy'] == 'short_segment':
                    params[2] = 1.0
                elif intent['strategy'] == 'connectivity':
                    params[3] = 1.0

        return params

    def _apply_action_to_state(self, current_state: GrowthState, action) -> GrowthState:
        """Apply action to state (simplified placeholder)."""
        # In full implementation, this would use the replay engine
        # For now, return state with incremented iteration
        return GrowthState(
            streets=current_state.streets,
            blocks=current_state.blocks,
            frontiers=current_state.frontiers,
            graph=current_state.graph,
            iteration=current_state.iteration + 1,
            city_bounds=current_state.city_bounds
        )

    def _compute_feature_stats(self) -> Dict[str, Any]:
        """Compute feature normalization statistics."""
        if not self.accepted_samples:
            return {}

        features = np.array([s.state_features for s in self.accepted_samples])
        return {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
        }

    def _compute_action_distribution(self) -> Dict[int, int]:
        """Compute action type distribution."""
        distribution = {}
        for sample in self.accepted_samples:
            action_type = sample.action_type
            distribution[action_type] = distribution.get(action_type, 0) + 1
        return distribution