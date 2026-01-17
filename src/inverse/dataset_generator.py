#!/usr/bin/env python3
"""
Dataset Generator
Phase 2: Generate ML training datasets from validated inverse growth traces.

This module provides a complete pipeline from city identifiers to ML-ready datasets:
1. Load city data and generate growth traces via inverse inference
2. Validate traces through replay verification
3. Extract state-action samples with feature vectors
4. Create train/val/test splits respecting city boundaries
5. Export datasets in multiple formats (pickle, CSV, npz)

Example canonical training row format:
{
    'state_features': np.ndarray(128,),  # Normalized city state features
    'action_type': int,                   # 0=EXTEND_FRONTIER, 1=SUBDIVIDE_BLOCK, etc.
    'action_params': np.ndarray(16,),     # Continuous parameters (direction, length, etc.)
    'confidence': float,                  # Inference confidence [0.0-1.0] for sample weighting
    'city_id': str,                       # Source city identifier
    'step_index': int,                    # Position in growth sequence
    'metadata': dict                      # Additional context (validation scores, etc.)
}
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import logging
import json
from dataclasses import asdict

from .data_structures import StateActionSample, MLDataset, ActionType, GrowthTrace
from .feature_extractor import StateFeatureExtractor
from .inference import BasicInferenceEngine
from .replay import ReplayEngine
from .validation import validate_trace_quality
from ..core.contracts import GrowthState
from ..core.loader import CityLoader

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Generate ML datasets from validated traces.
    
    This class manages the complete lifecycle from trace validation to dataset export,
    ensuring proper separation of cities across train/val/test splits to prevent leakage.
    """

    def __init__(self, output_dir: str, acceptance_criteria: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Directory for output files (traces, datasets, etc.)
            acceptance_criteria: Validation thresholds for trace acceptance
                - min_replay_fidelity: Minimum replay fidelity score (default: 0.7)
                - connectivity_required: Whether connectivity must be preserved (default: True)
                - min_actions: Minimum number of actions in trace (default: 5)
                - min_action_confidence: Minimum confidence per action (default: 0.5)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default acceptance criteria
        self.criteria = acceptance_criteria or {
            'min_replay_fidelity': 0.7,
            'connectivity_required': True,
            'min_actions': 5,
            'min_action_confidence': 0.5
        }
        
        self.feature_extractor = StateFeatureExtractor(use_per_city_normalization=True)

        self.accepted_samples: List[StateActionSample] = []
        self.rejected_traces: List[Dict[str, Any]] = []
        self.city_ids: set = set()
        
        # Initialize inference and replay engines
        self.inference_engine = BasicInferenceEngine()
        self.replay_engine = ReplayEngine()

    def generate_single_city_dataset(
        self,
        city_id: str,
        final_state: Optional[GrowthState] = None,
        initial_state: Optional[GrowthState] = None,
        max_steps: int = 10000,
        validate_trace: bool = True
    ) -> Tuple[Optional[GrowthTrace], Dict[str, Any]]:
        """
        Generate dataset from a single city using the full inverse pipeline.
        
        This is the main entry point for processing one city:
        1. Load city data (if final_state not provided)
        2. Run inverse inference to generate trace
        3. Optionally validate via replay
        4. Extract feature vectors for each action
        5. Store samples if validation passes
        
        Args:
            city_id: City identifier (e.g., 'piedmont_ca')
            final_state: Optional pre-loaded final city state
            initial_state: Optional known initial state (for testing)
            max_steps: Maximum inference steps
            validate_trace: Whether to validate trace via replay
            
        Returns:
            Tuple of (trace, validation_results)
            - trace: GrowthTrace object if accepted, None if rejected
            - validation_results: Dict with validation metrics
        """
        logger.info(f"Processing city: {city_id}")
        
        # Step 1: Load city data if not provided
        if final_state is None:
            try:
                loader = CityLoader()
                final_state = loader.load_city(city_id)
                logger.info(f"Loaded {city_id}: {len(final_state.streets)} streets")
            except Exception as e:
                logger.error(f"Failed to load city {city_id}: {e}")
                return None, {'error': str(e), 'stage': 'loading'}
        
        # Step 2: Run inverse inference
        try:
            trace = self.inference_engine.infer_trace(
                final_state=final_state,
                max_steps=max_steps,
                initial_state=initial_state
            )
            logger.info(f"Inferred {len(trace.actions)} actions for {city_id}")
        except Exception as e:
            logger.error(f"Inference failed for {city_id}: {e}")
            return None, {'error': str(e), 'stage': 'inference'}
        
        # Step 3: Validate trace via replay (if requested)
        validation_results = {}
        if validate_trace:
            try:
                validation_results = self._validate_trace(trace, final_state)
                logger.info(f"Validation for {city_id}: fidelity={validation_results.get('replay_fidelity', 0):.3f}")
            except Exception as e:
                logger.error(f"Validation failed for {city_id}: {e}")
                validation_results = {'error': str(e), 'stage': 'validation'}
        else:
            # Skip validation - assume trace is valid
            validation_results = {
                'replay_fidelity': 1.0,
                'connectivity_preserved': True,
                'validation_skipped': True
            }
        
        # Step 4: Check acceptance criteria
        if self.meets_criteria(validation_results, trace):
            self.add_trace(city_id, trace, final_state, validation_results)
            logger.info(f"Accepted {city_id}: {len(trace.actions)} actions")
            return trace, validation_results
        else:
            self.reject_trace(city_id, trace, validation_results)
            logger.warning(f"Rejected {city_id}: {validation_results}")
            return None, validation_results

    def _validate_trace(self, trace: GrowthTrace, original_final_state: GrowthState) -> Dict[str, Any]:
        """
        Validate trace by replaying it and comparing with original final state.
        
        Args:
            trace: Inferred growth trace
            original_final_state: Original final state to compare against
            
        Returns:
            Dict containing validation metrics:
            - replay_fidelity: Match percentage [0.0-1.0]
            - connectivity_preserved: Whether graph connectivity matches
            - num_streets_match: Number of streets that match
            - num_streets_expected: Expected number of streets
        """
        try:
            # Replay the trace
            replayed_state = self.replay_engine.replay_trace(trace)
            
            # Compute fidelity metrics
            validation = validate_trace_quality(
                original_state=original_final_state,
                replayed_state=replayed_state,
                trace=trace
            )
            
            return validation
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'replay_fidelity': 0.0,
                'connectivity_preserved': False,
                'error': str(e)
            }

    def meets_criteria(self, validation: Dict[str, Any], trace: Optional[GrowthTrace] = None) -> bool:
        """
        Check if trace meets acceptance criteria.
        
        Args:
            validation: Validation results from replay
            trace: Optional trace object for additional checks
            
        Returns:
            True if trace meets all acceptance criteria
        """
        # Check for validation errors
        if 'error' in validation:
            return False
        
        # Check replay fidelity
        meets_fidelity = validation.get('replay_fidelity', 0) >= self.criteria['min_replay_fidelity']
        
        # Check connectivity preservation
        meets_connectivity = (
            not self.criteria['connectivity_required'] or
            validation.get('connectivity_preserved', False)
        )
        
        # Check minimum number of actions
        meets_min_actions = (
            trace is None or
            len(trace.actions) >= self.criteria.get('min_actions', 0)
        )
        
        # Check minimum action confidence
        meets_confidence = (
            trace is None or
            all(action.confidence >= self.criteria.get('min_action_confidence', 0)
                for action in trace.actions)
        )

        return meets_fidelity and meets_connectivity and meets_min_actions and meets_confidence

    def add_trace(self, city_id: str, trace: GrowthTrace, original_state: GrowthState, 
                  validation: Dict[str, Any]):
        """
        Extract samples from validated trace and add to dataset.
        
        This method:
        1. Iterates through each action in the trace
        2. Extracts state features using StateFeatureExtractor
        3. Encodes action type and parameters
        4. Creates StateActionSample objects
        5. Saves trace metadata and samples to disk
        
        Args:
            city_id: City identifier
            trace: Validated growth trace
            original_state: Original final city state
            validation: Validation results
        """
        self.city_ids.add(city_id)

        # Create accepted traces directory
        accepted_dir = self.output_dir / 'accepted_traces' / city_id
        accepted_dir.mkdir(parents=True, exist_ok=True)

        # Save trace metadata
        trace_data = {
            'actions': [
                {
                    'action_type': action.action_type.value,
                    'confidence': action.confidence,
                    'target_id': action.target_id,
                    'intent_params': action.intent_params,
                    'timestamp': action.timestamp
                }
                for action in trace.actions
            ],
            'metadata': trace.metadata,
            'num_actions': len(trace.actions),
            'average_confidence': trace.average_confidence
        }
        
        with open(accepted_dir / 'trace.json', 'w') as f:
            json.dump(trace_data, f, indent=2)

        with open(accepted_dir / 'validation.json', 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            validation_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in validation.items()
            }
            json.dump(validation_serializable, f, indent=2)

        # Extract samples by iterating through trace with state progression
        current_state = trace.initial_state
        samples_data = []

        for step_idx, action in enumerate(trace.actions):
            try:
                # Extract state features (128-dim vector)
                state_features = self.feature_extractor.extract_features(
                    current_state,
                    city_name=city_id
                )

                # Encode action as classification target and parameters
                action_type_id = self._encode_action_type(action.action_type)
                action_params = self._encode_action_params(action)

                # Create training sample
                sample = StateActionSample(
                    state_features=state_features,
                    action_type=action_type_id,
                    action_params=action_params,
                    confidence=action.confidence,  # Used as training weight
                    city_id=city_id,
                    step_index=step_idx,
                    metadata={
                        'validation_fidelity': validation.get('replay_fidelity', 0),
                        'action_timestamp': action.timestamp
                    }
                )

                self.accepted_samples.append(sample)
                
                # Store serializable version
                samples_data.append({
                    'state_features': state_features.tolist(),
                    'action_type': action_type_id,
                    'action_params': action_params.tolist(),
                    'confidence': float(action.confidence),
                    'step_index': step_idx
                })

                # Advance state (use replay engine for accuracy)
                current_state = self._apply_action_to_state(current_state, action)
                
            except Exception as e:
                logger.error(f"Failed to extract sample at step {step_idx} for {city_id}: {e}")
                continue

        # Save samples as compressed numpy archive
        if samples_data:
            np.savez_compressed(
                accepted_dir / 'samples.npz',
                samples=samples_data,
                city_id=city_id,
                num_samples=len(samples_data)
            )
            
        logger.info(f"Added {len(samples_data)} samples from {city_id}")

    def reject_trace(self, city_id: str, trace: GrowthTrace, validation: Dict[str, Any]):
        """
        Store rejected trace for analysis and debugging.
        
        Rejected traces are saved separately to help diagnose:
        - Inference failures
        - Validation failures
        - Low confidence actions
        
        Args:
            city_id: City identifier
            trace: Rejected growth trace
            validation: Validation results explaining rejection
        """
        rejected_dir = self.output_dir / 'rejected_traces' / city_id
        rejected_dir.mkdir(parents=True, exist_ok=True)

        # Save trace
        trace_data = {
            'actions': [
                {
                    'action_type': action.action_type.value,
                    'confidence': action.confidence,
                    'target_id': action.target_id,
                    'intent_params': action.intent_params
                }
                for action in trace.actions
            ],
            'metadata': trace.metadata,
            'num_actions': len(trace.actions)
        }
        
        with open(rejected_dir / 'trace.json', 'w') as f:
            json.dump(trace_data, f, indent=2)

        # Save validation results
        with open(rejected_dir / 'validation.json', 'w') as f:
            validation_serializable = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in validation.items()
            }
            json.dump(validation_serializable, f, indent=2)

        self.rejected_traces.append({
            'city_id': city_id,
            'trace': trace,
            'validation': validation,
            'rejection_reason': self._get_rejection_reason(validation, trace)
        })
        
        logger.warning(f"Rejected {city_id}: {self._get_rejection_reason(validation, trace)}")

    def _get_rejection_reason(self, validation: Dict[str, Any], trace: GrowthTrace) -> str:
        """Generate human-readable rejection reason."""
        reasons = []
        
        if 'error' in validation:
            reasons.append(f"error: {validation['error']}")
        
        fidelity = validation.get('replay_fidelity', 0)
        if fidelity < self.criteria['min_replay_fidelity']:
            reasons.append(f"low_fidelity: {fidelity:.3f} < {self.criteria['min_replay_fidelity']}")
        
        if self.criteria['connectivity_required'] and not validation.get('connectivity_preserved', False):
            reasons.append("connectivity_not_preserved")
        
        if len(trace.actions) < self.criteria.get('min_actions', 0):
            reasons.append(f"too_few_actions: {len(trace.actions)} < {self.criteria['min_actions']}")
        
        low_conf_actions = [a for a in trace.actions if a.confidence < self.criteria.get('min_action_confidence', 0)]
        if low_conf_actions:
            reasons.append(f"low_confidence_actions: {len(low_conf_actions)}")
        
        return "; ".join(reasons) if reasons else "unknown"

    def generate_multi_city_dataset(
        self,
        city_ids: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: Optional[int] = 42,
        validate_traces: bool = True
    ) -> MLDataset:
        """
        Generate dataset from multiple cities with train/val/test splits.
        
        This function:
        1. Processes each city through the inverse pipeline
        2. Aggregates accepted samples
        3. Splits cities (not samples!) into train/val/test to prevent leakage
        4. Returns MLDataset object with proper splits
        
        Args:
            city_ids: List of city identifiers to process
            train_ratio: Fraction of cities for training (default: 0.7)
            val_ratio: Fraction of cities for validation (default: 0.15)
            test_ratio: Fraction of cities for testing (default: 0.15)
            random_seed: Random seed for reproducible splits
            validate_traces: Whether to validate each trace via replay
            
        Returns:
            MLDataset object with samples and split indices
        """
        logger.info(f"Generating multi-city dataset from {len(city_ids)} cities")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Process each city
        for city_id in city_ids:
            try:
                self.generate_single_city_dataset(
                    city_id=city_id,
                    validate_trace=validate_traces
                )
            except Exception as e:
                logger.error(f"Failed to process {city_id}: {e}")
                continue
        
        # Create train/val/test splits by city
        dataset = self._create_dataset_splits(train_ratio, val_ratio, test_ratio)
        
        logger.info(f"Dataset created: {len(dataset.train_indices)} train, "
                   f"{len(dataset.val_indices)} val, {len(dataset.test_indices)} test samples")
        
        return dataset

    def _create_dataset_splits(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> MLDataset:
        """
        Create train/val/test splits respecting city boundaries.
        
        IMPORTANT: Splits are by city, not by sample, to prevent leakage.
        All samples from the same city go into the same split.
        
        Args:
            train_ratio: Fraction of cities for training
            val_ratio: Fraction of cities for validation
            test_ratio: Fraction of cities for testing
            
        Returns:
            MLDataset object with populated splits
        """
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        
        # Shuffle cities for random assignment
        city_list = list(self.city_ids)
        np.random.shuffle(city_list)

        n_cities = len(city_list)
        n_train = int(train_ratio * n_cities)
        n_val = int(val_ratio * n_cities)
        
        # Assign cities to splits
        train_cities = set(city_list[:n_train])
        val_cities = set(city_list[n_train:n_train + n_val])
        test_cities = set(city_list[n_train + n_val:])

        # Assign sample indices based on city membership
        train_indices = [i for i, s in enumerate(self.accepted_samples) 
                        if s.city_id in train_cities]
        val_indices = [i for i, s in enumerate(self.accepted_samples) 
                      if s.city_id in val_cities]
        test_indices = [i for i, s in enumerate(self.accepted_samples) 
                       if s.city_id in test_cities]

        # Create dataset object
        dataset = MLDataset(
            samples=self.accepted_samples,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            city_ids=self.city_ids,
            feature_stats=self._compute_feature_stats(),
            action_distribution=self._compute_action_distribution()
        )
        
        logger.info(f"Split {n_cities} cities: {len(train_cities)} train, "
                   f"{len(val_cities)} val, {len(test_cities)} test")

        return dataset

    def save_dataset(
        self,
        dataset: Optional[MLDataset] = None,
        export_formats: List[str] = ['pickle', 'csv']
    ):
        """
        Save dataset to disk in multiple formats.
        
        Args:
            dataset: MLDataset to save (if None, creates from current samples)
            export_formats: List of formats to export ('pickle', 'csv', 'npz')
        """
        if dataset is None:
            dataset = self._create_dataset_splits(0.7, 0.15, 0.15)
        
        # Save pickle format (preserves full object structure)
        if 'pickle' in export_formats:
            pickle_path = self.output_dir / 'ml_dataset.pkl'
            with open(pickle_path, 'wb') as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved pickle dataset to {pickle_path}")
        
        # Save CSV format (for easy inspection and external tools)
        if 'csv' in export_formats:
            self._save_dataset_csv(dataset)
        
        # Save npz format (efficient for numpy-based loading)
        if 'npz' in export_formats:
            self._save_dataset_npz(dataset)

    def _save_dataset_csv(self, dataset: MLDataset):
        """Export dataset to CSV files (train.csv, val.csv, test.csv)."""
        csv_dir = self.output_dir / 'csv'
        csv_dir.mkdir(exist_ok=True)
        
        for split_name, indices in [
            ('train', dataset.train_indices),
            ('val', dataset.val_indices),
            ('test', dataset.test_indices)
        ]:
            if not indices:
                continue
            
            samples = [dataset.samples[i] for i in indices]
            
            # Create DataFrame
            rows = []
            for sample in samples:
                row = {
                    'city_id': sample.city_id,
                    'step_index': sample.step_index,
                    'action_type': sample.action_type,
                    'confidence': sample.confidence,
                    **{f'feature_{i}': sample.state_features[i] 
                       for i in range(len(sample.state_features))},
                    **{f'param_{i}': sample.action_params[i] 
                       for i in range(len(sample.action_params))}
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            csv_path = csv_dir / f'{split_name}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {split_name} CSV with {len(df)} rows to {csv_path}")

    def _save_dataset_npz(self, dataset: MLDataset):
        """Export dataset to npz format (efficient numpy arrays)."""
        npz_dir = self.output_dir / 'npz'
        npz_dir.mkdir(exist_ok=True)
        
        for split_name, indices in [
            ('train', dataset.train_indices),
            ('val', dataset.val_indices),
            ('test', dataset.test_indices)
        ]:
            if not indices:
                continue
            
            samples = [dataset.samples[i] for i in indices]
            
            # Stack arrays
            state_features = np.stack([s.state_features for s in samples])
            action_types = np.array([s.action_type for s in samples])
            action_params = np.stack([s.action_params for s in samples])
            confidences = np.array([s.confidence for s in samples])
            city_ids = np.array([s.city_id for s in samples])
            step_indices = np.array([s.step_index for s in samples])
            
            npz_path = npz_dir / f'{split_name}.npz'
            np.savez_compressed(
                npz_path,
                state_features=state_features,
                action_types=action_types,
                action_params=action_params,
                confidences=confidences,
                city_ids=city_ids,
                step_indices=step_indices
            )
            logger.info(f"Saved {split_name} npz with {len(samples)} samples to {npz_path}")

    def print_summary(self):
        """Print comprehensive dataset generation summary."""
        n_accepted = len(self.accepted_samples)
        n_rejected = len(self.rejected_traces)
        n_cities = len(self.city_ids)

        print("\n" + "="*60)
        print("Dataset Generation Summary")
        print("="*60)
        print(f"Cities processed: {n_cities + n_rejected}")
        print(f"  ✓ Accepted:  {n_cities}")
        print(f"  ✗ Rejected:  {n_rejected}")
        print(f"Acceptance rate: {n_cities / (n_cities + n_rejected) * 100:.1f}%")
        print(f"\nTotal samples: {n_accepted}")

        if self.accepted_samples:
            # Action distribution
            action_dist = self._compute_action_distribution()
            print(f"\nAction type distribution:")
            for action_type, count in sorted(action_dist.items()):
                action_name = self._decode_action_type(action_type)
                pct = count / n_accepted * 100
                print(f"  {action_name:20s}: {count:6d} ({pct:5.1f}%)")
            
            # Confidence statistics
            confidences = [s.confidence for s in self.accepted_samples]
            print(f"\nConfidence statistics:")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Std:  {np.std(confidences):.3f}")
            print(f"  Min:  {np.min(confidences):.3f}")
            print(f"  Max:  {np.max(confidences):.3f}")
            
            # Per-city statistics
            samples_per_city = {}
            for sample in self.accepted_samples:
                samples_per_city[sample.city_id] = samples_per_city.get(sample.city_id, 0) + 1
            
            print(f"\nSamples per city:")
            print(f"  Mean: {np.mean(list(samples_per_city.values())):.1f}")
            print(f"  Min:  {np.min(list(samples_per_city.values()))}")
            print(f"  Max:  {np.max(list(samples_per_city.values()))}")

        print("="*60 + "\n")

    def _encode_action_type(self, action_type: ActionType) -> int:
        """
        Encode action type as integer for classification.
        
        Action type mapping:
        0 = EXTEND_FRONTIER: Add street by extending a frontier edge
        1 = SUBDIVIDE_BLOCK: Split existing block with new street
        2 = REALIGN_STREET: Modify existing street geometry
        3 = REMOVE_STREET: Delete street (non-monotonic growth)
        """
        action_mapping = {
            ActionType.EXTEND_FRONTIER: 0,
            ActionType.SUBDIVIDE_BLOCK: 1,
            ActionType.REALIGN_STREET: 2,
            ActionType.REMOVE_STREET: 3,
        }
        return action_mapping.get(action_type, 0)

    def _decode_action_type(self, action_type_id: int) -> str:
        """Decode action type integer to string name."""
        type_names = {
            0: "EXTEND_FRONTIER",
            1: "SUBDIVIDE_BLOCK",
            2: "REALIGN_STREET",
            3: "REMOVE_STREET"
        }
        return type_names.get(action_type_id, "UNKNOWN")

    def _encode_action_params(self, action) -> np.ndarray:
        """
        Encode action parameters as fixed-size continuous vector.
        
        Parameter encoding (16-dim):
        [0-1]: Direction indicators (peripheral_expansion, central_growth)
        [2-3]: Strategy indicators (short_segment, connectivity)
        [4-7]: Geometric parameters (angle, length, curvature, etc.)
        [8-15]: Reserved for future parameters
        
        Args:
            action: InverseGrowthAction object
            
        Returns:
            16-dimensional parameter vector
        """
        params = np.zeros(16, dtype=np.float32)

        if not hasattr(action, 'intent_params') or not action.intent_params:
            return params
        
        intent = action.intent_params

        # Direction encoding (one-hot-like)
        if 'direction' in intent:
            if intent['direction'] == 'peripheral_expansion':
                params[0] = 1.0
            elif intent['direction'] == 'central_growth':
                params[1] = 1.0

        # Strategy encoding
        if 'strategy' in intent:
            if intent['strategy'] == 'short_segment':
                params[2] = 1.0
            elif intent['strategy'] == 'connectivity':
                params[3] = 1.0
        
        # Geometric parameters (if available)
        if 'angle' in intent:
            params[4] = float(intent['angle']) / 360.0  # Normalize to [0, 1]
        
        if 'length' in intent:
            params[5] = min(float(intent['length']) / 1000.0, 1.0)  # Normalize
        
        if 'curvature' in intent:
            params[6] = float(intent['curvature'])

        return params

    def _apply_action_to_state(self, current_state: GrowthState, action) -> GrowthState:
        """
        Apply action to state to get next state in sequence.
        
        In full implementation, this should use ReplayEngine.
        For now, returns a placeholder state with incremented iteration.
        
        Args:
            current_state: Current growth state
            action: Action to apply
            
        Returns:
            Next growth state after applying action
        """
        # TODO: Integrate with ReplayEngine for accurate state progression
        # For now, return simplified next state
        return GrowthState(
            streets=current_state.streets,
            blocks=current_state.blocks,
            frontiers=current_state.frontiers,
            graph=current_state.graph,
            iteration=current_state.iteration + 1,
            city_bounds=current_state.city_bounds
        )

    def _compute_feature_stats(self) -> Dict[str, Any]:
        """
        Compute feature normalization statistics across all samples.
        
        Returns:
            Dict with mean, std, min, max for each feature dimension
        """
        if not self.accepted_samples:
            return {}

        features = np.array([s.state_features for s in self.accepted_samples])
        return {
            'mean': np.mean(features, axis=0).tolist(),
            'std': np.std(features, axis=0).tolist(),
            'min': np.min(features, axis=0).tolist(),
            'max': np.max(features, axis=0).tolist(),
            'num_samples': len(self.accepted_samples),
            'feature_dim': features.shape[1]
        }

    def _compute_action_distribution(self) -> Dict[int, int]:
        """
        Compute action type distribution across all samples.
        
        Returns:
            Dict mapping action_type_id to count
        """
        distribution = {}
        for sample in self.accepted_samples:
            action_type = sample.action_type
            distribution[action_type] = distribution.get(action_type, 0) + 1
        return distribution


# Convenience functions for common use cases

def generate_dataset_from_city(
    city_id: str,
    output_dir: str,
    validate: bool = True
) -> Tuple[Optional[GrowthTrace], Dict[str, Any]]:
    """
    Convenience function to generate dataset from a single city.
    
    Args:
        city_id: City identifier
        output_dir: Output directory for dataset
        validate: Whether to validate trace via replay
        
    Returns:
        Tuple of (trace, validation_results)
    """
    generator = DatasetGenerator(output_dir)
    return generator.generate_single_city_dataset(
        city_id=city_id,
        validate_trace=validate
    )


def generate_dataset_from_cities(
    city_ids: List[str],
    output_dir: str,
    export_formats: List[str] = ['pickle', 'csv'],
    random_seed: int = 42
) -> MLDataset:
    """
    Convenience function to generate dataset from multiple cities.
    
    Args:
        city_ids: List of city identifiers
        output_dir: Output directory for dataset
        export_formats: List of export formats ('pickle', 'csv', 'npz')
        random_seed: Random seed for reproducible splits
        
    Returns:
        MLDataset object
    """
    generator = DatasetGenerator(output_dir)
    dataset = generator.generate_multi_city_dataset(
        city_ids=city_ids,
        random_seed=random_seed
    )
    generator.save_dataset(dataset, export_formats=export_formats)
    generator.print_summary()
    return dataset
