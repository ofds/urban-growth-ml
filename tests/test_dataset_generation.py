#!/usr/bin/env python3
"""Test dataset generation pipeline from traces to ML-ready datasets."""

import sys
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.inverse.dataset_generator import DatasetGenerator, generate_dataset_from_city
from src.inverse.data_structures import StateActionSample, MLDataset, ActionType, GrowthTrace, InverseGrowthAction
from src.core.contracts import GrowthState
from src.core.growth.new.growth_engine import GrowthEngine


class TestDatasetGenerator(unittest.TestCase):
    """Test DatasetGenerator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = DatasetGenerator(str(self.temp_dir))

        # Create mock city state
        self.mock_state = self._create_mock_growth_state()

        # Create mock trace
        self.mock_trace = self._create_mock_growth_trace()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_mock_growth_state(self) -> GrowthState:
        """Create a minimal mock GrowthState for testing."""
        # This is a simplified mock - in real tests you'd use actual data
        return MagicMock(spec=GrowthState)

    def _create_mock_growth_trace(self) -> GrowthTrace:
        """Create a mock GrowthTrace with sample actions."""
        actions = [
            InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id="frontier_1",
                intent_params={'direction': 'peripheral_expansion', 'strategy': 'short_segment'},
                confidence=0.8,
                state_diff={
                    'added_streets': [{
                        'u': 1, 'v': 2,
                        'geometry_wkt': 'LINESTRING (0 0, 10 0)',
                        'highway': 'residential'
                    }],
                    'removed_streets': [],
                    'graph_changes': {},
                    'frontier_changes': {}
                }
            ),
            InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id="frontier_2",
                intent_params={'direction': 'central_growth', 'strategy': 'connectivity'},
                confidence=0.9,
                state_diff={
                    'added_streets': [{
                        'u': 3, 'v': 4,
                        'geometry_wkt': 'LINESTRING (10 0, 20 0)',
                        'highway': 'primary'
                    }],
                    'removed_streets': [],
                    'graph_changes': {},
                    'frontier_changes': {}
                }
            )
        ]

        return GrowthTrace(
            actions=actions,
            initial_state=self.mock_state,
            final_state=self.mock_state,
            metadata={'test_trace': True}
        )

    def test_initialization(self):
        """Test DatasetGenerator initialization."""
        self.assertIsInstance(self.generator, DatasetGenerator)
        self.assertEqual(self.generator.criteria['min_replay_fidelity'], 0.7)
        self.assertTrue(self.generator.criteria['connectivity_required'])
        self.assertEqual(len(self.generator.accepted_samples), 0)
        self.assertEqual(len(self.generator.rejected_traces), 0)
        self.assertEqual(len(self.generator.city_ids), 0)

    def test_encode_action_type(self):
        """Test action type encoding."""
        self.assertEqual(self.generator._encode_action_type(ActionType.EXTEND_FRONTIER), 0)
        self.assertEqual(self.generator._encode_action_type(ActionType.SUBDIVIDE_BLOCK), 1)
        self.assertEqual(self.generator._encode_action_type(ActionType.REALIGN_STREET), 2)
        self.assertEqual(self.generator._encode_action_type(ActionType.REMOVE_STREET), 3)

    def test_encode_action_params(self):
        """Test action parameter encoding."""
        action = InverseGrowthAction(
            action_type=ActionType.EXTEND_FRONTIER,
            target_id="test",
            intent_params={
                'direction': 'peripheral_expansion',
                'strategy': 'short_segment',
                'angle': 45.0,
                'length': 100.0
            },
            confidence=0.8
        )

        params = self.generator._encode_action_params(action)

        self.assertEqual(len(params), 16)
        self.assertEqual(params[0], 1.0)  # peripheral_expansion
        self.assertEqual(params[2], 1.0)  # short_segment
        self.assertEqual(params[4], 45.0 / 360.0)  # normalized angle
        self.assertEqual(params[5], 100.0 / 1000.0)  # normalized length

    def test_meets_criteria(self):
        """Test acceptance criteria evaluation."""
        # Test valid case with relaxed criteria for testing
        generator = DatasetGenerator(str(self.temp_dir), {
            'min_replay_fidelity': 0.7,
            'connectivity_required': True,
            'min_actions': 2,  # Relaxed for test
            'min_action_confidence': 0.5
        })
        valid_validation = {
            'replay_fidelity': 0.8,
            'connectivity_preserved': True
        }
        self.assertTrue(generator.meets_criteria(valid_validation, self.mock_trace))

        # Test low fidelity
        invalid_validation = {
            'replay_fidelity': 0.5,
            'connectivity_preserved': True
        }
        self.assertFalse(self.generator.meets_criteria(invalid_validation, self.mock_trace))

        # Test connectivity not preserved
        invalid_validation2 = {
            'replay_fidelity': 0.8,
            'connectivity_preserved': False
        }
        self.assertFalse(self.generator.meets_criteria(invalid_validation2, self.mock_trace))

    @patch('src.inverse.dataset_generator.GrowthEngine')
    def test_generate_single_city_dataset_loading_failure(self, mock_engine_class):
        """Test handling of city loading failure."""
        mock_engine_class.side_effect = Exception("Loading failed")

        trace, validation = self.generator.generate_single_city_dataset("nonexistent_city")

        self.assertIsNone(trace)
        self.assertIn('error', validation)
        self.assertEqual(validation['stage'], 'loading')

    def test_generate_single_city_dataset_validation_failure(self):
        """Test handling of validation failure."""
        # Mock successful loading and inference
        with patch.object(self.generator, 'inference_engine') as mock_inference:
            mock_inference.infer_trace.return_value = self.mock_trace

            # Mock replay engine validate_trace_replay method
            with patch.object(self.generator.replay_engine, 'validate_trace_replay') as mock_validate:
                mock_validate.return_value = {
                    'replay_fidelity': 0.3,  # Below threshold
                    'connectivity_preserved': True,
                    'success': False
                }

                trace, validation = self.generator.generate_single_city_dataset(
                    "test_city",
                    final_state=self.mock_state,
                    validate_trace=True
                )

                self.assertIsNone(trace)  # Should be rejected
                self.assertEqual(validation['replay_fidelity'], 0.3)

    def test_add_trace(self):
        """Test adding a validated trace to the dataset."""
        # Mock feature extractor
        with patch.object(self.generator, 'feature_extractor') as mock_extractor:
            mock_extractor.extract_features.return_value = np.random.randn(128)

            # Add trace
            self.generator.add_trace("test_city", self.mock_trace, self.mock_state, {
                'replay_fidelity': 0.9,
                'connectivity_preserved': True
            })

            # Check that samples were added
            self.assertEqual(len(self.generator.accepted_samples), 2)  # 2 actions in mock trace
            self.assertIn("test_city", self.generator.city_ids)

            # Check sample properties
            sample = self.generator.accepted_samples[0]
            self.assertIsInstance(sample, StateActionSample)
            self.assertEqual(sample.city_id, "test_city")
            self.assertEqual(sample.action_type, 0)  # EXTEND_FRONTIER
            self.assertEqual(len(sample.state_features), 128)
            self.assertEqual(len(sample.action_params), 16)
            self.assertEqual(sample.confidence, 0.8)

    def test_create_dataset_splits(self):
        """Test creation of train/val/test splits."""
        # Add some mock samples from different cities
        cities = ['city1', 'city2', 'city3', 'city4', 'city5']

        for i, city in enumerate(cities):
            sample = StateActionSample(
                state_features=np.random.randn(128),
                action_type=0,
                action_params=np.random.randn(16),
                confidence=0.8,
                city_id=city,
                step_index=0
            )
            self.generator.accepted_samples.append(sample)
            self.generator.city_ids.add(city)

        # Create splits
        dataset = self.generator._create_dataset_splits(0.6, 0.2, 0.2)

        self.assertIsInstance(dataset, MLDataset)
        self.assertEqual(len(dataset.samples), 5)

        # Check that splits are disjoint and cover all samples
        all_indices = set(dataset.train_indices + dataset.val_indices + dataset.test_indices)
        self.assertEqual(len(all_indices), 5)

        # Check that cities are properly segregated
        train_cities = set(dataset.samples[i].city_id for i in dataset.train_indices)
        val_cities = set(dataset.samples[i].city_id for i in dataset.val_indices)
        test_cities = set(dataset.samples[i].city_id for i in dataset.test_indices)

        self.assertEqual(len(train_cities.intersection(val_cities)), 0)
        self.assertEqual(len(train_cities.intersection(test_cities)), 0)
        self.assertEqual(len(val_cities.intersection(test_cities)), 0)

    def test_save_dataset_formats(self):
        """Test saving dataset in different formats."""
        # Create a small dataset
        sample = StateActionSample(
            state_features=np.random.randn(128),
            action_type=0,
            action_params=np.random.randn(16),
            confidence=0.8,
            city_id="test_city",
            step_index=0
        )
        self.generator.accepted_samples.append(sample)
        self.generator.city_ids.add("test_city")

        dataset = self.generator._create_dataset_splits(1.0, 0.0, 0.0)

        # Test pickle format
        self.generator.save_dataset(dataset, export_formats=['pickle'])
        pickle_path = self.temp_dir / 'ml_dataset.pkl'
        self.assertTrue(pickle_path.exists())

        # Test CSV format
        self.generator.save_dataset(dataset, export_formats=['csv'])
        csv_dir = self.temp_dir / 'csv'
        self.assertTrue(csv_dir.exists())
        self.assertTrue((csv_dir / 'train.csv').exists())

        # Test npz format
        self.generator.save_dataset(dataset, export_formats=['npz'])
        npz_dir = self.temp_dir / 'npz'
        self.assertTrue(npz_dir.exists())
        self.assertTrue((npz_dir / 'train.npz').exists())

    def test_print_summary(self):
        """Test summary printing (just ensure it doesn't crash)."""
        # Add some test data
        sample = StateActionSample(
            state_features=np.random.randn(128),
            action_type=0,
            action_params=np.random.randn(16),
            confidence=0.8,
            city_id="test_city",
            step_index=0
        )
        self.generator.accepted_samples.append(sample)
        self.generator.city_ids.add("test_city")

        # This should not raise an exception
        try:
            self.generator.print_summary()
        except Exception as e:
            self.fail(f"print_summary() raised an exception: {e}")

    def test_convenience_functions(self):
        """Test convenience functions."""
        with patch('src.inverse.dataset_generator.DatasetGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            mock_generator.generate_single_city_dataset.return_value = (self.mock_trace, {})

            # Test generate_dataset_from_city
            trace, validation = generate_dataset_from_city("test_city", "/tmp")

            mock_generator.generate_single_city_dataset.assert_called_once_with(
                city_id="test_city",
                validate_trace=True
            )


class TestDatasetIntegration(unittest.TestCase):
    """Integration tests for dataset generation pipeline."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.core.growth.new.growth_engine.GrowthEngine')
    def test_full_pipeline_integration(self, mock_engine_class):
        """Test full dataset generation pipeline integration."""
        # Mock the growth engine
        mock_engine = MagicMock()
        mock_engine.load_initial_state.return_value = self._create_mock_state()
        mock_engine_class.return_value = mock_engine

        # Mock validation
        with patch.object(generator.replay_engine, 'validate_trace_replay') as mock_validate:
            mock_validate.return_value = {
                'replay_fidelity': 0.85,
                'connectivity_preserved': True,
                'morphological_valid': True
            }

        # Create generator and run pipeline
        generator = DatasetGenerator(str(self.temp_dir))

        # Mock inference to return our test trace
        with patch.object(generator, 'inference_engine') as mock_inference:
            mock_inference.infer_trace.return_value = self._create_test_trace()

            # Mock feature extraction
            with patch.object(generator, 'feature_extractor') as mock_extractor:
                mock_extractor.extract_features.return_value = np.random.randn(128)

                # Run the pipeline
                trace, validation = generator.generate_single_city_dataset(
                    "piedmont_ca",
                    validate_trace=True
                )

                # Verify results
                self.assertIsNotNone(trace)
                self.assertEqual(validation['replay_fidelity'], 0.85)
                self.assertEqual(len(generator.accepted_samples), 2)  # 2 actions in test trace

    def _create_mock_state(self):
        """Create a mock GrowthState for integration tests."""
        return MagicMock(spec=GrowthState)

    def _create_test_trace(self):
        """Create a test trace with realistic actions."""
        actions = [
            InverseGrowthAction(
                action_type=ActionType.EXTEND_FRONTIER,
                target_id="f1",
                intent_params={'direction': 'peripheral_expansion'},
                confidence=0.8,
                state_diff={'added_streets': [], 'removed_streets': []}
            ),
            InverseGrowthAction(
                action_type=ActionType.SUBDIVIDE_BLOCK,
                target_id="b1",
                intent_params={'strategy': 'connectivity'},
                confidence=0.7,
                state_diff={'added_streets': [], 'removed_streets': []}
            )
        ]

        return GrowthTrace(
            actions=actions,
            initial_state=self._create_mock_state(),
            final_state=self._create_mock_state()
        )


if __name__ == '__main__':
    unittest.main()
