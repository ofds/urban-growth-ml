#!/usr/bin/env python3
"""
Unit tests for Configuration System.
"""

import unittest
import tempfile
import os
import sys
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inverse.core.config import (
    InferenceConfig, InferenceLimits, PerformanceConfig, StrategyConfig,
    StrategyWeights, SpatialConfig, ValidationConfig, LoggingConfig,
    StrategyType, get_default_config, create_config_from_file, save_config_to_file
)


class TestInferenceConfig(unittest.TestCase):
    """Test cases for InferenceConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()

    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = InferenceConfig()

        # Test limits
        self.assertEqual(config.limits.max_steps, 1_000_000)
        self.assertEqual(config.limits.max_blocks, 50)
        self.assertEqual(config.limits.max_candidates, 50)

        # Test performance
        self.assertTrue(config.performance.enable_performance_tracking)
        self.assertTrue(config.performance.enable_spatial_indexing)

        # Test strategies
        self.assertTrue(config.strategies.enabled_strategies['fractal_pattern'])
        self.assertTrue(config.strategies.enabled_strategies['angle_harmonization'])
        self.assertTrue(config.strategies.enabled_strategies['block_centroid'])
        self.assertFalse(config.strategies.enabled_strategies['ml_augmented'])

        # Test strategy weights
        self.assertEqual(config.strategies.weights.fractal_pattern, 1.2)
        self.assertEqual(config.strategies.weights.block_centroid, 1.3)

    def test_validation_positive_limits(self):
        """Test validation of positive limits."""
        with self.assertRaises(ValueError):
            InferenceConfig(limits=InferenceLimits(max_steps=0))

        with self.assertRaises(ValueError):
            InferenceConfig(limits=InferenceLimits(max_blocks=-1))

        with self.assertRaises(ValueError):
            InferenceConfig(limits=InferenceLimits(max_candidates=0))

    def test_validation_strategy_weights(self):
        """Test validation of strategy weights."""
        with self.assertRaises(ValueError):
            InferenceConfig(strategies=StrategyConfig(weights=StrategyWeights(fractal_pattern=-1.0)))

    def test_validation_spatial_parameters(self):
        """Test validation of spatial parameters."""
        with self.assertRaises(ValueError):
            InferenceConfig(spatial=SpatialConfig(spatial_query_radius=0))

        with self.assertRaises(ValueError):
            InferenceConfig(spatial=SpatialConfig(geometry_similarity_threshold=1.5))

    def test_validation_logging_level(self):
        """Test validation of logging level."""
        with self.assertRaises(ValueError):
            InferenceConfig(logging=LoggingConfig(log_level="INVALID"))

        # Valid levels should work
        config = LoggingConfig(log_level="DEBUG")
        self.assertEqual(config.log_level, "DEBUG")

    def test_validation_strictness(self):
        """Test validation of validation strictness."""
        with self.assertRaises(ValueError):
            InferenceConfig(validation=ValidationConfig(validation_strictness="invalid"))

        # Valid strictness should work
        config = ValidationConfig(validation_strictness="high")
        self.assertEqual(config.validation_strictness, "high")

    def test_get_enabled_strategies(self):
        """Test getting list of enabled strategies."""
        enabled = self.config.get_enabled_strategies()
        self.assertIn('fractal_pattern', enabled)
        self.assertIn('angle_harmonization', enabled)
        self.assertIn('block_centroid', enabled)
        self.assertNotIn('ml_augmented', enabled)

    def test_get_strategy_weight(self):
        """Test getting strategy weights."""
        self.assertEqual(self.config.get_strategy_weight('fractal_pattern'), 1.2)
        self.assertEqual(self.config.get_strategy_weight('nonexistent'), 1.0)

    def test_is_strategy_enabled(self):
        """Test checking if strategy is enabled."""
        self.assertTrue(self.config.is_strategy_enabled('fractal_pattern'))
        self.assertFalse(self.config.is_strategy_enabled('ml_augmented'))
        self.assertFalse(self.config.is_strategy_enabled('nonexistent'))

    def test_create_legacy_config(self):
        """Test creating legacy configuration format."""
        legacy = self.config.create_legacy_config()
        expected_keys = ['fractal_pattern', 'angle_harmonization', 'block_centroid',
                        'ml_augmented', 'multi_resolution', 'advanced_search']
        for key in expected_keys:
            self.assertIn(key, legacy)
            self.assertIsInstance(legacy[key], bool)

    def test_to_dict_and_from_dict(self):
        """Test conversion to and from dictionary."""
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('limits', config_dict)
        self.assertIn('strategies', config_dict)
        self.assertIn('performance', config_dict)

        # Test round-trip
        new_config = InferenceConfig.from_dict(config_dict)
        self.assertEqual(new_config.limits.max_steps, self.config.limits.max_steps)
        self.assertEqual(new_config.strategies.weights.fractal_pattern,
                        self.config.strategies.weights.fractal_pattern)

    def test_validate_compatibility(self):
        """Test configuration compatibility validation."""
        # Test with config that enables Phase 2A strategies
        phase2a_config = InferenceConfig(
            strategies=StrategyConfig(
                enabled_strategies={
                    'fractal_pattern': True,
                    'angle_harmonization': True,
                    'block_centroid': True,
                    'ml_augmented': True,  # Phase 2A feature
                    'multi_resolution': True,  # Phase 2A feature
                    'advanced_search': True   # Phase 2A feature
                }
            )
        )
        warnings = phase2a_config.validate_compatibility()
        # Should have warnings about Phase 2A features
        phase2a_warnings = [w for w in warnings if 'Phase 2A' in w]
        self.assertTrue(len(phase2a_warnings) > 0)

        # Test with high limits
        high_limits_config = InferenceConfig(
            limits=InferenceLimits(max_candidates=150, max_blocks=150)
        )
        warnings = high_limits_config.validate_compatibility()
        high_limit_warnings = [w for w in warnings if 'high' in w.lower()]
        self.assertTrue(len(high_limit_warnings) > 0)

    def test_log_configuration_summary(self):
        """Test logging configuration summary."""
        # Should not raise any exceptions
        with patch('inverse.core.config.logger') as mock_logger:
            self.config.log_configuration_summary()
            mock_logger.info.assert_called()

    def test_custom_configuration(self):
        """Test custom configuration creation."""
        custom_config = InferenceConfig(
            limits=InferenceLimits(max_steps=50000, max_candidates=25),
            performance=PerformanceConfig(enable_performance_tracking=False),
            strategies=StrategyConfig(
                enabled_strategies={
                    'fractal_pattern': False,
                    'angle_harmonization': True,
                    'block_centroid': True,
                    'ml_augmented': True
                },
                weights=StrategyWeights(
                    angle_harmonization=2.0,
                    block_centroid=1.5
                )
            )
        )

        # Test custom limits
        self.assertEqual(custom_config.limits.max_steps, 50000)
        self.assertEqual(custom_config.limits.max_candidates, 25)

        # Test custom performance
        self.assertFalse(custom_config.performance.enable_performance_tracking)

        # Test custom strategies
        self.assertFalse(custom_config.is_strategy_enabled('fractal_pattern'))
        self.assertTrue(custom_config.is_strategy_enabled('angle_harmonization'))
        self.assertTrue(custom_config.is_strategy_enabled('ml_augmented'))

        # Test custom weights
        self.assertEqual(custom_config.get_strategy_weight('angle_harmonization'), 2.0)
        self.assertEqual(custom_config.get_strategy_weight('block_centroid'), 1.5)


class TestConfigFileOperations(unittest.TestCase):
    """Test cases for configuration file operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = InferenceConfig()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_save_config_to_json(self):
        """Test saving configuration to JSON file."""
        json_path = os.path.join(self.temp_dir, 'config.json')
        save_config_to_file(self.config, json_path)

        self.assertTrue(os.path.exists(json_path))

        # Verify content
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.assertIn('limits', data)
        self.assertIn('strategies', data)

    def test_create_config_from_json(self):
        """Test loading configuration from JSON file."""
        json_path = os.path.join(self.temp_dir, 'config.json')
        save_config_to_file(self.config, json_path)

        loaded_config = create_config_from_file(json_path)
        self.assertIsInstance(loaded_config, InferenceConfig)
        self.assertEqual(loaded_config.limits.max_steps, self.config.limits.max_steps)

    def test_create_config_from_nonexistent_file(self):
        """Test loading from nonexistent file."""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.json')
        with self.assertRaises(FileNotFoundError):
            create_config_from_file(nonexistent_path)

    def test_invalid_file_extension(self):
        """Test loading from file with invalid extension."""
        invalid_path = os.path.join(self.temp_dir, 'config.txt')
        with open(invalid_path, 'w') as f:
            f.write('invalid')

        with self.assertRaises(ValueError):
            create_config_from_file(invalid_path)

    def test_yaml_missing_yaml(self):
        """Test YAML operations when PyYAML is not available."""
        yaml_path = os.path.join(self.temp_dir, 'config.yaml')

        # Mock yaml import to raise ImportError
        with patch.dict('sys.modules', {'yaml': None}):
            with self.assertRaises(ImportError):
                save_config_to_file(self.config, yaml_path)

            # Create a YAML file manually for loading test
            with open(yaml_path, 'w') as f:
                f.write('limits:\n  max_steps: 1000\n')

            with self.assertRaises(ImportError):
                create_config_from_file(yaml_path)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global configuration functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        self.assertIsInstance(config, InferenceConfig)
        self.assertEqual(config.limits.max_steps, 1_000_000)


class TestStrategyType(unittest.TestCase):
    """Test cases for StrategyType enum."""

    def test_strategy_type_values(self):
        """Test StrategyType enum values."""
        self.assertEqual(StrategyType.FRACTAL_PATTERN.value, 'fractal_pattern')
        self.assertEqual(StrategyType.ANGLE_HARMONIZATION.value, 'angle_harmonization')
        self.assertEqual(StrategyType.BLOCK_CENTROID.value, 'block_centroid')
        self.assertEqual(StrategyType.ML_AUGMENTED.value, 'ml_augmented')


if __name__ == '__main__':
    unittest.main()
