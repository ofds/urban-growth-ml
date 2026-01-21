#!/usr/bin/env python3
"""
Configuration System for Urban Growth Inference

Centralized configuration management following SOLID principles.
Provides type-safe configuration with validation and defaults.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Enumeration of available inference strategies."""
    FRACTAL_PATTERN = "fractal_pattern"
    ANGLE_HARMONIZATION = "angle_harmonization"
    BLOCK_CENTROID = "block_centroid"
    MORPHOLOGICAL_PATTERN = "morphological_pattern"
    CONNECTIVITY_DRIVEN = "connectivity_driven"
    ADAPTIVE_LEARNING = "adaptive_learning"
    ML_AUGMENTED = "ml_augmented"
    MULTI_RESOLUTION = "multi_resolution"
    ADVANCED_SEARCH = "advanced_search"


@dataclass
class InferenceLimits:
    """Configuration for inference operation limits."""
    max_steps: int = 1_000_000
    max_blocks: int = 50
    max_frontiers: int = 100
    max_candidates: int = 50
    max_candidates_per_strategy: int = 20
    max_streets_per_block: int = 20
    strategy_timeout_seconds: int = 30
    full_rebuild_interval: int = 25


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking and optimization."""
    enable_performance_tracking: bool = True
    max_history_size: int = 1000
    enable_spatial_indexing: bool = True
    enable_incremental_updates: bool = True
    enable_batch_caching: bool = True
    cache_interval_steps: int = 50


@dataclass
class StrategyWeights:
    """Configuration for strategy weights and priorities."""
    fractal_pattern: float = 1.2
    angle_harmonization: float = 1.1
    block_centroid: float = 1.3
    morphological_pattern: float = 1.5
    connectivity_driven: float = 1.4
    adaptive_learning: float = 1.0
    ml_augmented: float = 1.0
    multi_resolution: float = 1.0
    advanced_search: float = 1.0


@dataclass
class StrategyConfig:
    """Configuration for strategy enablement and parameters."""
    enabled_strategies: Dict[str, bool] = field(default_factory=lambda: {
        StrategyType.FRACTAL_PATTERN.value: True,
        StrategyType.ANGLE_HARMONIZATION.value: True,
        StrategyType.BLOCK_CENTROID.value: True,
        StrategyType.MORPHOLOGICAL_PATTERN.value: False,
        StrategyType.CONNECTIVITY_DRIVEN.value: False,
        StrategyType.ADAPTIVE_LEARNING.value: False,
        StrategyType.ML_AUGMENTED.value: False,      # Phase 2A
        StrategyType.MULTI_RESOLUTION.value: False,  # Phase 2A
        StrategyType.ADVANCED_SEARCH.value: False    # Phase 2A
    })

    weights: StrategyWeights = field(default_factory=StrategyWeights)

    # Strategy-specific parameters
    block_centroid_max_block_area: float = 50000.0  # m²
    block_centroid_max_missing_edges: int = 2
    fractal_pattern_min_similarity: float = 0.5
    angle_harmonization_gmm_components: int = 4


@dataclass
class SpatialConfig:
    """Configuration for spatial indexing and operations."""
    enable_rtree_indexing: bool = True
    spatial_query_radius: float = 100.0  # meters
    frontier_freshness_enabled: bool = True
    geometry_similarity_threshold: float = 0.5
    angle_tolerance_degrees: float = 15.0


@dataclass
class ValidationConfig:
    """Configuration for state validation and error handling."""
    enable_state_validation: bool = True
    enable_frontier_freshness_check: bool = True
    enable_graph_consistency_check: bool = True
    max_no_progress_steps: int = 10
    validation_strictness: str = "medium"  # "low", "medium", "high"


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging."""
    log_level: str = "INFO"
    enable_debug_logging: bool = False
    enable_performance_logging: bool = True
    log_strategy_stats: bool = True
    log_step_progress: bool = True
    progress_log_interval: int = 100


@dataclass
class InferenceConfig:
    """
    Master configuration class for urban growth inference.

    Provides centralized configuration management with validation,
    defaults, and type safety following SOLID principles.
    """
    # Core components
    limits: InferenceLimits = field(default_factory=InferenceLimits)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate limits
        if self.limits.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.limits.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if self.limits.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        # Validate strategy weights
        for strategy_name in StrategyType:
            weight = getattr(self.strategies.weights, strategy_name.value, None)
            if weight is not None and weight < 0:
                raise ValueError(f"Strategy weight for {strategy_name.value} must be non-negative")

        # Validate spatial parameters
        if self.spatial.spatial_query_radius <= 0:
            raise ValueError("spatial_query_radius must be positive")
        if not (0 <= self.spatial.geometry_similarity_threshold <= 1):
            raise ValueError("geometry_similarity_threshold must be between 0 and 1")

        # Validate logging level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.logging.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")

        # Validate validation strictness
        valid_strictness = ["low", "medium", "high"]
        if self.validation.validation_strictness.lower() not in valid_strictness:
            raise ValueError(f"validation_strictness must be one of {valid_strictness}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferenceConfig':
        """
        Create configuration from dictionary.

        Supports nested dictionary structures for complex configurations.
        """
        # Handle nested configurations
        limits_dict = config_dict.get('limits', {})
        performance_dict = config_dict.get('performance', {})
        strategies_dict = config_dict.get('strategies', {})
        spatial_dict = config_dict.get('spatial', {})
        validation_dict = config_dict.get('validation', {})
        logging_dict = config_dict.get('logging', {})

        # Extract strategy-specific configs
        enabled_strategies = strategies_dict.get('enabled_strategies', {})
        weights_dict = strategies_dict.get('weights', {})

        return cls(
            limits=InferenceLimits(**limits_dict),
            performance=PerformanceConfig(**performance_dict),
            strategies=StrategyConfig(
                enabled_strategies=enabled_strategies,
                weights=StrategyWeights(**weights_dict),
                **{k: v for k, v in strategies_dict.items()
                   if k not in ['enabled_strategies', 'weights']}
            ),
            spatial=SpatialConfig(**spatial_dict),
            validation=ValidationConfig(**validation_dict),
            logging=LoggingConfig(**logging_dict)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'limits': {
                'max_steps': self.limits.max_steps,
                'max_blocks': self.limits.max_blocks,
                'max_frontiers': self.limits.max_frontiers,
                'max_candidates': self.limits.max_candidates,
                'max_candidates_per_strategy': self.limits.max_candidates_per_strategy,
                'max_streets_per_block': self.limits.max_streets_per_block,
                'strategy_timeout_seconds': self.limits.strategy_timeout_seconds,
                'full_rebuild_interval': self.limits.full_rebuild_interval
            },
            'performance': {
                'enable_performance_tracking': self.performance.enable_performance_tracking,
                'max_history_size': self.performance.max_history_size,
                'enable_spatial_indexing': self.performance.enable_spatial_indexing,
                'enable_incremental_updates': self.performance.enable_incremental_updates,
                'enable_batch_caching': self.performance.enable_batch_caching,
                'cache_interval_steps': self.performance.cache_interval_steps
            },
            'strategies': {
                'enabled_strategies': self.strategies.enabled_strategies.copy(),
                'weights': {
                    'fractal_pattern': self.strategies.weights.fractal_pattern,
                    'angle_harmonization': self.strategies.weights.angle_harmonization,
                    'block_centroid': self.strategies.weights.block_centroid,
                    'morphological_pattern': self.strategies.weights.morphological_pattern,
                    'connectivity_driven': self.strategies.weights.connectivity_driven,
                    'adaptive_learning': self.strategies.weights.adaptive_learning,
                    'ml_augmented': self.strategies.weights.ml_augmented,
                    'multi_resolution': self.strategies.weights.multi_resolution,
                    'advanced_search': self.strategies.weights.advanced_search
                },
                'block_centroid_max_block_area': self.strategies.block_centroid_max_block_area,
                'block_centroid_max_missing_edges': self.strategies.block_centroid_max_missing_edges,
                'fractal_pattern_min_similarity': self.strategies.fractal_pattern_min_similarity,
                'angle_harmonization_gmm_components': self.strategies.angle_harmonization_gmm_components
            },
            'spatial': {
                'enable_rtree_indexing': self.spatial.enable_rtree_indexing,
                'spatial_query_radius': self.spatial.spatial_query_radius,
                'frontier_freshness_enabled': self.spatial.frontier_freshness_enabled,
                'geometry_similarity_threshold': self.spatial.geometry_similarity_threshold,
                'angle_tolerance_degrees': self.spatial.angle_tolerance_degrees
            },
            'validation': {
                'enable_state_validation': self.validation.enable_state_validation,
                'enable_frontier_freshness_check': self.validation.enable_frontier_freshness_check,
                'enable_graph_consistency_check': self.validation.enable_graph_consistency_check,
                'max_no_progress_steps': self.validation.max_no_progress_steps,
                'validation_strictness': self.validation.validation_strictness
            },
            'logging': {
                'log_level': self.logging.log_level,
                'enable_debug_logging': self.logging.enable_debug_logging,
                'enable_performance_logging': self.logging.enable_performance_logging,
                'log_strategy_stats': self.logging.log_strategy_stats,
                'log_step_progress': self.logging.log_step_progress,
                'progress_log_interval': self.logging.progress_log_interval
            }
        }

    def get_enabled_strategies(self) -> List[str]:
        """Get list of enabled strategy names."""
        return [name for name, enabled in self.strategies.enabled_strategies.items() if enabled]

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get weight for a specific strategy."""
        if hasattr(self.strategies.weights, strategy_name):
            return getattr(self.strategies.weights, strategy_name)
        return 1.0  # Default weight

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled."""
        return self.strategies.enabled_strategies.get(strategy_name, False)

    def create_legacy_config(self) -> Dict[str, Any]:
        """
        Create legacy configuration format for backward compatibility.

        Returns configuration in the format expected by MultiStrategyInferenceEngine.__init__
        """
        return {
            'fractal_pattern': self.is_strategy_enabled('fractal_pattern'),
            'angle_harmonization': self.is_strategy_enabled('angle_harmonization'),
            'block_centroid': self.is_strategy_enabled('block_centroid'),
            'ml_augmented': self.is_strategy_enabled('ml_augmented'),
            'multi_resolution': self.is_strategy_enabled('multi_resolution'),
            'advanced_search': self.is_strategy_enabled('advanced_search')
        }

    def validate_compatibility(self) -> List[str]:
        """
        Validate configuration compatibility and return warnings.

        Returns list of warning messages for potential issues.
        """
        warnings = []

        # Check for conflicting settings
        if self.performance.enable_incremental_updates and not self.performance.enable_spatial_indexing:
            warnings.append("Incremental updates enabled but spatial indexing disabled - this may cause performance issues")

        if self.validation.validation_strictness == "high" and self.limits.strategy_timeout_seconds < 60:
            warnings.append("High validation strictness with short strategy timeout may cause frequent timeouts")

        # Check for Phase 2A features that aren't fully implemented
        phase2_strategies = ['ml_augmented', 'multi_resolution', 'advanced_search']
        for strategy in phase2_strategies:
            if self.is_strategy_enabled(strategy):
                warnings.append(f"Strategy '{strategy}' is enabled but may not be fully implemented (Phase 2A feature)")

        # Check for reasonable limits
        if self.limits.max_candidates > 100:
            warnings.append("Very high max_candidates may impact performance")

        if self.limits.max_blocks > 100:
            warnings.append("Very high max_blocks may cause O(N²) scaling issues")

        return warnings

    def log_configuration_summary(self):
        """Log a summary of the current configuration."""
        enabled_strategies = self.get_enabled_strategies()

        logger.info("=== INFERENCE CONFIGURATION SUMMARY ===")
        logger.info(f"Enabled strategies ({len(enabled_strategies)}): {', '.join(enabled_strategies)}")
        logger.info(f"Limits: max_steps={self.limits.max_steps}, max_candidates={self.limits.max_candidates}")
        logger.info(f"Performance: tracking={self.performance.enable_performance_tracking}, spatial_indexing={self.performance.enable_spatial_indexing}")
        logger.info(f"Validation: strictness={self.validation.validation_strictness}")

        warnings = self.validate_compatibility()
        if warnings:
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")


# Global default configuration instance
DEFAULT_CONFIG = InferenceConfig()


def get_default_config() -> InferenceConfig:
    """Get default configuration instance."""
    return DEFAULT_CONFIG


def create_config_from_file(config_path: str) -> InferenceConfig:
    """
    Create configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        InferenceConfig instance
    """
    import os
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Determine file type and load
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML configuration files")
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError("Configuration file must be .yaml, .yml, or .json")

    return InferenceConfig.from_dict(config_dict)


def save_config_to_file(config: InferenceConfig, config_path: str):
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    config_dict = config.to_dict()

    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        except ImportError:
            raise ImportError("PyYAML required for YAML configuration files")
    elif config_path.endswith('.json'):
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError("Configuration file must be .yaml, .yml, or .json")
