#!/usr/bin/env python3
"""
Dataset Generation CLI
======================

Command-line interface for generating ML-ready datasets from urban growth traces.

This script provides an easy-to-use interface for:
1. Generating datasets from one or more cities
2. Configuring validation thresholds and acceptance criteria
3. Specifying train/val/test split ratios
4. Exporting datasets in multiple formats (pickle, CSV, npz)
5. Viewing detailed generation statistics

Usage Examples
--------------

Generate dataset from a single city:
    python script/generate_datasets.py --cities piedmont_ca --output ./datasets/piedmont

Generate dataset from multiple cities:
    python script/generate_datasets.py \
        --cities piedmont_ca berkeley_ca oakland_ca \
        --output ./datasets/bay_area \
        --formats pickle csv npz

Generate with custom validation criteria:
    python script/generate_datasets.py \
        --cities piedmont_ca \
        --output ./datasets/strict \
        --min-fidelity 0.85 \
        --min-confidence 0.7 \
        --min-actions 10

Generate with custom split ratios:
    python script/generate_datasets.py \
        --cities city1 city2 city3 city4 city5 \
        --output ./datasets/custom_split \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --test-ratio 0.2

Skip validation (faster, but less reliable):
    python script/generate_datasets.py \
        --cities piedmont_ca \
        --output ./datasets/quick \
        --no-validate

Load cities from file:
    python script/generate_datasets.py \
        --cities-file cities.txt \
        --output ./datasets/from_file

Environment Variables
---------------------
URBAN_GROWTH_DATA_DIR: Base directory for city data files
URBAN_GROWTH_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inverse.dataset_generator import DatasetGenerator, generate_dataset_from_cities
from src.inverse.data_structures import MLDataset


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the script.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def load_cities_from_file(filepath: str) -> List[str]:
    """
    Load city IDs from a text file (one per line).
    
    Lines starting with # are treated as comments.
    Empty lines are ignored.
    
    Args:
        filepath: Path to text file containing city IDs
        
    Returns:
        List of city identifiers
    """
    cities = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                cities.append(line)
    return cities


def validate_split_ratios(train: float, val: float, test: float) -> bool:
    """
    Validate that split ratios sum to 1.0.
    
    Args:
        train: Training ratio
        val: Validation ratio
        test: Test ratio
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train}, val={val}, test={test})"
        )
    return True


def main():
    """Main entry point for dataset generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate ML training datasets from urban growth traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--cities',
        nargs='+',
        help='List of city identifiers to process (e.g., piedmont_ca berkeley_ca)'
    )
    input_group.add_argument(
        '--cities-file',
        type=str,
        help='Path to text file containing city IDs (one per line)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Output directory for datasets and traces'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['pickle', 'csv', 'npz'],
        default=['pickle', 'csv'],
        help='Export formats for dataset (default: pickle csv)'
    )
    
    # Validation criteria
    validation_group = parser.add_argument_group('Validation Criteria')
    validation_group.add_argument(
        '--min-fidelity',
        type=float,
        default=0.7,
        help='Minimum replay fidelity for trace acceptance (default: 0.7)'
    )
    validation_group.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence per action (default: 0.5)'
    )
    validation_group.add_argument(
        '--min-actions',
        type=int,
        default=5,
        help='Minimum number of actions in trace (default: 5)'
    )
    validation_group.add_argument(
        '--no-connectivity-check',
        action='store_true',
        help='Disable connectivity preservation requirement'
    )
    validation_group.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip trace validation via replay (faster but less reliable)'
    )
    
    # Dataset split configuration
    split_group = parser.add_argument_group('Dataset Splits')
    split_group.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Fraction of cities for training (default: 0.7)'
    )
    split_group.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Fraction of cities for validation (default: 0.15)'
    )
    split_group.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Fraction of cities for testing (default: 0.15)'
    )
    split_group.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible splits (default: 42)'
    )
    
    # Inference configuration
    inference_group = parser.add_argument_group('Inference Configuration')
    inference_group.add_argument(
        '--max-steps',
        type=int,
        default=10000,
        help='Maximum inference steps per city (default: 10000)'
    )
    
    # Logging configuration
    logging_group = parser.add_argument_group('Logging')
    logging_group.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    logging_group.add_argument(
        '--log-file',
        type=str,
        help='Optional file to write logs'
    )
    
    # Additional options
    parser.add_argument(
        '--save-rejected',
        action='store_true',
        default=True,
        help='Save rejected traces for analysis (default: True)'
    )
    parser.add_argument(
        '--save-metadata',
        action='store_true',
        default=True,
        help='Save detailed metadata and statistics (default: True)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Urban Growth ML - Dataset Generation")
    logger.info("="*60)
    
    # Load city list
    if args.cities:
        cities = args.cities
        logger.info(f"Processing {len(cities)} cities from command line")
    else:
        cities = load_cities_from_file(args.cities_file)
        logger.info(f"Loaded {len(cities)} cities from {args.cities_file}")
    
    if not cities:
        logger.error("No cities specified for processing")
        sys.exit(1)
    
    logger.info(f"Cities: {', '.join(cities)}")
    
    # Validate split ratios
    try:
        validate_split_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except ValueError as e:
        logger.error(f"Invalid split ratios: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Build acceptance criteria
    acceptance_criteria = {
        'min_replay_fidelity': args.min_fidelity,
        'min_action_confidence': args.min_confidence,
        'min_actions': args.min_actions,
        'connectivity_required': not args.no_connectivity_check
    }
    
    logger.info("Acceptance criteria:")
    for key, value in acceptance_criteria.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize dataset generator
    logger.info("Initializing dataset generator...")
    generator = DatasetGenerator(
        output_dir=str(output_dir),
        acceptance_criteria=acceptance_criteria
    )
    
    # Process cities
    logger.info(f"Processing {len(cities)} cities...")
    try:
        dataset = generator.generate_multi_city_dataset(
            city_ids=cities,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            validate_traces=not args.no_validate
        )
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}", exc_info=True)
        sys.exit(1)
    
    # Save dataset in requested formats
    logger.info(f"Saving dataset in formats: {', '.join(args.formats)}")
    try:
        generator.save_dataset(dataset, export_formats=args.formats)
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}", exc_info=True)
        sys.exit(1)
    
    # Save metadata
    if args.save_metadata:
        metadata_path = output_dir / 'dataset_metadata.json'
        metadata = {
            'cities': cities,
            'num_cities': len(cities),
            'num_accepted': len(generator.city_ids),
            'num_rejected': len(generator.rejected_traces),
            'num_samples': len(dataset.samples),
            'num_train': len(dataset.train_indices),
            'num_val': len(dataset.val_indices),
            'num_test': len(dataset.test_indices),
            'acceptance_criteria': acceptance_criteria,
            'split_ratios': {
                'train': args.train_ratio,
                'val': args.val_ratio,
                'test': args.test_ratio
            },
            'random_seed': args.random_seed,
            'feature_dim': 128,
            'action_param_dim': 16,
            'action_distribution': dataset.action_distribution,
            'export_formats': args.formats
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    # Print summary
    generator.print_summary()
    
    # Print file locations
    logger.info("\n" + "="*60)
    logger.info("Output Files")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")
    
    if 'pickle' in args.formats:
        logger.info(f"  - ml_dataset.pkl (Python pickle format)")
    
    if 'csv' in args.formats:
        csv_dir = output_dir / 'csv'
        logger.info(f"  - csv/train.csv, csv/val.csv, csv/test.csv")
    
    if 'npz' in args.formats:
        npz_dir = output_dir / 'npz'
        logger.info(f"  - npz/train.npz, npz/val.npz, npz/test.npz")
    
    logger.info(f"  - accepted_traces/ (per-city traces and samples)")
    
    if generator.rejected_traces:
        logger.info(f"  - rejected_traces/ (rejected traces for analysis)")
    
    if args.save_metadata:
        logger.info(f"  - dataset_metadata.json (generation metadata)")
    
    # Success message with basic stats
    logger.info("\n" + "="*60)
    logger.info("Dataset generation completed successfully!")
    logger.info("="*60)
    logger.info(f"Total samples: {len(dataset.samples)}")
    logger.info(f"  Train: {len(dataset.train_indices)} samples")
    logger.info(f"  Val:   {len(dataset.val_indices)} samples")
    logger.info(f"  Test:  {len(dataset.test_indices)} samples")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
