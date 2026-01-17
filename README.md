# Urban Growth ML

A machine learning system for reverse-engineering urban street network growth patterns from real cities, using inverse inference to create synthetic training data for predictive modeling.

## Problem Statement

Urban planning and city simulation require understanding how cities grow organically. Traditional approaches simulate growth forward from seeds, but real cities have complex histories. This project solves the inverse problem: given a final city state, reconstruct its growth history to create ML training data for predicting future urban development.

## Original Intent

The core innovation is using **growth traces as synthetic supervision** for machine learning:

- **Trace → ML Datasets**: Convert inferred growth sequences into feature-label pairs for supervised learning
- **Trace → Visualizations**: Generate growth movies and fidelity diagnostics for understanding and debugging

The inverse pipeline (skeleton extraction → inference → replay) achieves **100% fidelity** on real cities, providing ground truth for training growth prediction models.

## Core Capabilities

### Growth Engine (Forward Simulation)
Procedural generation system that simulates urban development through discrete growth actions.

- **Action Types**: Street extensions, block subdivisions, and frontier expansions
- **Frontier-Based Growth**: Maintains active growth boundaries for realistic expansion patterns
- **Deterministic Replay**: Reproduces exact city states from action sequences
- **Validation**: Ensures topological validity (connectivity, intersections, blocks)

### Inverse Inference Engine
Reconstructs historical growth sequences from final city states using morphological analysis.

- **Skeleton Extraction**: Identifies initial street grid that bootstrapped growth
- **Chronological Ordering**: Infers temporal sequence of street additions
- **Action Reconstruction**: Generates precise growth actions (extensions, connections) for each street
- **Confidence Scoring**: Assigns reliability scores based on geometric and topological evidence

### Trace Replay Validation
Validates inference quality by replaying growth sequences through the forward engine.

- **Action Execution**: Applies inferred actions sequentially to initial state
- **Morphological Comparison**: Measures geometric and topological fidelity
- **Fidelity Metrics**: Street-level accuracy, block preservation, frontier matching
- **Diagnostic Visualizations**: Side-by-side comparisons and trace summaries

### Feature Extraction & Dataset Generation
Creates ML-ready training datasets from validated growth traces.

- **State Features**: Spatial context, density metrics, frontier configurations
- **Action Labels**: Encoded growth decisions with parameters and confidence scores
- **Multi-City Support**: Processes diverse urban morphologies for robust training
- **Train/Val/Test Splits**: Balanced datasets with proper cross-validation

### Visualizations
Generates diagnostic visualizations for growth analysis and replay validation.

- **Growth Movies**: Animated sequences showing city evolution over time
- **Replay Comparisons**: Original vs. replayed city overlays with difference analysis
- **Trace Summaries**: Action timelines, confidence distributions, and type breakdowns

## Architecture & Structure

```
urban-growth-ml/
├── src/
│   ├── core/
│   │   ├── contracts.py              # Core data structures (GrowthState, FrontierEdge)
│   │   └── growth/new/
│   │       ├── growth_engine.py      # Main GrowthEngine API
│   │       ├── actions.py            # Growth action definitions
│   │       ├── state_updater.py      # State evolution logic
│   │       └── validators.py         # Action validation rules
│   ├── inverse/
│   │   ├── inference.py              # BasicInferenceEngine (backward reconstruction)
│   │   ├── replay.py                 # TraceReplayEngine (forward validation)
│   │   ├── rewind.py                 # Action reversal logic
│   │   ├── skeleton.py               # Initial skeleton extraction
│   │   ├── data_structures.py        # GrowthTrace, InverseGrowthAction
│   │   ├── feature_extractor.py      # ML feature generation
│   │   ├── dataset_generator.py      # Training dataset creation
│   │   ├── validation.py             # Morphological validation
│   │   └── visualization.py          # Diagnostic plots
│   └── utils/
│       └── osm_processor.py          # OSM data extraction
├── tests/                            # Test suite with real city validation
├── notebooks/                        # Jupyter exploration notebooks
├── script/                           # Data processing scripts
└── outputs/                          # Generated datasets and visualizations
```

## Pipeline Stages

### 1. Loading
Load real city street networks from processed OSM data.
- Parse street geometries, topologies, and metadata
- Build spatial graph representations
- Identify blocks and growth frontiers

### 2. Inference
Reconstruct growth history from final city state.
- Extract arterial skeleton as initial seed
- Infer chronological street addition sequence
- Generate action traces with confidence scores

### 3. Replay
Validate inference by replaying actions forward.
- Execute actions through growth engine
- Compare replayed geometry to original
- Compute fidelity metrics and generate diagnostics

### 4. Dataset Generation
Create ML training datasets from validated traces.
- Extract features from each growth state
- Label with inferred actions as ground truth
- Generate balanced train/validation/test splits

### 5. Visualization
Produce growth movies and validation diagnostics.
- Create animated city evolution sequences
- Generate replay comparison plots
- Produce trace summary statistics

## Usage

### Run Full Pipeline on Real City (Piedmont, CA)
```python
from core.growth.new.growth_engine import GrowthEngine
from inverse.inference import BasicInferenceEngine
from inverse.replay import TraceReplayEngine

# Load city data
engine = GrowthEngine('piedmont_ca', seed=42)
city = engine.load_initial_state()

# Infer growth trace
inference = BasicInferenceEngine()
trace = inference.infer_trace(city)

# Validate with replay
replay_engine = TraceReplayEngine()
validation = replay_engine.validate_trace_replay(
    trace=trace,
    original_state=city,
    city_name='piedmont_ca'
)

print(f"Actions replayed: {validation['replay_actions']}/{validation['trace_actions']}")
print(f"Morphological fidelity: {validation['replay_fidelity']:.2f}")
```

### Generate ML Dataset from Traces
```python
from inverse.dataset_generator import DatasetGenerator

# Create generator with acceptance criteria
criteria = {
    'min_replay_fidelity': 0.8,
    'min_action_confidence': 0.6,
    'connectivity_required': True
}
generator = DatasetGenerator(output_dir='outputs/datasets', acceptance_criteria=criteria)

# Add validated traces
generator.add_trace('piedmont_ca', trace, city, validation)
# ... add more cities ...

# Save dataset with train/val/test splits
generator.save_dataset()
```

### Generate Growth/Replay Visualizations
```python
from inverse.visualization import InverseGrowthVisualizer

visualizer = InverseGrowthVisualizer()

# Create replay comparison (automatic during validation)
comparison_path = visualizer.create_replay_comparison(
    original_state=city,
    replayed_state=replayed_state,
    validation_results=validation,
    trace_metadata=trace.metadata,
    filename="piedmont_replay_validation.png"
)

# Create trace summary
summary_path = visualizer.create_trace_summary_visualization(
    trace=trace,
    filename="piedmont_trace_summary.png"
)
```

## Research Intent & Design Philosophy

Growth traces serve as **synthetic supervision** for machine learning:

- **Synthetic Labels**: Inferred actions provide ground truth for training growth predictors without manual annotation
- **Visual Diagnostics**: Growth movies and replay comparisons enable qualitative assessment of model performance
- **Scalable Training**: Real city traces generate diverse training data for robust ML models

The inverse-forward pipeline ensures **trace validity**: only traces that replay perfectly become training data, guaranteeing high-quality supervision.

## Validation Results

**Piedmont, CA (1,404 streets)**:
- ✅ **Inference**: 1,402 actions inferred (100% coverage, 0.80 avg confidence)
- ✅ **Replay**: 100% action success rate, 100% street reproduction, 1.00 morphological fidelity
- ✅ **Status**: READY FOR ML TRAINING

## Setup & Requirements

### Dependencies
```
python >= 3.8
geopandas >= 0.10.0
networkx >= 2.6
shapely >= 1.8.0
matplotlib >= 3.4.0
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
```

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation
Extract city data from OpenStreetMap:
```bash
python script/01_extract_piedmont.py
```

### Run Tests
```bash
# Full pipeline test on Piedmont
python tests/test_piedmont_full_pipeline.py

# Run all tests
python run_tests.py
# or
python -m pytest tests/
```
