# Urban Growth ML: Center-Out Inference & Data Quality Enhancement

## Executive Summary

This document outlines proposed improvements to the urban-growth-ml system to implement center-out urban growth inference and enhanced data quality assessment tools. The current system uses a "peeling" approach that works backward from final city states. The proposed enhancements will add geographic prioritization to create more realistic growth patterns and comprehensive quality validation frameworks.

## Project Overview

**Urban Growth ML** is a machine learning system for reverse-engineering urban street network growth patterns. The core innovation uses **inverse inference** to reconstruct historical growth sequences from final city states, creating synthetic training data for predictive urban development models.

### Key Innovation: Synthetic Supervision
- **Trace → ML Datasets**: Convert inferred growth sequences into feature-label pairs
- **Trace → Visualizations**: Generate growth movies and fidelity diagnostics
- **Research Intent**: Provide ground truth for training growth prediction models without manual annotation

## Current System Architecture

### Core Components

```
urban-growth-ml/
├── Data Pipeline: OSM → Streets → Blocks → Frontiers
├── Growth Engine: Forward simulation through discrete actions
├── Inverse Inference: Multi-strategy backward reconstruction
├── Validation: Replay engine with morphological comparison
└── Visualization: Growth movies and diagnostic plots
```

### Current Inference Approach: "Peeling" Strategy

The system currently implements backward inference through:

1. **Skeleton Extraction**: Identifies arterial grid as initial seed
2. **Peripheral Removal**: Removes dead-end and short streets first
3. **Multi-Strategy Selection**: Uses mathematical approaches:
   - Fractal Pattern Strategy (self-similar patterns)
   - Angle Harmonization Strategy (dominant angle distributions)
   - Block Centroid Strategy (incomplete block closure)

### Data Quality Framework

**Current Quality Checks:**
- OSM highway type filtering (motorway, trunk, primary, etc.)
- Street length minimums (5m threshold)
- Block area filtering (100m² minimum)
- Basic validation metrics (connectivity, monotonic decrease)

**Current Validation:**
- Replay fidelity (action success rate, street reproduction)
- Morphological comparison (original vs. replayed geometry)
- Graph connectivity checks

## Analysis of Current Limitations

### Inference Pattern Issues
1. **Non-Geographic Prioritization**: Current strategies don't consider spatial location
2. **Arbitrary Removal Order**: No center-out growth logic
3. **Limited Geographic Context**: Missing distance-from-center weighting

### Data Quality Gaps
1. **Shallow Validation**: Basic connectivity checks only
2. **Limited Geometric Assessment**: No angle distribution analysis
3. **No Quality Scoring**: Binary pass/fail without confidence metrics
4. **Poor Visualization**: Limited diagnostic tools for quality issues

## Proposed Improvements

### 1. Center-Out Inference Strategy

#### Geographic Prioritization Framework
- **City Center Calculation**: Weighted centroid based on street density and connectivity
- **Radial Distance Weighting**: Prioritize removal of peripheral streets
- **Growth Wave Propagation**: Simulate realistic urban expansion patterns

#### Implementation Approach
```python
class CenterOutInferenceStrategy(InferenceStrategy):
    def generate_candidates(self, state, skeleton_edges, spatial_index):
        # Calculate city center
        center = self._calculate_weighted_city_center(state)

        # Score streets by distance from center
        candidates = []
        for street_id, street in state.streets.iterrows():
            distance_score = self._calculate_distance_weight(
                street.geometry, center
            )
            # Higher score for peripheral streets
            confidence = distance_score * geometric_validity_score
            candidates.append((action, confidence))
```

#### Mathematical Foundation
- **Distance Weighting**: `score = 1 / (1 + distance_from_center)`
- **Connectivity Bonus**: Additional weight for well-connected peripheral streets
- **Geometric Constraints**: Maintain angle and length distribution validity

### 2. Enhanced Data Quality Assessment

#### Multi-Metric Quality Framework

**Geometric Quality Metrics:**
- **Angle Distribution Analysis**: Compare to expected urban patterns
- **Street Length Statistics**: Mean, std, outliers detection
- **Intersection Density**: Crossroad vs. dead-end ratios
- **Block Completeness**: Closed vs. open block analysis

**Topological Quality Metrics:**
- **Graph Connectivity**: Component analysis and articulation points
- **Cycle Detection**: Urban block structure validation
- **Frontier Validity**: Growth boundary geometric consistency
- **Skeleton Integrity**: Arterial network connectivity

**Morphological Quality Metrics:**
- **Pattern Recognition**: Grid, radial, organic pattern detection
- **Scale Invariance**: Fractal dimension analysis
- **Density Gradients**: Center-to-edge density analysis

#### Quality Scoring Algorithm

```python
class DataQualityAssessor:
    def assess_city_quality(self, city_state):
        scores = {
            'geometric': self._assess_geometric_quality(city_state),
            'topological': self._assess_topological_quality(city_state),
            'morphological': self._assess_morphological_quality(city_state)
        }

        # Weighted composite score
        overall_score = (
            0.4 * scores['geometric'] +
            0.4 * scores['topological'] +
            0.2 * scores['morphological']
        )

        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'recommendations': self._generate_recommendations(scores)
        }
```

### 3. Advanced Visualization Tools

#### Quality Heatmaps
- **Geometric Quality Overlay**: Color-coded streets by quality metrics
- **Topological Health Map**: Connectivity and cycle structure visualization
- **Growth Pattern Analysis**: Radial expansion pattern detection

#### Diagnostic Dashboards
- **Multi-Metric Comparison**: Side-by-side quality assessment across cities
- **Temporal Quality Tracking**: Quality evolution during inference
- **Failure Mode Analysis**: Common data quality issues and solutions

#### Enhanced Growth Animations
- **Center-Out Visualization**: Radial expansion movies with quality indicators
- **Quality-Annotated Frames**: Real-time quality metrics overlay
- **Comparative Animations**: Original vs. inferred growth patterns

## Implementation Plan

### Phase 1: Center-Out Inference Strategy (2 weeks)

**Week 1: Core Implementation**
- [ ] Implement `CenterOutInferenceStrategy` class
- [ ] Add city center calculation algorithms
- [ ] Integrate radial distance weighting
- [ ] Test on Piedmont dataset

**Week 2: Integration & Optimization**
- [ ] Integrate with existing multi-strategy framework
- [ ] Performance optimization for large cities
- [ ] Validation against known growth patterns

### Phase 2: Quality Assessment Framework (3 weeks)

**Week 3: Geometric Quality**
- [ ] Implement angle distribution analysis
- [ ] Add street length statistical validation
- [ ] Create intersection density metrics

**Week 4: Topological Quality**
- [ ] Build graph connectivity analysis
- [ ] Implement cycle detection algorithms
- [ ] Add frontier validity checks

**Week 5: Morphological Quality**
- [ ] Pattern recognition (grid, radial, organic)
- [ ] Fractal dimension calculation
- [ ] Density gradient analysis

### Phase 3: Visualization Enhancements (2 weeks)

**Week 6: Quality Visualizations**
- [ ] Implement quality heatmap overlays
- [ ] Create diagnostic dashboard components
- [ ] Add quality-annotated growth animations

**Week 7: Integration & Testing**
- [ ] Integrate visualizations with inference pipeline
- [ ] Test on multiple cities (Piedmont, Berkeley, Cambridge)
- [ ] Performance optimization and documentation

## Technical Implementation Details

### Center-Out Strategy Algorithm

```python
def _calculate_city_center(self, state):
    """Calculate weighted city center based on street density and connectivity."""
    # Method 1: Density-weighted centroid
    total_length = 0
    weighted_x = 0
    weighted_y = 0

    for idx, street in state.streets.iterrows():
        length = street.geometry.length
        centroid = street.geometry.centroid

        total_length += length
        weighted_x += centroid.x * length
        weighted_y += centroid.y * length

    if total_length > 0:
        center_x = weighted_x / total_length
        center_y = weighted_y / total_length
        return Point(center_x, center_y)

    # Fallback: geometric median of intersections
    return self._calculate_geometric_median(state)

def _calculate_distance_weight(self, geometry, center):
    """Calculate distance-based weight for center-out prioritization."""
    if hasattr(geometry, 'centroid'):
        geom_center = geometry.centroid
        distance = center.distance(geom_center)

        # Normalize by city radius
        city_radius = self._estimate_city_radius(state, center)
        normalized_distance = distance / city_radius if city_radius > 0 else 0

        # Higher weight for peripheral streets
        return normalized_distance  # 0-1 scale, higher = more peripheral
    return 0.5
```

### Quality Assessment Metrics

```python
def _assess_geometric_quality(self, city_state):
    """Comprehensive geometric quality assessment."""
    metrics = {}

    # Angle distribution analysis
    angles = self._extract_street_angles(city_state.streets)
    metrics['angle_uniformity'] = self._calculate_angle_uniformity(angles)
    metrics['dominant_angles'] = self._detect_dominant_angles(angles)

    # Length distribution analysis
    lengths = self._extract_street_lengths(city_state.streets)
    metrics['length_distribution'] = self._analyze_length_distribution(lengths)

    # Intersection analysis
    metrics['intersection_density'] = self._calculate_intersection_density(city_state)

    # Composite geometric score
    return self._combine_geometric_scores(metrics)

def _assess_topological_quality(self, city_state):
    """Graph-theoretic topological quality assessment."""
    graph = city_state.graph

    metrics = {}
    metrics['connectivity'] = nx.is_connected(graph.to_undirected())
    metrics['components'] = nx.number_connected_components(graph.to_undirected())
    metrics['avg_degree'] = sum(dict(graph.degree()).values()) / len(graph)
    metrics['articulation_points'] = len(list(nx.articulation_points(graph.to_undirected())))

    return self._combine_topological_scores(metrics)
```

## Expected Outcomes

### Inference Quality Improvements
- **More Realistic Growth Patterns**: Center-out expansion matches observed urban development
- **Better Morphological Fidelity**: Improved replay validation scores
- **Enhanced Predictive Power**: Higher quality training data for ML models

### Data Quality Benefits
- **Automated Quality Assessment**: Systematic identification of data issues
- **Confidence Scoring**: Quantified reliability metrics for each inference
- **Diagnostic Capabilities**: Clear identification of quality problems and solutions

### Visualization Enhancements
- **Quality-Aware Analysis**: Visual quality overlays for better understanding
- **Comparative Diagnostics**: Easy comparison across cities and methods
- **Research Communication**: Publication-ready growth visualizations

## Success Metrics

### Quantitative Metrics
- **Replay Fidelity**: Target >95% morphological fidelity
- **Inference Speed**: Maintain <2x slowdown vs. current approach
- **Quality Detection**: >90% accuracy in identifying data quality issues

### Qualitative Metrics
- **Growth Realism**: Expert evaluation of center-out growth patterns
- **Diagnostic Utility**: Researcher feedback on quality assessment tools
- **Visualization Clarity**: Publication suitability of output graphics

## Risk Assessment & Mitigation

### Technical Risks
- **Performance Impact**: Center-out calculations may slow inference
  - *Mitigation*: Spatial indexing and caching optimizations
- **Algorithm Complexity**: Quality assessment may be computationally expensive
  - *Mitigation*: Progressive quality checks with early exit conditions

### Implementation Risks
- **Integration Complexity**: Multiple strategy coordination
  - *Mitigation*: Modular design with clear interfaces
- **Visualization Overhead**: Additional plotting may complicate codebase
  - *Mitigation*: Separate visualization module with optional imports

## Conclusion

The proposed center-out inference strategy and enhanced data quality assessment framework will significantly improve the urban-growth-ml system's ability to generate realistic urban growth patterns and provide researchers with comprehensive tools for evaluating data quality. These improvements maintain backward compatibility while adding powerful new capabilities for urban morphology analysis and machine learning training data generation.

The implementation follows a phased approach to minimize risk and ensure each component delivers value independently, with comprehensive testing and validation at each stage.
