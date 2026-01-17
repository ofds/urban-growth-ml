#!/usr/bin/env python3
"""
Phase 7.2 Contracts Module - Immutable Data Contracts

This module defines the formal contracts for Phase 7 data structures
with schema validation and immutability guarantees.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Literal
from shapely.geometry import LineString, Polygon
import geopandas as gpd
import networkx as nx
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrontierEdge:
    """Immutable frontier edge representation with validation."""
    frontier_id: str
    edge_id: Optional[Tuple[str, str]]
    block_id: Optional[int]
    geometry: LineString
    frontier_type: Literal['dead_end', 'block_edge']
    expansion_weight: float
    spatial_hash: str

    def __post_init__(self):
        """Validate contract invariants."""
        if not isinstance(self.geometry, LineString) or not self.geometry.is_valid:
            raise ValueError(f"FrontierEdge {self.frontier_id}: Invalid LineString geometry")

        if not self.frontier_id:
            raise ValueError("FrontierEdge: Empty frontier_id not allowed")

        # Generate and validate spatial hash
        coords_str = ",".join(f"{x:.3f},{y:.3f}" for x, y in self.geometry.coords)
        expected_hash = hashlib.sha256(coords_str.encode()).hexdigest()[:16]

        # If spatial_hash is empty or incorrect, set it correctly
        # This is the only exception to immutability - during initialization only
        if not self.spatial_hash or self.spatial_hash != expected_hash:
            object.__setattr__(self, 'spatial_hash', expected_hash)
            
        # Validate expansion weight range
        if not (0.0 <= self.expansion_weight <= 1.0):
            raise ValueError(f"FrontierEdge {self.frontier_id}: expansion_weight must be in [0, 1]")



@dataclass(frozen=True)
class GrowthState:
    """Immutable growth state snapshot with validation."""
    streets: gpd.GeoDataFrame
    blocks: gpd.GeoDataFrame
    frontiers: List[FrontierEdge]
    graph: nx.Graph
    iteration: int
    city_bounds: Polygon
    expected_iterations: int = 50

    def __post_init__(self):
        """Validate contract invariants."""
        if 'geometry' not in self.streets.columns or 'geometry' not in self.blocks.columns:
            raise ValueError("GrowthState: GeoDataFrames must have 'geometry' column")

        if not isinstance(self.city_bounds, Polygon) or not self.city_bounds.is_valid:
            raise ValueError("GrowthState: Invalid city_bounds polygon")

        if self.iteration < 0:
            raise ValueError("GrowthState: iteration must be non-negative")
        

    def _create_empty_observation(self) -> 'MLObservation':
        """Create a minimal valid observation for empty state."""
        from ..ml.state_representation import MLObservation
        
        try:
            import torch
            node_features = torch.zeros((1, 5), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_features = torch.zeros((0, 3), dtype=torch.float32)
            global_features = torch.tensor([0.0] * 7, dtype=torch.float32)
            frontier_mask = torch.zeros(0, dtype=torch.bool)
        except ImportError:
            node_features = np.zeros((1, 5), dtype=np.float32)
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_features = np.zeros((0, 3), dtype=np.float32)
            global_features = np.array([0.0] * 7, dtype=np.float32)
            frontier_mask = np.zeros(0, dtype=bool)
        
        return MLObservation(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            global_features=global_features,
            frontier_mask=frontier_mask,
            metadata={'empty': True}
        )

    def to_ml_observation(self) -> 'MLObservation':
        """
        Convert current GrowthState into an ML-ready observation.

        Returns:
            MLObservation: Structured tensors for GNN/RL training
        """
        from ..ml.state_representation import MLObservation, get_city_center, calculate_edge_angle

        # Get city center for coordinate normalization
        center = get_city_center(self.city_bounds)

        if not self.graph or len(self.graph.nodes()) == 0:
            logger.warning("to_ml_observation called on empty graph, returning zero observation")
            # Return minimal valid observation for empty state
            return self._create_empty_observation()
        
        # Extract node information
        nodes = list(self.graph.nodes(data=True))
        node_ids = [node_id for node_id, _ in nodes]
        node_positions = []
        node_degrees = []

        for node_id, node_data in nodes:
            # Get position from geometry
            geom = node_data.get('geometry')
            if geom and hasattr(geom, 'coords'):
                x, y = geom.coords[0]
            else:
                x, y = 0.0, 0.0
            node_positions.append((x, y))
            node_degrees.append(self.graph.degree[node_id])

        # Normalize coordinates
        normalized_positions = normalize_coordinates(node_positions, center)

        # Create node features [N, 5]: normalized_x, normalized_y, degree_norm, distance_norm, is_dead_end
        if node_degrees:
            max_degree = max(node_degrees)
            degree_norm = [d / max_degree if max_degree > 0 else 0.0 for d in node_degrees]
        else:
            degree_norm = []

        distances = [np.sqrt(x**2 + y**2) for x, y in normalized_positions]
        max_distance = max(distances) if distances else 1.0
        distance_norm = [d / max_distance if max_distance > 0 else 0.0 for d in distances]

        node_features = []
        for i, (x, y) in enumerate(normalized_positions):
            node_features.append([
                x,  # normalized_x
                y,  # normalized_y
                degree_norm[i],  # degree (normalized)
                distance_norm[i],  # distance from center (normalized)
                1.0 if node_degrees[i] == 1 else 0.0  # is_dead_end
            ])

        # Extract edge information
        edges = list(self.graph.edges(data=True))
        edge_index = []
        edge_features = []
        frontier_mask = []

        # Create mapping from node_id to index
        node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Get frontier edge IDs for masking
        frontier_edge_ids = set()
        for frontier in self.frontiers:
            if frontier.edge_id:
                frontier_edge_ids.add(tuple(sorted(frontier.edge_id)))

        for u, v, edge_data in edges:
            # Edge index in COO format
            edge_index.append([node_id_to_idx[u], node_id_to_idx[v]])

            # Edge features: length, angle, is_frontier
            length = edge_data.get('length', 50.0)
            angle = calculate_edge_angle(edge_data.get('geometry'))

            # Check if this edge is on frontier
            is_frontier = tuple(sorted([u, v])) in frontier_edge_ids

            edge_features.append([
                length / 100.0,  # normalize length (assume max ~100m)
                angle / np.pi,   # normalize angle to [-1, 1]
                1.0 if is_frontier else 0.0
            ])

            frontier_mask.append(is_frontier)

        # Global features [7]: iteration_norm, num_nodes, num_edges, avg_degree, total_length, num_blocks, avg_block_area
        iteration_norm = self.iteration / self.expected_iterations if self.expected_iterations > 0 else 0.0

        avg_degree = np.mean(node_degrees) if node_degrees else 0.0
        total_length = sum(length for _, _, data in edges for length in [data.get('length', 50.0)])

        num_blocks = len(self.blocks)
        block_areas = [geom.area for geom in self.blocks.geometry if hasattr(geom, 'area')]
        avg_block_area = np.mean(block_areas) if block_areas else 0.0

        global_features = [
            iteration_norm,
            len(nodes),
            len(edges),
            avg_degree,
            total_length,
            num_blocks,
            avg_block_area
        ]

        # Create tensors or numpy arrays
        try:
            import torch
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
            global_features_tensor = torch.tensor(global_features, dtype=torch.float32)
            frontier_mask_tensor = torch.tensor(frontier_mask, dtype=torch.bool)
        except ImportError:
            # Fallback to numpy arrays
            node_features_tensor = np.array(node_features, dtype=np.float32)
            edge_index_tensor = np.array(edge_index, dtype=np.int64).T
            edge_features_tensor = np.array(edge_features, dtype=np.float32)
            global_features_tensor = np.array(global_features, dtype=np.float32)
            frontier_mask_tensor = np.array(frontier_mask, dtype=bool)

        # Metadata
        metadata = {
            'node_id_to_idx': node_id_to_idx,
            'center': center,
            'city_bounds': self.city_bounds.bounds if hasattr(self.city_bounds, 'bounds') else None
        }

        return MLObservation(
            node_features=node_features_tensor,
            edge_index=edge_index_tensor,
            edge_features=edge_features_tensor,
            global_features=global_features_tensor,
            frontier_mask=frontier_mask_tensor,
            metadata=metadata
        )


@dataclass(frozen=True)
class BiasOutput:
    """Bias computation result with confidence and masking detection."""
    bias_multiplier: float
    confidence: float
    components: Dict[str, float]
    masking_detected: bool
    explanation: str

    def __post_init__(self):
        """Validate contract invariants."""
        if not (0.5 <= self.bias_multiplier <= 1.5):
            raise ValueError(f"BiasOutput: bias_multiplier {self.bias_multiplier} outside [0.5, 1.5]")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"BiasOutput: confidence {self.confidence} outside [0.0, 1.0]")


@dataclass
class Phase8Injection:
    """Regional constraints injected by Phase 8."""
    regional_biases: Dict[str, float]
    cross_city_frontiers: List[FrontierEdge]
    infrastructure_corridors: List[LineString]
    coordination_signals: Dict[str, Any]


def normalize_coordinates(coords: List[Tuple[float, float]], center: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Normalize coordinates relative to center with optional scaling.

    Args:
        coords: List of (x, y) coordinates
        center: (x, y) center point

    Returns:
        List of normalized (x, y) coordinates
    """
    if not coords:
        return []
    
    center_x, center_y = center
    normalized = [(x - center_x, y - center_y) for x, y in coords]
    
    # Optional: scale to [-1, 1] range for better ML training
    max_distance = max(np.sqrt(x**2 + y**2) for x, y in normalized)
    if max_distance > 1e-6:
        normalized = [(x / max_distance, y / max_distance) for x, y in normalized]
    
    return normalized


# Validation functions
def validate_frontier_edge_schema(frontier: FrontierEdge) -> bool:
    """Validate FrontierEdge against schema requirements."""
    try:
        # Trigger __post_init__ validation
        frontier.__post_init__()
        return True
    except ValueError:
        return False


def validate_growth_state_schema(state: GrowthState) -> bool:
    """Validate GrowthState against schema requirements."""
    try:
        # Trigger __post_init__ validation
        state.__post_init__()
        return True
    except ValueError:
        return False


def validate_bias_output_schema(output: BiasOutput) -> bool:
    """Validate BiasOutput against schema requirements."""
    try:
        # Trigger __post_init__ validation
        output.__post_init__()
        return True
    except ValueError:
        return False