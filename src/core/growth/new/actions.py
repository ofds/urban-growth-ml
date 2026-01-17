"""Growth action data contracts and type definitions.

This module defines the core action types and data structures used by the
procedural city growth engine. Actions represent validated proposals to
modify the city state (e.g., extending streets, subdividing blocks).
"""

from dataclasses import dataclass
from typing import Dict, Any
from shapely.geometry import LineString

# External contract imports
from src.core.contracts import FrontierEdge


# Action type constants
ACTION_GROW_TRAJECTORY = "grow_trajectory"
ACTION_SUBDIVIDE_BLOCK = "subdivide_block"


@dataclass(frozen=True)
class GrowthAction:
    """Represents a validated growth action proposal.
    
    Action Types:
    - grow_trajectory: Extends a dead-end frontier with a new straight street
      segment. Used when frontier_type == "dead_end". Effect: Adds 1 new street
      and 1 new node. Geometry: LineString from frontier endpoint to new endpoint.
    
    - subdivide_block: Bisects a large block with a new street. Used when
      frontier_type == "block_edge" and block area exceeds threshold. Effect:
      Adds 1 new street and splits 1 block into 2 blocks. Geometry: LineString
      crossing the block interior.
    """
    action_type: str  # ACTION_GROW_TRAJECTORY or ACTION_SUBDIVIDE_BLOCK
    frontier_edge: FrontierEdge  # The frontier being extended
    proposed_geometry: LineString  # The new street geometry
    parameters: Dict[str, Any]  # Additional action metadata


def validate_action_type(action_type: str) -> bool:
    """Validate if an action type is supported.
    
    Args:
        action_type: The action type string to validate
        
    Returns:
        True if action_type is valid, False otherwise
    """
    return action_type in (ACTION_GROW_TRAJECTORY, ACTION_SUBDIVIDE_BLOCK)
