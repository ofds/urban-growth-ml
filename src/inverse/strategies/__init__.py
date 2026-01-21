#!/usr/bin/env python3
"""
Strategies Module for Urban Growth Inference

Provides various mathematical strategies for inferring urban growth patterns.
All strategies follow the BaseInferenceStrategy interface for consistency.
"""

from .base_strategy import BaseInferenceStrategy
from .peripheral_strategy import PeripheralStrategy

__all__ = [
    'BaseInferenceStrategy',
    'PeripheralStrategy',
]
