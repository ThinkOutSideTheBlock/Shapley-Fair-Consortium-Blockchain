"""
Shapley-Fair Consortium Blockchain Research Package.

This package provides implementations for:
- Game-theoretic allocation mechanisms in consortium blockchains
- Incentive analysis and strategic behavior modeling
- Simulation and analysis tools for research
"""

# Import core modules for easy access
from .modules import (
    AllocationEngine,
    AllocationResult,
    DataGenerator,
    GameInstance,
    IncentiveAnalyzer,
    IncentiveMetrics,
    SimulationRunner,
    AnalysisPipeline
)
from .utils import ExperimentLogger

__all__ = [
    # Core modules
    'AllocationEngine',
    'AllocationResult',
    'DataGenerator',
    'GameInstance',
    'IncentiveAnalyzer',
    'IncentiveMetrics',
    'SimulationRunner',
    'AnalysisPipeline',
    'ExperimentLogger'
]

__version__ = "1.0.0"
__author__ = "Shapley-Fair Consortium Blockchain Research"