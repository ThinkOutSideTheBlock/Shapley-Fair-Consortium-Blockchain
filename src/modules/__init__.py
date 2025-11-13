"""
Modules package for Shapley-Fair Consortium Blockchain Research.

This package contains core modules for:
- Allocation: Implements various allocation rules (Shapley, Owen, Banzhaf, proportional, etc.)
- Data generation: Creates synthetic consortium game instances
- Incentive analysis: Computes strategic incentive metrics
- Simulation running: Orchestrates parameter sweeps
- Analysis: Generates figures and statistical summaries
"""

from .allocation import AllocationEngine, AllocationResult
from .data_gen import DataGenerator, GameInstance
from .incentives import IncentiveAnalyzer, IncentiveMetrics
from .runner import SimulationRunner
from .analysis import AnalysisPipeline

__all__ = [
    'AllocationEngine',
    'AllocationResult',
    'DataGenerator',
    'GameInstance',
    'IncentiveAnalyzer',
    'IncentiveMetrics',
    'SimulationRunner',
    'AnalysisPipeline'
]