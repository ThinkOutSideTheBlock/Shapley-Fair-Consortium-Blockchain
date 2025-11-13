# =============================================================================
# FILE: src/modules/runner.py
"""
Simulation Engine - Orchestrates parameter sweeps with robust checkpointing

New Features:
- Incremental checkpointing (save every N runs)
- Resume from partial checkpoint
- Parallel execution with ProcessPoolExecutor
- Progress tracking with tqdm

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .data_gen import DataGenerator, GameInstance
from .allocation import AllocationEngine, AllocationResult
from .incentives import IncentiveAnalyzer, IncentiveMetrics

logger = logging.getLogger(__name__)


class SimulationRunner:
    """
    Executes large-scale simulation sweeps with parallelization

    Features:
    - Incremental checkpointing (resume capability)
    - Parallel execution (multi-process)
    - Error handling and logging
    - Multiple output formats (CSV, pickle, JSON)
    """

    def __init__(self, output_dir: str, seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.results = []

    def run_single_iteration(
        self,
        run_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single Monte Carlo iteration

        Parameters from run_config:
        - run_idx, n_agents, mu, sigma_q, sigma_report
        - detection_prob, penalty, payoff_shape, allocation_method
        - seed, n_iterations, n_mc_samples
        - compute_nash (optional), check_core_lp (optional)
        """

        seed = run_config['seed']

        # Initialize modules with seeded RNG
        data_gen = DataGenerator(seed=seed)
        alloc_engine = AllocationEngine(seed=seed)
        incentive_analyzer = IncentiveAnalyzer(seed=seed)

        # Generate game instance
        game = data_gen.generate_instance(
            n_agents=run_config['n_agents'],
            mu=run_config['mu'],
            sigma_q=run_config['sigma_q'],
            sigma_report=run_config['sigma_report'],
            payoff_shape=run_config['payoff_shape'],
            q_dist=run_config.get('q_dist', 'lognormal'),
            report_model=run_config.get('report_model', 'truthful'),
            alpha=run_config.get('alpha', 1.0),
            beta=run_config.get('beta', 0.1),
            gamma=run_config.get('gamma', 0.8)
        )

        # Compute allocation
        alloc_result = alloc_engine.allocate(
            method=run_config['allocation_method'],
            n_agents=game.n_agents,
            payoff_function=game.payoff_function,
            reports=game.q_reported,
            weights=run_config.get('weights', None),
            n_samples=run_config.get('n_mc_samples', 10000)
        )

        # Compute incentive metrics
        incentive_metrics = incentive_analyzer.compute_metrics(
            game_instance=game,
            allocation_result=alloc_result,
            allocation_engine=alloc_engine,
            detection_prob=run_config['detection_prob'],
            penalty=run_config['penalty'],
            compute_nash=run_config.get('compute_nash', False),
            check_core_lp=run_config.get('check_core_lp', False)
        )

        # Package results
        result = {
            # Config
            'run_idx': run_config['run_idx'],
            'seed': seed,
            'n_agents': run_config['n_agents'],
            'mu': run_config['mu'],
            'sigma_q': run_config['sigma_q'],
            'sigma_report': run_config['sigma_report'],
            'detection_prob': run_config['detection_prob'],
            'penalty': run_config['penalty'],
            'payoff_shape': run_config['payoff_shape'],
            'allocation_method': run_config['allocation_method'],
            'q_dist': run_config.get('q_dist', 'lognormal'),
            'report_model': run_config.get('report_model', 'truthful'),

            # Game characteristics
            'q_true_mean': float(np.mean(game.q_true)),
            'q_true_std': float(np.std(game.q_true)),
            'q_true_gini': game.metadata['q_true_gini'],
            'total_value': alloc_result.metadata['total_value'],

            # Allocation results
            'allocations_mean': float(np.mean(alloc_result.allocations)),
            'allocations_std': float(np.std(alloc_result.allocations)),
            'allocations_min': float(np.min(alloc_result.allocations)),
            'allocations_max': float(np.max(alloc_result.allocations)),

            # Monte Carlo convergence (if applicable)
            'n_samples_used': alloc_result.n_samples_used,
            'converged': alloc_result.converged,

            # Incentive metrics
            'itm_score': incentive_metrics.incentive_to_misreport,
            'avg_max_gain': incentive_metrics.avg_max_gain,
            'max_max_gain': incentive_metrics.max_max_gain,
            'gini_allocation': incentive_metrics.gini_coefficient,
            'entropy_normalized': incentive_metrics.normalized_entropy,
            'envy_rate': incentive_metrics.envy_rate,
            'fairreward_composite': incentive_metrics.fairreward_composite,  # NEW
            'stability_index': incentive_metrics.stability_index,
            'num_profitable_coalitions': incentive_metrics.num_profitable_coalitions,

            # Core membership (if checked)
            'in_core': incentive_metrics.in_core,
            'core_epsilon': incentive_metrics.core_epsilon,

            # Nash equilibrium (if computed)
            'nash_converged': incentive_metrics.nash_converged,
            'nash_deviation_from_truth': incentive_metrics.nash_deviation_from_truth,

            # Timestamp
            'timestamp': datetime.now().isoformat()
        }

        return result

    def run_grid(
        self,
        run_manifest: List[Dict[str, Any]],
        parallel_workers: int = 4,
        checkpoint_interval: int = 100,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Execute full parameter grid with parallelization and checkpointing

        Parameters:
        -----------
        run_manifest : list of dict
            List of run configurations (each dict has parameters for one run)
        parallel_workers : int
            Number of parallel processes (set to 1 for serial execution)
        checkpoint_interval : int
            Save checkpoint every N runs
        resume : bool
            Resume from checkpoint if exists
        """

        checkpoint_path = self.output_dir / 'checkpoint.pkl'

        # Resume if requested
        if resume and checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
                completed_indices = set(checkpoint_data['completed_indices'])
                self.results = checkpoint_data['results']
        else:
            completed_indices = set()
            self.results = []

        # Filter manifest for incomplete runs
        runs_to_execute = [
            run for run in run_manifest
            if run['run_idx'] not in completed_indices
        ]

        if len(runs_to_execute) == 0:
            logger.info("All runs already completed!")
            return pd.DataFrame(self.results)

        logger.info(
            f"Executing {len(runs_to_execute)} runs with {parallel_workers} workers"
        )

        # Execute with progress bar
        if parallel_workers > 1:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(self.run_single_iteration, run): run['run_idx']
                    for run in runs_to_execute
                }

                with tqdm(total=len(runs_to_execute), desc="Simulation Progress") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            self.results.append(result)
                            completed_indices.add(result['run_idx'])

                            # Checkpoint periodically
                            if len(self.results) % checkpoint_interval == 0:
                                self._save_checkpoint(completed_indices)

                            pbar.update(1)

                        except Exception as e:
                            run_idx = futures[future]
                            logger.error(
                                f"Run {run_idx} failed: {e}", exc_info=True)
                            pbar.update(1)
        else:
            # Serial execution
            for run in tqdm(runs_to_execute, desc="Simulation Progress"):
                try:
                    result = self.run_single_iteration(run)
                    self.results.append(result)
                    completed_indices.add(result['run_idx'])

                    if len(self.results) % checkpoint_interval == 0:
                        self._save_checkpoint(completed_indices)

                except Exception as e:
                    logger.error(
                        f"Run {run['run_idx']} failed: {e}", exc_info=True)

        # Final save
        self._save_checkpoint(completed_indices)
        df_results = pd.DataFrame(self.results)
        self._save_results(df_results)

        logger.info(f"Completed {len(self.results)} runs")
        return df_results

    def _save_checkpoint(self, completed_indices: set):
        """Save checkpoint for resume capability"""
        checkpoint_path = self.output_dir / 'checkpoint.pkl'
        checkpoint_data = {
            'completed_indices': list(completed_indices),
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved: {len(self.results)} runs completed")

    def _save_results(self, df: pd.DataFrame):
        """Save final results in multiple formats"""
        # CSV for easy viewing
        csv_path = self.output_dir / 'results.csv'
        df.to_csv(csv_path, index=False)

        # Pickle for Python analysis
        pkl_path = self.output_dir / 'results.pkl'
        df.to_pickle(pkl_path)

        # JSON for interoperability
        json_path = self.output_dir / 'results.json'
        df.to_json(json_path, orient='records', indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"  - CSV: {csv_path}")
        logger.info(f"  - Pickle: {pkl_path}")
        logger.info(f"  - JSON: {json_path}")
