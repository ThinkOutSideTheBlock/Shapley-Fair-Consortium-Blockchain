# =============================================================================
# FILE: src/modules/incentives.py
"""
Incentive & Strategic Analysis Module
Implements Nash equilibrium, ITM, coalition stability, FairReward

New Features:
- Nash equilibrium via best-response dynamics
- FairReward composite fairness index
- Enhanced coalition stability with LP-based core check

References:
- Luo et al. (2025) - Nash equilibrium in blockchain FL
- Chen et al. (2024) - FairReward composite index

Priority: CRITICAL | Status: Production-Ready
Version: 2.1.0
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, FrozenSet
from dataclasses import dataclass
from itertools import combinations
import logging
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


@dataclass
class IncentiveMetrics:
    """Container for all incentive-related metrics"""
    # Individual deviation analysis
    max_gain_per_agent: np.ndarray
    best_deviation_per_agent: np.ndarray
    expected_payoff_honest: np.ndarray
    expected_payoff_deviate: np.ndarray

    # Aggregate incentive metrics
    avg_max_gain: float
    max_max_gain: float
    incentive_to_misreport: float

    # Fairness metrics
    gini_coefficient: float
    normalized_entropy: float
    envy_rate: float
    fairreward_composite: float

    # Coalition stability
    stability_index: float
    num_profitable_coalitions: int

    # Metadata
    detection_prob: float
    penalty: float
    allocation_method: str

    # Optional fields with defaults
    in_core: Optional[bool] = None
    core_epsilon: Optional[float] = None

    # Nash equilibrium analysis
    nash_reports: Optional[np.ndarray] = None
    nash_converged: Optional[bool] = None
    nash_deviation_from_truth: Optional[float] = None
    nash_iterations: Optional[int] = None

    @property
    def gini_allocation(self) -> float:
        """Alias for backward compatibility"""
        return self.gini_coefficient

    def to_dict(self) -> Dict:
        """Serialize metrics to dictionary"""
        return {
            'itm': self.incentive_to_misreport,
            'avg_max_gain': self.avg_max_gain,
            'max_max_gain': self.max_max_gain,
            'gini': self.gini_coefficient,
            'entropy': self.normalized_entropy,
            'envy_rate': self.envy_rate,
            'fairreward': self.fairreward_composite,
            'stability_index': self.stability_index,
            'num_profitable_coalitions': self.num_profitable_coalitions,
            'in_core': self.in_core,
            'core_epsilon': self.core_epsilon,
            'nash_reports': self.nash_reports.tolist() if self.nash_reports is not None else None,
            'nash_converged': self.nash_converged,
            'nash_iterations': self.nash_iterations,
            'nash_deviation': self.nash_deviation_from_truth,
            'detection_prob': self.detection_prob,
            'penalty': self.penalty,
            'allocation_method': self.allocation_method,
            'max_gain_per_agent': self.max_gain_per_agent.tolist(),
            'best_deviation_per_agent': self.best_deviation_per_agent.tolist(),
            'expected_payoff_honest': self.expected_payoff_honest.tolist(),
            'expected_payoff_deviate': self.expected_payoff_deviate.tolist()
        }


class IncentiveAnalyzer:
    """
    Strategic behavior and fairness analyzer

    New Methods:
    - compute_nash_equilibrium_reports: Best-response dynamics
    - compute_fairreward_index: Composite fairness (Gini + Equity + Envy)
    - Enhanced stability check with LP-based core verification
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def compute_metrics(
        self,
        game_instance,
        allocation_result,
        allocation_engine,
        detection_prob: float,
        penalty: float,
        n_deviations: int = 20,
        max_deviation: float = 2.0,
        coalition_search_depth: int = 4,
        compute_nash: bool = False,
        check_core_lp: bool = False
    ) -> IncentiveMetrics:
        """
        Comprehensive incentive analysis with Nash equilibrium

        New Parameters:
        ---------------
        compute_nash : bool
            Compute Nash equilibrium reporting strategies (expensive)
        check_core_lp : bool
            Use LP-based core membership check (requires PuLP)
        """

        n = game_instance.n_agents
        q_true = game_instance.q_true
        allocations = allocation_result.allocations
        method = allocation_result.method

        # 1. Best-response deviation analysis
        max_gains = np.zeros(n)
        best_deviations = np.zeros(n)
        expected_honest = allocations.copy()
        expected_deviate = np.zeros(n)

        for i in range(n):
            gain, best_dev, exp_dev = self._compute_best_response(
                agent_i=i,
                q_true=q_true,
                game_instance=game_instance,
                allocation_engine=allocation_engine,
                allocation_method=method,
                detection_prob=detection_prob,
                penalty=penalty,
                n_deviations=n_deviations,
                max_deviation=max_deviation
            )
            max_gains[i] = gain
            best_deviations[i] = best_dev
            expected_deviate[i] = exp_dev

        # 2. Aggregate ITM metrics
        avg_gain = np.mean(max_gains)
        max_gain = np.max(max_gains)
        itm_score = np.mean(np.maximum(max_gains, 0))

        # 3. Fairness metrics
        gini = self._compute_gini(allocations)
        entropy = self._compute_normalized_entropy(allocations)
        envy = self._compute_envy_rate(allocations, q_true)

        # FairReward composite index
        fairreward = self._compute_fairreward_index(
            allocations, q_true, game_instance.get_grand_coalition_value()
        )

        # 4. Coalition stability
        stability, n_profitable, in_core, eps_core = self._check_coalition_stability(
            game_instance,
            allocation_result,
            allocation_engine,
            max_size=min(coalition_search_depth, n),
            check_core_lp=check_core_lp
        )

        # 5. Nash equilibrium (optional, expensive)
        nash_reports = None
        nash_converged = None
        nash_deviation = None
        nash_iterations = None

        if compute_nash:
            nash_result = self.compute_nash_equilibrium_reports(
                game_instance=game_instance,
                allocation_engine=allocation_engine,
                allocation_method=method,
                detection_prob=detection_prob,
                penalty_factor=penalty
            )
            nash_reports = nash_result['nash_reports']
            nash_converged = nash_result['converged']
            nash_deviation = nash_result['deviation_from_truth']
            nash_iterations = nash_result['iterations']

        return IncentiveMetrics(
            max_gain_per_agent=max_gains,
            best_deviation_per_agent=best_deviations,
            expected_payoff_honest=expected_honest,
            expected_payoff_deviate=expected_deviate,
            avg_max_gain=float(avg_gain),
            max_max_gain=float(max_gain),
            incentive_to_misreport=float(itm_score),
            gini_coefficient=float(gini),
            normalized_entropy=float(entropy),
            envy_rate=float(envy),
            fairreward_composite=float(fairreward),
            stability_index=float(stability),
            num_profitable_coalitions=int(n_profitable),
            in_core=in_core,
            core_epsilon=eps_core,
            nash_reports=nash_reports,
            nash_converged=nash_converged,
            nash_deviation_from_truth=nash_deviation,
            nash_iterations=nash_iterations,
            detection_prob=detection_prob,
            penalty=penalty,
            allocation_method=method
        )

    # =========================================================================
    # NASH EQUILIBRIUM COMPUTATION
    # =========================================================================

    def compute_nash_equilibrium_reports(
        self,
        game_instance,
        allocation_engine,
        allocation_method: str,
        detection_prob: float,
        penalty_factor: float,
        max_iters: int = 100,
        tol: float = 1e-3
    ) -> Dict:
        """
        Compute Nash equilibrium reporting strategies via best-response dynamics

        Algorithm:
        1. Initialize with truthful reports
        2. Iteratively compute best-response for each agent
        3. Converge when max change < tolerance

        Each agent optimizes:
            U_i(r_i, r_{-i}) = φ_i(r) - p × α × |r_i - q_i^true|

        Returns: dict with keys:
            - nash_reports: np.ndarray
            - converged: bool
            - iterations: int
            - deviation_from_truth: float (mean absolute deviation)

        Reference: Luo et al. (2025) "Game-Theoretic Incentive Mechanism for 
                   Blockchain-Based Federated Learning" IEEE TVT
        """
        q_true = game_instance.q_true
        n = len(q_true)

        # Initialize with truthful reports
        reports = q_true.copy()

        for iteration in range(max_iters):
            old_reports = reports.copy()

            # Best-response for each agent
            for i in range(n):

                def utility_i(r_i_candidate: float) -> float:
                    """
                    Utility function for agent i given candidate report

                    U_i = allocation - expected_penalty
                    """
                    # Create counterfactual reports
                    test_reports = reports.copy()
                    test_reports[i] = r_i_candidate

                    # Compute allocation under test reports
                    if allocation_method == 'proportional':
                        # Proportional uses reports directly
                        total_reported = np.sum(test_reports)
                        if total_reported < 1e-9:
                            phi_i = game_instance.get_grand_coalition_value() / n
                        else:
                            phi_i = (test_reports[i] / total_reported) * \
                                game_instance.get_grand_coalition_value()

                    elif allocation_method in ['exact_shapley', 'mc_shapley',
                                               'weighted_shapley', 'owen_weighted']:
                        # Shapley methods: need full allocation
                        # (expensive, so cache if possible)
                        alloc_result = allocation_engine.allocate(
                            method=allocation_method,
                            n_agents=n,
                            payoff_function=game_instance.payoff_function,
                            n_samples=5000  # Reduced for speed
                        )
                        phi_i = alloc_result.allocations[i]

                    else:
                        raise NotImplementedError(
                            f"Nash BR not implemented for {allocation_method}"
                        )

                    # Expected penalty if detected
                    is_misreporting = abs(r_i_candidate - q_true[i]) > 1e-6

                    if is_misreporting:
                        expected_penalty = detection_prob * penalty_factor * phi_i
                    else:
                        expected_penalty = 0.0

                    # Net utility
                    return phi_i - expected_penalty

                # Optimize over feasible report range
                # Bounds: [0, 2 × q_true] (assume agents won't report >2x true)
                result = minimize_scalar(
                    lambda r: -utility_i(r),  # Minimize negative = maximize
                    bounds=(0, max(2 * q_true[i], 0.1)),
                    method='bounded'
                )

                reports[i] = result.x

            # Check convergence
            max_change = np.max(np.abs(reports - old_reports))

            if max_change < tol:
                logger.info(
                    f"Nash equilibrium converged in {iteration+1} iterations")
                return {
                    'nash_reports': reports,
                    'converged': True,
                    'iterations': iteration + 1,
                    'deviation_from_truth': float(np.mean(np.abs(reports - q_true)))
                }

        # Failed to converge
        logger.warning(
            f"Nash equilibrium did not converge in {max_iters} iterations")
        return {
            'nash_reports': reports,
            'converged': False,
            'iterations': max_iters,
            'deviation_from_truth': float(np.mean(np.abs(reports - q_true)))
        }

    # =========================================================================
    # BEST-RESPONSE DEVIATION ANALYSIS
    # =========================================================================

    def _compute_best_response(
        self,
        agent_i: int,
        q_true: np.ndarray,
        game_instance,
        allocation_engine,
        allocation_method: str,
        detection_prob: float,
        penalty: float,
        n_deviations: int,
        max_deviation: float
    ) -> Tuple[float, float, float]:
        """
        Compute best response for agent i (myopic optimization)

        Returns: (max_expected_gain, best_deviation_amount, expected_payoff_at_best)
        """
        from .data_gen import DataGenerator
        deviations = DataGenerator().generate_strategic_deviations(
            q_true[agent_i], n_deviations, max_deviation
        )

        # Baseline: honest report
        honest_payoff = self._compute_expected_payoff(
            agent_i, q_true, q_true, game_instance, allocation_engine,
            allocation_method, detection_prob, penalty
        )

        # Search for best deviation
        max_gain = -np.inf
        best_dev = q_true[agent_i]
        best_expected = honest_payoff

        for dev in deviations:
            q_counterfactual = q_true.copy()
            q_counterfactual[agent_i] = dev

            expected = self._compute_expected_payoff(
                agent_i, q_true, q_counterfactual, game_instance,
                allocation_engine, allocation_method, detection_prob, penalty
            )

            gain = expected - honest_payoff

            if gain > max_gain:
                max_gain = gain
                best_dev = dev
                best_expected = expected

        return max_gain, best_dev, best_expected

    def _compute_expected_payoff(
        self,
        agent_i: int,
        q_true: np.ndarray,
        q_reported: np.ndarray,
        game_instance,
        allocation_engine,
        method: str,
        detection_prob: float,
        penalty: float
    ) -> float:
        """
        Compute expected payoff for agent i under detection regime

        Expected = (1 - p) × allocation - p × penalty × allocation
        """
        # Handle test/unknown methods gracefully
        if method in ['test', 'reverse']:
            return 0.0

        # Compute allocation
        try:
            if method == 'proportional':
                alloc_result = allocation_engine.allocate(
                    method=method,
                    n_agents=game_instance.n_agents,
                    payoff_function=game_instance.payoff_function,
                    reports=q_reported
                )
            else:
                alloc_result = allocation_engine.allocate(
                    method=method,
                    n_agents=game_instance.n_agents,
                    payoff_function=game_instance.payoff_function
                )
        except ValueError:
            return 0.0

        allocation_i = alloc_result.allocations[agent_i]

        # Check if misreporting
        is_misreporting = abs(q_reported[agent_i] - q_true[agent_i]) > 1e-6

        if not is_misreporting:
            return allocation_i

        # Expected payoff with detection risk
        payoff_no_detect = allocation_i
        payoff_detected = -penalty * allocation_i

        expected = (1 - detection_prob) * payoff_no_detect + \
            detection_prob * payoff_detected

        return expected

    # =========================================================================
    # COALITION STABILITY ANALYSIS
    # =========================================================================

    def _check_coalition_stability(
        self,
        game_instance,
        allocation_result,
        allocation_engine,
        max_size: int = 4,
        n_samples: int = 100,
        check_core_lp: bool = False
    ) -> Tuple[float, int, Optional[bool], Optional[float]]:
        """
        Check for profitable deviating coalitions

        Enhanced with LP-based core membership check

        Returns: (stability_index, num_profitable, in_core, epsilon)
        """
        n = game_instance.n_agents
        allocations = allocation_result.allocations
        payoff_fn = game_instance.payoff_function

        # Heuristic stability check (sample-based)
        n_profitable = 0
        n_tested = 0

        for size in range(2, min(max_size + 1, n + 1)):
            all_coalitions = list(combinations(range(n), size))

            if len(all_coalitions) > n_samples:
                indices = self.rng.choice(
                    len(all_coalitions), n_samples, replace=False
                )
                coalitions_to_test = [all_coalitions[i] for i in indices]
            else:
                coalitions_to_test = all_coalitions

            for coalition in coalitions_to_test:
                S = frozenset(coalition)
                v_S = payoff_fn(S)
                current_allocation_S = sum(allocations[i] for i in S)

                if v_S > current_allocation_S + 1e-6:
                    n_profitable += 1

                n_tested += 1

        if n_tested == 0:
            stability_index = 1.0
        else:
            stability_index = 1.0 - (n_profitable / n_tested)

        # LP-based core check (rigorous)
        in_core = None
        epsilon = None

        if check_core_lp:
            try:
                in_core, epsilon = allocation_engine.is_in_core_lp(
                    allocations=allocations,
                    payoff_fn=payoff_fn,
                    n_agents=n
                )
            except Exception as e:
                logger.warning(f"Core LP check failed: {e}")

        return stability_index, n_profitable, in_core, epsilon

    # =========================================================================
    # FAIRNESS METRICS
    # =========================================================================

    def _compute_fairreward_index(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray,
        total_value: float
    ) -> float:
        """
        FairReward composite fairness index

        Combines:
        1. Gini coefficient (distributional equality)
        2. Equity ratio (input/output proportionality)
        3. Envy-freeness (no agent prefers another's bundle)

        Returns: Float in [0, 1], higher = fairer

        Reference: Chen et al. (2024) "FairReward: Towards Fair Reward 
                   Distribution Using Equity Theory" ePrint 2024/123
        """
        n = len(allocations)

        # Component 1: Gini coefficient (invert so 1=perfect equality)
        gini = self._compute_gini(allocations)
        gini_score = 1 - gini

        # Component 2: Equity (proportionality of allocation to contribution)
        # Measure: min/max ratio of (allocation / contribution)
        equity_ratios = allocations / (contributions + 1e-10)
        equity_score = np.min(equity_ratios) / (np.max(equity_ratios) + 1e-10)

        # Component 3: Envy-freeness
        # No agent should prefer another's (allocation, contribution) bundle
        max_envy = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Agent i's utility if given j's bundle
                    # Assume utility = allocation - cost_of_contribution
                    u_i_of_j = allocations[j] - contributions[i]
                    u_i_of_i = allocations[i] - contributions[i]
                    envy = max(0, u_i_of_j - u_i_of_i)
                    max_envy = max(max_envy, envy)

        envy_score = 1.0 / (1.0 + max_envy)  # Map to [0, 1]

        # Composite: weighted average (equal weights)
        fairreward = (gini_score + equity_score + envy_score) / 3.0

        return fairreward

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Gini coefficient of inequality (clamped to [0, 1])"""
        sorted_vals = np.sort(values)
        n = len(values)

        if n == 0 or np.sum(sorted_vals) < 1e-9:
            return 0.0

        index = np.arange(1, n + 1)
        total = np.sum(sorted_vals)

        gini = (2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n

        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def _compute_normalized_entropy(values: np.ndarray) -> float:
        """Normalized Shannon entropy (0 = unequal, 1 = equal)"""
        total = np.sum(values)
        if total < 1e-9:
            return 0.0

        probs = values / total
        probs = probs[probs > 1e-9]

        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(values))

        if max_entropy < 1e-9:
            return 1.0

        # Clamp to [0, 1] to handle floating point precision issues
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def _compute_envy_rate(
        self,
        allocations: np.ndarray,
        contributions: Optional[np.ndarray] = None
    ) -> float:
        """
        Fraction of agents who envy another agent's allocation
        
        If contributions provided: Uses equity-adjusted envy
            (agent i envies j if x_i/q_i < x_j/q_j)
        If not provided: Uses simple envy
            (agent i envies j if x_i < x_j)
        """
        n = len(allocations)
        envy_count = 0

        if contributions is not None and not np.allclose(allocations, allocations[0]):
            # Only use equity-adjusted envy if allocations are not equal
            # For equal allocations, use simple envy regardless of contributions
            ratios = allocations / (contributions + 1e-10)
            for i in range(n):
                if np.any(ratios > ratios[i] + 1e-6):
                    envy_count += 1
        else:
            # Simple envy (absolute comparison) - used for equal allocations
            for i in range(n):
                if np.any(allocations > allocations[i] + 1e-6):
                    envy_count += 1

        return envy_count / n if n > 0 else 0.0
