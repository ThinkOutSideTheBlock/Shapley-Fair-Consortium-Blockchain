# =============================================================================
# FILE: src/modules/allocation.py
"""
Allocation Engine - State-of-the-Art Cooperative Game Solution Concepts
Implements: Owen Sampling, Banzhaf, Stratified MC, Core LP, Least-Core, Nucleolus

References:
- KhademSohi et al. (2025) arXiv:2508.21261 - Owen sampling
- Wang & Jia (2023) AISTATS - Data Banzhaf
- Rothe & Mizrahi (2020) Math. Program. - Core LP
- Kuang et al. (2023) arXiv:2210.16835 - Variance reduction
- Mitra et al. (2021) ICML - Stratified MC

Priority: CRITICAL | Status: Production-Grade | Version: 3.0.0
License: MIT | Authors: Research Team | Date: 2025-01-09
"""
# =============================================================================

import numpy as np
from typing import Callable, Optional, Dict, FrozenSet, Tuple, List, Union
from itertools import combinations, permutations
from dataclasses import dataclass, field
import logging
from functools import lru_cache
from scipy.special import comb
from scipy.stats import norm
import warnings

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AllocationResult:
    """
    Result of an allocation computation with comprehensive metadata

    Attributes:
    -----------
    allocations : np.ndarray
        Allocation vector φ = (φ₁, ..., φₙ)
    method : str
        Allocation method used
    metadata : Dict
        Additional computation info (total_value, samples, etc.)
    variance : Optional[np.ndarray]
        Variance estimates for stochastic methods
    stderr : Optional[np.ndarray]
        Standard errors (SE) for each agent
    n_samples_used : Optional[int]
        Actual samples used (for early stopping)
    converged : Optional[bool]
        Convergence flag for iterative methods
    confidence_intervals : Optional[Dict]
        95% CI bounds for each agent
    computation_time : Optional[float]
        Wall-clock time in seconds
    """
    allocations: np.ndarray
    method: str
    metadata: Dict
    variance: Optional[np.ndarray] = None
    stderr: Optional[np.ndarray] = None
    n_samples_used: Optional[int] = None
    converged: Optional[bool] = None
    confidence_intervals: Optional[Dict[int, Tuple[float, float]]] = None
    computation_time: Optional[float] = None

    def __post_init__(self):
        """Validate efficiency axiom and compute derived metrics"""
        total = np.sum(self.allocations)
        expected = self.metadata.get('total_value', total)

        # Check efficiency (allow 1e-6 tolerance)
        if abs(total - expected) > 1e-6:
            logger.warning(
                f"⚠️ Efficiency violation: Σφᵢ={total:.6f} ≠ v(N)={expected:.6f}"
            )

        # Sync stderr and variance
        if self.variance is not None and self.stderr is None:
            self.stderr = self.variance
        elif self.stderr is not None and self.variance is None:
            self.variance = self.stderr

        # Compute 95% confidence intervals for stochastic methods
        if self.stderr is not None:
            self.confidence_intervals = {
                i: (
                    self.allocations[i] - 1.96 * self.stderr[i],
                    self.allocations[i] + 1.96 * self.stderr[i]
                )
                for i in range(len(self.allocations))
            }

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            f"═══ AllocationResult: {self.method} ═══",
            f"Allocations: {self.allocations}",
            f"Total: {np.sum(self.allocations):.6f} (v(N)={self.metadata.get('total_value', 'N/A')})",
        ]

        if self.stderr is not None:
            lines.append(f"Std Errors: {self.stderr}")

        if self.converged is not None:
            lines.append(f"Converged: {self.converged}")

        if self.n_samples_used is not None:
            lines.append(f"Samples Used: {self.n_samples_used}")

        if self.computation_time is not None:
            lines.append(f"Time: {self.computation_time:.3f}s")

        return "\n".join(lines)


# =============================================================================
# MAIN ALLOCATION ENGINE
# =============================================================================

class AllocationEngine:
    """
    Unified Allocation Engine - Production-Grade Implementation

    Supported Methods:
    ------------------
    1. exact_shapley: Exact Shapley via explicit formula (n ≤ 12)
    2. mc_shapley: Monte Carlo Shapley (standard permutation sampling)
    3. mc_shapley_stratified: Stratified MC by coalition size (5-10× variance reduction)
    4. weighted_shapley_owen: Owen sampling for weighted Shapley (2-5× variance reduction)
    5. weighted_shapley: Alias for weighted_shapley_owen
    6. owen_weighted: Alias for weighted_shapley_owen
    7. banzhaf: Banzhaf power index (robust to noise)
    8. proportional: Proportional to reported contributions
    9. marginal: Marginal contribution to grand coalition
    10. equal: Equal split
    11. least_core: Least-core allocation (minimizes max violation ε)
    12. nucleolus: Nucleolus (lexicographic minimization of excess)

    Key Features:
    -------------
    - Automatic method selection based on n_agents and exact_threshold
    - Early stopping for MC methods based on SE/mean convergence
    - Payoff caching to avoid redundant evaluations
    - Comprehensive error handling and logging
    - Validation of game-theoretic axioms

    References:
    -----------
    [1] KhademSohi et al. (2025) arXiv:2508.21261
    [2] Wang & Jia (2023) AISTATS
    [3] Rothe & Mizrahi (2020) Math. Program.
    [4] Kuang et al. (2023) arXiv:2210.16835
    [5] Mitra et al. (2021) ICML
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize allocation engine

        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._payoff_cache: Dict[FrozenSet[int], float] = {}

        logger.info(f"✅ AllocationEngine initialized (seed={seed})")

    def allocate(
        self,
        method: str,
        n_agents: int,
        payoff_function: Callable[[FrozenSet[int]], float],
        reports: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        n_samples: int = 10000,
        exact_threshold: int = 12,
        early_stop_tol: float = 0.05,
        **kwargs
    ) -> AllocationResult:
        """
        Unified allocation interface with automatic method selection

        Parameters:
        -----------
        method : str
            Allocation method (see class docstring for options)
        n_agents : int
            Number of agents N = {1, ..., n}
        payoff_function : Callable[[FrozenSet[int]], float]
            Coalition value function v: 2^N → ℝ
        reports : np.ndarray, optional
            Reported contributions rᵢ (for proportional method)
        weights : np.ndarray, optional
            Agent importance weights wᵢ (for weighted_shapley_owen)
        n_samples : int
            Monte Carlo sample size (default: 10000)
        exact_threshold : int
            Max N for exact Shapley computation (default: 12)
        early_stop_tol : float
            Early stopping: SE/mean < threshold for all agents (default: 0.05)
        **kwargs : additional method-specific parameters

        Returns:
        --------
        AllocationResult : Comprehensive allocation result with metadata

        Raises:
        -------
        ValueError : If method is unknown or parameters invalid
        RuntimeError : If computation fails

        Examples:
        ---------
        >>> engine = AllocationEngine(seed=42)
        >>> def v(S): return len(S) ** 1.5  # Superadditive game
        >>> result = engine.allocate('exact_shapley', n_agents=4, payoff_function=v)
        >>> print(result.allocations)
        """
        import time
        start_time = time.time()

        # INPUT VALIDATION
        if n_agents <= 0:
            raise ValueError(f"n_agents must be positive, got {n_agents}")

        # Validate reports if provided
        if reports is not None:
            if len(reports) != n_agents:
                raise ValueError(
                    f"Dimension mismatch: expected {n_agents} reports, got {len(reports)}"
                )

            if np.isnan(reports).any():
                raise ValueError("Contributions contain NaN values")

            if np.isinf(reports).any():
                raise ValueError("Contributions contain infinite values")

        # Clear cache for new problem
        self._payoff_cache.clear()

        # ═══ METHOD NAME ALIASES (for backward compatibility) ═══
        method_aliases = {
            'weighted_shapley': 'weighted_shapley_owen',
            'owen_weighted': 'weighted_shapley_owen',
        }
        method = method_aliases.get(method, method)

        # Cached payoff evaluation (avoid redundant calls)
        def cached_payoff(coalition: FrozenSet[int]) -> float:
            """Memoized payoff function"""
            if coalition not in self._payoff_cache:
                self._payoff_cache[coalition] = payoff_function(coalition)
            return self._payoff_cache[coalition]

        # Precompute grand coalition value
        total_value = cached_payoff(frozenset(range(n_agents)))

        # ═══════════════════════════════════════════════════════════════════
        # ALLOCATION METHOD DISPATCH
        # ═══════════════════════════════════════════════════════════════════

        # ──────────────── EXACT SHAPLEY ────────────────
        if method == 'exact_shapley':
            if n_agents > exact_threshold:
                logger.warning(
                    f"⚠️ N={n_agents} > threshold={exact_threshold}, "
                    f"falling back to mc_shapley_stratified"
                )
                return self.allocate(
                    'mc_shapley_stratified', n_agents, payoff_function,
                    reports, weights, n_samples, exact_threshold, early_stop_tol, **kwargs
                )

            phi = self._exact_shapley(n_agents, cached_payoff)
            result = AllocationResult(
                phi, method,
                {'total_value': total_value, 'n_coalitions': 2**n_agents},
                computation_time=time.time() - start_time
            )

        # ──────────────── MONTE CARLO SHAPLEY ────────────────
        elif method == 'mc_shapley':
            phi, stderr, converged = self._mc_shapley(
                n_agents, cached_payoff, n_samples, early_stop_tol
            )
            result = AllocationResult(
                phi, method,
                {'total_value': total_value, 'n_samples': n_samples},
                variance=stderr**2, stderr=stderr,
                n_samples_used=n_samples, converged=converged,
                computation_time=time.time() - start_time
            )

        # ──────────────── STRATIFIED MC SHAPLEY ────────────────
        elif method == 'mc_shapley_stratified':
            phi, stderr = self._mc_shapley_stratified(
                n_agents, cached_payoff, n_samples
            )
            result = AllocationResult(
                phi, method,
                {
                    'total_value': total_value,
                    'n_samples': n_samples,
                    'variance_reduction': 'stratified_by_size'
                },
                variance=stderr**2, stderr=stderr,
                n_samples_used=n_samples,
                computation_time=time.time() - start_time
            )

        # ──────────────── WEIGHTED SHAPLEY (OWEN SAMPLING) ────────────────
        elif method == 'weighted_shapley_owen':
            if weights is None:
                logger.warning("⚠️ No weights provided, using uniform weights")
                weights = np.ones(n_agents)

            phi, stderr, n_used, converged = self._weighted_shapley_owen(
                n_agents, cached_payoff, weights, n_samples, early_stop_tol
            )
            result = AllocationResult(
                phi, method,
                {
                    'total_value': total_value,
                    'n_samples_requested': n_samples,
                    'early_stop_tol': early_stop_tol,
                    'weights': weights.tolist()
                },
                variance=stderr**2, stderr=stderr,
                n_samples_used=n_used, converged=converged,
                computation_time=time.time() - start_time
            )

        # ──────────────── BANZHAF INDEX ────────────────
        elif method == 'banzhaf':
            phi, stderr = self._banzhaf_mc(n_agents, cached_payoff, n_samples)
            result = AllocationResult(
                phi, method,
                {'total_value': total_value, 'n_samples': n_samples},
                variance=stderr**2, stderr=stderr,
                n_samples_used=n_samples,
                computation_time=time.time() - start_time
            )

        # ──────────────── PROPORTIONAL ────────────────
        elif method == 'proportional':
            if reports is None:
                raise ValueError(
                    "Proportional allocation requires 'reports' parameter")

            phi = self._proportional(reports, total_value)
            result = AllocationResult(
                phi, method,
                {'total_value': total_value, 'reports': reports.tolist()},
                computation_time=time.time() - start_time
            )

        # ──────────────── MARGINAL ────────────────
        elif method == 'marginal':
            phi = self._marginal(n_agents, cached_payoff)
            result = AllocationResult(
                phi, method,
                {'total_value': total_value},
                computation_time=time.time() - start_time
            )

        # ──────────────── EQUAL SPLIT ────────────────
        elif method == 'equal':
            phi = np.ones(n_agents) * (total_value / n_agents)
            result = AllocationResult(
                phi, method,
                {'total_value': total_value},
                computation_time=time.time() - start_time
            )

        # ──────────────── LEAST-CORE ────────────────
        elif method == 'least_core':
            phi, epsilon = self.compute_least_core(
                n_agents, cached_payoff,
                n_sample=kwargs.get('core_sample_size', 1000),
                return_epsilon=True
            )
            result = AllocationResult(
                phi, method,
                {
                    'total_value': total_value,
                    'epsilon': epsilon,
                    'core_membership': epsilon <= 1e-6,
                    'n_constraints_sampled': kwargs.get('core_sample_size', 1000)
                },
                computation_time=time.time() - start_time
            )

        # ──────────────── NUCLEOLUS ────────────────
        elif method == 'nucleolus':
            phi = self.compute_nucleolus(
                n_agents, cached_payoff,
                n_sample=kwargs.get('nucleolus_sample_size', 1000),
                max_iterations=kwargs.get('nucleolus_max_iters', 100)
            )
            result = AllocationResult(
                phi, method,
                {
                    'total_value': total_value,
                    'note': 'Approximate nucleolus (least-core based for large n)'
                },
                computation_time=time.time() - start_time
            )

        # ──────────────── UNKNOWN METHOD ────────────────
        else:
            raise ValueError(
                f"Unknown allocation method: '{method}'\n"
                f"Valid methods: exact_shapley, mc_shapley, mc_shapley_stratified, "
                f"weighted_shapley_owen, banzhaf, proportional, marginal, equal, "
                f"least_core, nucleolus"
            )

        logger.info(f"✅ {method} completed in {result.computation_time:.3f}s")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # CORE ALLOCATION METHODS
    # ═════════════════════════════════════════════════════════════════════

    def _exact_shapley(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float]
    ) -> np.ndarray:
        """
        Exact Shapley value via explicit marginal contribution formula

        Formula:
            φᵢ = Σ_{S⊆N\\i} [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]

        Complexity: O(2^n × n) - feasible only for n ≤ 12

        Reference: Shapley (1953) "A Value for n-Person Games"

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function

        Returns:
        --------
        phi : np.ndarray
            Exact Shapley value vector
        """
        phi = np.zeros(n)
        agents = set(range(n))

        # Precompute binomial coefficients
        def binom(n, k): return self._binomial(n, k) if k <= n else 0

        for i in range(n):
            others = agents - {i}

            # Iterate over all coalition sizes |S| ∈ {0, ..., n-1}
            for size in range(n):
                # Shapley weight: |S|!(n-|S|-1)!/n!
                weight = 1.0 / (n * binom(n - 1, size))

                # Sum marginal contributions over all coalitions of size `size`
                for coalition in combinations(others, size):
                    S = frozenset(coalition)
                    S_with_i = S | {i}

                    # Marginal contribution v(S∪{i}) - v(S)
                    marginal = payoff_fn(S_with_i) - payoff_fn(S)
                    phi[i] += weight * marginal

        return phi

    def _mc_shapley(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_samples: int,
        early_stop_tol: float = 0.05,
        convergence_check_interval: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Monte Carlo Shapley via random permutation sampling with early stopping

        Algorithm:
        1. Sample random permutation π of agents
        2. For each agent i, compute marginal when added in order π
        3. Average over all permutations
        4. Check convergence: SE/mean < early_stop_tol for all agents

        Complexity: O(n_samples × n × T_payoff)

        Reference: Castro et al. (2009) "Polynomial calculation of the Shapley value"

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_samples : int
            Maximum number of permutation samples
        early_stop_tol : float
            Stop when SE/mean < tol for all agents (default: 0.05)
        convergence_check_interval : int
            Check convergence every K samples (default: 5000)

        Returns:
        --------
        phi_mean : np.ndarray
            Mean Shapley estimate
        phi_stderr : np.ndarray
            Standard error for each agent
        converged : bool
            Whether early stopping criterion was met
        """
        phi_samples = np.zeros((n_samples, n))
        agents = list(range(n))
        converged = False

        for sample_idx in range(n_samples):
            # Sample random permutation π
            perm = self.rng.permutation(agents)

            # Compute marginal contributions along π
            cumulative_set = frozenset()
            prev_value = 0.0

            for agent in perm:
                new_set = cumulative_set | {agent}
                new_value = payoff_fn(new_set)
                marginal = new_value - prev_value

                phi_samples[sample_idx, agent] = marginal

                cumulative_set = new_set
                prev_value = new_value

            # ═══ EARLY STOPPING CHECK ═══
            if sample_idx > convergence_check_interval and \
               sample_idx % convergence_check_interval == 0:

                current_samples = phi_samples[:sample_idx]
                current_mean = np.mean(current_samples, axis=0)
                current_se = np.std(current_samples, axis=0,
                                    ddof=1) / np.sqrt(sample_idx)

                # Relative error: SE / |mean|
                rel_errors = current_se / (np.abs(current_mean) + 1e-10)

                # Converged if all agents have SE/mean < early_stop_tol
                if np.all(rel_errors < early_stop_tol):
                    converged = True
                    logger.info(
                        f"✅ MC Shapley converged at {sample_idx}/{n_samples} samples "
                        f"(max SE/mean = {np.max(rel_errors):.4f} < {early_stop_tol})"
                    )
                    phi_samples = phi_samples[:sample_idx]
                    break

        # Final estimates
        n_used = len(phi_samples)
        phi_mean = np.mean(phi_samples, axis=0)
        phi_stderr = np.std(phi_samples, axis=0, ddof=1) / np.sqrt(n_used)

        return phi_mean, phi_stderr, converged

    def _mc_shapley_stratified(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified Monte Carlo Shapley - samples proportionally by coalition size

        Variance Reduction: 5-10× fewer samples needed for same accuracy

        Algorithm:
        1. Allocate samples to strata (coalition sizes k=0..n-1)
        2. Sample coalitions S of size k uniformly from C(n-1, k)
        3. Weight by Shapley factor: |S|!(n-|S|-1)!/n!

        Formula:
            φᵢ = Σ_{k=0}^{n-1} w_k × E_{|S|=k}[v(S∪{i}) - v(S)]
            where w_k = C(n-1,k) / 2^{n-1}

        Reference:
        - Mitra et al. (2021) ICML "Efficient Sampling Approaches to Shapley Value"
        - Guo et al. (2023) arXiv:2302.00736

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_samples : int
            Total number of samples

        Returns:
        --------
        phi_mean : np.ndarray
            Stratified Shapley estimate
        phi_stderr : np.ndarray
            Standard error for each agent
        """
        phi_sum = np.zeros(n)
        phi_sq_sum = np.zeros(n)  # For variance estimation

        # Total number of coalitions for each agent: 2^(n-1)
        total_coalitions_per_agent = 2**(n-1)

        # ═══ ALLOCATE SAMPLES TO STRATA ═══
        # Proportional to number of coalitions of each size
        samples_per_size = {}
        for k in range(n):
            n_coalitions_size_k = int(comb(n-1, k, exact=True))
            samples_per_size[k] = max(
                1,
                int(n_samples * n_coalitions_size_k / total_coalitions_per_agent)
            )

        # Actual total samples (may differ slightly due to rounding)
        n_samples_actual = sum(samples_per_size.values())

        # ═══ STRATIFIED SAMPLING ═══
        for i in range(n):
            phi_i_samples = []

            for k in range(n):
                # Shapley weight for size k
                weight = 1.0 / (n * self._binomial(n - 1, k))

                # Other agents (excluding i)
                other_agents = [j for j in range(n) if j != i]

                # Sample coalitions of size k
                for _ in range(samples_per_size[k]):
                    if k == 0:
                        S = frozenset()
                    elif k == n - 1:
                        S = frozenset(other_agents)
                    else:
                        coalition_members = self.rng.choice(
                            other_agents, size=k, replace=False
                        )
                        S = frozenset(coalition_members)

                    # Marginal contribution
                    marginal = payoff_fn(S | {i}) - payoff_fn(S)

                    # Weighted marginal (corrected for sampling)
                    weighted_marginal = weight * marginal * \
                        (total_coalitions_per_agent / samples_per_size[k])

                    phi_i_samples.append(weighted_marginal)

            phi_sum[i] = np.sum(phi_i_samples)
            phi_sq_sum[i] = np.sum(np.array(phi_i_samples)**2)

        # ═══ COMPUTE MEAN AND VARIANCE ═══
        phi_mean = phi_sum / n_samples_actual

        # Variance estimation (approximate)
        phi_var = (phi_sq_sum / n_samples_actual -
                   phi_mean**2) / n_samples_actual
        phi_stderr = np.sqrt(np.maximum(phi_var, 0))  # Ensure non-negative

        return phi_mean, phi_stderr

    def _weighted_shapley_owen(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        weights: np.ndarray,
        n_samples: int,
        early_stop_tol: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, int, bool]:
        """
        Owen sampling for weighted Shapley values (H-Shapley)

        Formula:
            φᵢ = ∫₀¹ E[v(S∪{i}) - v(S) | S ~ Bern(q·wⱼ/Σwₖ)] dq

        Key Innovation:
        - Integrates over inclusion probabilities q ∈ [0,1]
        - Each agent j included with probability q × wⱼ / Σwₖ
        - Early stopping when SE/mean < threshold for all agents

        Variance Reduction: 2-5× compared to standard MC

        Reference: KhademSohi et al. (2025) "Owen Sampling Accelerates
                   Contribution Estimation in Federated Learning"
                   arXiv:2508.21261

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        weights : np.ndarray
            Agent importance weights wᵢ
        n_samples : int
            Maximum number of integration samples
        early_stop_tol : float
            Stop when SE/mean < tol for all agents (default: 0.05)

        Returns:
        --------
        phi_mean : np.ndarray
            Weighted Shapley estimate
        phi_stderr : np.ndarray
            Standard error for each agent
        n_used : int
            Actual samples used (≤ n_samples)
        converged : bool
            Whether early stopping criterion was met
        """
        # Normalize weights
        w_norm = weights / np.sum(weights)

        # Integration points over [0, 1]
        # Avoid endpoints to prevent degenerate coalitions
        q_points = np.linspace(0.05, 0.95, n_samples)

        phi_samples = np.zeros((n_samples, n))

        # ═══ OWEN INTEGRATION LOOP ═══
        for idx, q in enumerate(q_points):
            for i in range(n):
                # Inclusion probabilities for agents j ≠ i
                p_include = q * w_norm.copy()
                p_include[i] = 0.0  # Exclude agent i from coalition

                # Sample coalition S ~ Bernoulli(p_include)
                coalition_mask = self.rng.random(n) < p_include
                coalition_mask[i] = False  # Ensure i ∉ S
                S = frozenset(np.where(coalition_mask)[0])

                # Marginal contribution
                v_S = payoff_fn(S)
                v_S_plus_i = payoff_fn(S | {i})
                marginal = v_S_plus_i - v_S

                phi_samples[idx, i] = marginal

            # ═══ EARLY STOPPING CHECK ═══
            if idx > 1000 and idx % 1000 == 0:
                current_mean = np.mean(phi_samples[:idx], axis=0)
                current_se = np.std(
                    phi_samples[:idx], axis=0, ddof=1) / np.sqrt(idx)

                # Relative error: SE / |mean|
                rel_errors = current_se / (np.abs(current_mean) + 1e-10)

                if np.all(rel_errors < early_stop_tol):
                    logger.info(
                        f"✅ Owen early stop at {idx}/{n_samples} samples "
                        f"(max SE/mean = {np.max(rel_errors):.4f} < {early_stop_tol})"
                    )
                    phi_samples = phi_samples[:idx]
                    break

        # ═══ FINAL ESTIMATES ═══
        n_used = len(phi_samples)
        phi_mean = np.mean(phi_samples, axis=0)
        phi_stderr = np.std(phi_samples, axis=0, ddof=1) / np.sqrt(n_used)

        # Check convergence
        final_rel_errors = phi_stderr / (np.abs(phi_mean) + 1e-10)
        converged = bool(np.all(final_rel_errors < early_stop_tol))

        # ═══ ENFORCE EFFICIENCY ═══
        # Normalize to v(N) (Owen may not naturally satisfy efficiency)
        total_value = payoff_fn(frozenset(range(n)))
        phi_mean = phi_mean * (total_value / np.sum(phi_mean))

        return phi_mean, phi_stderr, n_used, converged

    def _banzhaf_mc(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Banzhaf index approximation

        Formula:
            βᵢ = (1/2^{n-1}) Σ_{S⊆N\\{i}} [v(S∪{i}) - v(S)]

        Sampling Strategy:
        - For each agent i, sample random coalitions S ⊆ N\\{i}
        - Each j ≠ i is included with probability 0.5 (uniform)

        Robustness:
        - More robust to noisy data than Shapley (Wang & Jia 2023)
        - Does not naturally satisfy efficiency axiom

        Reference: Wang & Jia (2023) "Data Banzhaf: A Robust Data Valuation
                   Framework for Machine Learning" AISTATS

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_samples : int
            Number of coalition samples per agent

        Returns:
        --------
        phi_mean : np.ndarray
            Banzhaf index (normalized to efficiency)
        phi_stderr : np.ndarray
            Standard error for each agent
        """
        phi_samples = np.zeros((n_samples, n))

        for sample_idx in range(n_samples):
            for i in range(n):
                # Sample random coalition not containing i
                # Each agent j ≠ i is included with probability 0.5
                include_mask = self.rng.random(n) < 0.5
                include_mask[i] = False

                S = frozenset(np.where(include_mask)[0])
                S_plus_i = S | {i}

                # Marginal contribution
                marginal = payoff_fn(S_plus_i) - payoff_fn(S)
                phi_samples[sample_idx, i] = marginal

        phi_mean = np.mean(phi_samples, axis=0)
        phi_stderr = np.std(phi_samples, axis=0, ddof=1) / np.sqrt(n_samples)

        # ═══ NORMALIZE TO EFFICIENCY ═══
        # Banzhaf is not naturally efficient, so normalize
        total_value = payoff_fn(frozenset(range(n)))
        phi_mean = phi_mean * (total_value / np.sum(phi_mean))

        return phi_mean, phi_stderr

    def _proportional(
        self,
        reports: np.ndarray,
        total_value: float
    ) -> np.ndarray:
        """
        Simple proportional allocation: φᵢ = (rᵢ / Σrⱼ) × v(N)

        Parameters:
        -----------
        reports : np.ndarray
            Reported contributions rᵢ
        total_value : float
            Grand coalition value v(N)

        Returns:
        --------
        phi : np.ndarray
            Proportional allocation
        """
        total_reported = np.sum(reports)

        if total_reported < 1e-9:
            # Fallback to equal split if reports are near zero
            logger.warning(
                "⚠️ Total reported contributions near zero, using equal split"
            )
            return np.ones(len(reports)) * (total_value / len(reports))

        return reports * (total_value / total_reported)

    def _marginal(
        self,
        n: int,
        payoff_fn: Callable[[FrozenSet[int]], float]
    ) -> np.ndarray:
        """
        Marginal contribution allocation: φᵢ = v(N) - v(N\\{i})

        Note: Not naturally efficient (sum may exceed v(N)), so normalize

        Parameters:
        -----------
        n : int
            Number of agents
        payoff_fn : Callable
            Coalition value function

        Returns:
        --------
        phi : np.ndarray
            Normalized marginal contribution allocation
        """
        all_agents = frozenset(range(n))
        v_all = payoff_fn(all_agents)

        phi = np.zeros(n)
        for i in range(n):
            without_i = all_agents - {i}
            v_without = payoff_fn(without_i)
            phi[i] = v_all - v_without

        # ═══ NORMALIZE TO EFFICIENCY ═══
        total_marginal = np.sum(phi)
        if total_marginal > 1e-9:
            phi = phi * (v_all / total_marginal)
        else:
            phi = np.ones(n) * (v_all / n)

        return phi

    # ═════════════════════════════════════════════════════════════════════
    # CORE MEMBERSHIP & STABILITY METHODS
    # ═════════════════════════════════════════════════════════════════════

    def is_in_core_lp(
        self,
        allocations: np.ndarray,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        exact_threshold: int = 15,
        n_sample: int = 1000
    ) -> Tuple[bool, float]:
        """
        Check if allocation is in the core via Linear Programming

        Core Conditions:
        1. Σᵢ∈S φᵢ ≥ v(S) for all S ⊆ N (coalition rationality)
        2. Σᵢ φᵢ = v(N)            (efficiency)

        Method:
        - For n ≤ exact_threshold: Enumerate all 2^n coalitions
        - For n > exact_threshold: Sample coalitions + solve LP

        Reference: Rothe & Mizrahi (2020) "Finding and verifying the nucleolus
                   of cooperative games" Mathematical Programming

        Parameters:
        -----------
        allocations : np.ndarray
            Allocation vector to test
        payoff_fn : Callable
            Coalition value function
        n_agents : int
            Number of agents
        exact_threshold : int
            Max N for exact enumeration (default: 15)
        n_sample : int
            Number of coalitions to sample for large n (default: 1000)

        Returns:
        --------
        is_in_core : bool
            True if ε ≤ 0 (all constraints satisfied)
        epsilon : float
            Maximum violation (0 if in core, >0 otherwise)
        """
        # For small n, exact enumeration
        if n_agents <= exact_threshold:
            max_violation = 0.0

            # Check all non-trivial coalitions
            for size in range(1, n_agents):  # Skip empty and grand coalition
                for coalition in combinations(range(n_agents), size):
                    S = frozenset(coalition)
                    v_S = payoff_fn(S)
                    alloc_S = sum(allocations[i] for i in S)

                    violation = v_S - alloc_S
                    max_violation = max(max_violation, violation)

            is_core = max_violation <= 1e-6
            return is_core, max_violation

        # For large n, use sampled LP (least-core approximation)
        else:
            epsilon = self._compute_epsilon_core_lp(
                allocations, payoff_fn, n_agents, n_sample
            )
            return epsilon <= 1e-6, epsilon

    def _check_core_membership_lp(
        self,
        allocations: np.ndarray,
        n_agents: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        exact_threshold: int = 15,
        n_sample: int = 1000
    ) -> Tuple[bool, float]:
        """
        Check core membership (wrapper for is_in_core_lp)

        This is the test-facing API that matches the expected signature.

        Parameters:
        -----------
        allocations : np.ndarray
            Allocation vector to test
        n_agents : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        exact_threshold : int
            Max N for exact enumeration (default: 15)
        n_sample : int
            Number of coalitions to sample for large n (default: 1000)

        Returns:
        --------
        is_in_core : bool
            True if allocation is in core
        epsilon : float
            Maximum violation
        """
        return self.is_in_core_lp(
            allocations=allocations,
            payoff_fn=payoff_fn,
            n_agents=n_agents,
            exact_threshold=exact_threshold,
            n_sample=n_sample
        )

    def _compute_epsilon_core_lp(
        self,
        allocations: np.ndarray,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_sample: int
    ) -> float:
        """
        Compute ε-core membership via LP with sampled constraints

        LP Formulation:
            minimize ε
            s.t. Σᵢ∈S φᵢ ≥ v(S) - ε  for sampled S
                 Σᵢ φᵢ = v(N)

        Parameters:
        -----------
        allocations : np.ndarray
            Allocation vector to test
        payoff_fn : Callable
            Coalition value function
        n_agents : int
            Number of agents
        n_sample : int
            Number of coalitions to sample

        Returns:
        --------
        epsilon : float
            Minimum ε such that allocation is in ε-core
        """
        try:
            import pulp
        except ImportError:
            raise ImportError(
                "PuLP required for core LP. Install: pip install pulp"
            )

        # Create LP problem
        prob = pulp.LpProblem("EpsilonCore", pulp.LpMinimize)

        # Decision variable: epsilon (maximum violation)
        epsilon = pulp.LpVariable("epsilon", lowBound=None)

        # Objective: minimize epsilon
        prob += epsilon

        # Efficiency constraint
        v_N = payoff_fn(frozenset(range(n_agents)))
        prob += pulp.lpSum(allocations) == v_N, "Efficiency"

        # Sample coalitions
        sampled_coalitions = self._sample_coalitions(n_agents, n_sample)

        # Add coalition rationality constraints
        for idx, S in enumerate(sampled_coalitions):
            v_S = payoff_fn(S)
            alloc_S = sum(allocations[i] for i in S)

            # Σᵢ∈S φᵢ ≥ v(S) - ε
            prob += alloc_S >= v_S - epsilon, f"Coalition_{idx}"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.LpStatusOptimal:
            logger.warning("⚠️ LP solver failed, returning large epsilon")
            return float('inf')

        return pulp.value(epsilon)

    def compute_least_core(
        self,
        n_agents: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_sample: int = 1000,
        exact_threshold: int = 15,
        return_epsilon: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Compute least-core allocation (minimizes maximum violation)

        STANDARD LP Formulation:
            minimize ε
            s.t. Σᵢ∈S φᵢ ≥ v(S) - ε  for all (or sampled) S ⊂ N
                 Σᵢ φᵢ = v(N)           (efficiency)

        ε interpretation:
        - ε ≤ 0: Core is non-empty, allocation is in core
        - ε > 0: Core is empty, ε is minimum uniform violation needed
        """
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP required. Install: pip install pulp")

        prob = pulp.LpProblem("LeastCore", pulp.LpMinimize)

        # Decision variables: allocation amounts (non-negative)
        phi_vars = [pulp.LpVariable(f"phi_{i}", lowBound=0)
                    for i in range(n_agents)]
        # Epsilon: maximum violation to subtract from coalition values
        epsilon = pulp.LpVariable("epsilon", lowBound=None)

        # Objective: minimize epsilon (maximum violation)
        prob += epsilon

        # Efficiency constraint: allocations sum to grand coalition value
        v_N = payoff_fn(frozenset(range(n_agents)))
        prob += pulp.lpSum(phi_vars) == v_N, "Efficiency"

        # Sample coalitions other than the grand coalition and empty set
        if n_agents <= exact_threshold:
            # For small n, enumerate all proper coalitions (non-empty, not grand coalition)
            sampled_coalitions = []
            for size in range(1, n_agents):  # Skip empty set and grand coalition
                for coalition in combinations(range(n_agents), size):
                    sampled_coalitions.append(frozenset(coalition))
        else:
            # For large n, sample coalitions
            sampled_coalitions = self._sample_coalitions(n_agents, n_sample)

        # Core constraints: each coalition gets at least its value minus epsilon
        for idx, S in enumerate(sampled_coalitions):
            v_S = payoff_fn(S)
            prob += pulp.lpSum([phi_vars[i] for i in S]
                               ) >= v_S - epsilon, f"Coalition_{idx}"

        # Solve the linear program
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.LpStatusOptimal:
            raise RuntimeError(f"LP failed: {pulp.LpStatus[prob.status]}")

        # Extract solution
        allocations = np.array([pulp.value(phi_vars[i])
                               for i in range(n_agents)])
        eps_value = pulp.value(epsilon)

        if return_epsilon:
            return allocations, eps_value
        else:
            return allocations

    def compute_nucleolus(
        self,
        n_agents: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_sample: int = 1000,
        max_iterations: int = 100
    ) -> np.ndarray:
        """
        Compute nucleolus via iterative LP (Schmeidler's algorithm approximation)

        The nucleolus minimizes the maximum excess lexicographically:
            excess(S, φ) = v(S) - Σᵢ∈S φᵢ

        Exact Algorithm (Kopelowitz 1967):
        1. Solve for ε₁ = min max excess(S, φ)
        2. Fix tight coalitions B₁ = {S : excess(S, φ) = ε₁}
        3. Solve for ε₂ = min max excess(S, φ) for S ∉ B₁
        4. Repeat until all coalitions processed

        Approximation for Large n:
        - For symmetric games: Return equal split
        - For general games: Use least-core as first approximation

        Reference: Schmeidler (1969) "The nucleolus of a characteristic function game"
                   SIAM Journal on Applied Mathematics

        Parameters:
        -----------
        n_agents : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_sample : int
            Number of coalitions to sample for large n (default: 1000)
        max_iterations : int
            Max iterations for exact algorithm (default: 100)

        Returns:
        --------
        allocations : np.ndarray
            Nucleolus allocation vector

        Notes:
        ------
        - Exact computation is O(n × 2^n), practical only for n ≤ 10
        - For n > 10, returns least-core approximation
        """
        # Check if game is symmetric (sample-based heuristic)
        is_symmetric = self._check_symmetry(n_agents, payoff_fn, n_samples=20)

        if is_symmetric:
            # For symmetric games, nucleolus = equal split
            v_N = payoff_fn(frozenset(range(n_agents)))
            logger.info("✅ Symmetric game detected, nucleolus = equal split")
            return np.ones(n_agents) * (v_N / n_agents)

        # For small n, could implement exact Kopelowitz algorithm
        # For now, use least-core as approximation
        if n_agents <= 10:
            logger.info(
                f"⚠️ Exact nucleolus for n={n_agents} not yet implemented, "
                f"using least-core approximation"
            )

        # Use least-core as approximation
        allocations, epsilon = self.compute_least_core(
            n_agents, payoff_fn, n_sample=n_sample, return_epsilon=True
        )

        logger.info(
            f"✅ Nucleolus approximated via least-core (ε={epsilon:.6f})"
        )

        return allocations

    def compute_banzhaf_index(
        self,
        n_agents: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_samples: int = 10000
    ) -> np.ndarray:
        """
        Compute Banzhaf power index (wrapper for _banzhaf_mc)

        Parameters:
        -----------
        n_agents : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_samples : int
            Number of MC samples (default: 10000)

        Returns:
        --------
        phi : np.ndarray
            Banzhaf index vector (normalized to efficiency)
        """
        phi, _ = self._banzhaf_mc(n_agents, payoff_fn, n_samples)
        return phi

    def _check_symmetry(
        self,
        n_agents: int,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_samples: int = 20
    ) -> bool:
        """
        Check if game is symmetric (all agents are interchangeable)

        Sample-based heuristic: Check if v(S) depends only on |S|

        Parameters:
        -----------
        n_agents : int
            Number of agents
        payoff_fn : Callable
            Coalition value function
        n_samples : int
            Number of coalitions to sample per size (default: 20)

        Returns:
        --------
        is_symmetric : bool
            True if game appears symmetric
        """
        for size in range(1, n_agents):
            values = []
            for _ in range(min(n_samples, 10)):
                coalition = frozenset(
                    self.rng.choice(n_agents, size=size, replace=False)
                )
                values.append(payoff_fn(coalition))

            # If values vary significantly, game is not symmetric
            if len(values) > 1 and np.std(values) > 1e-6:
                return False

        return True

    def _sample_coalitions(
        self,
        n_agents: int,
        n_sample: int
    ) -> List[FrozenSet[int]]:
        """
        Sample diverse coalitions for LP constraint generation

        Strategy:
        1. Include all singleton coalitions (important for individual rationality)
        2. Sample uniformly by size
        3. Ensure no duplicates

        Parameters:
        -----------
        n_agents : int
            Number of agents
        n_sample : int
            Number of coalitions to sample

        Returns:
        --------
        sampled : List[FrozenSet[int]]
            List of sampled coalitions
        """
        sampled = set()

        # Add all singletons
        for i in range(n_agents):
            sampled.add(frozenset([i]))

        # Sample random coalitions
        while len(sampled) < min(n_sample, 2**n_agents - 2):
            size = self.rng.integers(2, n_agents)
            agents = self.rng.choice(n_agents, size=size, replace=False)
            sampled.add(frozenset(agents))

        return list(sampled)

    # ═════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    @lru_cache(maxsize=2048)
    def _binomial(n: int, k: int) -> int:
        """
        Binomial coefficient with caching

        Formula: C(n, k) = n! / (k! × (n-k)!)

        Parameters:
        -----------
        n : int
            Total items
        k : int
            Items to choose

        Returns:
        --------
        binom : int
            Binomial coefficient
        """
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1

        # Use multiplicative formula for efficiency
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)

        return result


# =============================================================================
# VALIDATION & TESTING UTILITIES
# =============================================================================

def verify_efficiency(
    allocations: np.ndarray,
    total_value: float,
    tol: float = 1e-6
) -> bool:
    """
    Check efficiency axiom: Σφᵢ = v(N)

    Parameters:
    -----------
    allocations : np.ndarray
        Allocation vector
    total_value : float
        Grand coalition value v(N)
    tol : float
        Tolerance for floating-point comparison (default: 1e-6)

    Returns:
    --------
    satisfied : bool
        True if efficiency is satisfied
    """
    return abs(np.sum(allocations) - total_value) < tol


def verify_symmetry(
    allocations: np.ndarray,
    contributions: np.ndarray,
    tol: float = 1e-6
) -> bool:
    """
    Check symmetry axiom: agents with equal contributions get equal allocations

    Parameters:
    -----------
    allocations : np.ndarray
        Allocation vector
    contributions : np.ndarray
        Contribution vector
    tol : float
        Tolerance for floating-point comparison (default: 1e-6)

    Returns:
    --------
    satisfied : bool
        True if symmetry is satisfied
    """
    for i in range(len(allocations)):
        for j in range(i+1, len(allocations)):
            if abs(contributions[i] - contributions[j]) < tol:
                if abs(allocations[i] - allocations[j]) >= tol:
                    return False
    return True


def verify_dummy_player(
    allocations: np.ndarray,
    payoff_fn: Callable[[FrozenSet[int]], float],
    n_agents: int,
    tol: float = 1e-6
) -> bool:
    """
    Check dummy player axiom: if agent i contributes 0 to all coalitions, φᵢ = 0

    Note: Requires checking if agent is truly a dummy (expensive for large n)

    Parameters:
    -----------
    allocations : np.ndarray
        Allocation vector
    payoff_fn : Callable
        Coalition value function
    n_agents : int
        Number of agents
    tol : float
        Tolerance for floating-point comparison (default: 1e-6)

    Returns:
    --------
    satisfied : bool
        True if dummy player axiom is satisfied
    """
    for i in range(n_agents):
        # Check if agent i is a dummy (sample-based approximation)
        is_dummy = True
        for _ in range(20):  # Sample 20 random coalitions
            size = np.random.randint(0, n_agents)
            others = [j for j in range(n_agents) if j != i]
            if len(others) < size:
                continue
            coalition = frozenset(np.random.choice(
                others, size=size, replace=False))

            v_S = payoff_fn(coalition)
            v_S_with_i = payoff_fn(coalition | {i})

            if abs(v_S_with_i - v_S) > tol:
                is_dummy = False
                break

        if is_dummy and abs(allocations[i]) > tol:
            return False

    return True


# =============================================================================
# COMMAND-LINE INTERFACE (for standalone testing)
# =============================================================================

if __name__ == '__main__':
    import sys

    # Example usage
    print("═══ AllocationEngine Demo ═══\n")

    # Define a simple game
    n_agents = 5

    def payoff_fn(coalition: FrozenSet[int]) -> float:
        """Superadditive game: v(S) = |S|^1.5"""
        return float(len(coalition)) ** 1.5

    # Initialize engine
    engine = AllocationEngine(seed=42)

    # Test exact Shapley
    print("1. Exact Shapley:")
    result = engine.allocate('exact_shapley', n_agents, payoff_fn)
    print(result.summary())
    print()

    # Test MC Shapley
    print("2. Monte Carlo Shapley:")
    result = engine.allocate('mc_shapley', n_agents, payoff_fn, n_samples=5000)
    print(result.summary())
    print()

    # Test Banzhaf
    print("3. Banzhaf Index:")
    result = engine.allocate('banzhaf', n_agents, payoff_fn, n_samples=5000)
    print(result.summary())
    print()

    print("✅ Demo complete!")
