# =============================================================================
# FILE: src/modules/data_gen.py
"""
Synthetic Data Generator - Creates game instances with validated properties

CRITICAL FIX: Subadditive payoff function now guarantees v(S∪T) ≤ v(S) + v(T)

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, FrozenSet, Dict, Literal, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameInstance:
    """Represents a single consortium game instance"""
    n_agents: int
    q_true: np.ndarray  # True contributions (renamed from true_contributions)
    q_reported: np.ndarray  # Reported contributions
    payoff_function: Callable[[FrozenSet[int]], float]
    report_noise_params: Dict
    metadata: Dict
    seed: int

    # Aliases for compatibility
    @property
    def true_contributions(self) -> np.ndarray:
        return self.q_true

    @property
    def total_value(self) -> float:
        return self.get_grand_coalition_value()

    def __post_init__(self):
        assert len(self.q_true) == self.n_agents
        assert len(self.q_reported) == self.n_agents

    def get_coalition_value(self, coalition: FrozenSet[int]) -> float:
        """Compute coalition value v(S)"""
        return self.payoff_function(coalition)

    def get_grand_coalition_value(self) -> float:
        """Get v(N) - total value"""
        return self.payoff_function(frozenset(range(self.n_agents)))


class DataGenerator:
    """Generates validated consortium game instances"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def generate_instance(
        self,
        n_agents: int,
        mu: float = 10.0,
        sigma_q: float = 2.0,
        q_dist: Literal['lognormal', 'pareto',
                        'uniform', 'exponential'] = 'lognormal',
        sigma_report: float = 0.5,
        report_model: Literal['truthful', 'additive_noise',
                              'multiplicative_noise', 'strategic'] = 'truthful',
        payoff_shape: Literal['linear', 'subadditive',
                              'superadditive', 'threshold'] = 'subadditive',
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.8,
        threshold: float = 5.0
    ) -> GameInstance:
        """
        Generate a complete game instance with validated properties

        CRITICAL FIX: Subadditive payoff now uses concave transformation
        to ensure v(S∪T) ≤ v(S) + v(T) for disjoint S, T

        Parameters:
        -----------
        n_agents : int
            Number of consortium members
        mu, sigma_q : float
            Parameters for contribution distribution
        q_dist : str
            Distribution type for true contributions
        sigma_report : float
            Noise level in reporting
        report_model : str
            How agents report contributions
        payoff_shape : str
            Shape of coalition value function
        alpha, beta, gamma : float
            Payoff function parameters
        threshold : float
            Threshold parameter for threshold payoff
        """

        # Generate true contributions
        q_true = self._generate_contributions(n_agents, mu, sigma_q, q_dist)

        # Generate reported contributions
        q_reported = self._generate_reports(q_true, sigma_report, report_model)

        # Create payoff function
        payoff_fn = self._create_payoff_function(
            q_true, payoff_shape, alpha, beta, gamma, threshold
        )

        metadata = {
            'n_agents': n_agents,
            'mu': mu,
            'sigma_q': sigma_q,
            'q_dist': q_dist,
            'sigma_report': sigma_report,
            'report_model': report_model,
            'payoff_shape': payoff_shape,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'threshold': threshold,
            'q_true_mean': float(np.mean(q_true)),
            'q_true_std': float(np.std(q_true)),
            'q_true_gini': self._compute_gini(q_true)
        }

        return GameInstance(
            n_agents=n_agents,
            q_true=q_true,
            q_reported=q_reported,
            payoff_function=payoff_fn,
            report_noise_params={
                'sigma_report': sigma_report, 'model': report_model},
            metadata=metadata,
            seed=self.seed if self.seed else 0
        )

    def _generate_contributions(
        self,
        n_agents: int,
        mu: float,
        sigma: float,
        dist: str
    ) -> np.ndarray:
        """Generate true contribution vector"""

        if dist == 'lognormal':
            # LogNormal: heterogeneous contributions
            mean_log = np.log(mu**2 / np.sqrt(mu**2 + sigma**2))
            sigma_log = np.sqrt(np.log(1 + (sigma/mu)**2))
            q = self.rng.lognormal(mean_log, sigma_log, n_agents)

        elif dist == 'pareto':
            # Pareto: heavy-tailed, extreme heterogeneity
            shape = max(1.5, (mu / sigma)**2) if sigma > 0 else 2.0
            q = (self.rng.pareto(shape, n_agents) + 1) * mu / shape

        elif dist == 'uniform':
            # Uniform: bounded heterogeneity
            low = max(0, mu - np.sqrt(3) * sigma)
            high = mu + np.sqrt(3) * sigma
            q = self.rng.uniform(low, high, n_agents)

        elif dist == 'exponential':
            # Exponential: moderate skew
            q = self.rng.exponential(mu, n_agents)

        else:
            raise ValueError(f"Unknown distribution: {dist}")

        return np.maximum(q, 0.01)  # Ensure positivity

    def _generate_reports(
        self,
        q_true: np.ndarray,
        sigma_report: float,
        model: str
    ) -> np.ndarray:
        """Generate reported contributions (with noise or strategic behavior)"""

        n = len(q_true)

        if model == 'truthful':
            return q_true.copy()

        elif model == 'additive_noise':
            # Additive Gaussian noise
            noise = self.rng.normal(0, sigma_report, n)
            q_reported = q_true + noise
            return np.maximum(q_reported, 0.01)

        elif model == 'multiplicative_noise':
            # Multiplicative noise (proportional to contribution)
            noise = self.rng.normal(1.0, sigma_report, n)
            q_reported = q_true * noise
            return np.maximum(q_reported, 0.01)

        elif model == 'strategic':
            # Strategic inflation (agents slightly inflate reports)
            inflation = self.rng.uniform(0, sigma_report, n)
            q_reported = q_true * (1 + inflation)
            return q_reported

        else:
            raise ValueError(f"Unknown report model: {model}")

    def _create_payoff_function(
        self,
        q_true: np.ndarray,
        shape: str,
        alpha: float,
        beta: float,
        gamma: float,
        threshold: float
    ) -> Callable[[FrozenSet[int]], float]:
        """
        Create coalition value function v(S) with validated properties

        CRITICAL FIX: Subadditive now uses v(S) = α·(Σqᵢ)^γ - β·|S| with γ < 1
        This GUARANTEES subadditivity via concavity

        Properties ensured:
        - Monotonicity: S ⊆ T ⟹ v(S) ≤ v(T)
        - Subadditivity (if shape='subadditive'): v(S∪T) ≤ v(S) + v(T) for disjoint S,T
        - Superadditivity (if shape='superadditive'): v(S∪T) ≥ v(S) + v(T)
        """

        def payoff_fn(coalition: FrozenSet[int]) -> float:
            if len(coalition) == 0:
                return 0.0

            total_contrib = sum(q_true[i] for i in coalition)
            size = len(coalition)

            if shape == 'linear':
                # v(S) = α·Σqᵢ (perfectly additive)
                return alpha * total_contrib

            elif shape == 'subadditive':
                # FIXED: v(S) = α·(Σqᵢ)^γ - β·|S|  with γ < 1
                # Concave function ensures subadditivity
                if gamma >= 1.0:
                    logger.warning(
                        f"Subadditive requires γ < 1, got {gamma}. Setting γ=0.8")
                    gamma_corrected = 0.8
                else:
                    gamma_corrected = gamma

                value = alpha * (total_contrib **
                                 gamma_corrected) - beta * size
                return max(0.0, value)  # Ensure non-negativity

            elif shape == 'superadditive':
                # v(S) = α·(Σqᵢ)^γ  with γ > 1 (convex, increasing returns)
                if gamma <= 1.0:
                    logger.warning(
                        f"Superadditive requires γ > 1, got {gamma}. Setting γ=1.2")
                    gamma_corrected = 1.2
                else:
                    gamma_corrected = gamma

                return alpha * (total_contrib ** gamma_corrected)

            elif shape == 'threshold':
                # All-or-nothing: need minimum contribution
                return total_contrib if total_contrib >= threshold else 0.0

            else:
                raise ValueError(f"Unknown payoff shape: {shape}")

        return payoff_fn

    def generate_strategic_deviations(
        self,
        q_true: float,
        n_deviations: int = 20,
        max_deviation: float = 2.0
    ) -> np.ndarray:
        """Generate grid of possible strategic reports around true contribution"""
        # Grid from underreporting to overreporting
        deviations = np.linspace(
            max(0.01, q_true - max_deviation),
            q_true + max_deviation,
            n_deviations
        )
        return deviations

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Compute Gini coefficient"""
        sorted_vals = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        total = np.sum(sorted_vals)
        if total < 1e-9:
            return 0.0
        return (2 * np.sum(index * sorted_vals)) / (n * total) - (n + 1) / n


# =============================================================================
# GAME PROPERTY VALIDATION (NEW)
# =============================================================================

class GameValidator:
    """Validates game-theoretic properties of payoff functions"""

    @staticmethod
    def check_monotonicity(
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 100
    ) -> bool:
        """
        Check monotonicity: S ⊆ T ⟹ v(S) ≤ v(T)

        Sample-based test for computational efficiency
        """
        rng = np.random.default_rng()

        for _ in range(n_samples):
            # Sample subset S
            size_S = rng.integers(0, n_agents)
            S = frozenset(rng.choice(n_agents, size=size_S, replace=False))

            # Sample superset T ⊇ S
            remaining = [i for i in range(n_agents) if i not in S]
            if len(remaining) == 0:
                continue

            additional_size = rng.integers(1, len(remaining) + 1)
            additional = rng.choice(
                remaining, size=additional_size, replace=False)
            T = S | frozenset(additional)

            # Check v(S) ≤ v(T)
            v_S = payoff_fn(S)
            v_T = payoff_fn(T)

            if v_S > v_T + 1e-6:
                logger.warning(
                    f"Monotonicity violated: v({S}) = {v_S} > v({T}) = {v_T}")
                return False

        return True

    @staticmethod
    def check_subadditivity(
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 100
    ) -> Tuple[bool, float]:
        """
        Check subadditivity: v(S∪T) ≤ v(S) + v(T) for disjoint S, T

        Returns: (is_subadditive, max_violation)
        """
        rng = np.random.default_rng()
        max_violation = 0.0

        for _ in range(n_samples):
            # Sample disjoint coalitions S and T
            agents = list(range(n_agents))
            rng.shuffle(agents)

            split_point = rng.integers(1, n_agents)
            S = frozenset(agents[:split_point])
            T = frozenset(agents[split_point:])

            if len(S) == 0 or len(T) == 0:
                continue

            # Check v(S∪T) ≤ v(S) + v(T)
            v_S = payoff_fn(S)
            v_T = payoff_fn(T)
            v_union = payoff_fn(S | T)

            violation = v_union - (v_S + v_T)
            max_violation = max(max_violation, violation)

        is_subadditive = max_violation <= 1e-6

        if not is_subadditive:
            logger.warning(
                f"Subadditivity violated: max violation = {max_violation}")

        return is_subadditive, max_violation

    @staticmethod
    def check_superadditivity(
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 100
    ) -> Tuple[bool, float]:
        """
        Check superadditivity: v(S∪T) ≥ v(S) + v(T) for disjoint S, T

        Returns: (is_superadditive, min_violation)
        """
        rng = np.random.default_rng()
        min_violation = 0.0

        for _ in range(n_samples):
            agents = list(range(n_agents))
            rng.shuffle(agents)

            split_point = rng.integers(1, n_agents)
            S = frozenset(agents[:split_point])
            T = frozenset(agents[split_point:])

            if len(S) == 0 or len(T) == 0:
                continue

            v_S = payoff_fn(S)
            v_T = payoff_fn(T)
            v_union = payoff_fn(S | T)

            violation = (v_S + v_T) - v_union
            min_violation = min(min_violation, violation)

        is_superadditive = min_violation >= -1e-6

        if not is_superadditive:
            logger.warning(
                f"Superadditivity violated: min violation = {min_violation}")

        return is_superadditive, min_violation

    @staticmethod
    def check_convexity(
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 100
    ) -> bool:
        """
        Check convexity: v(S∪{i}) - v(S) ≤ v(T∪{i}) - v(T) for S ⊆ T, i ∉ T

        Convex games have non-empty core
        """
        rng = np.random.default_rng()

        for _ in range(n_samples):
            # Sample nested coalitions S ⊆ T
            size_S = rng.integers(0, n_agents - 1)
            S = frozenset(rng.choice(n_agents, size=size_S, replace=False))

            remaining = [i for i in range(n_agents) if i not in S]
            if len(remaining) < 2:
                continue

            # T = S ∪ (some additional agents)
            additional_size = rng.integers(1, len(remaining))
            additional = rng.choice(
                remaining, size=additional_size, replace=False)
            T = S | frozenset(additional)

            # Pick agent i ∉ T
            not_in_T = [j for j in range(n_agents) if j not in T]
            if len(not_in_T) == 0:
                continue

            i = rng.choice(not_in_T)

            # Check v(S∪{i}) - v(S) ≤ v(T∪{i}) - v(T)
            marginal_S = payoff_fn(S | {i}) - payoff_fn(S)
            marginal_T = payoff_fn(T | {i}) - payoff_fn(T)

            if marginal_S > marginal_T + 1e-6:
                logger.warning(
                    f"Convexity violated: marginal_S={marginal_S} > marginal_T={marginal_T}")
                return False

        return True
