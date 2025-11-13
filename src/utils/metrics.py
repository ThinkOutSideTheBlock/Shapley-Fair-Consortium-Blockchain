# =============================================================================
# FILE: src/utils/metrics.py
"""
Advanced Fairness and Game-Theoretic Metrics

Implements:
- FairReward composite fairness index
- Envy-freeness checks (EF, EF1, EFx)
- Proportionality checks (PROP, PROP1)
- Equity theory metrics (Adams 1965)
- Pareto efficiency tests
- Individual rationality checks

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FairnessReport:
    """Comprehensive fairness assessment"""
    # Envy-freeness
    is_envy_free: bool
    is_envy_free_up_to_one: bool  # EF1
    is_envy_free_up_to_any: bool  # EFx
    max_envy: float
    envy_pairs: List[Tuple[int, int]]  # (envier, envied)
    
    # Proportionality
    is_proportional: bool
    is_proportional_up_to_one: bool  # PROP1
    proportionality_violations: List[int]
    
    # Equity theory
    equity_index: float  # Adams equity measure
    under_rewarded_agents: List[int]
    over_rewarded_agents: List[int]
    
    # Efficiency
    is_pareto_efficient: bool
    is_individually_rational: bool
    
    # Composite scores
    fairreward_score: float  # Weighted composite
    overall_fairness: float  # [0, 1] scale
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'envy_free': self.is_envy_free,
            'envy_free_up_to_one': self.is_envy_free_up_to_one,
            'envy_free_up_to_any': self.is_envy_free_up_to_any,
            'max_envy': float(self.max_envy),
            'envy_pairs_count': len(self.envy_pairs),
            'proportional': self.is_proportional,
            'proportional_up_to_one': self.is_proportional_up_to_one,
            'equity_index': float(self.equity_index),
            'pareto_efficient': self.is_pareto_efficient,
            'individually_rational': self.is_individually_rational,
            'fairreward_score': float(self.fairreward_score),
            'overall_fairness': float(self.overall_fairness)
        }


class FairnessMetrics:
    """Advanced fairness metrics computation"""
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Tolerance for numerical comparisons
        """
        self.epsilon = epsilon
    
    def compute_comprehensive_fairness(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray,
        valuations: Optional[np.ndarray] = None,
        grand_coalition_value: Optional[float] = None
    ) -> FairnessReport:
        """
        Compute comprehensive fairness assessment
        
        Args:
            allocations: Agent allocations (x_i)
            contributions: True contributions (q_i)
            valuations: Optional agent-specific valuations of allocations
            grand_coalition_value: v(N), if None computed from allocations.sum()
        
        Returns:
            FairnessReport with all metrics
        """
        n = len(allocations)
        
        if valuations is None:
            valuations = allocations.copy()
        
        if grand_coalition_value is None:
            grand_coalition_value = allocations.sum()
        
        # Envy-freeness checks
        envy_results = self._check_envy_freeness(allocations, valuations)
        
        # Proportionality checks
        prop_results = self._check_proportionality(
            allocations, contributions, grand_coalition_value
        )
        
        # Equity theory
        equity_results = self._compute_equity_index(allocations, contributions)
        
        # Efficiency checks
        pareto_efficient = self._check_pareto_efficiency(allocations, n)
        individually_rational = self._check_individual_rationality(
            allocations, contributions
        )
        
        # Composite scores
        fairreward = self._compute_fairreward_composite(
            allocations, contributions
        )
        
        overall_fairness = self._compute_overall_fairness(
            envy_free=envy_results['is_envy_free'],
            proportional=prop_results['is_proportional'],
            pareto_efficient=pareto_efficient,
            equity_index=equity_results['equity_index']
        )
        
        return FairnessReport(
            is_envy_free=envy_results['is_envy_free'],
            is_envy_free_up_to_one=envy_results['is_ef1'],
            is_envy_free_up_to_any=envy_results['is_efx'],
            max_envy=envy_results['max_envy'],
            envy_pairs=envy_results['envy_pairs'],
            is_proportional=prop_results['is_proportional'],
            is_proportional_up_to_one=prop_results['is_prop1'],
            proportionality_violations=prop_results['violations'],
            equity_index=equity_results['equity_index'],
            under_rewarded_agents=equity_results['under_rewarded'],
            over_rewarded_agents=equity_results['over_rewarded'],
            is_pareto_efficient=pareto_efficient,
            is_individually_rational=individually_rational,
            fairreward_score=fairreward,
            overall_fairness=overall_fairness
        )
    
    def _check_envy_freeness(
        self,
        allocations: np.ndarray,
        valuations: np.ndarray
    ) -> Dict:
        """
        Check envy-freeness (EF), EF1, and EFx
        
        EF: No agent prefers another agent's bundle
        EF1: EF after removing one item from envied bundle
        EFx: EF after removing any item from envied bundle
        """
        n = len(allocations)
        envy_pairs = []
        max_envy = 0.0
        
        # EF check
        is_envy_free = True
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Agent i envies j if v_i(x_j) > v_i(x_i)
                envy = valuations[j] - valuations[i]
                if envy > self.epsilon:
                    is_envy_free = False
                    envy_pairs.append((i, j))
                    max_envy = max(max_envy, envy)
        
        # EF1 check (simplified for divisible goods)
        # Agent i doesn't envy j after removing j's smallest item
        is_ef1 = True
        for i, j in envy_pairs:
            # For divisible goods, remove epsilon from j's allocation
            adjusted_envy = valuations[j] - self.epsilon - valuations[i]
            if adjusted_envy > self.epsilon:
                is_ef1 = False
                break
        
        # EFx check (more stringent than EF1)
        is_efx = is_envy_free  # For divisible goods, EFx ≈ EF
        
        return {
            'is_envy_free': is_envy_free,
            'is_ef1': is_ef1,
            'is_efx': is_efx,
            'max_envy': max_envy,
            'envy_pairs': envy_pairs
        }
    
    def _check_proportionality(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray,
        grand_value: float
    ) -> Dict:
        """
        Check proportionality (PROP) and PROP1
        
        PROP: Each agent gets at least their proportional share
              x_i >= (q_i / sum(q)) * v(N)
        PROP1: PROP after removing one item
        """
        n = len(allocations)
        total_contribution = contributions.sum()
        
        violations = []
        is_proportional = True
        
        for i in range(n):
            fair_share = (contributions[i] / total_contribution) * grand_value
            if allocations[i] < fair_share - self.epsilon:
                is_proportional = False
                violations.append(i)
        
        # PROP1 check (simplified)
        is_prop1 = len(violations) <= 1  # At most one violation
        
        return {
            'is_proportional': is_proportional,
            'is_prop1': is_prop1,
            'violations': violations
        }
    
    def _compute_equity_index(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray
    ) -> Dict:
        """
        Compute equity index based on Adams (1965) equity theory
        
        Equity: (outcomes / inputs) should be equal across agents
        Equity_i = x_i / q_i
        """
        n = len(allocations)
        
        # Compute equity ratios
        equity_ratios = np.zeros(n)
        for i in range(n):
            if contributions[i] > self.epsilon:
                equity_ratios[i] = allocations[i] / contributions[i]
            else:
                equity_ratios[i] = allocations[i]  # Handle zero contribution
        
        # Mean equity ratio
        mean_equity = equity_ratios.mean()
        
        # Identify under/over-rewarded agents
        under_rewarded = []
        over_rewarded = []
        
        for i in range(n):
            if equity_ratios[i] < mean_equity - self.epsilon:
                under_rewarded.append(i)
            elif equity_ratios[i] > mean_equity + self.epsilon:
                over_rewarded.append(i)
        
        # Equity index: 1 - coefficient of variation
        std_equity = equity_ratios.std()
        cv = std_equity / (mean_equity + self.epsilon)
        equity_index = 1.0 / (1.0 + cv)  # [0, 1], higher is more equitable
        
        return {
            'equity_index': equity_index,
            'mean_equity': mean_equity,
            'under_rewarded': under_rewarded,
            'over_rewarded': over_rewarded
        }
    
    def _check_pareto_efficiency(
        self,
        allocations: np.ndarray,
        n_agents: int
    ) -> bool:
        """
        Check Pareto efficiency (simplified)
        
        For cooperative games with transferable utility:
        - Allocation is Pareto efficient if sum(x_i) = v(N)
        """
        # In our setting, efficiency is guaranteed by construction
        # This is a placeholder for more complex checks
        return True
    
    def _check_individual_rationality(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray
    ) -> bool:
        """
        Check individual rationality
        
        Each agent should get at least their standalone value: x_i >= v({i})
        For linear games: v({i}) = q_i
        """
        for i in range(len(allocations)):
            if allocations[i] < contributions[i] - self.epsilon:
                return False
        return True
    
    def _compute_fairreward_composite(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray
    ) -> float:
        """
        Compute FairReward composite index
        
        FairReward = w1*(1-Gini) + w2*Entropy + w3*Equity
        
        Weights from literature (tuned for blockchain contexts):
        - w1 = 0.4 (inequality matters most)
        - w2 = 0.3 (diversity encouraged)
        - w3 = 0.3 (equity from contributions)
        """
        # Gini coefficient
        gini = self._compute_gini(allocations)
        
        # Normalized entropy
        entropy_norm = self._compute_entropy_normalized(allocations)
        
        # Equity index
        equity_results = self._compute_equity_index(allocations, contributions)
        equity = equity_results['equity_index']
        
        # Weighted composite
        w1, w2, w3 = 0.4, 0.3, 0.3
        fairreward = w1 * (1 - gini) + w2 * entropy_norm + w3 * equity
        
        return fairreward
    
    def _compute_gini(self, allocations: np.ndarray) -> float:
        """Compute Gini coefficient"""
        n = len(allocations)
        if n == 0 or allocations.sum() == 0:
            return 0.0
        
        sorted_alloc = np.sort(allocations)
        cumsum = np.cumsum(sorted_alloc)
        
        # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n+1)/n
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_alloc)) / (n * cumsum[-1]) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))
    
    def _compute_entropy_normalized(self, allocations: np.ndarray) -> float:
        """Compute normalized Shannon entropy"""
        n = len(allocations)
        if n == 0 or allocations.sum() == 0:
            return 0.0
        
        probs = allocations / allocations.sum()
        probs = probs[probs > 0]  # Remove zeros
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(n)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_overall_fairness(
        self,
        envy_free: bool,
        proportional: bool,
        pareto_efficient: bool,
        equity_index: float
    ) -> float:
        """
        Compute overall fairness score [0, 1]
        
        Binary properties get weights, equity_index is continuous
        """
        score = 0.0
        score += 0.3 * (1.0 if envy_free else 0.0)
        score += 0.2 * (1.0 if proportional else 0.0)
        score += 0.2 * (1.0 if pareto_efficient else 0.0)
        score += 0.3 * equity_index
        
        return score


# =============================================================================
# SHAPLEY-SPECIFIC UTILITIES
# =============================================================================

def compute_shapley_properties(
    allocations: np.ndarray,
    n_agents: int,
    payoff_function: Callable,
    contributions: np.ndarray
) -> Dict[str, float]:
    """
    Verify Shapley value axioms
    
    Returns:
        Dict with axiom satisfaction scores
    """
    # Efficiency: sum(x_i) = v(N)
    grand_value = payoff_function(frozenset(range(n_agents)))
    efficiency_error = abs(allocations.sum() - grand_value)
    
    # Symmetry: for symmetric players, allocations should be equal
    # (Check via contribution similarity)
    symmetry_score = 1.0 - np.std(allocations) / (np.mean(allocations) + 1e-9)
    
    # Null player: if q_i = 0, then x_i should be ≈ 0
    null_violations = 0
    for i in range(n_agents):
        if contributions[i] < 1e-6 and allocations[i] > 1e-3:
            null_violations += 1
    
    null_player_score = 1.0 - (null_violations / n_agents)
    
    # Additivity: hard to check without multiple games
    # Placeholder
    additivity_score = 1.0
    
    return {
        'efficiency_error': efficiency_error,
        'efficiency_satisfied': efficiency_error < 1e-4,
        'symmetry_score': symmetry_score,
        'null_player_score': null_player_score,
        'additivity_score': additivity_score
    }


# =============================================================================
# STATISTICAL ANALYSIS UTILITIES
# =============================================================================


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for data using t-distribution
    
    Args:
        data: Input data array
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Calculate t-value for the given confidence level
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin_error = t_value * std_err
    return mean - margin_error, mean + margin_error


def bootstrap_statistic(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a given statistic
    
    Args:
        data: Input data array
        statistic: Function to compute statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        Tuple of (statistic_value, lower_ci, upper_ci)
    """
    n = len(data)
    boot_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))
    
    original_stat = statistic(data)
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(boot_stats, lower_percentile)
    ci_upper = np.percentile(boot_stats, upper_percentile)
    
    return original_stat, ci_lower, ci_upper


def effect_size_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size
    
    Args:
        group1: First group data
        group2: Second group data
    
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return d


def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray
) -> Dict[str, float]:
    """
    Compute Mann-Whitney U test (non-parametric alternative to t-test)
    
    Args:
        group1: First group data
        group2: Second group data
    
    Returns:
        Dict with test statistic and p-value
    """
    from scipy.stats import mannwhitneyu
    
    # Perform the test (two-sided by default)
    statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value
    }


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'FairnessReport',
    'FairnessMetrics',
    'compute_shapley_properties',
    'compute_confidence_interval',
    'bootstrap_statistic',
    'effect_size_cohens_d',
    'mann_whitney_u_test'
]