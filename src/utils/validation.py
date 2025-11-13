"""
Validation utilities for Shapley-Fair Consortium Blockchain Research
"""
import numpy as np
from typing import Callable, FrozenSet, Dict, Any, Optional
from dataclasses import dataclass
from scipy.special import comb
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class GameValidator:
    """
    Validates cooperative game properties (monotonicity, superadditivity, etc.)
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def validate_monotonicity(
        self,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 1000
    ) -> ValidationResult:
        """
        Validate monotonicity: v(S) ≤ v(T) for S ⊆ T
        
        Args:
            payoff_fn: Coalition value function
            n_agents: Number of agents
            n_samples: Number of random samples to test
        
        Returns:
            ValidationResult with validation status
        """
        from itertools import combinations
        
        # For small n, check all pairs of subsets
        if n_agents <= 8:
            agents = list(range(n_agents))
            for size_s in range(n_agents):
                for s_agents in combinations(agents, size_s):
                    S = set(s_agents)
                    v_s = payoff_fn(frozenset(S))
                    
                    for size_t in range(size_s + 1, n_agents + 1):
                        for t_agents in combinations(agents, size_t):
                            T = set(t_agents)
                            if S.issubset(T):
                                v_t = payoff_fn(frozenset(T))
                                if v_s > v_t + self.tolerance:
                                    return ValidationResult(
                                        is_valid=False,
                                        error=f"Monotonicity violation: v({S})={v_s:.4f} > v({T})={v_t:.4f}"
                                    )
        else:
            # For larger n, sample randomly
            rng = np.random.default_rng()
            for _ in range(n_samples):
                # Sample random subset S
                S_size = rng.integers(0, n_agents)
                S_agents = set(rng.choice(n_agents, size=S_size, replace=False))
                
                # Sample larger superset T
                remaining_agents = [i for i in range(n_agents) if i not in S_agents]
                add_count = rng.integers(0, len(remaining_agents))
                add_agents = rng.choice(remaining_agents, size=add_count, replace=False)
                T_agents = S_agents.union(set(add_agents))
                
                v_s = payoff_fn(frozenset(S_agents))
                v_t = payoff_fn(frozenset(T_agents))
                
                if v_s > v_t + self.tolerance:
                    return ValidationResult(
                        is_valid=False,
                        error=f"Monotonicity violation: v({S_agents})={v_s:.4f} > v({T_agents})={v_t:.4f}"
                    )
        
        return ValidationResult(is_valid=True)
    
    def validate_superadditivity(
        self,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 1000
    ) -> ValidationResult:
        """
        Validate superadditivity: v(S ∪ T) ≥ v(S) + v(T) for disjoint S, T
        
        Args:
            payoff_fn: Coalition value function
            n_agents: Number of agents
            n_samples: Number of random samples to test
        
        Returns:
            ValidationResult with validation status
        """
        rng = np.random.default_rng()
        
        for _ in range(n_samples):
            # Sample two disjoint sets S, T
            agents = list(range(n_agents))
            S_size = rng.integers(0, n_agents // 2 + 1)
            S = set(rng.choice(agents, size=S_size, replace=False))
            
            remaining_agents = [i for i in agents if i not in S]
            T_size = rng.integers(0, len(remaining_agents))
            T = set(rng.choice(remaining_agents, size=T_size, replace=False))
            
            if len(S) == 0 and len(T) == 0:
                continue  # Skip if both empty
            
            v_s = payoff_fn(frozenset(S))
            v_t = payoff_fn(frozenset(T))
            v_union = payoff_fn(frozenset(S.union(T)))
            
            if v_union < v_s + v_t - self.tolerance:
                return ValidationResult(
                    is_valid=False,
                    error=f"Superadditivity violation: v({S}∪{T})={v_union:.4f} < v({S})+v({T})={v_s + v_t:.4f}"
                )
        
        return ValidationResult(is_valid=True)
    
    def validate_subadditivity(
        self,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int,
        n_samples: int = 1000
    ) -> ValidationResult:
        """
        Validate subadditivity: v(S ∪ T) ≤ v(S) + v(T) for disjoint S, T
        
        Args:
            payoff_fn: Coalition value function
            n_agents: Number of agents
            n_samples: Number of random samples to test
        
        Returns:
            ValidationResult with validation status
        """
        rng = np.random.default_rng()
        
        for _ in range(n_samples):
            # Sample two disjoint sets S, T
            agents = list(range(n_agents))
            S_size = rng.integers(0, n_agents // 2 + 1)
            S = set(rng.choice(agents, size=S_size, replace=False))
            
            remaining_agents = [i for i in agents if i not in S]
            T_size = rng.integers(0, len(remaining_agents))
            T = set(rng.choice(remaining_agents, size=T_size, replace=False))
            
            if len(S) == 0 and len(T) == 0:
                continue  # Skip if both empty
            
            v_s = payoff_fn(frozenset(S))
            v_t = payoff_fn(frozenset(T))
            v_union = payoff_fn(frozenset(S.union(T)))
            
            if v_union > v_s + v_t + self.tolerance:
                return ValidationResult(
                    is_valid=False,
                    error=f"Subadditivity violation: v({S}∪{T})={v_union:.4f} > v({S})+v({T})={v_s + v_t:.4f}"
                )
        
        return ValidationResult(is_valid=True)


class ShapleyValidator:
    """
    Validates Shapley value properties and computations
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def validate_efficiency(
        self,
        allocations: np.ndarray,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int
    ) -> ValidationResult:
        """
        Validate efficiency: sum of allocations equals v(N)
        
        Args:
            allocations: Computed allocations for each agent
            payoff_fn: Coalition value function
            n_agents: Number of agents
        
        Returns:
            ValidationResult with validation status
        """
        grand_coalition_value = payoff_fn(frozenset(range(n_agents)))
        allocation_sum = np.sum(allocations)
        
        if abs(allocation_sum - grand_coalition_value) > self.tolerance:
            return ValidationResult(
                is_valid=False,
                error=f"Efficiency violation: sum(x)={allocation_sum:.4f} != v(N)={grand_coalition_value:.4f}",
                details={
                    'allocation_sum': allocation_sum,
                    'grand_coalition_value': grand_coalition_value,
                    'difference': allocation_sum - grand_coalition_value
                }
            )
        
        return ValidationResult(is_valid=True)
    
    def validate_symmetry(
        self,
        allocations: np.ndarray,
        contributions: np.ndarray
    ) -> ValidationResult:
        """
        Validate symmetry: agents with identical contributions get identical allocations
        
        Args:
            allocations: Computed allocations for each agent
            contributions: Agent contributions
        
        Returns:
            ValidationResult with validation status
        """
        # Group agents by contribution similarity
        n_agents = len(allocations)
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if abs(contributions[i] - contributions[j]) < self.tolerance:
                    # If contributions are equal, allocations should be equal
                    if abs(allocations[i] - allocations[j]) > self.tolerance:
                        return ValidationResult(
                            is_valid=False,
                            error=f"Symmetry violation: agents {i} and {j} have similar contributions "
                                  f"({contributions[i]:.4f} vs {contributions[j]:.4f}) "
                                  f"but different allocations ({allocations[i]:.4f} vs {allocations[j]:.4f})"
                        )
        
        return ValidationResult(is_valid=True)
    
    def validate_null_player(
        self,
        allocations: np.ndarray,
        payoff_fn: Callable[[FrozenSet[int]], float],
        n_agents: int
    ) -> ValidationResult:
        """
        Validate null player property: if an agent contributes 0 to all coalitions, allocation should be 0
        
        Args:
            allocations: Computed allocations for each agent
            payoff_fn: Coalition value function
            n_agents: Number of agents
        
        Returns:
            ValidationResult with validation status
        """
        for i in range(n_agents):
            # Check if agent i is a null player by testing marginal contributions
            is_null = True
            for size in range(n_agents):
                for coalition_indices in np.random.choice(
                    [j for j in range(n_agents) if j != i], 
                    size=size, 
                    replace=False
                ):
                    coalition = frozenset([coalition_indices] if isinstance(coalition_indices, int) else coalition_indices)
                    v_without_i = payoff_fn(coalition)
                    v_with_i = payoff_fn(coalition | {i})
                    marginal_contribution = v_with_i - v_without_i
                    
                    if abs(marginal_contribution) > self.tolerance:
                        is_null = False
                        break
                if not is_null:
                    break
            
            if is_null and abs(allocations[i]) > self.tolerance:
                return ValidationResult(
                    is_valid=False,
                    error=f"Null player violation: agent {i} is a null player but gets allocation {allocations[i]:.4f}"
                )
        
        return ValidationResult(is_valid=True)


# Global instances for easy use
game_validator = GameValidator()
shapley_validator = ShapleyValidator()