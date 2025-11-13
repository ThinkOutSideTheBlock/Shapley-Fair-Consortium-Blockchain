# =============================================================================
# FILE: src/utils/statistical.py
"""
Statistical Testing and Analysis

Implements:
- ANOVA (one-way, two-way)
- Tukey HSD post-hoc tests
- Cohen's d effect sizes
- Confidence intervals
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney)

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import logging

logger = logging.getLogger(__name__)


class StatisticalTests:
    """Statistical hypothesis testing utilities"""

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for hypothesis tests
        """
        self.alpha = alpha

    def one_way_anova(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor: str
    ) -> Dict:
        """
        Perform one-way ANOVA

        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable column
            factor: Name of factor/group column

        Returns:
            Dict with F-statistic, p-value, effect size
        """
        # Group data
        groups = [group[dependent_var].values
                  for name, group in data.groupby(factor)]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        grand_mean = data[dependent_var].mean()
        ss_between = sum(
            len(group) * (group[dependent_var].mean() - grand_mean)**2
            for name, group in data.groupby(factor)
        )
        ss_total = sum((data[dependent_var] - grand_mean)**2)
        eta_squared = ss_between / ss_total

        # Interpret result
        significant = p_value < self.alpha

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': significant,
            'eta_squared': eta_squared,
            'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
            'n_groups': len(groups),
            'n_observations': len(data)
        }

    def two_way_anova(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor1: str,
        factor2: str
    ) -> Dict:
        """
        Perform two-way ANOVA with interaction

        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable
            factor1: First factor/group column
            factor2: Second factor/group column

        Returns:
            Dict with main effects and interaction results
        """
        # Build formula
        formula = f'{dependent_var} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})'

        # Fit OLS model
        model = ols(formula, data=data).fit()

        # Perform ANOVA
        anova_table = anova_lm(model, typ=2)

        # Extract results
        results = {
            'main_effect_1': {
                'factor': factor1,
                'f_statistic': anova_table.loc[f'C({factor1})', 'F'],
                'p_value': anova_table.loc[f'C({factor1})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor1})', 'PR(>F)'] < self.alpha
            },
            'main_effect_2': {
                'factor': factor2,
                'f_statistic': anova_table.loc[f'C({factor2})', 'F'],
                'p_value': anova_table.loc[f'C({factor2})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor2})', 'PR(>F)'] < self.alpha
            },
            'interaction': {
                'factors': f'{factor1} × {factor2}',
                'f_statistic': anova_table.loc[f'C({factor1}):C({factor2})', 'F'],
                'p_value': anova_table.loc[f'C({factor1}):C({factor2})', 'PR(>F)'],
                'significant': anova_table.loc[f'C({factor1}):C({factor2})', 'PR(>F)'] < self.alpha
            },
            'model_r_squared': model.rsquared,
            'anova_table': anova_table
        }

        return results

    def tukey_hsd(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor: str
    ) -> Dict:
        """
        Perform Tukey HSD post-hoc test

        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable
            factor: Name of factor/group column

        Returns:
            Dict with pairwise comparison results
        """
        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(
            endog=data[dependent_var],
            groups=data[factor],
            alpha=self.alpha
        )

        # Extract pairwise comparisons
        comparisons = []
        for i in range(len(tukey_result.summary().data) - 1):  # Skip header
            row = tukey_result.summary().data[i + 1]
            comparisons.append({
                'group1': row[0],
                'group2': row[1],
                'mean_diff': float(row[2]),
                'lower_ci': float(row[3]),
                'upper_ci': float(row[4]),
                'reject_null': row[5],
                'significant': row[5]
            })

        return {
            'comparisons': comparisons,
            'summary': str(tukey_result),
            'n_comparisons': len(comparisons)
        }

    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True
    ) -> Dict:
        """
        Compute Cohen's d effect size

        Args:
            group1: First group data
            group2: Second group data
            pooled: Use pooled standard deviation

        Returns:
            Dict with Cohen's d and interpretation
        """
        mean1, mean2 = group1.mean(), group2.mean()

        if pooled:
            n1, n2 = len(group1), len(group2)
            var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
            pooled_std = np.sqrt(
                ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            d = (mean1 - mean2) / pooled_std
        else:
            d = (mean1 - mean2) / group1.std(ddof=1)

        interpretation = self._interpret_cohens_d(abs(d))

        return {
            'cohens_d': d,
            'mean_diff': mean1 - mean2,
            'interpretation': interpretation,
            'magnitude': abs(d)
        }

    def confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95
    ) -> Dict:
        """
        Compute confidence interval for mean

        Args:
            data: Sample data
            confidence: Confidence level (0-1)

        Returns:
            Dict with mean, CI bounds, and margin of error
        """
        n = len(data)
        mean = data.mean()
        std_err = stats.sem(data)

        # t-distribution for small samples
        df = n - 1
        t_crit = stats.t.ppf((1 + confidence) / 2, df)
        margin_error = t_crit * std_err

        ci_lower = mean - margin_error
        ci_upper = mean + margin_error

        return {
            'mean': mean,
            'confidence_level': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_of_error': margin_error,
            'std_error': std_err,
            'n': n
        }

    def kruskal_wallis(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        factor: str
    ) -> Dict:
        """
        Perform Kruskal-Wallis H-test (non-parametric ANOVA)

        Args:
            data: DataFrame with experimental data
            dependent_var: Name of dependent variable
            factor: Name of factor/group column

        Returns:
            Dict with H-statistic, p-value
        """
        groups = [group[dependent_var].values
                  for name, group in data.groupby(factor)]

        h_stat, p_value = stats.kruskal(*groups)
        significant = p_value < self.alpha

        return {
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant': significant,
            'n_groups': len(groups),
            'test_type': 'non-parametric'
        }

    def mann_whitney_u(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """
        Perform Mann-Whitney U test (non-parametric t-test)

        Args:
            group1: First group data
            group2: Second group data
            alternative: 'two-sided', 'less', or 'greater'

        Returns:
            Dict with U-statistic, p-value
        """
        u_stat, p_value = stats.mannwhitneyu(
            group1, group2, alternative=alternative
        )

        significant = p_value < self.alpha

        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        r = 1 - (2 * u_stat) / (n1 * n2)

        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'significant': significant,
            'effect_size_r': r,
            'alternative': alternative
        }

    def normality_test(
        self,
        data: np.ndarray,
        test: str = 'shapiro'
    ) -> Dict:
        """
        Test normality of data

        Args:
            data: Sample data
            test: 'shapiro' or 'kstest'

        Returns:
            Dict with test statistic, p-value, normality conclusion
        """
        if test == 'shapiro':
            stat, p_value = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
        elif test == 'kstest':
            stat, p_value = stats.kstest(data, 'norm')
            test_name = 'Kolmogorov-Smirnov'
        else:
            raise ValueError(f"Unknown test: {test}")

        normal = p_value > self.alpha

        return {
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'is_normal': normal,
            'conclusion': 'Data appears normal' if normal else 'Data not normal'
        }

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_sq < 0.01:
            return 'negligible'
        elif eta_sq < 0.06:
            return 'small'
        elif eta_sq < 0.14:
            return 'medium'
        else:
            return 'large'

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'


# =============================================================================
# POWER ANALYSIS
# =============================================================================

class PowerAnalysis:
    """Statistical power analysis for sample size determination"""

    @staticmethod
    def power_t_test(
        effect_size: float,
        n: int,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Compute statistical power for t-test

        Args:
            effect_size: Cohen's d
            n: Sample size per group
            alpha: Significance level
            alternative: 'two-sided' or 'one-sided'

        Returns:
            Statistical power (0-1)
        """
        from statsmodels.stats.power import ttest_power

        return ttest_power(
            effect_size=effect_size,
            nobs=n,
            alpha=alpha,
            alternative=alternative
        )

    @staticmethod
    def sample_size_t_test(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> int:
        """
        Compute required sample size for desired power

        Args:
            effect_size: Cohen's d
            power: Desired statistical power
            alpha: Significance level
            alternative: 'two-sided' or 'one-sided'

        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import tt_solve_power

        n = tt_solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative=alternative
        )

        return int(np.ceil(n))


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    'StatisticalTests',
    'PowerAnalysis'
]
