# =============================================================================
# FILE: src/modules/analysis.py
"""
Analysis & Visualization Module with Statistical Testing

New Features:
- ANOVA + Tukey HSD for method comparison
- Effect size computation (Cohen's d, η²)
- Publication-quality figures with statistical annotations

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')


logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

sns.set_style("whitegrid")
sns.set_palette("colorblind")


class AnalysisPipeline:
    """
    Generates publication-ready figures and statistical analyses

    New Methods:
    - compute_statistical_tests(): ANOVA + post-hoc tests
    - compute_effect_sizes(): Cohen's d, η²
    - Enhanced figure generation with significance annotations
    """

    def __init__(self, results_df: pd.DataFrame, output_dir: str):
        self.df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir = self.output_dir / 'tables'
        self.tables_dir.mkdir(exist_ok=True)

    def generate_all_figures(self):
        """Generate complete figure suite for paper"""
        logger.info("Generating all figures...")

        self.figure_1_itm_comparison()
        self.figure_2_fairness_tradeoffs()
        self.figure_3_stability_heatmap()
        self.figure_4_parameter_sensitivity()
        self.figure_5_method_comparison_boxplots()

        # NEW: Statistical comparison figure
        self.figure_6_statistical_comparison()

        logger.info(f"All figures saved to {self.figures_dir}")

    def compute_statistical_tests(
        self,
        metric: str = 'itm_score',
        group_col: str = 'allocation_method'
    ) -> Dict:
        """
        Perform ANOVA + Tukey HSD for method comparison

        Returns:
        --------
        dict with keys:
            - anova_f: F-statistic
            - anova_p: p-value
            - tukey_results: DataFrame of pairwise comparisons
            - effect_size_eta_squared: η² (proportion of variance explained)
        """
        # One-way ANOVA
        groups = [
            self.df[self.df[group_col] == method][metric].dropna()
            for method in self.df[group_col].unique()
        ]

        f_stat, p_value = stats.f_oneway(*groups)

        # Tukey HSD post-hoc test
        tukey_result = pairwise_tukeyhsd(
            endog=self.df[metric],
            groups=self.df[group_col],
            alpha=0.05
        )

        # Convert Tukey results to DataFrame
        tukey_df = pd.DataFrame(
            data=tukey_result.summary().data[1:],
            columns=tukey_result.summary().data[0]
        )

        # Compute effect size (η²)
        # η² = SS_between / SS_total
        grand_mean = self.df[metric].mean()
        ss_total = np.sum((self.df[metric] - grand_mean) ** 2)

        ss_between = 0
        for method in self.df[group_col].unique():
            group_data = self.df[self.df[group_col] == method][metric]
            n_group = len(group_data)
            group_mean = group_data.mean()
            ss_between += n_group * (group_mean - grand_mean) ** 2

        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return {
            'anova_f': f_stat,
            'anova_p': p_value,
            'tukey_results': tukey_df,
            'effect_size_eta_squared': eta_squared,
            'significant': p_value < 0.05
        }

    def compute_effect_sizes(
        self,
        metric: str,
        method1: str,
        method2: str,
        group_col: str = 'allocation_method'
    ) -> Dict:
        """
        Compute Cohen's d effect size between two methods

        Cohen's d interpretation:
            - Small: |d| ~ 0.2
            - Medium: |d| ~ 0.5
            - Large: |d| ~ 0.8+
        """
        data1 = self.df[self.df[group_col] == method1][metric].dropna()
        data2 = self.df[self.df[group_col] == method2][metric].dropna()

        # Pooled standard deviation
        n1, n2 = len(data1), len(data2)
        var1, var2 = data1.var(), data2.var()
        pooled_std = np.sqrt(
            ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (data1.mean() - data2.mean()) / \
            pooled_std if pooled_std > 0 else 0

        # Confidence interval (95%)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d

        # Interpret magnitude
        magnitude = 'negligible' if abs(cohens_d) < 0.2 else \
                    'small' if abs(cohens_d) < 0.5 else \
                    'medium' if abs(cohens_d) < 0.8 else 'large'

        return {
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'magnitude': magnitude,
            'mean_diff': data1.mean() - data2.mean(),
            'pooled_std': pooled_std
        }

    # =========================================================================
    # FIGURE GENERATION
    # =========================================================================

    def figure_1_itm_comparison(self):
        """
        Figure 1: ITM Score Comparison Across Methods

        Shows incentive-compatibility of different allocation methods
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: ITM by method
        sns.boxplot(
            data=self.df,
            x='allocation_method',
            y='itm_score',
            ax=axes[0]
        )
        axes[0].set_xlabel('Allocation Method')
        axes[0].set_ylabel('Incentive to Misreport (ITM)')
        axes[0].set_title('(A) ITM Score by Method')
        axes[0].tick_params(axis='x', rotation=45)

        # Add statistical annotations (ANOVA p-value)
        stats_result = self.compute_statistical_tests('itm_score')
        if stats_result['significant']:
            axes[0].text(
                0.02, 0.98,
                f"ANOVA: F={stats_result['anova_f']:.2f}, p={stats_result['anova_p']:.4f}*",
                transform=axes[0].transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        # Panel B: ITM vs Detection Probability
        for method in self.df['allocation_method'].unique():
            method_data = self.df[self.df['allocation_method'] == method]
            grouped = method_data.groupby('detection_prob')['itm_score'].mean()
            axes[1].plot(grouped.index, grouped.values,
                         marker='o', label=method, linewidth=2)

        axes[1].set_xlabel('Detection Probability')
        axes[1].set_ylabel('Mean ITM Score')
        axes[1].set_title('(B) ITM vs Detection Probability')
        axes[1].legend(title='Method', fontsize=9)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_itm_comparison.pdf')
        plt.savefig(self.figures_dir / 'fig1_itm_comparison.png')
        plt.close()

        logger.info("Generated Figure 1: ITM Comparison")

    def figure_2_fairness_tradeoffs(self):
        """
        Figure 2: Fairness Metrics Tradeoff Analysis

        Multi-panel showing Gini, Entropy, Envy, and FairReward composite
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        metrics = [
            ('gini_allocation', 'Gini Coefficient', axes[0, 0]),
            ('entropy_normalized', 'Normalized Entropy', axes[0, 1]),
            ('envy_rate', 'Envy Rate', axes[1, 0]),
            ('fairreward_composite', 'FairReward Index', axes[1, 1])
        ]

        for metric, title, ax in metrics:
            sns.violinplot(
                data=self.df,
                x='allocation_method',
                y=metric,
                ax=ax,
                inner='box'
            )
            ax.set_xlabel('Allocation Method')
            ax.set_ylabel(title)
            ax.set_title(
                f'({chr(65 + metrics.index((metric, title, ax)))}) {title}')
            ax.tick_params(axis='x', rotation=45)

            # Add mean values as text
            means = self.df.groupby('allocation_method')[metric].mean()
            for i, (method, mean_val) in enumerate(means.items()):
                ax.text(i, ax.get_ylim()[1] * 0.95, f'{mean_val:.3f}',
                        ha='center', va='top', fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_fairness_tradeoffs.pdf')
        plt.savefig(self.figures_dir / 'fig2_fairness_tradeoffs.png')
        plt.close()

        logger.info("Generated Figure 2: Fairness Tradeoffs")

    def figure_3_stability_heatmap(self):
        """
        Figure 3: Coalition Stability Analysis

        Heatmap showing stability_index across parameter combinations
        """
        # Pivot for heatmap: detection_prob × penalty → stability_index
        pivot_data = self.df.pivot_table(
            values='stability_index',
            index='penalty',
            columns='detection_prob',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Stability Index'},
            ax=ax
        )
        ax.set_xlabel('Detection Probability')
        ax.set_ylabel('Penalty Factor')
        ax.set_title(
            'Coalition Stability Index Heatmap\n(Higher = More Stable)')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_stability_heatmap.pdf')
        plt.savefig(self.figures_dir / 'fig3_stability_heatmap.png')
        plt.close()

        logger.info("Generated Figure 3: Stability Heatmap")

    def figure_4_parameter_sensitivity(self):
        """
        Figure 4: Parameter Sensitivity Analysis

        Shows how key metrics vary with heterogeneity (sigma_q) and noise (sigma_report)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel A: ITM vs Heterogeneity (sigma_q)
        for method in self.df['allocation_method'].unique():
            method_data = self.df[self.df['allocation_method'] == method]
            grouped = method_data.groupby(
                'sigma_q')['itm_score'].agg(['mean', 'std'])
            axes[0, 0].plot(grouped.index, grouped['mean'],
                            marker='o', label=method)
            axes[0, 0].fill_between(
                grouped.index,
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                alpha=0.2
            )
        axes[0, 0].set_xlabel('Contribution Heterogeneity (σ_q)')
        axes[0, 0].set_ylabel('Mean ITM Score')
        axes[0, 0].set_title('(A) ITM vs Heterogeneity')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.3)

        # Panel B: FairReward vs Heterogeneity
        for method in self.df['allocation_method'].unique():
            method_data = self.df[self.df['allocation_method'] == method]
            grouped = method_data.groupby(
                'sigma_q')['fairreward_composite'].mean()
            axes[0, 1].plot(grouped.index, grouped.values,
                            marker='s', label=method)
        axes[0, 1].set_xlabel('Contribution Heterogeneity (σ_q)')
        axes[0, 1].set_ylabel('FairReward Index')
        axes[0, 1].set_title('(B) FairReward vs Heterogeneity')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(alpha=0.3)

        # Panel C: Stability vs Reporting Noise
        for method in self.df['allocation_method'].unique():
            method_data = self.df[self.df['allocation_method'] == method]
            grouped = method_data.groupby('sigma_report')[
                'stability_index'].mean()
            axes[1, 0].plot(grouped.index, grouped.values,
                            marker='^', label=method)
        axes[1, 0].set_xlabel('Reporting Noise (σ_report)')
        axes[1, 0].set_ylabel('Mean Stability Index')
        axes[1, 0].set_title('(C) Stability vs Reporting Noise')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.3)

        # Panel D: Gini vs Number of Agents
        for method in self.df['allocation_method'].unique():
            method_data = self.df[self.df['allocation_method'] == method]
            grouped = method_data.groupby('n_agents')['gini_allocation'].mean()
            axes[1, 1].plot(grouped.index, grouped.values,
                            marker='d', label=method)
        axes[1, 1].set_xlabel('Number of Agents (n)')
        axes[1, 1].set_ylabel('Mean Gini Coefficient')
        axes[1, 1].set_title('(D) Gini vs Number of Agents')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig4_parameter_sensitivity.pdf')
        plt.savefig(self.figures_dir / 'fig4_parameter_sensitivity.png')
        plt.close()

        logger.info("Generated Figure 4: Parameter Sensitivity")

    def figure_5_method_comparison_boxplots(self):
        """
        Figure 5: Comprehensive Method Comparison

        Side-by-side boxplots for multiple metrics
        """
        metrics_to_plot = [
            'itm_score',
            'fairreward_composite',
            'stability_index',
            'gini_allocation'
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            sns.boxplot(
                data=self.df,
                x='allocation_method',
                y=metric,
                ax=ax,
                palette='Set2'
            )

            # Title
            metric_names = {
                'itm_score': 'Incentive to Misreport',
                'fairreward_composite': 'FairReward Index',
                'stability_index': 'Stability Index',
                'gini_allocation': 'Gini Coefficient'
            }
            ax.set_title(
                f'({chr(65 + idx)}) {metric_names.get(metric, metric)}')
            ax.set_xlabel('Allocation Method')
            ax.set_ylabel(metric_names.get(metric, metric))
            ax.tick_params(axis='x', rotation=45)

            # Add horizontal line at ideal value (if applicable)
            if metric in ['stability_index', 'fairreward_composite']:
                ax.axhline(y=1.0, color='red', linestyle='--',
                           linewidth=1, alpha=0.5, label='Ideal')
            elif metric in ['itm_score', 'gini_allocation']:
                ax.axhline(y=0.0, color='red', linestyle='--',
                           linewidth=1, alpha=0.5, label='Ideal')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig5_method_comparison.pdf')
        plt.savefig(self.figures_dir / 'fig5_method_comparison.png')
        plt.close()

        logger.info("Generated Figure 5: Method Comparison")

    def figure_6_statistical_comparison(self):
        """
        Figure 6: Statistical Comparison with Effect Sizes (NEW)

        Shows Cohen's d effect sizes between Shapley and other methods
        """
        if 'exact_shapley' not in self.df['allocation_method'].values:
            logger.warning("Exact Shapley not in dataset, skipping Figure 6")
            return

        metrics = ['itm_score', 'fairreward_composite', 'stability_index']
        methods = [m for m in self.df['allocation_method'].unique()
                   if m != 'exact_shapley']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            effect_sizes = []
            method_labels = []

            for method in methods:
                effect = self.compute_effect_sizes(
                    metric=metric,
                    method1='exact_shapley',
                    method2=method
                )
                effect_sizes.append(effect['cohens_d'])
                method_labels.append(method)

            # Bar plot
            colors = ['green' if abs(d) < 0.5 else 'orange' if abs(d) < 0.8 else 'red'
                      for d in effect_sizes]
            ax.barh(method_labels, effect_sizes, color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linewidth=1)

            # Reference lines for effect size magnitude
            for val, label in [(-0.8, 'Large'), (-0.5, 'Medium'), (-0.2, 'Small'),
                               (0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
                ax.axvline(x=val, color='gray', linestyle='--',
                           linewidth=0.5, alpha=0.5)

            ax.set_xlabel("Cohen's d Effect Size")
            ax.set_title(
                f'({chr(65 + idx)}) {metric.replace("_", " ").title()}')
            ax.set_ylabel('Comparison Method')
            ax.grid(alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig6_effect_sizes.pdf')
        plt.savefig(self.figures_dir / 'fig6_effect_sizes.png')
        plt.close()

        logger.info("Generated Figure 6: Statistical Comparison")

    # =========================================================================
    # SUMMARY TABLES
    # =========================================================================

    def generate_summary_tables(self):
        """Generate LaTeX and CSV tables for paper"""

        # Table 1: Method Performance Summary
        summary = self.df.groupby('allocation_method').agg({
            'itm_score': ['mean', 'std'],
            'fairreward_composite': ['mean', 'std'],
            'stability_index': ['mean', 'std'],
            'gini_allocation': ['mean', 'std'],
            'in_core': lambda x: x.sum() / x.count() if x.count() > 0 else np.nan
        }).round(4)

        # Flatten MultiIndex columns
        summary.columns = ['_'.join(col).strip()
                           for col in summary.columns.values]
        summary.to_csv(self.tables_dir / 'table1_method_summary.csv')
        summary.to_latex(self.tables_dir / 'table1_method_summary.tex')

        logger.info("Generated Table 1: Method Summary")

        # Table 2: Statistical Test Results
        test_results = []
        for metric in ['itm_score', 'fairreward_composite', 'stability_index']:
            stats_result = self.compute_statistical_tests(metric)
            test_results.append({
                'metric': metric,
                'F_statistic': stats_result['anova_f'],
                'p_value': stats_result['anova_p'],
                'eta_squared': stats_result['effect_size_eta_squared'],
                'significant': 'Yes' if stats_result['significant'] else 'No'
            })

        df_tests = pd.DataFrame(test_results)
        df_tests.to_csv(self.tables_dir /
                        'table2_statistical_tests.csv', index=False)
        df_tests.to_latex(self.tables_dir /
                          'table2_statistical_tests.tex', index=False)

        logger.info("Generated Table 2: Statistical Tests")

        # Table 3: Effect Sizes (Shapley vs Others)
        if 'exact_shapley' in self.df['allocation_method'].values:
            effect_results = []
            for method in self.df['allocation_method'].unique():
                if method == 'exact_shapley':
                    continue

                for metric in ['itm_score', 'fairreward_composite']:
                    effect = self.compute_effect_sizes(
                        metric=metric,
                        method1='exact_shapley',
                        method2=method
                    )
                    effect_results.append({
                        'comparison': f'Shapley vs {method}',
                        'metric': metric,
                        'cohens_d': effect['cohens_d'],
                        'magnitude': effect['magnitude'],
                        'mean_diff': effect['mean_diff']
                    })

            df_effects = pd.DataFrame(effect_results)
            df_effects.to_csv(self.tables_dir /
                              'table3_effect_sizes.csv', index=False)
            df_effects.to_latex(
                self.tables_dir / 'table3_effect_sizes.tex', index=False)

            logger.info("Generated Table 3: Effect Sizes")

    # =========================================================================
    # SPECIALIZED ANALYSES
    # =========================================================================

    def analyze_nash_equilibria(self):
        """
        Analyze Nash equilibrium results (if computed)

        Generates:
        - Convergence rate statistics
        - Deviation from truth vs method
        - Figure showing Nash reports distribution
        """
        if 'nash_converged' not in self.df.columns:
            logger.warning("No Nash equilibrium data available")
            return

        nash_df = self.df[self.df['nash_converged'].notna()].copy()

        if len(nash_df) == 0:
            logger.warning("No Nash equilibrium runs found")
            return

        # Convergence statistics
        conv_rate = nash_df['nash_converged'].mean()
        mean_deviation = nash_df['nash_deviation_from_truth'].mean()

        logger.info(f"Nash Equilibrium Analysis:")
        logger.info(f"  Convergence rate: {conv_rate:.2%}")
        logger.info(f"  Mean deviation from truth: {mean_deviation:.4f}")

        # Figure: Deviation from truth by method
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=nash_df,
            x='allocation_method',
            y='nash_deviation_from_truth',
            ax=ax
        )
        ax.set_xlabel('Allocation Method')
        ax.set_ylabel('Nash Deviation from Truthful Reporting')
        ax.set_title('Strategic Reporting at Nash Equilibrium')
        ax.tick_params(axis='x', rotation=45)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig_nash_deviation.pdf')
        plt.savefig(self.figures_dir / 'fig_nash_deviation.png')
        plt.close()

        logger.info("Generated Nash Equilibrium Analysis Figure")

    def analyze_core_membership(self):
        """
        Analyze core membership results (if LP checks performed)

        Generates:
        - Core membership rate by method
        - Epsilon (core violation) distribution
        """
        if 'in_core' not in self.df.columns:
            logger.warning("No core membership data available")
            return

        core_df = self.df[self.df['in_core'].notna()].copy()

        if len(core_df) == 0:
            logger.warning("No core membership checks found")
            return

        # Core membership rate by method
        core_rates = core_df.groupby('allocation_method')['in_core'].mean()

        logger.info("Core Membership Analysis:")
        for method, rate in core_rates.items():
            logger.info(f"  {method}: {rate:.2%} in core")

        # Figure: Epsilon distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Core membership rate
        core_rates.plot(kind='bar', ax=axes[0], color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Allocation Method')
        axes[0].set_ylabel('Core Membership Rate')
        axes[0].set_title('(A) Fraction of Allocations in Core')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=1.0, color='green', linestyle='--',
                        linewidth=1, alpha=0.5)

        # Panel B: Epsilon (violation) for out-of-core allocations
        out_of_core = core_df[core_df['in_core'] == False]
        if len(out_of_core) > 0:
            sns.boxplot(
                data=out_of_core,
                x='allocation_method',
                y='core_epsilon',
                ax=axes[1]
            )
            axes[1].set_xlabel('Allocation Method')
            axes[1].set_ylabel('Core Violation (ε)')
            axes[1].set_title('(B) Magnitude of Core Violation')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'All allocations in core',
                         ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('(B) Core Violation Analysis')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig_core_membership.pdf')
        plt.savefig(self.figures_dir / 'fig_core_membership.png')
        plt.close()

        logger.info("Generated Core Membership Analysis Figure")

    def generate_correlation_matrix(self):
        """
        Generate correlation matrix of key metrics

        Useful for identifying relationships between fairness, incentives, stability
        """
        metrics_of_interest = [
            'itm_score',
            'fairreward_composite',
            'stability_index',
            'gini_allocation',
            'entropy_normalized',
            'envy_rate',
            'detection_prob',
            'penalty',
            'sigma_q',
            'sigma_report'
        ]

        # Filter to available metrics
        available_metrics = [
            m for m in metrics_of_interest if m in self.df.columns]

        corr_matrix = self.df[available_metrics].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        ax.set_title('Correlation Matrix of Key Metrics')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig_correlation_matrix.pdf')
        plt.savefig(self.figures_dir / 'fig_correlation_matrix.png')
        plt.close()

        logger.info("Generated Correlation Matrix")

    def generate_latex_summary(self):
        """
        Generate comprehensive LaTeX summary for paper appendix
        """
        latex_content = []

        latex_content.append(r"\section{Simulation Results Summary}")
        latex_content.append(r"\subsection{Dataset Overview}")
        latex_content.append(f"Total simulation runs: {len(self.df)}")
        latex_content.append(
            f"Number of allocation methods: {self.df['allocation_method'].nunique()}")
        latex_content.append(f"Parameter ranges:")
        latex_content.append(
            f"  - Agents (n): {self.df['n_agents'].min()}-{self.df['n_agents'].max()}")
        latex_content.append(
            f"  - Detection prob: {self.df['detection_prob'].min()}-{self.df['detection_prob'].max()}")
        latex_content.append(
            f"  - Penalty: {self.df['penalty'].min()}-{self.df['penalty'].max()}")

        latex_content.append(r"\subsection{Statistical Test Results}")
        for metric in ['itm_score', 'fairreward_composite', 'stability_index']:
            stats_result = self.compute_statistical_tests(metric)
            latex_content.append(f"\\textbf{{{metric}}}:")
            latex_content.append(
                f"  F-statistic = {stats_result['anova_f']:.4f}")
            latex_content.append(f"  p-value = {stats_result['anova_p']:.4e}")
            latex_content.append(
                f"  $\\eta^2$ = {stats_result['effect_size_eta_squared']:.4f}")
            latex_content.append("")

        # Write to file
        with open(self.output_dir / 'latex_summary.txt', 'w') as f:
            f.write('\n'.join(latex_content))

        logger.info("Generated LaTeX summary")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_and_analyze(results_path: str, output_dir: str):
    """
    Convenience function to load results and run full analysis pipeline

    Usage:
    ------
    from src.modules.analysis import load_and_analyze
    load_and_analyze('results/results.pkl', 'results/analysis')
    """
    df = pd.read_pickle(results_path)
    pipeline = AnalysisPipeline(df, output_dir)

    # Generate all outputs
    pipeline.generate_all_figures()
    pipeline.generate_summary_tables()
    pipeline.analyze_nash_equilibria()
    pipeline.analyze_core_membership()
    pipeline.generate_correlation_matrix()
    pipeline.generate_latex_summary()

    logger.info("Analysis pipeline complete!")
