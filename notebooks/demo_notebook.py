# =============================================================================
# FILE: notebooks/demo_notebook.py
"""
Demo Jupyter Notebook (save as .ipynb)
Quick start guide and visualization examples
"""

from pathlib import Path
import sys
demo_notebook_content = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Shapley-Fair Consortium Blockchain - Demo\\n",
    "\\n",
    "Quick start guide for running experiments and generating figures."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "from src.modules.data_gen import DataGenerator\\n",
    "from src.modules.allocation import AllocationEngine\\n",
    "from src.modules.incentives import IncentiveAnalyzer\\n",
    "\\n",
    "sns.set_style('whitegrid')\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Generate Synthetic Game"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create data generator\\n",
    "data_gen = DataGenerator(seed=42)\\n",
    "\\n",
    "# Generate game instance\\n",
    "game = data_gen.generate_instance(\\n",
    "    n_agents=6,\\n",
    "    mu=10.0,\\n",
    "    sigma_q=3.0,\\n",
    "    sigma_report=0.5,\\n",
    "    payoff_shape='subadditive',\\n",
    "    gamma=0.8\\n",
    ")\\n",
    "\\n",
    "print(f'True contributions: {game.q_true}')\\n",
    "print(f'Reported contributions: {game.q_reported}')\\n",
    "print(f'Total value: {game.get_grand_coalition_value():.2f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Compare Allocation Methods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "alloc_engine = AllocationEngine(seed=42)\\n",
    "\\n",
    "methods = ['proportional', 'exact_shapley', 'weighted_shapley']\\n",
    "results = {}\\n",
    "\\n",
    "for method in methods:\\n",
    "    result = alloc_engine.allocate(\\n",
    "        method=method,\\n",
    "        n_agents=game.n_agents,\\n",
    "        payoff_function=game.payoff_function,\\n",
    "        reports=game.q_reported,\\n",
    "        n_samples=10000\\n",
    "    )\\n",
    "    results[method] = result.allocations\\n",
    "\\n",
    "# Visualize\\n",
    "df_alloc = pd.DataFrame(results)\\n",
    "df_alloc.plot(kind='bar', figsize=(10, 6))\\n",
    "plt.xlabel('Agent')\\n",
    "plt.ylabel('Allocation')\\n",
    "plt.title('Allocation Comparison')\\n",
    "plt.legend(title='Method')\\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Analyze Strategic Incentives"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "incentive_analyzer = IncentiveAnalyzer(seed=42)\\n",
    "\\n",
    "# Compare ITM across methods\\n",
    "itm_results = {}\\n",
    "\\n",
    "for method in methods:\\n",
    "    alloc_result = alloc_engine.allocate(\\n",
    "        method=method,\\n",
    "        n_agents=game.n_agents,\\n",
    "        payoff_function=game.payoff_function,\\n",
    "        reports=game.q_reported,\\n",
    "        n_samples=10000\\n",
    "    )\\n",
    "    \\n",
    "    metrics = incentive_analyzer.compute_metrics(\\n",
    "        game_instance=game,\\n",
    "        allocation_result=alloc_result,\\n",
    "        allocation_engine=alloc_engine,\\n",
    "        detection_prob=0.3,\\n",
    "        penalty=1.0\\n",
    "    )\\n",
    "    \\n",
    "    itm_results[method] = {\\n",
    "        'ITM': metrics.incentive_to_misreport,\\n",
    "        'Gini': metrics.gini_coefficient,\\n",
    "        'Envy': metrics.envy_rate,\\n",
    "        'Stability': metrics.stability_index\\n",
    "    }\\n",
    "\\n",
    "pd.DataFrame(itm_results).T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Run Full Experiment\\n",
    "\\n",
    "```bash\\n",
    "# From command line:\\n",
    "python -m src.main --config configs/experiment_small.yaml --mode full\\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''

# Save configs


def save_configs():
    """Save configuration files"""
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)

    with open(config_dir / 'experiment_small.yaml', 'w') as f:
        f.write(experiment_config_small)

    with open(config_dir / 'experiment_full.yaml', 'w') as f:
        f.write(experiment_config_full)

    print("Configuration files saved to configs/")


if __name__ == '__main__':
    save_configs()
