#!/usr/bin/env python3
# =============================================================================
# FILE: src/main.py
"""
Main CLI Entry Point

Provides command-line interface for:
- Running experiments
- Analyzing results
- Validating game properties
- Generating reports

Priority: HIGH | Status: Production-Ready
Version: 2.0.0
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd

from modules.runner import SimulationRunner
from modules.analysis import AnalysisPipeline
from utils.validation import GameValidator, ShapleyValidator
from utils.metrics import FairnessMetrics


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('experiment.log')
        ]
    )


def load_config(config_path: Path) -> dict:
    """Load experiment configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiments(args):
    """Run experiments based on configuration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment run...")

    # Load config
    config = load_config(Path(args.config))

    # Initialize runner
    from modules.runner import SimulationRunner
    runner = SimulationRunner(
        output_dir=Path(args.output)
    )

    # Generate run manifest based on config
    from modules.data_gen import DataGenerator
    import itertools
    import numpy as np
    
    # Create parameter combinations
    # Check if parameters are nested (new format) or flat (old format)
    params = config.get('parameters', config)

    param_grid = {
        'n_agents': params.get('n_agents', [3]),
        'mu': params.get('mu', [10.0]),
        'sigma_q': params.get('sigma_q', [2.0]),
        'sigma_report': params.get('sigma_report', [0.5]),
        'detection_prob': params.get('detection_prob', [0.0]),
        'penalty': params.get('penalty', [0.0]),
        'payoff_shape': params.get('payoff_shapes', ['linear']),
        'allocation_method': params.get('allocation_methods', ['exact_shapley']),
        'report_model': params.get('report_model', params.get('reporting_models', ['truthful']))
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    # Create run manifest
    run_manifest = []
    for idx, combo in enumerate(combinations):
        run_config = dict(zip(keys, combo))
        run_config['run_idx'] = idx
        run_config['seed'] = 42 + idx  # Different seed for each run
        run_config['n_mc_samples'] = config.get('n_mc_samples', 10000)
        run_config['n_iterations'] = config.get('n_iterations', 1)
        run_manifest.append(run_config)
    
    # Run experiments
    results = runner.run_grid(
        run_manifest=run_manifest,
        parallel_workers=args.jobs,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume
    )

    logger.info(f"Experiments complete. Results saved to {args.output}")

    return results


def analyze_results(args):
    """Analyze experimental results"""
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis...")

    # Load results
    results_path = Path(args.results)

    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    results_df = pd.read_csv(results_path)

    # Load CSV file
    results_df = pd.read_csv(results_path)
    
    # Run analysis pipeline
    pipeline = AnalysisPipeline(
        results_df=results_df,
        output_dir=str(Path(args.output))
    )
    
    # Generate all outputs
    pipeline.generate_all_figures()
    pipeline.generate_summary_tables()
    pipeline.analyze_nash_equilibria()
    pipeline.analyze_core_membership()
    pipeline.generate_correlation_matrix()
    pipeline.generate_latex_summary()

    logger.info(f"Analysis complete. Output saved to {args.output}")

    return results_df


def validate_game(args):
    """Validate game theory properties"""
    logger = logging.getLogger(__name__)
    logger.info("Running game validation...")

    # This would load a specific game instance
    # For demo, we'll create a simple test

    from modules.data_gen import DataGenerator

    generator = DataGenerator(seed=42)
    game = generator.generate_instance(
        n_agents=args.n_agents,
        mu=10.0,
        sigma_q=2.0,
        payoff_shape=args.payoff_shape
    )

    # Validate game properties
    validator = GameValidator()
    validation_results = validator.validate_all_properties(
        n_agents=game.n_agents,
        payoff_function=game.payoff_function,
        sample_coalitions=args.n_samples
    )

    # Print summary
    print("\n" + "="*60)
    print(validation_results['summary'])
    print("="*60 + "\n")

    # Detailed results
    for prop, result in validation_results.items():
        if prop != 'summary' and isinstance(result, dict):
            print(f"\n{prop.upper()}:")
            print(f"  Satisfied: {result.get('satisfied', 'N/A')}")
            print(f"  Score: {result.get('score', 'N/A'):.3f}")
            if 'violations' in result and len(result['violations']) > 0:
                print(f"  Violations: {len(result['violations'])}")

    return validation_results


def validate_shapley(args):
    """Validate Shapley value axioms"""
    logger = logging.getLogger(__name__)
    logger.info("Validating Shapley axioms...")

    from modules.data_gen import DataGenerator
    from modules.allocation import AllocationEngine

    # Generate game
    generator = DataGenerator(seed=42)
    game = generator.generate_instance(
        n_agents=args.n_agents,
        mu=10.0,
        sigma_q=2.0,
        payoff_shape='linear'
    )

    # Compute Shapley values
    engine = AllocationEngine(seed=42)
    shapley_result = engine.allocate(
        method='exact_shapley' if args.n_agents <= 10 else 'mc_shapley',
        n_agents=game.n_agents,
        payoff_function=game.payoff_function,
        reports=game.q_true
    )

    # Validate axioms
    validator = ShapleyValidator()
    validation_results = validator.validate_shapley_axioms(
        allocations=shapley_result.allocations,
        n_agents=game.n_agents,
        payoff_function=game.payoff_function,
        contributions=game.q_true
    )

    # Print summary
    print("\n" + "="*60)
    print(validation_results['summary'])
    print("="*60 + "\n")

    for axiom, result in validation_results.items():
        if axiom not in ['all_axioms_satisfied', 'summary'] and isinstance(result, dict):
            print(f"\n{axiom.upper()}:")
            print(f"  Satisfied: {result['satisfied']}")
            if 'error' in result:
                print(f"  Error: {result['error']:.6f}")
            if 'violations' in result and len(result['violations']) > 0:
                print(f"  Violations: {len(result['violations'])}")

    return validation_results


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Shapley-Fair Consortium Blockchain Research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments
  python src/main.py run --config configs/experiment.yaml --jobs 4
  
  # Analyze results
  python src/main.py analyze --results experiments/results/results.csv
  
  # Validate game properties
  python src/main.py validate-game --n-agents 5 --payoff-shape superadditive
  
  # Validate Shapley axioms
  python src/main.py validate-shapley --n-agents 4
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run experiments command
    run_parser = subparsers.add_parser('run', help='Run experiments')
    run_parser.add_argument('--config', '-c', type=str,
                            default='configs/experiment.yaml',
                            help='Path to experiment config file')
    run_parser.add_argument('--output', '-o', type=str,
                            default='experiments',
                            help='Output directory')
    run_parser.add_argument('--jobs', '-j', type=int, default=1,
                            help='Number of parallel jobs')
    run_parser.add_argument('--resume', action='store_true',
                            help='Resume from checkpoint')
    run_parser.add_argument('--checkpoint-interval', type=int, default=10,
                            help='Checkpoint interval (iterations)')

    # Analyze results command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument('--results', '-r', type=str, required=True,
                                help='Path to results CSV file')
    analyze_parser.add_argument('--output', '-o', type=str,
                                default='experiments/analysis',
                                help='Output directory')
    analyze_parser.add_argument('--no-figures', action='store_true',
                                help='Skip figure generation')
    analyze_parser.add_argument('--no-tables', action='store_true',
                                help='Skip table generation')

    # Validate game command
    validate_game_parser = subparsers.add_parser(
        'validate-game', help='Validate game properties'
    )
    validate_game_parser.add_argument('--n-agents', '-n', type=int, default=5,
                                      help='Number of agents')
    validate_game_parser.add_argument('--payoff-shape', type=str,
                                      default='superadditive',
                                      choices=['linear', 'subadditive',
                                               'superadditive', 'threshold'],
                                      help='Payoff function shape')
    validate_game_parser.add_argument('--n-samples', type=int, default=100,
                                      help='Number of coalitions to sample')

    # Validate Shapley command
    validate_shapley_parser = subparsers.add_parser(
        'validate-shapley', help='Validate Shapley axioms'
    )
    validate_shapley_parser.add_argument('--n-agents', '-n', type=int, default=4,
                                         help='Number of agents')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Dispatch to appropriate command
    if args.command == 'run':
        run_experiments(args)
    elif args.command == 'analyze':
        analyze_results(args)
    elif args.command == 'validate-game':
        validate_game(args)
    elif args.command == 'validate-shapley':
        validate_shapley(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
