"""
Enhanced logger with experiment tracking for Shapley-Fair Consortium Blockchain Research
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Enhanced logger with experiment tracking capabilities
    """
    
    def __init__(self, name: str = "experiment_logger", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the experiment logger
        
        Args:
            name: Logger name
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.config = config or {}
        self.start_time = None
        
    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """
        Log the start of an experiment
        
        Args:
            config: Experiment configuration
        """
        self.start_time = datetime.now()
        self.logger.info("="*80)
        self.logger.info("EXPERIMENT START")
        self.logger.info(f"Time: {self.start_time.isoformat()}")
        self.logger.info(f"Config: {config}")
        self.logger.info("="*80)
    
    def log_experiment_end(self, results_summary: Dict[str, Any]) -> None:
        """
        Log the end of an experiment
        
        Args:
            results_summary: Summary of results
        """
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None
        
        self.logger.info("EXPERIMENT END")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Results: {results_summary}")
        self.logger.info("="*80)
    
    def log_milestone(self, message: str) -> None:
        """
        Log a significant milestone in the experiment
        
        Args:
            message: Milestone description
        """
        self.logger.info(f"checkpoint: {message}")