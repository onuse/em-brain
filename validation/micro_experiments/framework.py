#!/usr/bin/env python3
"""
Micro-Experiment Framework for Brain Assumption Validation

This framework enables rapid testing of specific brain assumptions with
focused experiments that run in 5-10 minutes and provide clear pass/fail results.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# Add paths for brain modules
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

@dataclass
class MicroExperimentResult:
    """Result from a single micro-experiment."""
    experiment_name: str
    assumption_tested: str
    start_time: float
    duration_seconds: float
    
    # Test results
    passed: bool
    confidence: float  # 0.0 to 1.0
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    
    # Data
    measurements: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    # Metadata
    sample_size: int = 0
    random_seed: int = 42
    
    def to_dict(self):
        return asdict(self)

class MicroExperiment(ABC):
    """Base class for micro-experiments."""
    
    def __init__(self, name: str, assumption: str, timeout_seconds: int = 300):
        self.name = name
        self.assumption = assumption
        self.timeout_seconds = timeout_seconds
        self.client = None
        self.environment = None
        self.start_time = None
        
    @abstractmethod
    def setup(self) -> bool:
        """Setup experiment. Return True if successful."""
        pass
    
    @abstractmethod
    def run_test(self) -> MicroExperimentResult:
        """Run the actual test. Return result."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass
    
    def execute(self) -> MicroExperimentResult:
        """Execute the complete micro-experiment."""
        print(f"🧪 Running micro-experiment: {self.name}")
        print(f"   Assumption: {self.assumption}")
        
        self.start_time = time.time()
        
        try:
            # Setup
            if not self.setup():
                return MicroExperimentResult(
                    experiment_name=self.name,
                    assumption_tested=self.assumption,
                    start_time=self.start_time,
                    duration_seconds=time.time() - self.start_time,
                    passed=False,
                    confidence=0.0,
                    error_message="Setup failed"
                )
            
            # Run with timeout
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Experiment timeout after {self.timeout_seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            try:
                result = self.run_test()
                signal.alarm(0)  # Cancel timeout
            except TimeoutError as e:
                result = MicroExperimentResult(
                    experiment_name=self.name,
                    assumption_tested=self.assumption,
                    start_time=self.start_time,
                    duration_seconds=self.timeout_seconds,
                    passed=False,
                    confidence=0.0,
                    error_message=str(e)
                )
            
            # Set timing
            result.start_time = self.start_time
            result.duration_seconds = time.time() - self.start_time
            
            return result
            
        except Exception as e:
            return MicroExperimentResult(
                experiment_name=self.name,
                assumption_tested=self.assumption,
                start_time=self.start_time,
                duration_seconds=time.time() - self.start_time,
                passed=False,
                confidence=0.0,
                error_message=str(e)
            )
        
        finally:
            self.cleanup()
    
    def connect_to_brain(self) -> bool:
        """Helper to connect to brain server."""
        try:
            self.client = MinimalBrainClient()
            return self.client.connect()
        except Exception as e:
            print(f"   ❌ Failed to connect to brain: {e}")
            return False
    
    def create_environment(self, **kwargs) -> bool:
        """Helper to create test environment."""
        try:
            self.environment = SensoryMotorWorld(**kwargs)
            return True
        except Exception as e:
            print(f"   ❌ Failed to create environment: {e}")
            return False

class MicroExperimentSuite:
    """Collection of micro-experiments for systematic validation."""
    
    def __init__(self, results_dir: Path = None):
        self.experiments: List[MicroExperiment] = []
        self.results: List[MicroExperimentResult] = []
        self.results_dir = results_dir or Path("validation/micro_experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def add_experiment(self, experiment: MicroExperiment):
        """Add experiment to suite."""
        self.experiments.append(experiment)
    
    def run_all(self, stop_on_failure: bool = False) -> Dict[str, Any]:
        """Run all experiments in suite."""
        print(f"🔬 Running {len(self.experiments)} micro-experiments")
        print("=" * 60)
        
        self.results = []
        failed_experiments = []
        
        for i, experiment in enumerate(self.experiments):
            print(f"\\n[{i+1}/{len(self.experiments)}] {experiment.name}")
            
            result = experiment.execute()
            self.results.append(result)
            
            # Print result
            if result.passed:
                print(f"   ✅ PASSED (confidence: {result.confidence:.3f})")
            else:
                print(f"   ❌ FAILED (confidence: {result.confidence:.3f})")
                if result.error_message:
                    print(f"      Error: {result.error_message}")
                failed_experiments.append(experiment.name)
                
                if stop_on_failure:
                    print("   ⏹️ Stopping on failure")
                    break
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save results
        self._save_results(summary)
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        # Group by assumption
        assumptions = {}
        for result in self.results:
            assumption = result.assumption_tested
            if assumption not in assumptions:
                assumptions[assumption] = []
            assumptions[assumption].append(result)
        
        # Calculate assumption validation rates
        assumption_scores = {}
        for assumption, results in assumptions.items():
            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)
            assumption_scores[assumption] = {
                'passed': passed_count,
                'total': total_count,
                'rate': passed_count / total_count,
                'avg_confidence': np.mean([r.confidence for r in results])
            }
        
        return {
            'total_experiments': total,
            'passed_experiments': passed,
            'failed_experiments': failed,
            'success_rate': passed / total if total > 0 else 0.0,
            'avg_confidence': np.mean([r.confidence for r in self.results]),
            'assumption_scores': assumption_scores,
            'experiment_results': [r.to_dict() for r in self.results],
            'timestamp': time.time()
        }
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save results to file."""
        timestamp = int(time.time())
        results_file = self.results_dir / f"micro_experiment_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary_serializable = convert_numpy_types(summary)
        
        with open(results_file, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        
        print(f"\\n💾 Results saved to {results_file}")
    
    def print_summary(self):
        """Print experiment summary."""
        if not self.results:
            print("No experiments run yet.")
            return
        
        summary = self._generate_summary()
        
        print(f"\\n📊 Micro-Experiment Summary")
        print("=" * 60)
        print(f"Total Experiments: {summary['total_experiments']}")
        print(f"Passed: {summary['passed_experiments']} ✅")
        print(f"Failed: {summary['failed_experiments']} ❌")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Confidence: {summary['avg_confidence']:.3f}")
        
        print(f"\\n📋 Assumption Validation:")
        for assumption, scores in summary['assumption_scores'].items():
            rate = scores['rate']
            confidence = scores['avg_confidence']
            status = "✅" if rate >= 0.8 else "⚠️" if rate >= 0.5 else "❌"
            print(f"   {status} {assumption}: {rate:.1%} ({scores['passed']}/{scores['total']}) - confidence: {confidence:.3f}")
        
        print(f"\\n📈 Failed Experiments:")
        for result in self.results:
            if not result.passed:
                print(f"   ❌ {result.experiment_name}: {result.error_message or 'Unknown error'}")

# Statistical utilities
def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for data."""
    try:
        import scipy.stats as stats
        
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # t-distribution for small samples
        t_val = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_error = t_val * std_err
        
        return mean - margin_error, mean + margin_error
    except ImportError:
        # Fallback without scipy
        mean = np.mean(data)
        std = np.std(data)
        margin_error = 1.96 * std / np.sqrt(len(data))  # Approximate 95% CI
        return mean - margin_error, mean + margin_error

def perform_t_test(group1: List[float], group2: List[float]) -> Tuple[float, float]:
    """Perform t-test between two groups."""
    try:
        import scipy.stats as stats
        statistic, p_value = stats.ttest_ind(group1, group2)
        return statistic, p_value
    except ImportError:
        # Fallback without scipy - simple comparison
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1), np.std(group2)
        
        # Simple t-statistic approximation
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(2/len(group1)))
        
        # Rough p-value approximation
        p_value = 2 * (1 - abs(t_stat) / 3)  # Very rough approximation
        p_value = max(0.0, min(1.0, p_value))
        
        return t_stat, p_value

def calculate_effect_size(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d

def is_statistically_significant(p_value: float, alpha: float = 0.05) -> bool:
    """Check if p-value is statistically significant."""
    return p_value < alpha

def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"