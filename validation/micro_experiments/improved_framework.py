#!/usr/bin/env python3
"""
Improved Micro-Experiment Framework - Phase 1 Optimizations

Key improvements:
1. Persistent connections - reuse same client across tests
2. World reuse - avoid excessive environment resets
3. Retry logic - handle brain error 5.0 gracefully
4. Increased timeouts - better for intensive tests
5. Better error handling - continue testing despite failures
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
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from src.communication.client import MinimalBrainClient
from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

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
    retry_count: int = 0
    
    def to_dict(self):
        return asdict(self)

class BrainProcessingError(Exception):
    """Exception for brain processing errors (error code 5.0)."""
    pass

class ImprovedMicroExperiment(ABC):
    """Improved base class for micro-experiments with persistent connections."""
    
    def __init__(self, name: str, assumption: str, timeout_seconds: int = 600):
        self.name = name
        self.assumption = assumption
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        
        # Persistent resources (set by suite)
        self.client = None
        self.environment = None
        self.reuse_resources = True
        
    @abstractmethod
    def setup(self) -> bool:
        """Setup experiment. Return True if successful."""
        pass
    
    @abstractmethod
    def run_test(self) -> MicroExperimentResult:
        """Run the actual test. Return result."""
        pass
    
    def cleanup(self):
        """Clean up resources (only if not reusing)."""
        if not self.reuse_resources:
            if self.client:
                self.client.disconnect()
            self.client = None
            self.environment = None
    
    def execute(self, shared_client=None, shared_environment=None) -> MicroExperimentResult:
        """Execute the complete micro-experiment with shared resources."""
        print(f"🧪 Running micro-experiment: {self.name}")
        print(f"   Assumption: {self.assumption}")
        
        self.start_time = time.time()
        
        # Use shared resources if available
        if shared_client:
            self.client = shared_client
            self.reuse_resources = True
        if shared_environment:
            self.environment = shared_environment
            self.reuse_resources = True
        
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
    
    def get_action_with_retry(self, sensory_input: List[float], max_retries: int = 3, timeout: float = 10.0) -> Optional[List[float]]:
        """Get action from brain with retry logic for error 5.0."""
        for attempt in range(max_retries):
            try:
                action = self.client.get_action(sensory_input, timeout=timeout)
                if action is not None:
                    return action
                    
                # None response might be error 5.0, try again
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Brain returned None, retrying ({attempt + 1}/{max_retries})")
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                    
            except Exception as e:
                # Check for brain processing errors (5.x codes)
                error_str = str(e)
                if ("5." in error_str or "Brain processing error" in error_str or 
                    "BRAIN_PROCESSING_ERROR" in error_str or
                    "SIMILARITY_ENGINE_FAILURE" in error_str or
                    "PREDICTION_ENGINE_FAILURE" in error_str or
                    "MEMORY_PRESSURE_ERROR" in error_str):
                    if attempt < max_retries - 1:
                        print(f"   ⚠️  Brain processing error ({error_str[:50]}...), retrying ({attempt + 1}/{max_retries})")
                        time.sleep(0.1)
                        continue
                    else:
                        raise BrainProcessingError(f"Brain processing failed after {max_retries} attempts: {error_str}")
                else:
                    raise
        
        return None
    
    def reset_environment_if_needed(self, force_reset: bool = False):
        """Reset environment only if needed or forced."""
        if force_reset or not hasattr(self.environment, '_initialized'):
            self.environment.reset()
            self.environment._initialized = True
        # For most tests, reuse existing environment state

class ImprovedMicroExperimentSuite:
    """Improved suite with persistent connections and resource reuse."""
    
    def __init__(self, results_dir: Path = None):
        self.experiments: List[ImprovedMicroExperiment] = []
        self.results: List[MicroExperimentResult] = []
        self.results_dir = results_dir or Path("micro_experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Persistent resources
        self.persistent_client = None
        self.persistent_environment = None
        
    def add_experiment(self, experiment: ImprovedMicroExperiment):
        """Add experiment to suite."""
        self.experiments.append(experiment)
    
    def setup_persistent_resources(self) -> bool:
        """Setup persistent client and environment for all tests."""
        print("🔧 Setting up persistent resources...")
        
        # Create persistent client
        try:
            self.persistent_client = MinimalBrainClient()
            if not self.persistent_client.connect():
                print("   ❌ Failed to connect to brain server")
                return False
            print("   ✅ Persistent brain connection established")
        except Exception as e:
            print(f"   ❌ Failed to create persistent client: {e}")
            return False
        
        # Create persistent environment
        try:
            self.persistent_environment = SensoryMotorWorld(random_seed=42)
            print("   ✅ Persistent test environment created")
        except Exception as e:
            print(f"   ❌ Failed to create persistent environment: {e}")
            return False
        
        return True
    
    def cleanup_persistent_resources(self):
        """Clean up persistent resources."""
        print("🧹 Cleaning up persistent resources...")
        
        if self.persistent_client:
            self.persistent_client.disconnect()
            self.persistent_client = None
            print("   ✅ Disconnected from brain server")
        
        self.persistent_environment = None
        print("   ✅ Cleaned up test environment")
    
    def run_all(self, stop_on_failure: bool = False) -> Dict[str, Any]:
        """Run all experiments with persistent resources."""
        print(f"🔬 Running {len(self.experiments)} micro-experiments")
        print("=" * 60)
        
        # Setup persistent resources
        if not self.setup_persistent_resources():
            return {
                'total_experiments': len(self.experiments),
                'successful_experiments': 0,
                'failed_experiments': len(self.experiments),
                'success_rate': 0.0,
                'error': 'Failed to setup persistent resources'
            }
        
        try:
            self.results = []
            failed_experiments = []
            
            for i, experiment in enumerate(self.experiments):
                print(f"\\n[{i+1}/{len(self.experiments)}] {experiment.name}")
                
                # Run with persistent resources
                result = experiment.execute(
                    shared_client=self.persistent_client,
                    shared_environment=self.persistent_environment
                )
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
            summary = self.generate_summary()
            self.save_results(summary)
            
            return summary
            
        finally:
            self.cleanup_persistent_resources()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of all results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        success_rate = passed / total if total > 0 else 0.0
        avg_confidence = np.mean([r.confidence for r in self.results]) if self.results else 0.0
        
        # Calculate per-assumption scores
        assumption_scores = {}
        for result in self.results:
            assumption = result.assumption_tested
            if assumption not in assumption_scores:
                assumption_scores[assumption] = []
            assumption_scores[assumption].append(result.confidence)
        
        # Average confidence per assumption
        for assumption in assumption_scores:
            assumption_scores[assumption] = np.mean(assumption_scores[assumption])
        
        return {
            'total_experiments': total,
            'successful_experiments': passed,
            'failed_experiments': failed,
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'assumption_scores': assumption_scores,
            'experiment_results': [r.to_dict() for r in self.results]
        }
    
    def save_results(self, summary: Dict[str, Any]):
        """Save results to file."""
        timestamp = int(time.time())
        results_file = self.results_dir / f"micro_experiment_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\\n📊 Results saved to {results_file}")