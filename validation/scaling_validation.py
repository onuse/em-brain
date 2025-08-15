#!/usr/bin/env python3
"""
Scaling Validation Study

Extends the existing validation framework to test large-scale performance (10k+ experiences)
while integrating with existing infrastructure and methodologies.

This complements the existing biological_embodied_learning.py by focusing specifically
on scalability and performance characteristics at large volumes.

Usage:
    python3 scaling_validation.py --scale 10000 --use-existing-environment
    python3 scaling_validation.py --scale 25000 --parallel --duration 2.0
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

# Import existing validation infrastructure
from embodied_learning.experiments.biological_embodied_learning import (
    BiologicalEmbodiedLearningExperiment, 
    ExperimentConfig
)
from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
from src.communication.client import MinimalBrainClient


@dataclass
class ScalingConfig:
    """Configuration for scaling validation study."""
    target_experiences: int = 10000
    batch_size: int = 100
    progress_checkpoints: List[int] = None
    use_existing_environment: bool = True
    parallel_batches: bool = False
    duration_hours: float = 2.0
    random_seed: int = 42
    
    def __post_init__(self):
        if self.progress_checkpoints is None:
            # Default checkpoints at 10%, 25%, 50%, 75%, 90%, 100%
            self.progress_checkpoints = [
                int(self.target_experiences * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
            ]


class ScalingValidationStudy:
    """
    Large-scale validation study that integrates with existing infrastructure.
    
    This extends the biological learning experiment to test scalability while
    reusing existing environments, metrics, and analysis frameworks.
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.results_dir = Path(f"validation/scaling_results_{int(time.time())}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize existing infrastructure
        self.client = MinimalBrainClient()
        
        # Reuse existing environment if requested
        if config.use_existing_environment:
            self.environment = SensoryMotorWorld(random_seed=config.random_seed)
            print(f"📊 Using existing SensoryMotorWorld environment")
        else:
            self.environment = self._create_scaling_environment()
            print(f"📊 Created new scaling environment")
        
        # Scaling metrics
        self.scaling_metrics = []
        self.performance_samples = []
        self.memory_samples = []
        
        # Error tracking aligned with ERROR_CODES.md
        self.error_summary = {
            'total_errors': 0,
            'retryable_errors': 0,
            'non_retryable_errors': 0,
            'error_codes': {}  # Track specific error codes encountered
        }
        
        print(f"📈 Scaling Validation Study")
        print(f"   Target experiences: {config.target_experiences:,}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Duration: {config.duration_hours} hours")
        print(f"   Results: {self.results_dir}")
    
    def _create_scaling_environment(self) -> SensoryMotorWorld:
        """Create environment optimized for scaling tests."""
        # Create a more complex environment for scaling tests
        return SensoryMotorWorld(
            world_size=15.0,  # Larger world
            num_light_sources=3,  # More complexity
            num_obstacles=8,
            random_seed=self.config.random_seed
        )
    
    def run_scaling_study(self) -> Dict[str, Any]:
        """Run the complete scaling validation study."""
        print(f"\\n🚀 Starting scaling validation study...")
        
        # Connect to brain
        if not self.client.connect():
            raise ConnectionError("Failed to connect to brain server")
        
        baseline_metrics = {}
        scaling_results = {}
        analysis_results = {}
        integration_results = {}
        
        try:
            start_time = time.time()
            
            # Phase 1: Baseline measurement
            print(f"\\n📊 Phase 1: Baseline Performance Measurement")
            baseline_metrics = self._measure_baseline_performance()
            
            # Phase 2: Scaling performance test
            print(f"\\n📈 Phase 2: Large-Scale Experience Processing")
            scaling_results = self._run_scaling_test()
            
            # Phase 3: Performance analysis
            print(f"\\n🔍 Phase 3: Scaling Analysis")
            analysis_results = self._analyze_scaling_performance(baseline_metrics, scaling_results)
            
            # Phase 4: Integration with existing experiments
            print(f"\\n🔗 Phase 4: Integration Validation")
            integration_results = self._validate_integration()
            
        except KeyboardInterrupt:
            print(f"\\n⏹️  Scaling validation interrupted by user")
            print(f"   📊 Partial results will be saved...")
            
            # Try to generate analysis with partial data
            try:
                if baseline_metrics and scaling_results:
                    analysis_results = self._analyze_scaling_performance(baseline_metrics, scaling_results)
                else:
                    analysis_results = {'interrupted': True, 'message': 'Insufficient data for analysis'}
            except Exception as e:
                analysis_results = {'interrupted': True, 'error': str(e)}
            
            integration_results = {'interrupted': True, 'message': 'Integration test skipped due to interruption'}
            
        except Exception as e:
            print(f"\\n❌ Scaling validation failed: {e}")
            analysis_results = {'error': str(e)}
            integration_results = {'error': str(e)}
            
        finally:
            try:
                # Always try to save results, even if partial
                final_results = {
                    'config': {
                        'target_experiences': self.config.target_experiences,
                        'batch_size': self.config.batch_size,
                        'duration_hours': self.config.duration_hours,
                        'parallel_batches': self.config.parallel_batches
                    },
                    'baseline_metrics': baseline_metrics,
                    'scaling_results': scaling_results,
                    'analysis': analysis_results,
                    'integration_results': integration_results,
                    'study_duration': time.time() - start_time if 'start_time' in locals() else 0,
                    'total_experiences_processed': len(self.scaling_metrics),
                    'interrupted': 'interrupted' in analysis_results,
                    'error_summary': self.error_summary  # Include error tracking per ERROR_CODES.md
                }
                
                self._save_results(final_results)
                self._generate_scaling_report(final_results)
                
                return final_results
                
            except Exception as cleanup_error:
                print(f"   ⚠️  Error during cleanup: {cleanup_error}")
                # Return minimal results to avoid complete failure
                return {
                    'config': {'target_experiences': self.config.target_experiences},
                    'baseline_metrics': baseline_metrics,
                    'scaling_results': scaling_results,
                    'analysis': analysis_results,
                    'integration_results': integration_results,
                    'cleanup_error': str(cleanup_error),
                    'interrupted': True
                }
            finally:
                self.client.disconnect()
    
    def _get_action_with_retry(self, sensory_input, max_retries: int = 3, timeout: float = 10.0):
        """Get action from brain with retry logic for Brain Processing Errors (5.x codes)."""
        for attempt in range(max_retries):
            try:
                action = self.client.get_action(sensory_input, timeout=timeout)
                if action is not None:
                    return action
                    
                # None response might be a brain processing error, try again
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                    
            except Exception as e:
                error_str = str(e)
                
                # Track error in summary
                self.error_summary['total_errors'] += 1
                
                # Check for specific 5.x brain processing errors
                if any(code in error_str for code in ["5.0", "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7"]):
                    self.error_summary['retryable_errors'] += 1
                    
                    # Extract and track specific error code
                    detected_code = None
                    error_names = {
                        "5.1": "SIMILARITY_ENGINE_FAILURE",
                        "5.2": "PREDICTION_ENGINE_FAILURE", 
                        "5.3": "EXPERIENCE_STORAGE_FAILURE",
                        "5.4": "ACTIVATION_DYNAMICS_FAILURE",
                        "5.5": "PATTERN_ANALYSIS_FAILURE",
                        "5.6": "MEMORY_PRESSURE_ERROR",
                        "5.7": "GPU_PROCESSING_ERROR",
                        "5.0": "BRAIN_PROCESSING_ERROR"
                    }
                    
                    for code, name in error_names.items():
                        if code in error_str:
                            detected_code = code
                            self.error_summary['error_codes'][code] = self.error_summary['error_codes'].get(code, 0) + 1
                            break
                    
                    if attempt < max_retries - 1:
                        # Extract specific error code for better reporting
                        if detected_code:
                            error_name = error_names[detected_code]
                            print(f"   ⚠️  {error_name} ({detected_code}) - attempt {attempt + 1}/{max_retries}, retrying...")
                            
                            # Special handling for generic error 5.0
                            if detected_code == "5.0":
                                print(f"      💡 Note: Error 5.0 is generic - check logs/brain_errors.jsonl for details")
                                self._check_error_logs()
                        else:
                            print(f"   ⚠️  Brain processing error - attempt {attempt + 1}/{max_retries}, retrying...")
                        
                        time.sleep(0.1)
                        continue
                    else:
                        # Final attempt failed - report specific error
                        if detected_code:
                            error_name = error_names[detected_code]
                            print(f"   ❌ {error_name} ({detected_code}) - failed after {max_retries} attempts")
                            
                            # For error 5.0, provide more investigation guidance
                            if detected_code == "5.0":
                                print(f"      💡 Generic error 5.0 indicates unclassified failure")
                                print(f"      📋 Check logs/brain_errors.jsonl for detailed error information")
                                print(f"      🔍 Use: python3 server/tools/view_errors.py --limit 5")
                                self._check_error_logs()
                        else:
                            print(f"   ❌ Brain processing failed after {max_retries} attempts: {error_str}")
                        return None
                else:
                    # Non-retryable error
                    self.error_summary['non_retryable_errors'] += 1
                    print(f"   ❌ Non-retryable error: {e}")
                    return None
        
        return None
    
    def _check_error_logs(self):
        """Check recent error logs for more details about error 5.0."""
        try:
            error_log_file = Path("logs/brain_errors.jsonl")
            if error_log_file.exists():
                # Read last few errors
                with open(error_log_file, 'r') as f:
                    lines = f.readlines()
                    recent_errors = [json.loads(line.strip()) for line in lines[-3:]]
                    
                if recent_errors:
                    print(f"      📋 Recent errors from log:")
                    for error in recent_errors:
                        print(f"         - {error.get('error_name', 'Unknown')}: {error.get('message', 'No message')}")
                        if error.get('exception'):
                            print(f"           Exception: {error.get('exception_type', 'Unknown')}: {error.get('exception')}")
        except Exception as e:
            # Don't fail if we can't read logs
            pass

    def _measure_baseline_performance(self) -> Dict[str, Any]:
        """Measure baseline performance with small number of experiences."""
        print(f"   Measuring baseline with 100 experiences...")
        
        baseline_experiences = 100
        start_time = time.time()
        
        response_times = []
        memory_usage = []
        failed_actions = 0
        
        for i in range(baseline_experiences):
            # Get sensory input
            sensory_input = self.environment.get_sensory_input()
            
            # Measure response time with retry logic
            action_start = time.time()
            action = self._get_action_with_retry(sensory_input, max_retries=3, timeout=5.0)
            response_time = time.time() - action_start
            
            if action is not None:
                response_times.append(response_time)
                self.environment.execute_action(action)
            else:
                failed_actions += 1
            
            # Sample memory usage periodically
            if i % 10 == 0:
                memory_usage.append(self._estimate_memory_usage())
        
        baseline_duration = time.time() - start_time
        
        if failed_actions > 0:
            print(f"   ⚠️  {failed_actions} failed actions during baseline measurement")
        
        return {
            'experiences': baseline_experiences,
            'duration': baseline_duration,
            'avg_response_time': np.mean(response_times) if response_times else 0.0,
            'response_time_std': np.std(response_times) if response_times else 0.0,
            'experiences_per_second': len(response_times) / baseline_duration,  # Use successful experiences
            'memory_usage_baseline': np.mean(memory_usage) if memory_usage else 0.0,
            'response_times': response_times[-10:],  # Last 10 for comparison
            'failed_actions': failed_actions,
            'success_rate': len(response_times) / baseline_experiences if baseline_experiences > 0 else 0.0
        }
    
    def _run_scaling_test(self) -> Dict[str, Any]:
        """Run the main scaling test."""
        experiences_processed = 0
        start_time = time.time()
        
        try:
            while experiences_processed < self.config.target_experiences:
                # Process batch
                batch_start = time.time()
                batch_results = self._process_experience_batch()
                batch_duration = time.time() - batch_start
                
                experiences_processed += len(batch_results)
                
                # Record scaling metrics
                scaling_metric = {
                    'experiences_processed': experiences_processed,
                    'batch_size': len(batch_results),
                    'batch_duration': batch_duration,
                    'experiences_per_second': len(batch_results) / batch_duration if batch_duration > 0 else 0,
                    'timestamp': time.time(),
                    'elapsed_time': time.time() - start_time
                }
                self.scaling_metrics.append(scaling_metric)
                
                # Check progress checkpoints
                if experiences_processed in self.config.progress_checkpoints:
                    checkpoint_results = self._checkpoint_analysis(experiences_processed)
                    self.performance_samples.append(checkpoint_results)
                    
                    print(f"   📊 Checkpoint {experiences_processed:,}: "
                          f"Rate={scaling_metric['experiences_per_second']:.1f} exp/s, "
                          f"Memory={checkpoint_results['memory_usage_mb']:.1f}MB, "
                          f"Success={checkpoint_results['success_rate']:.1%}")
                
                # Time limit check
                if (time.time() - start_time) > (self.config.duration_hours * 3600):
                    print(f"   ⏰ Time limit reached: {self.config.duration_hours} hours")
                    break
                    
        except KeyboardInterrupt:
            print(f"\\n   ⏹️  Scaling test interrupted at {experiences_processed:,} experiences")
            # Let the exception propagate to be handled by the main method
            raise
        
        return {
            'experiences_processed': experiences_processed,
            'total_duration': time.time() - start_time,
            'scaling_metrics': self.scaling_metrics,
            'performance_samples': self.performance_samples
        }
    
    def _process_experience_batch(self) -> List[Dict[str, Any]]:
        """Process a batch of experiences."""
        batch_results = []
        failed_count = 0
        
        for _ in range(self.config.batch_size):
            # Generate experience using existing environment
            sensory_input = self.environment.get_sensory_input()
            
            # Get brain response with retry logic
            action = self._get_action_with_retry(sensory_input, max_retries=2, timeout=3.0)
            if action is None:
                failed_count += 1
                continue
            
            # Execute in environment
            result = self.environment.execute_action(action)
            
            # Record experience
            experience_data = {
                'sensory_input': sensory_input,
                'action': action,
                'result': result,
                'timestamp': time.time()
            }
            batch_results.append(experience_data)
        
        # Report failures if significant
        if failed_count > 0:
            print(f"   ⚠️  {failed_count}/{self.config.batch_size} actions failed in batch")
        
        return batch_results
    
    def _checkpoint_analysis(self, experiences_count: int) -> Dict[str, Any]:
        """Analyze performance at a checkpoint."""
        # Get recent performance metrics
        recent_metrics = self.scaling_metrics[-10:] if len(self.scaling_metrics) >= 10 else self.scaling_metrics
        
        if not recent_metrics:
            return {'insufficient_data': True}
        
        # Calculate current performance
        current_rate = np.mean([m['experiences_per_second'] for m in recent_metrics])
        memory_usage = self._estimate_memory_usage()
        
        # Test response time with retry logic
        response_times = []
        failed_responses = 0
        for _ in range(10):
            sensory_input = self.environment.get_sensory_input()
            start = time.time()
            action = self._get_action_with_retry(sensory_input, max_retries=2, timeout=3.0)
            if action is not None:
                response_times.append(time.time() - start)
            else:
                failed_responses += 1
        
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        return {
            'experiences_count': experiences_count,
            'current_rate': current_rate,
            'memory_usage_mb': memory_usage,
            'avg_response_time': avg_response_time,
            'response_time_samples': len(response_times),
            'failed_responses': failed_responses,
            'success_rate': len(response_times) / 10.0,  # Out of 10 attempts
            'timestamp': time.time()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _analyze_scaling_performance(self, baseline: Dict, scaling: Dict) -> Dict[str, Any]:
        """Analyze scaling performance characteristics."""
        
        if not self.performance_samples:
            return {'insufficient_data': True}
        
        # Extract trends
        experience_counts = [s['experiences_count'] for s in self.performance_samples]
        processing_rates = [s['current_rate'] for s in self.performance_samples]
        memory_usage = [s['memory_usage_mb'] for s in self.performance_samples]
        response_times = [s['avg_response_time'] for s in self.performance_samples]
        
        # Calculate scaling efficiency
        baseline_rate = baseline['experiences_per_second']
        final_rate = processing_rates[-1] if processing_rates else baseline_rate
        scaling_efficiency = final_rate / baseline_rate
        
        # Memory scaling analysis
        baseline_memory = baseline['memory_usage_baseline']
        final_memory = memory_usage[-1] if memory_usage else baseline_memory
        memory_scaling_factor = final_memory / baseline_memory
        
        # Response time analysis
        baseline_response = baseline['avg_response_time']
        final_response = response_times[-1] if response_times else baseline_response
        response_time_degradation = final_response / baseline_response
        
        # Performance trend analysis
        if len(processing_rates) >= 3:
            # Linear regression on processing rates
            rate_trend = np.polyfit(experience_counts, processing_rates, 1)[0]
            memory_trend = np.polyfit(experience_counts, memory_usage, 1)[0]
            response_trend = np.polyfit(experience_counts, response_times, 1)[0]
        else:
            rate_trend = 0.0
            memory_trend = 0.0
            response_trend = 0.0
        
        return {
            'scaling_efficiency': scaling_efficiency,
            'memory_scaling_factor': memory_scaling_factor,
            'response_time_degradation': response_time_degradation,
            'performance_trends': {
                'rate_trend': rate_trend,
                'memory_trend': memory_trend,
                'response_trend': response_trend
            },
            'final_metrics': {
                'processing_rate': final_rate,
                'memory_usage_mb': final_memory,
                'response_time_ms': final_response * 1000
            },
            'scaling_assessment': self._assess_scaling_quality(scaling_efficiency, memory_scaling_factor, response_time_degradation)
        }
    
    def _assess_scaling_quality(self, efficiency: float, memory_factor: float, response_degradation: float) -> str:
        """Assess overall scaling quality."""
        if efficiency > 0.8 and memory_factor < 2.0 and response_degradation < 1.5:
            return "excellent"
        elif efficiency > 0.6 and memory_factor < 3.0 and response_degradation < 2.0:
            return "good"
        elif efficiency > 0.4 and memory_factor < 5.0 and response_degradation < 3.0:
            return "acceptable"
        else:
            return "poor"
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate integration with existing experiment framework."""
        print(f"   Testing integration with biological learning experiment...")
        
        try:
            # Create a mini biological learning experiment
            bio_config = ExperimentConfig(
                duration_hours=0.1,  # 6 minutes
                session_duration_minutes=5,
                consolidation_duration_minutes=1,
                random_seed=self.config.random_seed
            )
            
            # This reuses the existing environment and client
            bio_experiment = BiologicalEmbodiedLearningExperiment(bio_config)
            
            # Run just the baseline session
            baseline_session = bio_experiment._run_learning_session(session_id=0, is_baseline=True)
            
            integration_success = True
            integration_metrics = {
                'session_actions': baseline_session.total_actions,
                'session_performance': bio_experiment._calculate_session_performance(baseline_session),
                'avg_light_distance': baseline_session.avg_light_distance,
                'exploration_score': baseline_session.exploration_score
            }
            
        except Exception as e:
            print(f"   ❌ Integration test failed: {e}")
            integration_success = False
            integration_metrics = {'error': str(e)}
        
        return {
            'integration_success': integration_success,
            'integration_metrics': integration_metrics,
            'framework_compatibility': integration_success
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        results_file = self.results_dir / "scaling_validation_results.json"
        
        # Convert numpy types to JSON-serializable
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
    
    def _generate_scaling_report(self, results: Dict[str, Any]):
        """Generate scaling validation report."""
        
        analysis = results['analysis']
        config = results['config']
        interrupted = results.get('interrupted', False)
        
        report = f"""# Scaling Validation Report
        
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Target Experiences**: {config['target_experiences']:,}  
**Actual Experiences**: {results['total_experiences_processed']:,}  
**Duration**: {results['study_duration']/3600:.1f} hours  
**Status**: {'INTERRUPTED' if interrupted else 'COMPLETED'}  

"""
        
        if interrupted:
            report += f"""
## ⚠️  Study Interrupted
This study was interrupted before completion. Results are based on partial data.

"""
        
        # Only include detailed analysis if we have valid data
        if 'error' not in analysis and 'interrupted' not in analysis:
            report += f"""## Scaling Performance

**Scaling Efficiency**: {analysis['scaling_efficiency']:.3f}  
**Memory Scaling Factor**: {analysis['memory_scaling_factor']:.2f}x  
**Response Time Degradation**: {analysis['response_time_degradation']:.2f}x  
**Overall Assessment**: {analysis['scaling_assessment']}  

## Final Metrics

- **Processing Rate**: {analysis['final_metrics']['processing_rate']:.1f} experiences/sec
- **Memory Usage**: {analysis['final_metrics']['memory_usage_mb']:.1f} MB
- **Response Time**: {analysis['final_metrics']['response_time_ms']:.1f} ms

## Performance Trends

- **Rate Trend**: {analysis['performance_trends']['rate_trend']:+.2f} exp/s per 1000 experiences
- **Memory Trend**: {analysis['performance_trends']['memory_trend']:+.2f} MB per 1000 experiences  
- **Response Trend**: {analysis['performance_trends']['response_trend']:+.4f} s per 1000 experiences

"""
        else:
            report += f"""## Analysis Status

**Analysis Error**: {analysis.get('error', 'Analysis incomplete due to interruption')}  
**Message**: {analysis.get('message', 'Partial data collected but insufficient for full analysis')}  

"""
        
        # Integration results
        integration = results['integration_results']
        if 'error' not in integration and 'interrupted' not in integration:
            report += f"""## Integration Validation

**Framework Compatibility**: {integration['framework_compatibility']}  
**Integration Success**: {integration['integration_success']}  

"""
        else:
            report += f"""## Integration Validation

**Status**: {integration.get('message', 'Integration test not completed')}  

"""
        
        # Baseline metrics (if available)
        baseline = results.get('baseline_metrics', {})
        if baseline:
            report += f"""## Baseline Performance

- **Success Rate**: {baseline.get('success_rate', 0.0):.1%}
- **Average Response Time**: {baseline.get('avg_response_time', 0.0)*1000:.1f} ms
- **Processing Rate**: {baseline.get('experiences_per_second', 0.0):.1f} experiences/sec
- **Failed Actions**: {baseline.get('failed_actions', 0)}

"""
        
        # Error summary (per ERROR_CODES.md)
        error_summary = results.get('error_summary', {})
        if error_summary.get('total_errors', 0) > 0:
            report += f"""## Error Summary (per ERROR_CODES.md)

- **Total Errors**: {error_summary.get('total_errors', 0)}
- **Retryable Errors**: {error_summary.get('retryable_errors', 0)}
- **Non-retryable Errors**: {error_summary.get('non_retryable_errors', 0)}

### Error Codes Encountered:
"""
            
            error_codes = error_summary.get('error_codes', {})
            error_names = {
                "5.0": "BRAIN_PROCESSING_ERROR",
                "5.1": "SIMILARITY_ENGINE_FAILURE",
                "5.2": "PREDICTION_ENGINE_FAILURE", 
                "5.3": "EXPERIENCE_STORAGE_FAILURE",
                "5.4": "ACTIVATION_DYNAMICS_FAILURE",
                "5.5": "PATTERN_ANALYSIS_FAILURE",
                "5.6": "MEMORY_PRESSURE_ERROR",
                "5.7": "GPU_PROCESSING_ERROR"
            }
            
            for code, count in error_codes.items():
                error_name = error_names.get(code, "UNKNOWN_ERROR")
                report += f"- **{error_name} ({code})**: {count} occurrences\\n"
            
            report += f"""
### Resolution Guidance:
"""
            
            # Add specific resolution guidance based on encountered errors
            if "5.0" in error_codes:
                report += "- **5.0 BRAIN_PROCESSING_ERROR**: Generic error - check logs/brain_errors.jsonl for root cause\\n"
                report += "  - Run: `python3 server/tools/view_errors.py --limit 10`\\n"
                report += "  - This indicates the server couldn't classify the specific error type\\n"
                report += "  - Common causes: timeout, memory issues, or unexpected exceptions\\n"
            if "5.1" in error_codes:
                report += "- **5.1 SIMILARITY_ENGINE_FAILURE**: Check similarity engine cache and GPU status\\n"
            if "5.2" in error_codes:
                report += "- **5.2 PREDICTION_ENGINE_FAILURE**: Check prediction engine and pattern analysis\\n"
            if "5.3" in error_codes:
                report += "- **5.3 EXPERIENCE_STORAGE_FAILURE**: Check experience storage and memory limits\\n"
            if "5.4" in error_codes:
                report += "- **5.4 ACTIVATION_DYNAMICS_FAILURE**: Check activation system and utility calculations\\n"
            if "5.5" in error_codes:
                report += "- **5.5 PATTERN_ANALYSIS_FAILURE**: Check GPU status and pattern discovery cache\\n"
            if "5.6" in error_codes:
                report += "- **5.6 MEMORY_PRESSURE_ERROR**: Reduce cache sizes or increase available memory\\n"
            if "5.7" in error_codes:
                report += "- **5.7 GPU_PROCESSING_ERROR**: Check GPU status and MPS availability\\n"
            
            report += f"""
"""
        
        report += f"""## Recommendations

"""
        
        # Add recommendations based on available data
        if 'error' not in analysis and 'interrupted' not in analysis:
            if analysis['scaling_efficiency'] > 0.8:
                report += "- ✅ Excellent scaling efficiency - system handles large volumes well\\n"
            elif analysis['scaling_efficiency'] > 0.6:
                report += "- ⚠️ Good scaling efficiency - minor optimizations may help\\n"
            else:
                report += "- ❌ Poor scaling efficiency - significant optimizations needed\\n"
            
            if analysis['memory_scaling_factor'] < 2.0:
                report += "- ✅ Memory usage scales well\\n"
            elif analysis['memory_scaling_factor'] < 3.0:
                report += "- ⚠️ Memory usage acceptable but monitor for larger scales\\n"
            else:
                report += "- ❌ Memory usage grows too quickly - optimization needed\\n"
            
            if analysis['response_time_degradation'] < 1.5:
                report += "- ✅ Response times remain stable at scale\\n"
            elif analysis['response_time_degradation'] < 2.0:
                report += "- ⚠️ Response times degrade moderately - acceptable for most use cases\\n"
            else:
                report += "- ❌ Response times degrade significantly - performance optimization needed\\n"
                
        elif interrupted:
            report += "- 🔄 Rerun study with longer duration to get complete results\\n"
            if baseline.get('success_rate', 1.0) < 0.9:
                report += "- ⚠️ Consider investigating Brain Processing Error (5.0) causes\\n"
        else:
            report += "- ❌ Study failed - check error logs and system status\\n"
        
        # Save report
        report_file = self.results_dir / "scaling_validation_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📄 Report saved to {report_file}")


def main():
    """Main entry point for scaling validation."""
    parser = argparse.ArgumentParser(description="Scaling validation study")
    parser.add_argument('--scale', type=int, default=10000, help='Target number of experiences')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--duration', type=float, default=2.0, help='Duration limit in hours')
    parser.add_argument('--parallel', action='store_true', help='Use parallel batch processing')
    parser.add_argument('--use-existing-environment', action='store_true', default=True,
                       help='Use existing SensoryMotorWorld environment')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ScalingConfig(
        target_experiences=args.scale,
        batch_size=args.batch_size,
        duration_hours=args.duration,
        parallel_batches=args.parallel,
        use_existing_environment=args.use_existing_environment,
        random_seed=args.seed
    )
    
    # Run scaling study
    study = ScalingValidationStudy(config)
    
    try:
        results = study.run_scaling_study()
        
        # Print summary
        print(f"\\n📊 Scaling Validation Summary:")
        print(f"   Experiences processed: {results['total_experiences_processed']:,}")
        print(f"   Duration: {results['study_duration']/3600:.1f} hours")
        
        # Analysis results (if available)
        if 'scaling_efficiency' in results['analysis']:
            print(f"   Scaling efficiency: {results['analysis']['scaling_efficiency']:.3f}")
            print(f"   Assessment: {results['analysis']['scaling_assessment']}")
        
        # Baseline metrics (if available)
        if 'success_rate' in results['baseline_metrics']:
            print(f"   Baseline success rate: {results['baseline_metrics']['success_rate']:.1%}")
        
        # Error summary (per ERROR_CODES.md)
        error_summary = results.get('error_summary', {})
        if error_summary.get('total_errors', 0) > 0:
            print(f"   Total errors: {error_summary['total_errors']} ({error_summary['retryable_errors']} retryable)")
            
            # Show most common error codes
            error_codes = error_summary.get('error_codes', {})
            if error_codes:
                most_common = max(error_codes.items(), key=lambda x: x[1])
                error_names = {
                    "5.0": "BRAIN_PROCESSING_ERROR",
                    "5.1": "SIMILARITY_ENGINE_FAILURE",
                    "5.2": "PREDICTION_ENGINE_FAILURE", 
                    "5.6": "MEMORY_PRESSURE_ERROR",
                    "5.7": "GPU_PROCESSING_ERROR"
                }
                error_name = error_names.get(most_common[0], "UNKNOWN_ERROR")
                print(f"   Most common error: {error_name} ({most_common[0]}) - {most_common[1]} times")
        
        print(f"   Results saved to: {study.results_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\\n❌ Scaling validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())