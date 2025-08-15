#!/usr/bin/env python3
"""
Comprehensive Brain Verification Suite - Phase 1 Improved

This suite verifies that the optimized brain behaves as expected by:
1. Running extended validation tests with performance monitoring
2. Validating memory bounds and cache management
3. Testing biological realism with longer timescales
4. Monitoring for performance degradation
5. Ensuring learning persistence and consistency

Phase 1 Improvements:
- Persistent connections to reduce connection overhead
- World reuse pattern to minimize environment resets
- Retry logic for brain error 5.0 handling
- Increased timeouts for intensive tests
- Better error handling and reporting

Usage:
  python3 validation/comprehensive_verification.py --duration 60 --clear-logs
"""

import sys
import os
import time
import json
import psutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import shutil
import numpy as np
import matplotlib.pyplot as plt

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from test_integration import IntegrationTestSuite
from micro_experiments.improved_core_assumptions import create_improved_core_assumption_suite

class ComprehensiveBrainVerification:
    """Comprehensive verification suite for the optimized brain with Phase 1 improvements."""
    
    def __init__(self, duration_minutes: int = 60, clear_logs: bool = True):
        self.duration_minutes = duration_minutes
        
        # Persistent resources to avoid excessive reconnections
        self.persistent_client = None
        self.persistent_environment = None
        self.clear_logs = clear_logs
        self.start_time = None
        self.results = {}
        self.performance_metrics = []
        self.memory_metrics = []
        self.learning_metrics = []
        
        # Create results directory
        self.results_dir = Path("verification_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🔍 Comprehensive Brain Verification Suite")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Clear logs: {clear_logs}")
        print(f"   Results: {self.results_dir}")
    
    def _setup_persistent_resources(self) -> bool:
        """Setup persistent client and environment to avoid excessive reconnections."""
        print("🔧 Setting up persistent resources...")
        
        # Create persistent client
        try:
            from src.communication.client import MinimalBrainClient
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
            from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            self.persistent_environment = SensoryMotorWorld(random_seed=42)
            print("   ✅ Persistent test environment created")
        except Exception as e:
            print(f"   ❌ Failed to create persistent environment: {e}")
            return False
        
        return True
    
    def _cleanup_persistent_resources(self):
        """Clean up persistent resources."""
        print("🧹 Cleaning up persistent resources...")
        
        if self.persistent_client:
            self.persistent_client.disconnect()
            self.persistent_client = None
            print("   ✅ Disconnected from brain server")
        
        self.persistent_environment = None
        print("   ✅ Test environment cleaned up")
    
    def run_verification(self) -> Dict[str, Any]:
        """Run comprehensive verification suite."""
        print(f"\\n🎯 Starting Comprehensive Brain Verification")
        print("=" * 60)
        
        try:
            # Phase 1: Clean slate preparation
            if self.clear_logs:
                self._clear_logs_and_memory()
            
            # Phase 2: Start server and monitoring
            self._start_server_and_monitoring()
            
            # Phase 3: Run verification tests
            self._run_verification_tests()
            
            # Phase 4: Generate analysis
            self._generate_analysis()
            
            return self.results
            
        except KeyboardInterrupt:
            print("\\n⏹️ Verification interrupted by user")
            return self.results
        except Exception as e:
            print(f"\\n❌ Verification failed: {e}")
            return self.results
        finally:
            self._cleanup()
    
    def _clear_logs_and_memory(self):
        """Clear logs and memory to start fresh."""
        print("\\n🧹 Clearing logs and memory...")
        
        # Clear logs
        logs_dir = brain_root / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
            print(f"   ✅ Cleared logs directory: {logs_dir}")
        
        # Clear robot memory
        memory_dir = brain_root / "robot_memory"
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
            print(f"   ✅ Cleared memory directory: {memory_dir}")
        
        # Clear validation results
        validation_results = Path("micro_experiments/results")
        if validation_results.exists():
            shutil.rmtree(validation_results)
            print(f"   ✅ Cleared validation results")
        
        print("   🆕 Starting with clean slate")
    
    def _start_server_and_monitoring(self):
        """Start server and begin monitoring."""
        print("\\n🚀 Starting server and monitoring...")
        
        # Start server
        self.integration_suite = IntegrationTestSuite()
        self.integration_suite._test_server_startup()
        
        if not self.integration_suite._is_server_ready():
            raise RuntimeError("Failed to start server")
        
        print("   ✅ Server started successfully")
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("   ✅ Performance monitoring started")
        
        self.start_time = time.time()
    
    def _run_verification_tests(self):
        """Run comprehensive verification tests."""
        print("\\n🧪 Running verification tests...")
        
        # Test 1: Integration test
        print("\\n📋 Test 1: Integration Verification")
        integration_results = self._run_integration_test()
        self.results['integration'] = integration_results
        
        # Test 2: Micro-experiments over time
        print("\\n📋 Test 2: Micro-Experiments Over Time")
        micro_results = self._run_micro_experiments_over_time()
        self.results['micro_experiments'] = micro_results
        
        # Test 3: Extended learning test
        print("\\n📋 Test 3: Extended Learning Test")
        learning_results = self._run_extended_learning_test()
        self.results['extended_learning'] = learning_results
        
        # Test 4: Memory bounds test
        print("\\n📋 Test 4: Memory Bounds Test")
        memory_results = self._run_memory_bounds_test()
        self.results['memory_bounds'] = memory_results
        
        # Test 5: Performance consistency test
        print("\\n📋 Test 5: Performance Consistency Test")
        performance_results = self._run_performance_consistency_test()
        self.results['performance_consistency'] = performance_results
    
    def _run_integration_test(self) -> Dict[str, Any]:
        """Run integration test to verify basic functionality."""
        print("   Running integration test...")
        
        # Use the existing integration test
        integration_suite = IntegrationTestSuite()
        integration_suite.server_process = self.integration_suite.server_process
        
        # Run tests (skip server startup/shutdown)
        integration_suite._test_import_resolution()
        integration_suite._test_client_connection()
        integration_suite._test_environment_creation()
        integration_suite._test_sensory_input_generation()
        integration_suite._test_action_execution()
        integration_suite._test_brain_sensory_input()
        integration_suite._test_brain_action_output()
        integration_suite._test_round_trip_communication()
        integration_suite._test_connection_stability()
        integration_suite._test_consolidation_survival()
        
        # Calculate results
        passed = sum(1 for r in integration_suite.results if r.passed)
        total = len(integration_suite.results)
        
        return {
            'passed': passed,
            'total': total,
            'success_rate': passed / total if total > 0 else 0.0,
            'details': [r.test_name for r in integration_suite.results if r.passed]
        }
    
    def _run_micro_experiments_over_time(self) -> Dict[str, Any]:
        """Run micro-experiments at different time intervals."""
        print("   Running micro-experiments over time...")
        
        intervals = [0, 15, 30, 45]  # Minutes
        results = {}
        
        for interval in intervals:
            if interval > 0:
                print(f"   Waiting {interval} minutes for next test...")
                time.sleep(interval * 60)
            
            print(f"   Running micro-experiments at {interval} minutes...")
            suite = create_improved_core_assumption_suite()
            summary = suite.run_all(stop_on_failure=False)
            
            results[f"interval_{interval}min"] = {
                'success_rate': summary.get('success_rate', 0.0),
                'avg_confidence': summary.get('avg_confidence', 0.0),
                'assumption_scores': summary.get('assumption_scores', {}),
                'timestamp': time.time()
            }
        
        return results
    
    def _run_extended_learning_test(self) -> Dict[str, Any]:
        """Run extended learning test to verify biological realism."""
        print("   Running extended learning test...")
        
        from src.communication.client import MinimalBrainClient
        from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
        
        # Use persistent connection for improved performance
        client = MinimalBrainClient()
        environment = SensoryMotorWorld(random_seed=42)
        
        if not client.connect():
            return {'error': 'Failed to connect to brain'}
        
        try:
            # Run learning episodes with reduced resets
            episodes = 20
            episode_results = []
            
            for episode in range(episodes):
                print(f"   Episode {episode + 1}/{episodes}")
                
                # Reset environment only every 5 episodes (world reuse)
                if episode % 5 == 0:
                    environment.reset()
                
                # Run episode
                episode_data = self._run_learning_episode(client, environment, episode)
                episode_results.append(episode_data)
                
                # Brief pause between episodes
                time.sleep(1)  # Reduced pause
            
            # Analyze learning progression
            learning_analysis = self._analyze_learning_progression(episode_results)
            
            return {
                'episodes': len(episode_results),
                'episode_results': episode_results,
                'learning_analysis': learning_analysis
            }
            
        finally:
            client.disconnect()
    
    def _run_learning_episode(self, client, environment, episode_num: int) -> Dict[str, Any]:
        """Run a single learning episode."""
        actions_per_episode = 50
        prediction_errors = []
        light_distances = []
        failed_actions = 0
        
        for action_num in range(actions_per_episode):
            # Get sensory input
            sensory_input = environment.get_sensory_input()
            
            # Get brain action with retry logic
            start_time = time.time()
            action = self._get_action_with_retry(client, sensory_input, max_retries=3, timeout=10.0)
            response_time = time.time() - start_time
            
            if action is None:
                failed_actions += 1
                continue
            
            # Execute action
            result = environment.execute_action(action)
            
            # Calculate prediction error
            next_sensory = environment.get_sensory_input()
            prediction_error = np.mean(np.abs(np.array(action) - np.array(next_sensory[:4])))
            prediction_errors.append(prediction_error)
            
            # Get light distance
            metrics = result.get('metrics', {})
            light_distance = metrics.get('min_light_distance', 1.0)
            light_distances.append(light_distance)
        
        return {
            'episode': episode_num,
            'actions': len(prediction_errors),
            'failed_actions': failed_actions,
            'avg_prediction_error': np.mean(prediction_errors) if prediction_errors else 0.0,
            'avg_light_distance': np.mean(light_distances) if light_distances else 1.0,
            'prediction_errors': prediction_errors,
            'light_distances': light_distances
        }
    
    def _get_action_with_retry(self, client, sensory_input, max_retries: int = 3, timeout: float = 10.0):
        """Get action from brain with retry logic for error 5.0."""
        for attempt in range(max_retries):
            try:
                action = client.get_action(sensory_input, timeout=timeout)
                if action is not None:
                    return action
                    
                # None response might be error 5.0, try again
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Brief pause before retry
                    continue
                    
            except Exception as e:
                if "5.0" in str(e) or "Brain processing error" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        print(f"   ⚠️  Brain processing failed after {max_retries} attempts")
                        return None
                else:
                    print(f"   ⚠️  Unexpected error: {e}")
                    return None
        
        return None
    
    def _analyze_learning_progression(self, episode_results: List[Dict]) -> Dict[str, Any]:
        """Analyze learning progression across episodes."""
        if len(episode_results) < 3:
            return {'insufficient_data': True}
        
        # Extract metrics
        prediction_errors = [ep['avg_prediction_error'] for ep in episode_results]
        light_distances = [ep['avg_light_distance'] for ep in episode_results]
        
        # Calculate learning trends
        error_trend = np.polyfit(range(len(prediction_errors)), prediction_errors, 1)[0]
        distance_trend = np.polyfit(range(len(light_distances)), light_distances, 1)[0]
        
        # Early vs late performance
        early_episodes = episode_results[:5]
        late_episodes = episode_results[-5:]
        
        early_error = np.mean([ep['avg_prediction_error'] for ep in early_episodes])
        late_error = np.mean([ep['avg_prediction_error'] for ep in late_episodes])
        
        early_distance = np.mean([ep['avg_light_distance'] for ep in early_episodes])
        late_distance = np.mean([ep['avg_light_distance'] for ep in late_episodes])
        
        return {
            'error_trend_slope': error_trend,
            'distance_trend_slope': distance_trend,
            'error_improvement': early_error - late_error,
            'distance_improvement': early_distance - late_distance,
            'learning_detected': error_trend < 0 and (early_error - late_error) > 0,
            'navigation_improvement': distance_trend < 0 and (early_distance - late_distance) > 0
        }
    
    def _run_memory_bounds_test(self) -> Dict[str, Any]:
        """Test memory bounds and cache management."""
        print("   Testing memory bounds...")
        
        # Monitor memory usage over time
        memory_samples = []
        
        # Generate load to test memory bounds
        from src.communication.client import MinimalBrainClient
        from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
        
        client = MinimalBrainClient()
        environment = SensoryMotorWorld(random_seed=42)
        
        if not client.connect():
            return {'error': 'Failed to connect to brain'}
        
        try:
            # Generate many experiences to test memory bounds
            for i in range(200):  # Generate substantial load
                if i % 50 == 0:
                    print(f"   Memory test progress: {i}/200")
                
                # Get system memory
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_samples.append({
                    'iteration': i,
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'timestamp': time.time()
                })
                
                # Generate brain activity with retry logic
                sensory_input = environment.get_sensory_input()
                action = self._get_action_with_retry(client, sensory_input, max_retries=2, timeout=5.0)
                
                if action is not None:
                    environment.execute_action(action)
                
                # Small delay
                time.sleep(0.1)
            
            # Analyze memory usage
            initial_memory = memory_samples[0]['rss_mb']
            final_memory = memory_samples[-1]['rss_mb']
            max_memory = max(sample['rss_mb'] for sample in memory_samples)
            
            return {
                'samples': len(memory_samples),
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'max_memory_mb': max_memory,
                'memory_growth_mb': final_memory - initial_memory,
                'memory_bounded': max_memory < initial_memory + 200  # Allow 200MB growth
            }
            
        finally:
            client.disconnect()
    
    def _run_performance_consistency_test(self) -> Dict[str, Any]:
        """Test performance consistency over time."""
        print("   Testing performance consistency...")
        
        response_times = []
        
        from src.communication.client import MinimalBrainClient
        from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
        
        client = MinimalBrainClient()
        environment = SensoryMotorWorld(random_seed=42)
        
        if not client.connect():
            return {'error': 'Failed to connect to brain'}
        
        try:
            # Test response times over extended period
            for i in range(100):
                sensory_input = environment.get_sensory_input()
                
                start_time = time.time()
                action = self._get_action_with_retry(client, sensory_input, max_retries=2, timeout=10.0)
                response_time = time.time() - start_time
                
                if action is not None:
                    response_times.append(response_time)
                    environment.execute_action(action)
                
                time.sleep(0.5)  # Half second between requests
            
            # Analyze performance consistency
            early_times = response_times[:25]
            late_times = response_times[-25:]
            
            early_avg = np.mean(early_times)
            late_avg = np.mean(late_times)
            
            performance_degradation = (late_avg - early_avg) / early_avg
            
            return {
                'samples': len(response_times),
                'early_avg_ms': early_avg * 1000,
                'late_avg_ms': late_avg * 1000,
                'performance_degradation': performance_degradation,
                'consistent_performance': abs(performance_degradation) < 0.5  # Less than 50% degradation
            }
            
        finally:
            client.disconnect()
    
    def _monitor_performance(self):
        """Monitor system performance in background."""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Get process metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                
                metrics = {
                    'timestamp': time.time(),
                    'elapsed_minutes': (time.time() - self.start_time) / 60,
                    'cpu_percent': cpu_percent,
                    'system_memory_percent': memory.percent,
                    'process_memory_mb': process_memory.rss / 1024 / 1024,
                    'system_memory_available_gb': memory.available / 1024 / 1024 / 1024
                }
                
                self.performance_metrics.append(metrics)
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                print(f"   ⚠️ Monitoring error: {e}")
                time.sleep(30)
    
    def _generate_analysis(self):
        """Generate comprehensive analysis of verification results."""
        print("\\n📊 Generating analysis...")
        
        # Save raw results
        timestamp = int(time.time())
        results_file = self.results_dir / f"verification_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = self._convert_to_json_serializable(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Generate performance plots
        self._generate_performance_plots()
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"   ✅ Analysis saved to {self.results_dir}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_performance_plots(self):
        """Generate performance monitoring plots."""
        if not self.performance_metrics:
            return
        
        times = [m['elapsed_minutes'] for m in self.performance_metrics]
        cpu_usage = [m['cpu_percent'] for m in self.performance_metrics]
        memory_usage = [m['process_memory_mb'] for m in self.performance_metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU usage plot
        ax1.plot(times, cpu_usage, 'b-', label='CPU Usage %')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Usage Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Memory usage plot
        ax2.plot(times, memory_usage, 'r-', label='Memory Usage (MB)')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Memory Usage (MB)')
        ax2.set_title('Memory Usage Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_monitoring.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate summary report."""
        total_duration = (time.time() - self.start_time) / 60
        
        report = f"""
# Comprehensive Brain Verification Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Duration**: {total_duration:.1f} minutes

## Summary

"""
        
        # Integration test summary
        if 'integration' in self.results:
            integration = self.results['integration']
            report += f"**Integration Test**: {integration['passed']}/{integration['total']} tests passed ({integration['success_rate']:.1%})\\n\\n"
        
        # Micro-experiments summary
        if 'micro_experiments' in self.results:
            micro = self.results['micro_experiments']
            report += f"**Micro-experiments**: Tracked over {len(micro)} intervals\\n\\n"
        
        # Extended learning summary
        if 'extended_learning' in self.results:
            learning = self.results['extended_learning']
            if 'learning_analysis' in learning:
                analysis = learning['learning_analysis']
                report += f"**Extended Learning**: {learning['episodes']} episodes\\n"
                report += f"  - Learning detected: {analysis.get('learning_detected', False)}\\n"
                report += f"  - Error improvement: {analysis.get('error_improvement', 0):.4f}\\n\\n"
        
        # Memory bounds summary
        if 'memory_bounds' in self.results:
            memory = self.results['memory_bounds']
            report += f"**Memory Bounds**: {memory['samples']} samples\\n"
            report += f"  - Memory growth: {memory.get('memory_growth_mb', 0):.1f} MB\\n"
            report += f"  - Bounded: {memory.get('memory_bounded', False)}\\n\\n"
        
        # Performance consistency summary
        if 'performance_consistency' in self.results:
            perf = self.results['performance_consistency']
            report += f"**Performance Consistency**: {perf['samples']} samples\\n"
            report += f"  - Performance degradation: {perf.get('performance_degradation', 0):.1%}\\n"
            report += f"  - Consistent: {perf.get('consistent_performance', False)}\\n\\n"
        
        # Save report
        with open(self.results_dir / 'verification_report.md', 'w') as f:
            f.write(report)
    
    def _cleanup(self):
        """Clean up resources."""
        self.monitoring_active = False
        
        if hasattr(self, 'integration_suite'):
            self.integration_suite._test_server_shutdown()
        
        print("\\n🧹 Cleanup complete")

def main():
    """Main verification runner."""
    parser = argparse.ArgumentParser(description='Comprehensive Brain Verification Suite')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes')
    parser.add_argument('--clear-logs', action='store_true', help='Clear logs before starting')
    
    args = parser.parse_args()
    
    verification = ComprehensiveBrainVerification(
        duration_minutes=args.duration,
        clear_logs=args.clear_logs
    )
    
    results = verification.run_verification()
    
    print(f"\\n✅ Verification complete!")
    print(f"Results saved to: {verification.results_dir}")

if __name__ == "__main__":
    main()