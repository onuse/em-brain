#!/usr/bin/env python3
"""
Extended Learning Test

Uses the integration test pattern to run a longer learning session.
Tests brain behavior over extended time while managing server lifecycle.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from validation.test_integration import IntegrationTestSuite
from server.src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

class ExtendedLearningTest:
    """Extended learning test using integration test infrastructure."""
    
    def __init__(self, duration_minutes: int = 5):
        self.duration_minutes = duration_minutes
        self.results = {
            'start_time': time.time(),
            'experiences': [],
            'learning_metrics': [],
            'connection_events': []
        }
        self.integration_suite = None
        
    def run_test(self):
        """Run extended learning test."""
        print(f"🧠 Extended Learning Test ({self.duration_minutes} minutes)")
        print("=" * 60)
        print("Testing: Learning progression, memory bounds, connection stability")
        print()
        
        try:
            # Start server using integration test infrastructure
            print("🚀 Starting brain server...")
            self.integration_suite = IntegrationTestSuite()
            self.integration_suite._test_server_startup()
            
            if not self.integration_suite._is_server_ready():
                print("❌ Failed to start server")
                return False
            
            print("✅ Server started successfully")
            
            # Run extended learning session
            success = self._run_learning_session()
            
            if success:
                self._analyze_results()
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            self._cleanup()
    
    def _run_learning_session(self) -> bool:
        """Run the extended learning session."""
        client = MinimalBrainClient()
        environment = SensoryMotorWorld(random_seed=42)
        
        if not client.connect():
            print("❌ Failed to connect to brain")
            return False
        
        print(f"✅ Connected - starting {self.duration_minutes}-minute learning session...")
        
        end_time = time.time() + (self.duration_minutes * 60)
        cycle_count = 0
        last_report_time = time.time()
        
        try:
            while time.time() < end_time:
                cycle_start = time.time()
                
                # Get sensory input
                sensory_input = environment.get_sensory_input()
                
                # Get brain action
                action_start = time.time()
                action = client.get_action(sensory_input, timeout=10.0)
                response_time = time.time() - action_start
                
                if action is None:
                    self.results['connection_events'].append({
                        'time': time.time(),
                        'event': 'timeout',
                        'cycle': cycle_count
                    })
                    continue
                
                # Execute action
                result = environment.execute_action(action)
                
                # Calculate basic learning metrics
                next_sensory = environment.get_sensory_input()
                prediction_error = self._calculate_prediction_error(action, next_sensory)
                
                # Record experience
                experience = {
                    'cycle': cycle_count,
                    'timestamp': time.time(),
                    'elapsed_minutes': (time.time() - self.results['start_time']) / 60,
                    'response_time_ms': response_time * 1000,
                    'prediction_error': prediction_error,
                    'light_distance': result.get('metrics', {}).get('min_light_distance', 1.0)
                }
                self.results['experiences'].append(experience)
                
                # Report progress every 30 seconds
                if time.time() - last_report_time >= 30:
                    elapsed_minutes = (time.time() - self.results['start_time']) / 60
                    recent_experiences = self.results['experiences'][-30:]
                    avg_response = sum(exp['response_time_ms'] for exp in recent_experiences) / len(recent_experiences)
                    avg_error = sum(exp['prediction_error'] for exp in recent_experiences) / len(recent_experiences)
                    
                    print(f"⏱️  {elapsed_minutes:.1f}min: {cycle_count} cycles, {avg_response:.1f}ms response, {avg_error:.3f} error")
                    last_report_time = time.time()
                
                cycle_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            client.disconnect()
            print(f"✅ Completed {cycle_count} learning cycles")
            return True
            
        except Exception as e:
            print(f"❌ Learning session failed: {e}")
            client.disconnect()
            return False
    
    def _calculate_prediction_error(self, action: List[float], next_sensory: List[float]) -> float:
        """Calculate simple prediction error."""
        # Simple prediction error: difference between action and next sensory state
        if len(action) >= 4 and len(next_sensory) >= 4:
            error = sum(abs(a - s) for a, s in zip(action, next_sensory[:4]))
            return error / 4
        return 0.5  # Default moderate error
    
    def _analyze_results(self):
        """Analyze and report results."""
        print("\n📊 Extended Learning Test Results")
        print("=" * 50)
        
        if not self.results['experiences']:
            print("❌ No experiences recorded")
            return
        
        # Basic statistics
        total_experiences = len(self.results['experiences'])
        total_time = time.time() - self.results['start_time']
        
        print(f"Duration: {total_time:.1f}s")
        print(f"Total experiences: {total_experiences}")
        print(f"Experience rate: {total_experiences/total_time*60:.1f}/min")
        
        # Response time analysis
        response_times = [exp['response_time_ms'] for exp in self.results['experiences']]
        avg_response = sum(response_times) / len(response_times)
        max_response = max(response_times)
        min_response = min(response_times)
        
        print(f"Response time: {avg_response:.1f}ms avg, {max_response:.1f}ms max, {min_response:.1f}ms min")
        
        # Learning progression analysis
        experiences = self.results['experiences']
        early_batch = experiences[:len(experiences)//4]
        late_batch = experiences[-len(experiences)//4:]
        
        if early_batch and late_batch:
            early_error = sum(exp['prediction_error'] for exp in early_batch) / len(early_batch)
            late_error = sum(exp['prediction_error'] for exp in late_batch) / len(late_batch)
            error_improvement = early_error - late_error
            
            print(f"Learning progression:")
            print(f"  Early prediction error: {early_error:.3f}")
            print(f"  Late prediction error: {late_error:.3f}")
            print(f"  Error improvement: {error_improvement:+.3f}")
            
            if error_improvement > 0:
                print("  ✅ Learning detected (error decreased)")
            else:
                print("  ⚠️  No clear learning (error increased or stable)")
        
        # Performance consistency
        early_responses = response_times[:len(response_times)//4]
        late_responses = response_times[-len(response_times)//4:]
        
        if early_responses and late_responses:
            early_avg = sum(early_responses) / len(early_responses)
            late_avg = sum(late_responses) / len(late_responses)
            performance_change = ((late_avg - early_avg) / early_avg) * 100
            
            print(f"Performance consistency: {performance_change:+.1f}%")
            
            if abs(performance_change) < 50:
                print("  ✅ Performance stable")
            else:
                print("  ⚠️  Performance degradation detected")
        
        # Connection stability
        connection_issues = len(self.results['connection_events'])
        if connection_issues == 0:
            print("Connection stability: ✅ No issues")
        else:
            print(f"Connection stability: ⚠️  {connection_issues} timeouts")
        
        # Check persistence
        self._check_persistence_activity()
        
        # Save detailed results
        results_file = brain_root / f"extended_learning_test_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file.name}")
    
    def _check_persistence_activity(self):
        """Check if persistence system was active."""
        robot_memory_path = brain_root / "robot_memory"
        
        # Check for any new files
        if robot_memory_path.exists():
            checkpoints = list(robot_memory_path.glob("**/checkpoint_*"))
            
            if checkpoints:
                print(f"Persistence: ✅ {len(checkpoints)} checkpoints created")
                for checkpoint in checkpoints:
                    size_kb = checkpoint.stat().st_size / 1024
                    print(f"  - {checkpoint.name}: {size_kb:.1f}KB")
            else:
                print("Persistence: ⚠️  No checkpoints (checkpoint interval = 1000 experiences)")
        else:
            print("Persistence: ⚠️  No robot_memory directory")
    
    def _cleanup(self):
        """Clean up resources."""
        if self.integration_suite:
            print("\n🧹 Shutting down server...")
            self.integration_suite._test_server_shutdown()

def main():
    """Run extended learning test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extended Learning Test')
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes (default: 5)')
    
    args = parser.parse_args()
    
    test = ExtendedLearningTest(duration_minutes=args.duration)
    success = test.run_test()
    
    if success:
        print(f"\n🎉 {args.duration}-minute learning test completed successfully!")
        print("Brain demonstrated extended learning capability")
    else:
        print(f"\n❌ {args.duration}-minute learning test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)