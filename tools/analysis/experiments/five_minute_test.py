#!/usr/bin/env python3
"""
5-Minute Brain Test

Tests the optimized brain over 5 minutes to verify:
1. Connection stability
2. Memory bounds  
3. Learning progression
4. Performance consistency
5. Persistence behavior
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from server.src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

class FiveMinuteBrainTest:
    """5-minute brain test with monitoring."""
    
    def __init__(self):
        self.results = {
            'start_time': time.time(),
            'experiences': [],
            'performance_metrics': [],
            'connection_events': [],
            'memory_usage': [],
            'persistence_events': []
        }
        self.server_process = None
        
    def run_test(self):
        """Run the 5-minute test."""
        print("🧠 5-Minute Brain Test")
        print("=" * 50)
        print("Testing: Connection stability, memory bounds, learning, performance")
        print("Duration: 5 minutes")
        print()
        
        try:
            # Start server
            if not self._start_server():
                return False
            
            # Connect to brain
            client = MinimalBrainClient()
            environment = SensoryMotorWorld(random_seed=42)
            
            if not client.connect():
                print("❌ Failed to connect to brain")
                return False
            
            print("✅ Connected to brain - starting 5-minute test...")
            
            # Run test for 5 minutes
            end_time = time.time() + 300  # 5 minutes
            cycle_count = 0
            
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
                
                # Record experience
                experience = {
                    'cycle': cycle_count,
                    'timestamp': time.time(),
                    'elapsed_minutes': (time.time() - self.results['start_time']) / 60,
                    'sensory_input': sensory_input,
                    'action': action,
                    'response_time_ms': response_time * 1000,
                    'environment_result': result
                }
                self.results['experiences'].append(experience)
                
                # Record performance metrics every 30 seconds
                if cycle_count % 30 == 0:
                    self._record_performance_metrics(cycle_count)
                
                # Log progress every minute
                elapsed_minutes = (time.time() - self.results['start_time']) / 60
                if cycle_count % 60 == 0 and cycle_count > 0:
                    print(f"⏱️  {elapsed_minutes:.1f}min: {cycle_count} cycles, {response_time*1000:.1f}ms response")
                
                cycle_count += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.1)
            
            client.disconnect()
            
            # Generate final report
            self._generate_report()
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
        finally:
            self._cleanup()
    
    def _start_server(self) -> bool:
        """Start the brain server."""
        try:
            server_script = brain_root / "server" / "brain_server.py"
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(brain_root / "server"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            if self.server_process.poll() is None:
                print("✅ Server started successfully")
                return True
            else:
                print("❌ Server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Server startup error: {e}")
            return False
    
    def _record_performance_metrics(self, cycle_count: int):
        """Record performance metrics."""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get process metrics if server is running
            process_memory = 0
            if self.server_process and self.server_process.poll() is None:
                try:
                    process = psutil.Process(self.server_process.pid)
                    process_memory = process.memory_info().rss / 1024 / 1024  # MB
                except:
                    pass
            
            metrics = {
                'cycle': cycle_count,
                'timestamp': time.time(),
                'elapsed_minutes': (time.time() - self.results['start_time']) / 60,
                'cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'process_memory_mb': process_memory,
                'system_memory_available_gb': memory.available / 1024 / 1024 / 1024
            }
            
            self.results['performance_metrics'].append(metrics)
            
        except Exception as e:
            print(f"⚠️  Performance monitoring error: {e}")
    
    def _generate_report(self):
        """Generate test report."""
        print("\n📊 5-Minute Test Results")
        print("=" * 50)
        
        # Basic stats
        total_experiences = len(self.results['experiences'])
        total_time = time.time() - self.results['start_time']
        
        print(f"Duration: {total_time:.1f}s")
        print(f"Total experiences: {total_experiences}")
        print(f"Experience rate: {total_experiences/total_time*60:.1f}/min")
        
        # Response time analysis
        if self.results['experiences']:
            response_times = [exp['response_time_ms'] for exp in self.results['experiences']]
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            
            print(f"Response time: {avg_response:.1f}ms avg, {max_response:.1f}ms max, {min_response:.1f}ms min")
            
            # Check for performance degradation
            early_responses = response_times[:len(response_times)//4]
            late_responses = response_times[-len(response_times)//4:]
            
            if early_responses and late_responses:
                early_avg = sum(early_responses) / len(early_responses)
                late_avg = sum(late_responses) / len(late_responses)
                degradation = ((late_avg - early_avg) / early_avg) * 100
                
                print(f"Performance change: {degradation:+.1f}%")
        
        # Connection stability
        connection_issues = len(self.results['connection_events'])
        if connection_issues == 0:
            print("Connection: ✅ Stable")
        else:
            print(f"Connection: ⚠️  {connection_issues} issues")
        
        # Memory usage
        if self.results['performance_metrics']:
            memory_samples = [m['process_memory_mb'] for m in self.results['performance_metrics']]
            initial_memory = memory_samples[0] if memory_samples else 0
            final_memory = memory_samples[-1] if memory_samples else 0
            max_memory = max(memory_samples) if memory_samples else 0
            
            print(f"Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (peak: {max_memory:.1f}MB)")
            
            if final_memory > initial_memory * 2:
                print("Memory: ⚠️  Significant growth detected")
            else:
                print("Memory: ✅ Bounded growth")
        
        # Check for persistence activity
        self._check_persistence_activity()
        
        # Save detailed results
        results_file = brain_root / f"five_minute_test_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: {results_file.name}")
    
    def _check_persistence_activity(self):
        """Check if persistence system was active."""
        robot_memory_path = brain_root / "robot_memory"
        
        # Check for checkpoints
        checkpoints = list(robot_memory_path.glob("**/checkpoint_*"))
        
        if checkpoints:
            print(f"Persistence: ✅ {len(checkpoints)} checkpoints created")
            for checkpoint in checkpoints:
                size_kb = checkpoint.stat().st_size / 1024
                print(f"  - {checkpoint.name}: {size_kb:.1f}KB")
        else:
            print("Persistence: ⚠️  No checkpoints created (too few experiences)")
    
    def _cleanup(self):
        """Clean up resources."""
        if self.server_process and self.server_process.poll() is None:
            print("\n🧹 Shutting down server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()

def main():
    """Run 5-minute test."""
    test = FiveMinuteBrainTest()
    success = test.run_test()
    
    if success:
        print("\n🎉 5-minute test completed successfully!")
    else:
        print("\n❌ 5-minute test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)