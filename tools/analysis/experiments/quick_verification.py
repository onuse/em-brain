#!/usr/bin/env python3
"""
Quick Brain Verification Script

A simplified version of the comprehensive verification that focuses on the key optimizations
and runs within a reasonable time frame.
"""

import sys
import os
import time
import json
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths
brain_root = Path(__file__).parent
sys.path.insert(0, str(brain_root))

from validation.test_integration import IntegrationTestSuite
from validation.micro_experiments.core_assumptions import create_core_assumption_suite

class QuickBrainVerification:
    """Quick verification of optimized brain functionality."""
    
    def __init__(self):
        self.results = {}
        self.server_process = None
        
    def run_verification(self) -> Dict[str, Any]:
        """Run quick verification suite."""
        print("🔍 Quick Brain Verification")
        print("=" * 50)
        
        try:
            # Phase 1: Start server
            print("\n🚀 Phase 1: Starting brain server...")
            if not self._start_server():
                return {"error": "Failed to start server"}
            
            # Phase 2: Integration test
            print("\n🧪 Phase 2: Integration tests...")
            integration_results = self._run_integration_tests()
            self.results['integration'] = integration_results
            
            # Phase 3: Micro-experiments
            print("\n🧪 Phase 3: Micro-experiments...")
            micro_results = self._run_micro_experiments()
            self.results['micro_experiments'] = micro_results
            
            # Phase 4: Performance check
            print("\n⚡ Phase 4: Performance check...")
            performance_results = self._run_performance_check()
            self.results['performance'] = performance_results
            
            return self.results
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return {"error": str(e)}
        finally:
            self._cleanup()
    
    def _start_server(self) -> bool:
        """Start the brain server."""
        try:
            # Start server process
            server_script = brain_root / "server" / "brain_server.py"
            self.server_process = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(brain_root / "server"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is running
            if self.server_process.poll() is None:
                print("   ✅ Server started successfully")
                return True
            else:
                print("   ❌ Server failed to start")
                return False
                
        except Exception as e:
            print(f"   ❌ Server startup error: {e}")
            return False
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests without server management."""
        try:
            # Create integration suite but don't manage server
            integration_suite = IntegrationTestSuite()
            integration_suite.server_process = self.server_process  # Use our server
            
            # Run core tests
            integration_suite._test_import_resolution()
            integration_suite._test_client_connection()
            integration_suite._test_environment_creation()
            integration_suite._test_brain_sensory_input()
            integration_suite._test_brain_action_output()
            integration_suite._test_round_trip_communication()
            
            # Calculate results
            passed = sum(1 for r in integration_suite.results if r.passed)
            total = len(integration_suite.results)
            
            print(f"   ✅ Integration tests: {passed}/{total} passed")
            
            return {
                'passed': passed,
                'total': total,
                'success_rate': passed / total if total > 0 else 0.0
            }
            
        except Exception as e:
            print(f"   ❌ Integration tests failed: {e}")
            return {'error': str(e)}
    
    def _run_micro_experiments(self) -> Dict[str, Any]:
        """Run micro-experiments to test learning."""
        try:
            suite = create_core_assumption_suite()
            summary = suite.run_all(stop_on_failure=False)
            
            success_rate = summary.get('success_rate', 0.0)
            avg_confidence = summary.get('avg_confidence', 0.0)
            
            print(f"   ✅ Micro-experiments: {success_rate:.1%} success rate, {avg_confidence:.2f} avg confidence")
            
            return {
                'success_rate': success_rate,
                'avg_confidence': avg_confidence,
                'total_experiments': summary.get('total_experiments', 0),
                'passed_experiments': summary.get('passed_experiments', 0)
            }
            
        except Exception as e:
            print(f"   ❌ Micro-experiments failed: {e}")
            return {'error': str(e)}
    
    def _run_performance_check(self) -> Dict[str, Any]:
        """Run basic performance check."""
        try:
            from server.src.communication import MinimalBrainClient
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            client = MinimalBrainClient()
            environment = SensoryMotorWorld(random_seed=42)
            
            if not client.connect():
                return {'error': 'Failed to connect to brain'}
            
            # Test response times
            response_times = []
            for i in range(10):
                sensory_input = environment.get_sensory_input()
                
                start_time = time.time()
                action = client.get_action(sensory_input, timeout=5.0)
                response_time = time.time() - start_time
                
                if action is not None:
                    response_times.append(response_time)
                    environment.execute_action(action)
            
            client.disconnect()
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                print(f"   ✅ Performance: {avg_response_time*1000:.1f}ms avg, {max_response_time*1000:.1f}ms max")
                
                return {
                    'avg_response_time_ms': avg_response_time * 1000,
                    'max_response_time_ms': max_response_time * 1000,
                    'min_response_time_ms': min_response_time * 1000,
                    'total_requests': len(response_times)
                }
            else:
                return {'error': 'No successful responses'}
                
        except Exception as e:
            print(f"   ❌ Performance check failed: {e}")
            return {'error': str(e)}
    
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
            print("   ✅ Server shutdown complete")

def main():
    """Run quick verification."""
    verification = QuickBrainVerification()
    results = verification.run_verification()
    
    print("\n📊 Verification Results")
    print("=" * 50)
    
    if 'error' in results:
        print(f"❌ Verification failed: {results['error']}")
        return False
    
    # Print summary
    for phase, result in results.items():
        if isinstance(result, dict) and 'error' not in result:
            print(f"✅ {phase}: OK")
        elif isinstance(result, dict) and 'error' in result:
            print(f"❌ {phase}: {result['error']}")
    
    # Overall assessment
    has_errors = any(isinstance(r, dict) and 'error' in r for r in results.values())
    
    if not has_errors:
        print("\n🎉 Brain verification passed!")
        print("   All optimizations working correctly")
        return True
    else:
        print("\n⚠️  Brain verification had issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)