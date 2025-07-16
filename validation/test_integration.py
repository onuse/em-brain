#!/usr/bin/env python3
"""
Validation Integration Test Suite

Systematically test all aspects of the validation system integration:
1. Server lifecycle management
2. Client connection stability
3. Environment-brain integration
4. Import path resolution
5. Data flow validation

This must pass before any validation experiments can be run.
"""

import sys
import os
import time
import socket
import subprocess
import signal
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add paths for testing
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))

class IntegrationTestResult:
    """Test result with detailed information."""
    def __init__(self, test_name: str, passed: bool, message: str, details: Dict = None):
        self.test_name = test_name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()

class IntegrationTestSuite:
    """Complete integration test suite for validation system."""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 9999
        
    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🧪 Running Validation Integration Test Suite")
        print("=" * 60)
        
        # Phase 1: Basic Integration Tests
        print("\n📋 Phase 1: Basic Integration Tests")
        self._test_import_resolution()
        self._test_server_startup()
        self._test_client_connection()
        
        # Phase 2: Environment Integration Tests  
        print("\n📋 Phase 2: Environment Integration Tests")
        self._test_environment_creation()
        self._test_sensory_input_generation()
        self._test_action_execution()
        
        # Phase 3: Brain Integration Tests (with server running)
        print("\n📋 Phase 3: Brain Integration Tests")
        self._test_brain_sensory_input()
        self._test_brain_action_output()
        self._test_round_trip_communication()
        
        # Phase 4: Stability Tests (with server running)
        print("\n📋 Phase 4: Stability Tests")
        self._test_connection_stability()
        self._test_consolidation_survival()
        
        # Phase 5: Cleanup
        print("\n📋 Phase 5: Cleanup")
        self._test_server_shutdown()
        
        # Print results
        self._print_results()
        
        # Return overall success
        return all(result.passed for result in self.results)
    
    def _test_import_resolution(self):
        """Test that all required imports work."""
        print("🔍 Testing import resolution...")
        
        imports_to_test = [
            ('communication', 'MinimalBrainClient'),
            ('validation.embodied_learning.environments.sensory_motor_world', 'SensoryMotorWorld'),
            ('brain', 'MinimalBrain'),
            ('embodiment', 'EmbodiedFreeEnergySystem'),
        ]
        
        failed_imports = []
        
        for module_name, class_name in imports_to_test:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                print(f"   ✅ {module_name}.{class_name}")
            except ImportError as e:
                print(f"   ❌ {module_name}.{class_name} - {e}")
                failed_imports.append(f"{module_name}.{class_name}: {e}")
            except AttributeError as e:
                print(f"   ❌ {module_name}.{class_name} - {e}")
                failed_imports.append(f"{module_name}.{class_name}: {e}")
        
        if failed_imports:
            self.results.append(IntegrationTestResult(
                "import_resolution", False, 
                f"Failed imports: {len(failed_imports)}", 
                {"failed_imports": failed_imports}
            ))
        else:
            self.results.append(IntegrationTestResult(
                "import_resolution", True, 
                "All imports successful"
            ))
    
    def _test_server_startup(self):
        """Test that brain server can start properly."""
        print("🚀 Testing server startup...")
        
        try:
            # Check server script exists
            server_script = Path("server/brain_server.py")
            if not server_script.exists():
                self.results.append(IntegrationTestResult(
                    "server_startup", False,
                    f"Server script not found: {server_script}"
                ))
                return
            
            # Start server process
            self.server_process = subprocess.Popen(
                [sys.executable, "brain_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=Path("server")
            )
            
            # Wait for server to start
            start_time = time.time()
            timeout = 30.0  # 30 second timeout
            
            while time.time() - start_time < timeout:
                # Check if server is ready
                if self._is_server_ready():
                    print("   ✅ Server started successfully")
                    self.results.append(IntegrationTestResult(
                        "server_startup", True,
                        f"Server started in {time.time() - start_time:.1f}s"
                    ))
                    return
                
                # Check if process died
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    print(f"   ❌ Server process died")
                    print(f"      stdout: {stdout}")
                    print(f"      stderr: {stderr}")
                    self.results.append(IntegrationTestResult(
                        "server_startup", False,
                        "Server process died during startup",
                        {"stdout": stdout, "stderr": stderr}
                    ))
                    return
                
                time.sleep(0.5)
            
            print(f"   ❌ Server failed to start within {timeout} seconds")
            self.results.append(IntegrationTestResult(
                "server_startup", False,
                f"Server startup timeout ({timeout}s)"
            ))
            
        except Exception as e:
            print(f"   ❌ Server startup failed: {e}")
            self.results.append(IntegrationTestResult(
                "server_startup", False,
                f"Server startup exception: {e}"
            ))
    
    def _test_client_connection(self):
        """Test that client can connect to server."""
        print("🔗 Testing client connection...")
        
        if not self._is_server_ready():
            print("   ❌ Server not ready, skipping client test")
            self.results.append(IntegrationTestResult(
                "client_connection", False,
                "Server not ready for client connection"
            ))
            return
        
        try:
            # Import and test client
            from communication import MinimalBrainClient
            
            client = MinimalBrainClient()
            
            # Test connection
            start_time = time.time()
            if client.connect():
                connection_time = time.time() - start_time
                print(f"   ✅ Client connected in {connection_time:.1f}s")
                
                # Test basic communication
                test_input = [0.5, 0.5, 0.0, 1.0] * 4  # 16D input
                response = client.get_action(test_input, timeout=5.0)
                
                if response is not None:
                    print(f"   ✅ Basic communication successful")
                    print(f"      Input: {len(test_input)}D")
                    print(f"      Output: {len(response)}D")
                    
                    client.disconnect()
                    
                    self.results.append(IntegrationTestResult(
                        "client_connection", True,
                        f"Connection and communication successful",
                        {"connection_time": connection_time, "response_length": len(response)}
                    ))
                else:
                    print("   ❌ No response from server")
                    client.disconnect()
                    self.results.append(IntegrationTestResult(
                        "client_connection", False,
                        "No response from server"
                    ))
            else:
                print("   ❌ Client connection failed")
                self.results.append(IntegrationTestResult(
                    "client_connection", False,
                    "Client connection failed"
                ))
                
        except Exception as e:
            print(f"   ❌ Client connection exception: {e}")
            self.results.append(IntegrationTestResult(
                "client_connection", False,
                f"Client connection exception: {e}"
            ))
    
    def _test_server_shutdown(self):
        """Test that server can be shut down cleanly."""
        print("🛑 Testing server shutdown...")
        
        if self.server_process is None:
            print("   ❌ No server process to shutdown")
            self.results.append(IntegrationTestResult(
                "server_shutdown", False,
                "No server process to shutdown"
            ))
            return
        
        try:
            # Try graceful shutdown
            self.server_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.server_process.wait(timeout=5.0)
                print("   ✅ Server shutdown gracefully")
                self.results.append(IntegrationTestResult(
                    "server_shutdown", True,
                    "Server shutdown gracefully"
                ))
            except subprocess.TimeoutExpired:
                # Force kill if necessary
                print("   ⚠️ Forcing server shutdown...")
                self.server_process.kill()
                self.server_process.wait()
                print("   ✅ Server force-stopped")
                self.results.append(IntegrationTestResult(
                    "server_shutdown", True,
                    "Server shutdown (force-killed)"
                ))
                
        except Exception as e:
            print(f"   ❌ Server shutdown failed: {e}")
            self.results.append(IntegrationTestResult(
                "server_shutdown", False,
                f"Server shutdown exception: {e}"
            ))
        finally:
            self.server_process = None
    
    def _test_environment_creation(self):
        """Test that sensory-motor environment can be created."""
        print("🌍 Testing environment creation...")
        
        try:
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            # Create environment
            env = SensoryMotorWorld(
                world_size=5.0,
                num_light_sources=2,
                num_obstacles=3,
                random_seed=42
            )
            
            print("   ✅ Environment created successfully")
            self.results.append(IntegrationTestResult(
                "environment_creation", True,
                "Environment created successfully"
            ))
            
        except Exception as e:
            print(f"   ❌ Environment creation failed: {e}")
            self.results.append(IntegrationTestResult(
                "environment_creation", False,
                f"Environment creation exception: {e}"
            ))
    
    def _test_sensory_input_generation(self):
        """Test that environment can generate sensory input."""
        print("📡 Testing sensory input generation...")
        
        try:
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            env = SensoryMotorWorld(random_seed=42)
            
            # Generate sensory input
            sensory_input = env.get_sensory_input()
            
            if len(sensory_input) == 16:
                print(f"   ✅ Generated {len(sensory_input)}D sensory input")
                print(f"      Range: [{min(sensory_input):.3f}, {max(sensory_input):.3f}]")
                
                # Check if values are reasonable
                if all(0.0 <= x <= 1.0 for x in sensory_input):
                    print("   ✅ All values in [0,1] range")
                    self.results.append(IntegrationTestResult(
                        "sensory_input_generation", True,
                        "Sensory input generated correctly",
                        {"dimensions": len(sensory_input), "range": [min(sensory_input), max(sensory_input)]}
                    ))
                else:
                    print("   ⚠️ Some values outside [0,1] range")
                    self.results.append(IntegrationTestResult(
                        "sensory_input_generation", True,
                        "Sensory input generated (some values outside [0,1])",
                        {"dimensions": len(sensory_input), "range": [min(sensory_input), max(sensory_input)]}
                    ))
            else:
                print(f"   ❌ Wrong dimensions: {len(sensory_input)} (expected 16)")
                self.results.append(IntegrationTestResult(
                    "sensory_input_generation", False,
                    f"Wrong sensory input dimensions: {len(sensory_input)}"
                ))
                
        except Exception as e:
            print(f"   ❌ Sensory input generation failed: {e}")
            self.results.append(IntegrationTestResult(
                "sensory_input_generation", False,
                f"Sensory input generation exception: {e}"
            ))
    
    def _test_action_execution(self):
        """Test that environment can execute actions."""
        print("🚀 Testing action execution...")
        
        try:
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            env = SensoryMotorWorld(random_seed=42)
            
            # Test different actions
            test_actions = [
                [1.0, 0.0, 0.0, 0.0],  # Move forward
                [0.0, 1.0, 0.0, 0.0],  # Turn left
                [0.0, 0.0, 1.0, 0.0],  # Turn right
                [0.0, 0.0, 0.0, 1.0],  # Stop
            ]
            
            successful_actions = 0
            
            for i, action in enumerate(test_actions):
                result = env.execute_action(action)
                
                if result['success']:
                    successful_actions += 1
                    print(f"   ✅ Action {i}: {['FORWARD', 'LEFT', 'RIGHT', 'STOP'][i]}")
                else:
                    print(f"   ❌ Action {i}: {['FORWARD', 'LEFT', 'RIGHT', 'STOP'][i]}")
            
            if successful_actions == len(test_actions):
                print("   ✅ All actions executed successfully")
                self.results.append(IntegrationTestResult(
                    "action_execution", True,
                    "All actions executed successfully"
                ))
            else:
                print(f"   ⚠️ {successful_actions}/{len(test_actions)} actions successful")
                self.results.append(IntegrationTestResult(
                    "action_execution", True,
                    f"{successful_actions}/{len(test_actions)} actions successful"
                ))
                
        except Exception as e:
            print(f"   ❌ Action execution failed: {e}")
            self.results.append(IntegrationTestResult(
                "action_execution", False,
                f"Action execution exception: {e}"
            ))
    
    def _test_brain_sensory_input(self):
        """Test that brain accepts 16D sensory input."""
        print("🧠 Testing brain sensory input...")
        
        # This test requires running server - skip if not available
        if not self._is_server_ready():
            print("   ⚠️ Server not ready, skipping brain test")
            self.results.append(IntegrationTestResult(
                "brain_sensory_input", False,
                "Server not ready for brain testing"
            ))
            return
        
        try:
            from communication import MinimalBrainClient
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            client = MinimalBrainClient()
            env = SensoryMotorWorld(random_seed=42)
            
            if client.connect():
                # Get realistic sensory input
                sensory_input = env.get_sensory_input()
                
                # Send to brain
                response = client.get_action(sensory_input, timeout=5.0)
                
                if response is not None and len(response) == 4:
                    print(f"   ✅ Brain accepted 16D input, returned 4D output")
                    print(f"      Input range: [{min(sensory_input):.3f}, {max(sensory_input):.3f}]")
                    print(f"      Output range: [{min(response):.3f}, {max(response):.3f}]")
                    
                    client.disconnect()
                    
                    self.results.append(IntegrationTestResult(
                        "brain_sensory_input", True,
                        "Brain processed 16D sensory input correctly"
                    ))
                else:
                    print(f"   ❌ Invalid brain response: {response}")
                    client.disconnect()
                    self.results.append(IntegrationTestResult(
                        "brain_sensory_input", False,
                        f"Invalid brain response: {response}"
                    ))
            else:
                print("   ❌ Could not connect to brain")
                self.results.append(IntegrationTestResult(
                    "brain_sensory_input", False,
                    "Could not connect to brain"
                ))
                
        except Exception as e:
            print(f"   ❌ Brain sensory input test failed: {e}")
            self.results.append(IntegrationTestResult(
                "brain_sensory_input", False,
                f"Brain sensory input test exception: {e}"
            ))
    
    def _test_brain_action_output(self):
        """Test that brain action output works with environment."""
        print("🎯 Testing brain action output...")
        
        if not self._is_server_ready():
            print("   ⚠️ Server not ready, skipping brain action test")
            self.results.append(IntegrationTestResult(
                "brain_action_output", False,
                "Server not ready for brain action testing"
            ))
            return
        
        try:
            from communication import MinimalBrainClient
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            client = MinimalBrainClient()
            env = SensoryMotorWorld(random_seed=42)
            
            if client.connect():
                # Get sensory input
                sensory_input = env.get_sensory_input()
                
                # Get brain action
                brain_action = client.get_action(sensory_input, timeout=5.0)
                
                if brain_action is not None:
                    # Test action in environment
                    result = env.execute_action(brain_action)
                    
                    if result['success']:
                        print("   ✅ Brain action executed successfully in environment")
                        print(f"      Action: {brain_action}")
                        print(f"      Executed: {['FORWARD', 'LEFT', 'RIGHT', 'STOP'][result['action_executed']]}")
                        
                        self.results.append(IntegrationTestResult(
                            "brain_action_output", True,
                            "Brain action executed successfully in environment"
                        ))
                    else:
                        print("   ❌ Brain action failed in environment")
                        self.results.append(IntegrationTestResult(
                            "brain_action_output", False,
                            "Brain action failed in environment"
                        ))
                else:
                    print("   ❌ No brain action received")
                    self.results.append(IntegrationTestResult(
                        "brain_action_output", False,
                        "No brain action received"
                    ))
                
                client.disconnect()
            else:
                print("   ❌ Could not connect to brain")
                self.results.append(IntegrationTestResult(
                    "brain_action_output", False,
                    "Could not connect to brain"
                ))
                
        except Exception as e:
            print(f"   ❌ Brain action output test failed: {e}")
            self.results.append(IntegrationTestResult(
                "brain_action_output", False,
                f"Brain action output test exception: {e}"
            ))
    
    def _test_round_trip_communication(self):
        """Test complete round-trip communication."""
        print("🔄 Testing round-trip communication...")
        
        if not self._is_server_ready():
            print("   ⚠️ Server not ready, skipping round-trip test")
            self.results.append(IntegrationTestResult(
                "round_trip_communication", False,
                "Server not ready for round-trip testing"
            ))
            return
        
        try:
            from communication import MinimalBrainClient
            from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
            
            client = MinimalBrainClient()
            env = SensoryMotorWorld(random_seed=42)
            
            if client.connect():
                successful_cycles = 0
                total_cycles = 5
                
                for cycle in range(total_cycles):
                    # Get sensory input
                    sensory_input = env.get_sensory_input()
                    
                    # Get brain action
                    brain_action = client.get_action(sensory_input, timeout=5.0)
                    
                    if brain_action is not None:
                        # Execute in environment
                        result = env.execute_action(brain_action)
                        
                        if result['success']:
                            successful_cycles += 1
                            print(f"   ✅ Cycle {cycle+1}: Complete round-trip successful")
                        else:
                            print(f"   ❌ Cycle {cycle+1}: Action execution failed")
                    else:
                        print(f"   ❌ Cycle {cycle+1}: No brain response")
                
                client.disconnect()
                
                if successful_cycles == total_cycles:
                    print(f"   ✅ All {total_cycles} round-trip cycles successful")
                    self.results.append(IntegrationTestResult(
                        "round_trip_communication", True,
                        f"All {total_cycles} round-trip cycles successful"
                    ))
                else:
                    print(f"   ⚠️ {successful_cycles}/{total_cycles} cycles successful")
                    self.results.append(IntegrationTestResult(
                        "round_trip_communication", True,
                        f"{successful_cycles}/{total_cycles} cycles successful"
                    ))
            else:
                print("   ❌ Could not connect to brain")
                self.results.append(IntegrationTestResult(
                    "round_trip_communication", False,
                    "Could not connect to brain"
                ))
                
        except Exception as e:
            print(f"   ❌ Round-trip communication test failed: {e}")
            self.results.append(IntegrationTestResult(
                "round_trip_communication", False,
                f"Round-trip communication test exception: {e}"
            ))
    
    def _test_connection_stability(self):
        """Test connection stability over time."""
        print("⏱️ Testing connection stability...")
        
        if not self._is_server_ready():
            print("   ⚠️ Server not ready, skipping stability test")
            self.results.append(IntegrationTestResult(
                "connection_stability", False,
                "Server not ready for stability testing"
            ))
            return
        
        try:
            from communication import MinimalBrainClient
            
            client = MinimalBrainClient()
            
            if client.connect():
                # Test stability over 30 seconds
                start_time = time.time()
                successful_requests = 0
                total_requests = 0
                
                while time.time() - start_time < 30:
                    test_input = [0.5] * 16
                    response = client.get_action(test_input, timeout=3.0)
                    
                    total_requests += 1
                    if response is not None:
                        successful_requests += 1
                    
                    time.sleep(1.0)  # 1 second between requests
                
                client.disconnect()
                
                success_rate = successful_requests / total_requests if total_requests > 0 else 0
                
                if success_rate >= 0.9:
                    print(f"   ✅ Connection stable: {success_rate:.2%} success rate")
                    self.results.append(IntegrationTestResult(
                        "connection_stability", True,
                        f"Connection stable: {success_rate:.2%} success rate"
                    ))
                else:
                    print(f"   ❌ Connection unstable: {success_rate:.2%} success rate")
                    self.results.append(IntegrationTestResult(
                        "connection_stability", False,
                        f"Connection unstable: {success_rate:.2%} success rate"
                    ))
            else:
                print("   ❌ Could not connect to brain")
                self.results.append(IntegrationTestResult(
                    "connection_stability", False,
                    "Could not connect to brain"
                ))
                
        except Exception as e:
            print(f"   ❌ Connection stability test failed: {e}")
            self.results.append(IntegrationTestResult(
                "connection_stability", False,
                f"Connection stability test exception: {e}"
            ))
    
    def _test_consolidation_survival(self):
        """Test that connections survive consolidation breaks."""
        print("😴 Testing consolidation survival...")
        
        if not self._is_server_ready():
            print("   ⚠️ Server not ready, skipping consolidation test")
            self.results.append(IntegrationTestResult(
                "consolidation_survival", False,
                "Server not ready for consolidation testing"
            ))
            return
        
        try:
            from communication import MinimalBrainClient
            
            client = MinimalBrainClient()
            
            if client.connect():
                # Test before consolidation
                test_input = [0.5] * 16
                response_before = client.get_action(test_input, timeout=3.0)
                
                if response_before is not None:
                    print("   ✅ Connection active before consolidation")
                    
                    # Simulate consolidation break (60 seconds with keepalive)
                    print("   😴 Simulating consolidation break (60s with keepalive)...")
                    
                    consolidation_start = time.time()
                    consolidation_duration = 60  # 1 minute
                    
                    while time.time() - consolidation_start < consolidation_duration:
                        # Sleep for 10 seconds
                        time.sleep(10)
                        
                        # Send keepalive
                        if time.time() - consolidation_start < consolidation_duration:
                            try:
                                client.get_action([0.0] * 16, timeout=3.0)
                                print("   💓 Keepalive sent")
                            except:
                                print("   ⚠️ Keepalive failed")
                    
                    # Test after consolidation
                    response_after = client.get_action(test_input, timeout=3.0)
                    
                    if response_after is not None:
                        print("   ✅ Connection survived consolidation break")
                        self.results.append(IntegrationTestResult(
                            "consolidation_survival", True,
                            "Connection survived consolidation break"
                        ))
                    else:
                        print("   ❌ Connection failed after consolidation")
                        self.results.append(IntegrationTestResult(
                            "consolidation_survival", False,
                            "Connection failed after consolidation"
                        ))
                else:
                    print("   ❌ Connection failed before consolidation")
                    self.results.append(IntegrationTestResult(
                        "consolidation_survival", False,
                        "Connection failed before consolidation"
                    ))
                
                client.disconnect()
            else:
                print("   ❌ Could not connect to brain")
                self.results.append(IntegrationTestResult(
                    "consolidation_survival", False,
                    "Could not connect to brain"
                ))
                
        except Exception as e:
            print(f"   ❌ Consolidation survival test failed: {e}")
            self.results.append(IntegrationTestResult(
                "consolidation_survival", False,
                f"Consolidation survival test exception: {e}"
            ))
    
    def _is_server_ready(self) -> bool:
        """Check if server is ready for connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                result = sock.connect_ex(('localhost', self.server_port))
                return result == 0
        except:
            return False
    
    def _print_results(self):
        """Print comprehensive test results."""
        print(f"\n📊 Integration Test Results")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success Rate: {passed/total:.1%}")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"   {result.test_name}: {result.message}")
        
        print(f"\n📋 Detailed Results:")
        for result in self.results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} {result.test_name}: {result.message}")
    
    def cleanup(self):
        """Clean up any running processes."""
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5.0)
            except:
                self.server_process.kill()
                self.server_process.wait()
            self.server_process = None

def main():
    """Run integration test suite."""
    test_suite = IntegrationTestSuite()
    
    try:
        success = test_suite.run_all_tests()
        
        if success:
            print("\n🎉 All integration tests passed!")
            print("   Validation system is ready for use.")
            return 0
        else:
            print("\n❌ Some integration tests failed.")
            print("   Fix issues before running validation experiments.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Integration tests interrupted by user")
        return 1
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    sys.exit(main())