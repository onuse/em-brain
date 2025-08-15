#!/usr/bin/env python3
"""
Field Brain TCP Integration Test

Comprehensive test to validate that the field brain integration works correctly
with the MinimalTCPServer infrastructure. Tests the adapter pattern, interface
compatibility, and field brain functionality through the TCP server interface.
"""

import sys
import os

# Add the server/src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
server_src_dir = os.path.join(script_dir, '..', '..', 'server', 'src')
sys.path.insert(0, server_src_dir)

import time
import traceback
import json
from typing import Dict, List, Any

# Now we can import directly since server/src is in path
try:
    import brain
    import communication.tcp_server
    MinimalBrain = brain.MinimalBrain
    MinimalTCPServer = communication.tcp_server.MinimalTCPServer
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print(f"Server src path: {server_src_dir}")
    print(f"Path exists: {os.path.exists(server_src_dir)}")
    sys.exit(1)


class FieldBrainIntegrationTester:
    """Comprehensive tester for field brain TCP integration."""
    
    def __init__(self, quiet_mode: bool = False):
        """Initialize the integration tester."""
        self.quiet_mode = quiet_mode
        self.test_results = {}
        self.field_brain = None
        self.tcp_server = None
        
        if not quiet_mode:
            print("🧪 Field Brain TCP Integration Tester")
            print("=" * 50)
    
    def log_test(self, test_name: str, result: bool, details: str = ""):
        """Log a test result."""
        self.test_results[test_name] = {
            'passed': result,
            'details': details,
            'timestamp': time.time()
        }
        
        if not self.quiet_mode:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
            if details:
                print(f"    {details}")
    
    def test_field_brain_factory(self) -> bool:
        """Test 1: Import and instantiate the brain factory with brain_type='field'."""
        try:
            # Test field brain instantiation through factory
            config = {
                'brain': {
                    'type': 'field',
                    'sensory_dim': 16,
                    'motor_dim': 4,
                    'field_spatial_resolution': 20,
                    'field_temporal_window': 10.0,
                    'field_evolution_rate': 0.1,
                    'constraint_discovery_rate': 0.15
                }
            }
            
            self.field_brain = MinimalBrain(
                config=config,
                brain_type="field",
                sensory_dim=16,
                motor_dim=4,
                enable_logging=False,
                quiet_mode=True
            )
            
            # Verify it's a field brain
            is_field_brain = (
                hasattr(self.field_brain, 'vector_brain') and
                hasattr(self.field_brain.vector_brain, 'field_brain') and
                self.field_brain.brain_type == "field"
            )
            
            details = f"Brain type: {self.field_brain.brain_type}, Vector brain: {type(self.field_brain.vector_brain).__name__}"
            self.log_test("Field Brain Factory Instantiation", is_field_brain, details)
            return is_field_brain
            
        except Exception as e:
            self.log_test("Field Brain Factory Instantiation", False, f"Exception: {str(e)}")
            return False
    
    def test_process_sensory_input(self) -> bool:
        """Test 2: Test the process_sensory_input method with sample sensory data."""
        if not self.field_brain:
            self.log_test("Process Sensory Input", False, "Field brain not initialized")
            return False
        
        try:
            # Test with various sensory inputs
            test_cases = [
                # Normal case
                {"input": [0.5] * 16, "expected_output_len": 4, "name": "normal_input"},
                # Boundary case - all zeros
                {"input": [0.0] * 16, "expected_output_len": 4, "name": "zero_input"},
                # Boundary case - all ones
                {"input": [1.0] * 16, "expected_output_len": 4, "name": "max_input"},
                # Variable input
                {"input": [i * 0.1 for i in range(16)], "expected_output_len": 4, "name": "variable_input"},
                # Wrong input size (should be handled gracefully)
                {"input": [0.5] * 10, "expected_output_len": 4, "name": "short_input"},
                {"input": [0.5] * 20, "expected_output_len": 4, "name": "long_input"}
            ]
            
            all_tests_passed = True
            results = []
            
            for test_case in test_cases:
                try:
                    action, brain_state = self.field_brain.process_sensory_input(
                        test_case["input"], 
                        action_dimensions=test_case["expected_output_len"]
                    )
                    
                    # Validate output format
                    output_valid = (
                        isinstance(action, list) and
                        len(action) == test_case["expected_output_len"] and
                        all(isinstance(x, (int, float)) for x in action) and
                        isinstance(brain_state, dict)
                    )
                    
                    # Validate brain state contains expected fields
                    required_fields = ['prediction_confidence', 'total_cycles', 'prediction_method']
                    brain_state_valid = all(field in brain_state for field in required_fields)
                    
                    test_passed = output_valid and brain_state_valid
                    
                    results.append({
                        'test': test_case["name"],
                        'passed': test_passed,
                        'action_len': len(action) if action else 0,
                        'brain_state_keys': list(brain_state.keys()) if brain_state else [],
                        'prediction_method': brain_state.get('prediction_method', 'unknown')
                    })
                    
                    if not test_passed:
                        all_tests_passed = False
                        
                except Exception as e:
                    all_tests_passed = False
                    results.append({
                        'test': test_case["name"],
                        'passed': False,
                        'error': str(e)
                    })
            
            details = f"Processed {len(results)} test cases, all passed: {all_tests_passed}"
            self.log_test("Process Sensory Input", all_tests_passed, details)
            
            # Store detailed results for later inspection
            self.test_results["sensory_input_details"] = results
            return all_tests_passed
            
        except Exception as e:
            self.log_test("Process Sensory Input", False, f"Exception: {str(e)}")
            return False
    
    def test_get_brain_stats(self) -> bool:
        """Test 3: Test the get_brain_stats method."""
        if not self.field_brain:
            self.log_test("Get Brain Stats", False, "Field brain not initialized")
            return False
        
        try:
            stats = self.field_brain.get_brain_stats()
            
            # Validate stats structure
            required_top_level = ['brain_summary', 'vector_brain']
            required_summary = ['total_cycles', 'total_experiences', 'total_predictions', 'uptime_seconds']
            
            top_level_valid = all(key in stats for key in required_top_level)
            summary_valid = all(key in stats['brain_summary'] for key in required_summary)
            
            # Check for field-specific stats
            field_specific_valid = (
                'field_brain' in stats.get('vector_brain', {}) or
                any('field' in str(value).lower() for value in stats.values() if isinstance(value, (str, dict)))
            )
            
            all_valid = top_level_valid and summary_valid and field_specific_valid
            
            details = f"Stats keys: {list(stats.keys())}, Field-specific: {field_specific_valid}"
            self.log_test("Get Brain Stats", all_valid, details)
            
            # Store stats for analysis
            self.test_results["brain_stats"] = stats
            return all_valid
            
        except Exception as e:
            self.log_test("Get Brain Stats", False, f"Exception: {str(e)}")
            return False
    
    def test_tcp_server_integration(self) -> bool:
        """Test 4: Verify the field brain adapter works with the TCP server interface."""
        if not self.field_brain:
            self.log_test("TCP Server Integration", False, "Field brain not initialized")
            return False
        
        try:
            # Test TCP server creation with field brain
            self.tcp_server = MinimalTCPServer(self.field_brain, host='127.0.0.1', port=0)  # port 0 = auto-assign
            
            # Verify server can be created
            server_created = (
                self.tcp_server is not None and
                hasattr(self.tcp_server, 'brain') and
                self.tcp_server.brain == self.field_brain
            )
            
            if not server_created:
                self.log_test("TCP Server Integration", False, "Failed to create TCP server")
                return False
            
            # Test protocol compatibility without actually starting server
            # (Starting server would require proper threading and shutdown)
            protocol_compatible = (
                hasattr(self.tcp_server, 'protocol') and
                hasattr(self.tcp_server, 'clients') and
                hasattr(self.tcp_server, 'handle_client_connection')
            )
            
            # Test that brain interface methods are available
            brain_interface_valid = (
                hasattr(self.field_brain, 'process_sensory_input') and
                hasattr(self.field_brain, 'get_brain_stats') and
                hasattr(self.field_brain, 'store_experience')
            )
            
            integration_success = server_created and protocol_compatible and brain_interface_valid
            
            details = f"Server created: {server_created}, Protocol: {protocol_compatible}, Interface: {brain_interface_valid}"
            self.log_test("TCP Server Integration", integration_success, details)
            return integration_success
            
        except Exception as e:
            self.log_test("TCP Server Integration", False, f"Exception: {str(e)}")
            return False
    
    def test_import_compatibility(self) -> bool:
        """Test 5: Check for any import errors or compatibility issues."""
        try:
            import_tests = []
            
            # Test field brain specific imports
            try:
                import brains.field.tcp_adapter
                import_tests.append(("FieldBrainTCPAdapter", True, ""))
            except Exception as e:
                import_tests.append(("FieldBrainTCPAdapter", False, str(e)))
            
            try:
                import brains.field.generic_brain
                import_tests.append(("GenericFieldBrain", True, ""))
            except Exception as e:
                import_tests.append(("GenericFieldBrain", False, str(e)))
            
            try:
                import brains.field.dynamics.constraint_field_dynamics
                import_tests.append(("ConstraintField4D", True, ""))
            except Exception as e:
                import_tests.append(("ConstraintField4D", False, str(e)))
            
            try:
                import communication.tcp_server
                import_tests.append(("MinimalTCPServer", True, ""))
            except Exception as e:
                import_tests.append(("MinimalTCPServer", False, str(e)))
            
            # Check if all imports succeeded
            all_imports_ok = all(test[1] for test in import_tests)
            failed_imports = [test[0] for test in import_tests if not test[1]]
            
            if failed_imports:
                details = f"Failed imports: {', '.join(failed_imports)}"
            else:
                details = f"All {len(import_tests)} imports successful"
            
            self.log_test("Import Compatibility", all_imports_ok, details)
            
            # Store import test details
            self.test_results["import_tests"] = import_tests
            return all_imports_ok
            
        except Exception as e:
            self.log_test("Import Compatibility", False, f"Exception: {str(e)}")
            return False
    
    def test_field_brain_specific_features(self) -> bool:
        """Test 6: Validate field brain specific functionality."""
        if not self.field_brain:
            self.log_test("Field Brain Features", False, "Field brain not initialized")
            return False
        
        try:
            # Test field brain adapter specific methods
            adapter = self.field_brain.vector_brain
            
            features_tested = []
            
            # Test if adapter has field brain specific methods
            if hasattr(adapter, 'get_field_capabilities'):
                try:
                    capabilities = adapter.get_field_capabilities()
                    features_tested.append(("get_field_capabilities", True, f"Keys: {list(capabilities.keys())}"))
                except Exception as e:
                    features_tested.append(("get_field_capabilities", False, str(e)))
            else:
                features_tested.append(("get_field_capabilities", False, "Method not found"))
            
            # Test field brain configuration
            if hasattr(adapter, 'config'):
                config_valid = (
                    hasattr(adapter.config, 'spatial_resolution') and
                    hasattr(adapter.config, 'temporal_window') and
                    hasattr(adapter.config, 'sensory_dimensions')
                )
                features_tested.append(("field_config", config_valid, f"Config type: {type(adapter.config).__name__}"))
            else:
                features_tested.append(("field_config", False, "Config not found"))
            
            # Test field brain presence
            if hasattr(adapter, 'field_brain'):
                field_brain_valid = adapter.field_brain is not None
                features_tested.append(("field_brain_instance", field_brain_valid, f"Type: {type(adapter.field_brain).__name__}"))
            else:
                features_tested.append(("field_brain_instance", False, "Field brain instance not found"))
            
            # Test processing with field-specific outputs
            try:
                _, brain_state = self.field_brain.process_sensory_input([0.5] * 16)
                field_specific_keys = [key for key in brain_state.keys() if 'field' in key.lower()]
                has_field_output = len(field_specific_keys) > 0
                features_tested.append(("field_output_keys", has_field_output, f"Field keys: {field_specific_keys}"))
            except Exception as e:
                features_tested.append(("field_output_keys", False, str(e)))
            
            all_features_ok = all(test[1] for test in features_tested)
            
            details = f"Features tested: {len(features_tested)}, all passed: {all_features_ok}"
            self.log_test("Field Brain Features", all_features_ok, details)
            
            # Store feature test details
            self.test_results["field_features"] = features_tested
            return all_features_ok
            
        except Exception as e:
            self.log_test("Field Brain Features", False, f"Exception: {str(e)}")
            return False
    
    def run_integration_test_suite(self) -> Dict[str, Any]:
        """Run the complete integration test suite."""
        if not self.quiet_mode:
            print("\n🚀 Starting Field Brain TCP Integration Test Suite")
            print("-" * 50)
        
        start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_import_compatibility,
            self.test_field_brain_factory,
            self.test_process_sensory_input,
            self.test_get_brain_stats,
            self.test_tcp_server_integration,
            self.test_field_brain_specific_features
        ]
        
        test_results = []
        for test_func in tests:
            try:
                result = test_func()
                test_results.append(result)
            except Exception as e:
                if not self.quiet_mode:
                    print(f"❌ EXCEPTION in {test_func.__name__}: {str(e)}")
                    print(f"   Traceback: {traceback.format_exc()}")
                test_results.append(False)
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(test_results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        duration = time.time() - start_time
        
        # Compile final results
        final_results = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate,
                'duration_seconds': duration,
                'all_tests_passed': passed_tests == total_tests
            },
            'detailed_results': self.test_results,
            'integration_status': 'PASS' if passed_tests == total_tests else 'FAIL'
        }
        
        if not self.quiet_mode:
            print("\n" + "=" * 50)
            print("🏁 Field Brain TCP Integration Test Results")
            print("=" * 50)
            print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Integration Status: {final_results['integration_status']}")
            
            if passed_tests == total_tests:
                print("\n✅ ALL TESTS PASSED!")
                print("🎉 Field brain integration is working correctly with TCP server")
                print("🚀 Ready for production deployment")
            else:
                print(f"\n❌ {total_tests - passed_tests} TEST(S) FAILED")
                print("🔧 Integration requires attention before deployment")
        
        return final_results
    
    def cleanup(self):
        """Cleanup resources."""
        if self.field_brain:
            try:
                if hasattr(self.field_brain, 'finalize_session'):
                    self.field_brain.finalize_session()
            except:
                pass
        
        if self.tcp_server:
            try:
                if hasattr(self.tcp_server, 'stop'):
                    self.tcp_server.stop()
            except:
                pass


def main():
    """Main test execution."""
    print("🧪 Field Brain TCP Integration Test")
    print("Testing field brain compatibility with MinimalTCPServer")
    print("=" * 60)
    
    # Create and run tester
    tester = FieldBrainIntegrationTester(quiet_mode=False)
    
    try:
        results = tester.run_integration_test_suite()
        
        # Save results to file
        results_file = "/Users/jkarlsson/Documents/Projects/robot-project/brain/logs/field_brain_integration_test_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Return exit code based on results
        return 0 if results['summary']['all_tests_passed'] else 1
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)