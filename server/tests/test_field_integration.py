#!/usr/bin/env python3
"""
Field Brain TCP Integration Test - Simplified Version

Test to validate that the field brain integration works correctly
with the MinimalTCPServer infrastructure.
"""

import sys
import os
import time
import traceback
import json
from typing import Dict, List, Any

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add src directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_field_brain_integration():
    """Test field brain integration with TCP server."""
    print("🧪 Field Brain TCP Integration Test")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Import test
    print("\n1️⃣ Testing imports...")
    try:
        from server.src.brain_factory import MinimalBrain
        from src.communication.tcp_server import MinimalTCPServer
        print("✅ Core imports successful")
        test_results["imports"] = True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        test_results["imports"] = False
        return test_results
    
    # Test 2: Field brain creation
    print("\n2️⃣ Creating field brain...")
    try:
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
        
        field_brain = MinimalBrain(
            config=config,
            brain_type="field",
            sensory_dim=16,
            motor_dim=4,
            enable_logging=False,
            quiet_mode=True
        )
        
        print(f"✅ Field brain created: {field_brain.brain_type}")
        print(f"   Vector brain type: {type(field_brain.vector_brain).__name__}")
        test_results["brain_creation"] = True
        
    except Exception as e:
        print(f"❌ Field brain creation failed: {e}")
        traceback.print_exc()
        test_results["brain_creation"] = False
        return test_results
    
    # Test 3: Sensory processing
    print("\n3️⃣ Testing sensory processing...")
    try:
        # Test various input scenarios
        test_cases = [
            ([0.5] * 16, "normal_input"),
            ([0.0] * 16, "zero_input"),
            ([1.0] * 16, "max_input"),
            ([i * 0.1 for i in range(16)], "variable_input")
        ]
        
        processing_results = []
        for sensory_input, case_name in test_cases:
            action, brain_state = field_brain.process_sensory_input(sensory_input, 4)
            
            # Validate output
            valid_output = (
                isinstance(action, list) and
                len(action) == 4 and
                all(isinstance(x, (int, float)) for x in action) and
                isinstance(brain_state, dict) and
                'prediction_confidence' in brain_state
            )
            
            processing_results.append({
                'case': case_name,
                'valid': valid_output,
                'action_length': len(action) if action else 0,
                'prediction_method': brain_state.get('prediction_method', 'unknown')
            })
        
        all_valid = all(result['valid'] for result in processing_results)
        print(f"✅ Processed {len(test_cases)} cases, all valid: {all_valid}")
        
        if not all_valid:
            for result in processing_results:
                if not result['valid']:
                    print(f"   ❌ Failed case: {result['case']}")
        
        test_results["sensory_processing"] = all_valid
        test_results["processing_details"] = processing_results
        
    except Exception as e:
        print(f"❌ Sensory processing failed: {e}")
        traceback.print_exc()
        test_results["sensory_processing"] = False
    
    # Test 4: Brain statistics
    print("\n4️⃣ Testing brain statistics...")
    try:
        stats = field_brain.get_brain_stats()
        
        required_keys = ['brain_summary', 'vector_brain']
        stats_valid = all(key in stats for key in required_keys)
        
        # Check for field-specific information
        field_info_present = (
            any('field' in str(value).lower() for value in stats.values() if isinstance(value, (str, dict))) or
            'field_brain' in str(stats)
        )
        
        print(f"✅ Stats retrieved: {stats_valid}, Field info: {field_info_present}")
        print(f"   Main keys: {list(stats.keys())}")
        
        test_results["brain_stats"] = stats_valid
        test_results["field_info_present"] = field_info_present
        
    except Exception as e:
        print(f"❌ Brain stats failed: {e}")
        traceback.print_exc()
        test_results["brain_stats"] = False
    
    # Test 5: TCP server compatibility
    print("\n5️⃣ Testing TCP server compatibility...")
    try:
        tcp_server = MinimalTCPServer(field_brain, host='127.0.0.1', port=0)
        
        server_valid = (
            tcp_server is not None and
            hasattr(tcp_server, 'brain') and
            tcp_server.brain == field_brain and
            hasattr(tcp_server, 'protocol') and
            hasattr(tcp_server, 'clients')
        )
        
        print(f"✅ TCP server created: {server_valid}")
        print(f"   Server brain type: {tcp_server.brain.brain_type}")
        
        test_results["tcp_server"] = server_valid
        
    except Exception as e:
        print(f"❌ TCP server creation failed: {e}")
        traceback.print_exc()
        test_results["tcp_server"] = False
    
    # Test 6: Field brain specific features
    print("\n6️⃣ Testing field brain specific features...")
    try:
        adapter = field_brain.vector_brain
        
        features = []
        
        # Test adapter type
        is_adapter = 'Adapter' in type(adapter).__name__
        features.append(("is_field_adapter", is_adapter))
        
        # Test field brain presence
        has_field_brain = hasattr(adapter, 'field_brain')
        features.append(("has_field_brain", has_field_brain))
        
        # Test configuration
        has_config = hasattr(adapter, 'config')
        features.append(("has_config", has_config))
        
        # Test field-specific methods
        has_field_methods = (
            hasattr(adapter, 'get_field_capabilities') or
            hasattr(adapter, 'save_field_state') or
            hasattr(adapter, 'consolidate_field')
        )
        features.append(("has_field_methods", has_field_methods))
        
        all_features_ok = all(feature[1] for feature in features)
        
        print(f"✅ Field features test: {all_features_ok}")
        for feature_name, feature_ok in features:
            status = "✅" if feature_ok else "❌"
            print(f"   {status} {feature_name}")
        
        test_results["field_features"] = all_features_ok
        test_results["feature_details"] = features
        
    except Exception as e:
        print(f"❌ Field features test failed: {e}")
        traceback.print_exc()
        test_results["field_features"] = False
    
    # Cleanup
    try:
        if 'field_brain' in locals():
            field_brain.finalize_session()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    total_tests = len([k for k in test_results.keys() if isinstance(test_results[k], bool)])
    passed_tests = sum([v for k, v in test_results.items() if isinstance(v, bool)])
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Field brain integration is working correctly")
        print("🚀 Ready for TCP server deployment")
        status = "PASS"
    else:
        print(f"❌ {total_tests - passed_tests} TEST(S) FAILED")
        print("🔧 Integration requires attention")
        status = "FAIL"
    
    # Save results
    final_results = {
        'status': status,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        },
        'test_results': test_results,
        'timestamp': time.time()
    }
    
    # Save to logs
    try:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        results_file = os.path.join(logs_dir, 'field_brain_integration_test.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    return final_results


if __name__ == "__main__":
    results = test_field_brain_integration()
    exit_code = 0 if results['status'] == 'PASS' else 1
    exit(exit_code)