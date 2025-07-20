#!/usr/bin/env python3
"""
Basic Field Brain Integration Test - Quick Version

Tests the basic integration without complex processing to identify issues.
"""

import sys
import os
import time
import json

# Add src directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def basic_field_brain_test():
    """Basic test of field brain integration."""
    print("🧪 Basic Field Brain Integration Test")
    print("=" * 40)
    
    results = {}
    
    # Test 1: Imports
    print("1️⃣ Testing imports...")
    try:
        from brain import MinimalBrain
        from communication.tcp_server import MinimalTCPServer
        print("✅ Core imports successful")
        results["imports"] = True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results["imports"] = False
        return results
    
    # Test 2: Field brain instantiation
    print("\n2️⃣ Testing field brain instantiation...")
    try:
        config = {
            'brain': {
                'type': 'field',
                'sensory_dim': 8,  # Smaller dimensions for faster processing
                'motor_dim': 2,
                'field_spatial_resolution': 5,  # Much smaller for testing
                'field_temporal_window': 2.0,
                'field_evolution_rate': 0.05,
                'constraint_discovery_rate': 0.05
            }
        }
        
        field_brain = MinimalBrain(
            config=config,
            brain_type="field",
            sensory_dim=8,
            motor_dim=2,
            enable_logging=False,
            quiet_mode=True
        )
        
        print(f"✅ Field brain created successfully")
        print(f"   Brain type: {field_brain.brain_type}")
        print(f"   Vector brain: {type(field_brain.vector_brain).__name__}")
        results["brain_creation"] = True
        
    except Exception as e:
        print(f"❌ Field brain creation failed: {e}")
        import traceback
        traceback.print_exc()
        results["brain_creation"] = False
        return results
    
    # Test 3: TCP Server compatibility (without processing)
    print("\n3️⃣ Testing TCP server compatibility...")
    try:
        tcp_server = MinimalTCPServer(field_brain, host='127.0.0.1', port=0)
        
        # Check server properties
        server_ok = (
            tcp_server is not None and
            hasattr(tcp_server, 'brain') and
            tcp_server.brain == field_brain and
            hasattr(tcp_server, 'protocol')
        )
        
        print(f"✅ TCP server created: {server_ok}")
        results["tcp_server"] = server_ok
        
    except Exception as e:
        print(f"❌ TCP server creation failed: {e}")
        results["tcp_server"] = False
    
    # Test 4: Interface compatibility (method existence)
    print("\n4️⃣ Testing interface compatibility...")
    try:
        # Check required methods exist
        required_methods = [
            'process_sensory_input',
            'get_brain_stats', 
            'store_experience'
        ]
        
        method_checks = []
        for method in required_methods:
            has_method = hasattr(field_brain, method)
            method_checks.append(has_method)
            status = "✅" if has_method else "❌"
            print(f"   {status} {method}")
        
        all_methods_ok = all(method_checks)
        results["interface_compatibility"] = all_methods_ok
        
    except Exception as e:
        print(f"❌ Interface check failed: {e}")
        results["interface_compatibility"] = False
    
    # Test 5: Basic stats without processing
    print("\n5️⃣ Testing brain stats...")
    try:
        stats = field_brain.get_brain_stats()
        
        # Check basic stats structure
        stats_ok = (
            isinstance(stats, dict) and
            'brain_summary' in stats and
            'vector_brain' in stats
        )
        
        print(f"✅ Stats retrieved: {stats_ok}")
        if stats_ok:
            print(f"   Summary keys: {list(stats['brain_summary'].keys())}")
        
        results["brain_stats"] = stats_ok
        
    except Exception as e:
        print(f"❌ Brain stats failed: {e}")
        results["brain_stats"] = False
    
    # Test 6: Field brain specific features
    print("\n6️⃣ Testing field brain features...")
    try:
        adapter = field_brain.vector_brain
        
        # Check adapter features
        is_adapter = 'Adapter' in type(adapter).__name__
        has_field_brain = hasattr(adapter, 'field_brain')
        has_config = hasattr(adapter, 'config')
        
        features_ok = is_adapter and has_field_brain and has_config
        
        print(f"✅ Field features: {features_ok}")
        print(f"   Is adapter: {is_adapter}")
        print(f"   Has field brain: {has_field_brain}")
        print(f"   Has config: {has_config}")
        
        results["field_features"] = features_ok
        
    except Exception as e:
        print(f"❌ Field features test failed: {e}")
        results["field_features"] = False
    
    # Cleanup
    try:
        field_brain.finalize_session()
    except:
        pass
    
    # Summary
    print("\n" + "=" * 40)
    print("🏁 Basic Test Results")
    print("=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ BASIC INTEGRATION WORKING!")
        print("   Field brain can be instantiated")
        print("   TCP server compatibility confirmed")
        print("   Interface methods available") 
        status = "PASS"
    else:
        failed_tests = [test for test, result in results.items() if not result]
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        status = "FAIL"
    
    # Save results
    final_results = {
        'test_type': 'basic_integration',
        'status': status,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        },
        'test_results': results,
        'timestamp': time.time(),
        'notes': [
            "Basic integration test without complex field processing",
            "Tests instantiation, TCP compatibility, and interface methods",
            "Does not test actual sensory processing (may cause hangs)"
        ]
    }
    
    return final_results


if __name__ == "__main__":
    results = basic_field_brain_test()
    
    # Save results
    try:
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        results_file = os.path.join(logs_dir, 'field_brain_basic_test.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")
    
    exit_code = 0 if results['status'] == 'PASS' else 1
    exit(exit_code)