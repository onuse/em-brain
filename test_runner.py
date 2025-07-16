#!/usr/bin/env python3
"""
Root Test Runner

Run specific tests or test suites from anywhere in the project.
Usage: 
  python3 test_runner.py <test_name>  # Run specific test
  python3 test_runner.py all          # Run all tests
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Run specific test by name or all tests."""
    
    if len(sys.argv) < 2:
        print("Usage: python3 test_runner.py <test_name|all>")
        print("\nAvailable tests:")
        list_available_tests()
        return
    
    test_name = sys.argv[1].lower()
    
    # Define test locations
    test_locations = {
        # Integration tests (server/tests/integration/)
        'biological_timescales': 'cd server && python3 tests/integration/test_biological_timescales.py',
        'brain_learning': 'cd server && python3 tests/integration/test_brain_learning.py', 
        'brain_server': 'cd server && python3 tests/integration/test_brain_server.py',
        'extended_learning': 'cd server && python3 tests/integration/test_extended_learning.py',
        'socket_demo': 'cd server && python3 tests/integration/test_socket_demo.py',
        
        # Server unit tests (server/tests/)
        'embodied_free_energy': 'cd server && python3 tests/test_embodied_free_energy.py',
        'minimal_brain': 'cd server && python3 tests/test_minimal_brain.py',
        'client_server': 'cd server && python3 tests/test_client_server.py',
        'hardware_adaptation': 'cd server && python3 tests/test_hardware_adaptation.py',
        'utility_activation': 'cd server && python3 tests/test_utility_activation.py',
        'gpu_activation': 'cd server && python3 tests/test_gpu_activation_dynamics.py',
        'meta_learning': 'cd server && python3 tests/test_meta_learning.py',
        'prediction': 'cd server && python3 tests/test_prediction.py',
        'persistence': 'cd server && python3 tests/test_persistence.py',
        
        # Client tests
        'vocal_system': 'cd client_picarx && python3 tests/test_vocal_system.py',
    }
    
    if test_name == 'all':
        print("🧪 Running all tests...")
        failed_tests = []
        
        for name, command in test_locations.items():
            print(f"\n▶️  Running {name}...")
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ {name} passed")
                else:
                    print(f"❌ {name} failed")
                    print(f"Error: {result.stderr}")
                    failed_tests.append(name)
            except Exception as e:
                print(f"❌ {name} failed to run: {e}")
                failed_tests.append(name)
        
        print(f"\n📊 Test Summary:")
        print(f"   Passed: {len(test_locations) - len(failed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        
        if failed_tests:
            print(f"   Failed tests: {', '.join(failed_tests)}")
            sys.exit(1)
        else:
            print("🎉 All tests passed!")
            
    elif test_name in test_locations:
        print(f"🧪 Running {test_name}...")
        command = test_locations[test_name]
        result = subprocess.run(command, shell=True)
        sys.exit(result.returncode)
    else:
        print(f"❌ Unknown test: {test_name}")
        print("\nAvailable tests:")
        list_available_tests()
        sys.exit(1)

def list_available_tests():
    """List all available tests."""
    print("\nIntegration tests (server/tests/integration/):")
    print("  biological_timescales - Test biological learning timescales")
    print("  brain_learning - Test core brain learning")
    print("  brain_server - Test server connectivity")
    print("  extended_learning - Test extended learning scenarios") 
    print("  socket_demo - Test socket communication")
    
    print("\nServer unit tests:")
    print("  embodied_free_energy - Test embodied Free Energy system")
    print("  minimal_brain - Test core brain functionality")
    print("  client_server - Test client-server communication")
    print("  hardware_adaptation - Test hardware adaptation")
    print("  utility_activation - Test utility-based activation")
    print("  gpu_activation - Test GPU activation dynamics")
    print("  meta_learning - Test meta-learning capabilities")
    print("  prediction - Test prediction engine")
    print("  persistence - Test memory persistence")
    
    print("\nClient tests (client_picarx/tests/):")
    print("  vocal_system - Test vocal system")
    
    print("\nSpecial:")
    print("  all - Run all tests")

if __name__ == "__main__":
    main()