#!/usr/bin/env python3
"""
Test Runner

Run specific tests or test suites directly from command line.
Usage: python3 test_runner.py <test_name|all>
"""

import sys
import subprocess
import os

def main():
    """Run specific test by name or all tests."""
    
    # Discover all test files
    test_files = []
    tests_dir = "tests"
    
    if os.path.exists(tests_dir):
        for file in os.listdir(tests_dir):
            if file.startswith('test_') and file.endswith('.py'):
                test_name = file[5:-3]  # Remove 'test_' prefix and '.py' suffix
                test_files.append((test_name, f"python3 tests/{file}"))
    
    # Add organized test categories
    test_categories = {
        'strategy1': 'python3 tests/test_strategy1_pure_streams.py',
        'strategy2': 'python3 tests/test_learnable_similarity_emergence.py',
        'strategy4': 'python3 tests/test_utility_activation.py', 
        'strategy5': 'python3 tests/test_meta_learning.py',
        'brain': 'python3 tests/test_minimal_brain.py',
        'prediction': 'python3 tests/test_prediction.py',
        'client-server': 'python3 tests/test_client_server.py',
    }
    
    # Combine individual tests and categories
    all_tests = {}
    for name, cmd in test_files:
        all_tests[name] = cmd
    
    for name, cmd in test_categories.items():
        all_tests[name] = cmd
    
    if len(sys.argv) != 2:
        print("🧪 Test Runner")
        print("\nUsage: python3 test_runner.py <test_name|all>")
        print("\nAvailable tests:")
        
        print("\n  Strategy Tests:")
        strategy_tests = [k for k in all_tests.keys() if k.startswith('strategy')]
        for test in sorted(strategy_tests):
            print(f"   {test:20} - Test {test.upper()}")
        
        print("\n  System Tests:")
        system_tests = [k for k in all_tests.keys() if not k.startswith('strategy')]
        for test in sorted(system_tests):
            print(f"   {test:20} - Test {test.replace('-', ' ').replace('_', ' ')}")
        
        print(f"\n   {'all':20} - Run all tests")
        print("\nExamples:")
        print("   python3 test_runner.py strategy5")
        print("   python3 test_runner.py brain") 
        print("   python3 test_runner.py all")
        return
    
    test_name = sys.argv[1].lower()
    
    if test_name == 'all':
        print("🧪 Running all tests...")
        failed_tests = []
        passed_tests = []
        
        for name, command in sorted(all_tests.items()):
            print(f"\n🔍 Running {name} test...")
            try:
                result = subprocess.run(command, shell=True, check=True, 
                                      capture_output=True, text=True)
                passed_tests.append(name)
                print(f"✅ {name} test passed")
            except subprocess.CalledProcessError as e:
                failed_tests.append((name, e.returncode))
                print(f"❌ {name} test failed with return code {e.returncode}")
                if e.stderr:
                    print(f"   Error: {e.stderr.strip()}")
        
        print(f"\n📊 Test Summary:")
        print(f"   Passed: {len(passed_tests)}")
        print(f"   Failed: {len(failed_tests)}")
        
        if passed_tests:
            print(f"   ✅ Passed: {', '.join(passed_tests)}")
        if failed_tests:
            print(f"   ❌ Failed: {', '.join([name for name, _ in failed_tests])}")
        
        return 0 if not failed_tests else 1
    
    if test_name not in all_tests:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available: {', '.join(sorted(all_tests.keys()))}, all")
        return 1
    
    command = all_tests[test_name]
    print(f"🧪 Running {test_name} test...")
    print(f"   Command: {command}")
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"✅ {test_name} test completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name} test failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"⏹️  {test_name} test interrupted")
        return 1

if __name__ == "__main__":
    sys.exit(main())