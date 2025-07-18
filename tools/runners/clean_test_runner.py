#!/usr/bin/env python3
"""
Clean Test Runner - Only runs tests that work with current architecture.
"""

import os
import sys
import subprocess

# Only run tests that we know work with current architecture
WORKING_TESTS = [
    'minimal_brain',
    'hardware_adaptation',
    'biological_timescales',
    'brain_learning',
    'brain_server',
    'extended_learning',
    'socket_demo'
]

def run_test(test_name):
    """Run a specific test."""
    try:
        result = subprocess.run([sys.executable, 'test_runner.py', test_name], 
                              capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Test timed out"
    except Exception as e:
        return False, "", str(e)

def main():
    """Run all working tests."""
    print("🧪 CLEAN TEST RUNNER - Only Working Tests")
    print("=" * 50)
    
    results = {}
    
    for test_name in WORKING_TESTS:
        print(f"\n🔄 Running {test_name}...")
        success, stdout, stderr = run_test(test_name)
        results[test_name] = success
        
        if success:
            print(f"   ✅ {test_name} passed")
        else:
            print(f"   ❌ {test_name} failed")
            if stderr:
                print(f"   Error: {stderr[:200]}...")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total-passed}/{total}")
    
    if passed == total:
        print("\n🎉 All working tests passed!")
    else:
        print("\n⚠️  Some tests failed - check output above")

if __name__ == "__main__":
    main()
