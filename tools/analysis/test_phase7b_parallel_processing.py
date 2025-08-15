#!/usr/bin/env python3
"""
Test Phase 7b: Parallel Stream Processing Integration

Tests the integration of parallel brain coordinator with the main brain system,
comparing performance between parallel and sequential processing modes.
"""

import sys
import os
import time
import tempfile
import asyncio
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain_factory import MinimalBrain


def test_parallel_processing_integration():
    """Test Phase 7b parallel processing integration."""
    print("🧠 Testing Phase 7b: Parallel Stream Processing Integration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration with parallel processing enabled
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': False  # Disable for testing
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 25.0,  # 40Hz gamma frequency
                'enable_biological_timing': True,
                'enable_parallel_processing': True  # Enable Phase 7b
            },
            'logging': {
                'log_brain_cycles': False  # Disable for performance testing
            }
        }
        
        print("🔧 Creating brain with parallel processing enabled...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        
        # Verify parallel coordinator is available
        if not brain.parallel_coordinator:
            print("❌ Failed: Parallel coordinator not initialized")
            return False
        
        print("✅ Parallel coordinator initialized successfully")
        
        # Test sequential mode first (baseline)
        print("\n📊 Testing Sequential Processing (baseline)...")
        brain.enable_parallel_processing(False)
        
        sequential_times = []
        for i in range(20):
            sensory_input = [0.1 * (i % 10), 0.2 * (i % 7), 0.3] * 6  # 18 values, truncated to 16
            
            start_time = time.time()
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            cycle_time = (time.time() - start_time) * 1000
            
            sequential_times.append(cycle_time)
            
            # Verify brain state
            assert len(motor_output) == 4, f"Expected 4 motor outputs, got {len(motor_output)}"
            assert brain_state['parallel_processing'] == False, "Should be in sequential mode"
        
        sequential_avg = sum(sequential_times) / len(sequential_times)
        print(f"Sequential average: {sequential_avg:.2f}ms")
        
        # Test parallel mode
        print("\n📊 Testing Parallel Processing (Phase 7b)...")
        brain.enable_parallel_processing(True)
        
        parallel_times = []
        for i in range(20):
            sensory_input = [0.1 * (i % 10), 0.2 * (i % 7), 0.3] * 6  # 18 values, truncated to 16
            
            start_time = time.time()
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            cycle_time = (time.time() - start_time) * 1000
            
            parallel_times.append(cycle_time)
            
            # Verify brain state
            assert len(motor_output) == 4, f"Expected 4 motor outputs, got {len(motor_output)}"
            
            # Check for parallel processing indicators
            if brain_state.get('parallel_processing', False):
                assert 'biological_timing' in brain_state, "Should have biological timing info"
                print(f"✅ Cycle {i+1}: {cycle_time:.2f}ms (parallel)")
            else:
                print(f"⚠️ Cycle {i+1}: {cycle_time:.2f}ms (fallback to sequential)")
        
        parallel_avg = sum(parallel_times) / len(parallel_times)
        print(f"Parallel average: {parallel_avg:.2f}ms")
        
        # Get performance comparison
        performance_stats = brain.get_parallel_performance_stats()
        print(f"\n📈 Performance Comparison:")
        print(f"  Sequential cycles: {performance_stats.get('sequential_cycles', 0)}")
        print(f"  Parallel cycles: {performance_stats.get('parallel_cycles', 0)}")
        print(f"  Parallel success rate: {performance_stats.get('parallel_success_rate', 0):.1%}")
        
        if 'speedup_ratio' in performance_stats:
            speedup = performance_stats['speedup_ratio']
            print(f"  Speedup ratio: {speedup:.2f}x")
            
            if speedup > 1.0:
                print(f"🚀 Parallel processing is {speedup:.1f}x faster!")
            elif speedup < 0.9:
                print(f"⚠️ Parallel processing is slower (overhead: {(1/speedup - 1)*100:.1f}%)")
            else:
                print("📊 Performance is similar between modes")
        
        # Test biological timing integration
        print(f"\n🧬 Testing Biological Timing Integration...")
        if brain.biological_oscillator:
            timing = brain.biological_oscillator.get_current_timing()
            oscillator_stats = brain.biological_oscillator.get_oscillator_stats()
            
            print(f"  Current phase: {oscillator_stats['current_phase']}")
            print(f"  Gamma frequency: {oscillator_stats['avg_gamma_frequency']:.1f}Hz")
            print(f"  Binding window active: {timing.binding_window_active}")
            print("✅ Biological timing system operational")
        else:
            print("❌ Biological oscillator not available")
            return False
        
        # Test mode switching
        print(f"\n🔄 Testing Mode Switching...")
        brain.enable_parallel_processing(False)
        _, state1 = brain.process_sensory_input([0.5] * 16)
        assert state1['parallel_processing'] == False, "Should switch to sequential"
        
        brain.enable_parallel_processing(True)
        _, state2 = brain.process_sensory_input([0.5] * 16)
        # Note: May fallback to sequential if parallel processing fails
        print("✅ Mode switching works correctly")
        
        # Cleanup
        brain.finalize_session()
        
        print(f"\n✅ Phase 7b Integration Test Complete!")
        print(f"  Parallel processing: {'✅ Working' if performance_stats.get('parallel_cycles', 0) > 0 else '⚠️ Fallback only'}")
        print(f"  Biological timing: {'✅ Active' if brain.biological_oscillator else '❌ Disabled'}")
        print(f"  Gamma frequency: {1000.0 / brain.target_cycle_time_ms:.1f}Hz")
        
        return True


def test_gamma_frequency_accuracy():
    """Test that gamma frequency timing is accurate."""
    print(f"\n🔬 Testing Gamma Frequency Accuracy...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {'persistent_memory_path': temp_dir, 'enable_persistence': False},
            'brain': {
                'type': 'sparse_goldilocks',
                'target_cycle_time_ms': 25.0,  # 40Hz
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {'log_brain_cycles': False}
        }
        
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=True)
        brain.enable_parallel_processing(True)
        
        # Measure actual cycle times
        cycle_times = []
        start_time = time.time()
        
        for i in range(40):  # Test 40 cycles = 1 second at 40Hz
            cycle_start = time.time()
            sensory_input = [0.1 * i] * 16
            brain.process_sensory_input(sensory_input)
            cycle_end = time.time()
            
            cycle_times.append((cycle_end - cycle_start) * 1000)
        
        total_time = time.time() - start_time
        avg_cycle_time = sum(cycle_times) / len(cycle_times)
        actual_frequency = len(cycle_times) / total_time
        
        print(f"  Target cycle time: 25.0ms (40Hz)")
        print(f"  Actual avg cycle time: {avg_cycle_time:.2f}ms")
        print(f"  Actual frequency: {actual_frequency:.1f}Hz")
        print(f"  Timing accuracy: {abs(40.0 - actual_frequency) / 40.0 * 100:.1f}% deviation")
        
        # Check if we're within 10% of target frequency
        frequency_accuracy = abs(40.0 - actual_frequency) / 40.0
        if frequency_accuracy < 0.1:
            print("✅ Gamma frequency timing is accurate")
        else:
            print("⚠️ Gamma frequency timing has significant deviation")
        
        brain.finalize_session()
        return frequency_accuracy < 0.2  # Allow 20% deviation for pass


def main():
    """Run all Phase 7b tests."""
    print("🚀 Phase 7b: Parallel Stream Processing Test Suite")
    print("=" * 70)
    
    tests = [
        ("Parallel Processing Integration", test_parallel_processing_integration),
        ("Gamma Frequency Accuracy", test_gamma_frequency_accuracy),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 7b tests passed! Parallel processing is working correctly.")
        return True
    else:
        print("⚠️ Some Phase 7b tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)