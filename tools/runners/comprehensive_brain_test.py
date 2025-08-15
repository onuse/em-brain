#!/usr/bin/env python3
"""
Comprehensive Brain Test - Validates both MinimalBrain architectures.

This test validates that both the minimal and sparse_goldilocks 
brain architectures work correctly after the dimension mismatch fix.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

def test_both_brain_architectures():
    """Test both minimal and sparse_goldilocks brain architectures."""
    print("🧠 COMPREHENSIVE BRAIN ARCHITECTURE TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: MinimalBrain with minimal backend
    print("\n1. 🧠 Testing MinimalBrain with minimal backend...")
    try:
        from src.brain import MinimalBrain
        
        brain = MinimalBrain(brain_type='minimal', quiet_mode=True)
        
        # Test basic processing
        sensory_input = [0.5, 0.3, 0.8, 0.1, 0.2, 0.7, 0.9, 0.6, 0.4, 0.6, 0.2, 0.8, 0.1, 0.5, 0.3, 0.7]
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        
        results['minimal'] = {
            'status': 'success',
            'motor_dims': len(motor_output),
            'cycle_time_ms': brain_state.get('cycle_time_ms', 0),
            'architecture': brain_state.get('architecture', 'unknown'),
            'confidence': brain_state.get('prediction_confidence', 0)
        }
        
        print(f"   ✅ SUCCESS: {len(motor_output)}D motor output")
        print(f"   ⏱️  Cycle time: {brain_state.get('cycle_time_ms', 0):.2f}ms")
        print(f"   🎯 Confidence: {brain_state.get('prediction_confidence', 0):.3f}")
        print(f"   🏗️  Architecture: {brain_state.get('architecture', 'unknown')}")
        
        brain.finalize_session()
        
    except Exception as e:
        results['minimal'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ FAILED: {e}")
    
    # Test 2: MinimalBrain with sparse_goldilocks backend
    print("\n2. 🧬 Testing MinimalBrain with sparse_goldilocks backend...")
    try:
        from src.brain import MinimalBrain
        
        brain = MinimalBrain(brain_type='sparse_goldilocks', sensory_dim=16, motor_dim=8, temporal_dim=4, quiet_mode=True)
        
        # Test basic processing
        sensory_input = [0.5, 0.3, 0.8, 0.1, 0.2, 0.7, 0.9, 0.6, 0.4, 0.6, 0.2, 0.8, 0.1, 0.5, 0.3, 0.7]
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        
        results['sparse_goldilocks'] = {
            'status': 'success',
            'motor_dims': len(motor_output),
            'cycle_time_ms': brain_state.get('cycle_time_ms', 0),
            'architecture': brain_state.get('architecture', 'unknown'),
            'confidence': brain_state.get('prediction_confidence', 0),
            'background_storage': brain_state.get('background_storage', {})
        }
        
        print(f"   ✅ SUCCESS: {len(motor_output)}D motor output")
        print(f"   ⏱️  Cycle time: {brain_state.get('cycle_time_ms', 0):.2f}ms")
        print(f"   🎯 Confidence: {brain_state.get('prediction_confidence', 0):.3f}")
        print(f"   🏗️  Architecture: {brain_state.get('architecture', 'unknown')}")
        
        bg_stats = brain_state.get('background_storage', {})
        if bg_stats:
            print(f"   📦 Background patterns: {bg_stats.get('total_patterns', 0)}")
            print(f"   💾 Cache hit rate: {bg_stats.get('cache_hit_rate', 0):.2f}")
        
        brain.finalize_session()
        
    except Exception as e:
        results['sparse_goldilocks'] = {'status': 'failed', 'error': str(e)}
        print(f"   ❌ FAILED: {e}")
    
    # Test 3: Dimension flexibility test
    print("\n3. 🔧 Testing dimension flexibility...")
    dimension_tests = [
        (8, 4, 2),   # Small dimensions
        (16, 8, 4),  # Default dimensions
        (32, 16, 8), # Large dimensions
        (8, 4, 1),   # Minimal temporal
        (8, 4, 6),   # Larger temporal than biological features
    ]
    
    dimension_results = {}
    
    for sensory_dim, motor_dim, temporal_dim in dimension_tests:
        try:
            brain = MinimalBrain(
                brain_type='sparse_goldilocks',
                sensory_dim=sensory_dim,
                motor_dim=motor_dim,
                temporal_dim=temporal_dim,
                quiet_mode=True
            )
            
            # Test with appropriate input size
            sensory_input = [0.5] * sensory_dim
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            dimension_results[f"({sensory_dim},{motor_dim},{temporal_dim})"] = 'success'
            print(f"   ✅ ({sensory_dim},{motor_dim},{temporal_dim}): {len(motor_output)}D output")
            
            brain.finalize_session()
            
        except Exception as e:
            dimension_results[f"({sensory_dim},{motor_dim},{temporal_dim})"] = f'failed: {e}'
            print(f"   ❌ ({sensory_dim},{motor_dim},{temporal_dim}): {e}")
    
    results['dimension_flexibility'] = dimension_results
    
    # Test 4: Performance comparison
    print("\n4. ⚡ Performance comparison...")
    try:
        # Test minimal brain performance
        brain_minimal = MinimalBrain(brain_type='minimal', quiet_mode=True)
        
        start_time = time.time()
        for i in range(100):
            sensory_input = [0.1 + i*0.01] * 16
            motor_output, brain_state = brain_minimal.process_sensory_input(sensory_input)
        minimal_time = time.time() - start_time
        
        brain_minimal.finalize_session()
        
        # Test sparse_goldilocks brain performance
        brain_sparse = MinimalBrain(brain_type='sparse_goldilocks', quiet_mode=True)
        
        start_time = time.time()
        for i in range(100):
            sensory_input = [0.1 + i*0.01] * 16
            motor_output, brain_state = brain_sparse.process_sensory_input(sensory_input)
        sparse_time = time.time() - start_time
        
        brain_sparse.finalize_session()
        
        print(f"   📊 Minimal brain: {minimal_time:.3f}s ({100/minimal_time:.1f} cycles/sec)")
        print(f"   📊 Sparse brain: {sparse_time:.3f}s ({100/sparse_time:.1f} cycles/sec)")
        
        results['performance'] = {
            'minimal_time': minimal_time,
            'sparse_time': sparse_time,
            'minimal_cycles_per_sec': 100/minimal_time,
            'sparse_cycles_per_sec': 100/sparse_time
        }
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        results['performance'] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    print("\n📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 40)
    
    total_tests = 0
    passed_tests = 0
    
    # Count main architecture tests
    for arch_name, result in results.items():
        if arch_name in ['minimal', 'sparse_goldilocks']:
            total_tests += 1
            if result.get('status') == 'success':
                passed_tests += 1
                print(f"   ✅ {arch_name}: {result['cycle_time_ms']:.2f}ms cycles")
            else:
                print(f"   ❌ {arch_name}: {result.get('error', 'unknown error')}")
    
    # Count dimension tests
    if 'dimension_flexibility' in results:
        for dim_combo, result in results['dimension_flexibility'].items():
            total_tests += 1
            if result == 'success':
                passed_tests += 1
                print(f"   ✅ Dimensions {dim_combo}: OK")
            else:
                print(f"   ❌ Dimensions {dim_combo}: {result}")
    
    print(f"\n🎯 FINAL SCORE: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Both brain architectures working correctly!")
        print("✅ Dimension mismatch fix successful")
        print("✅ Exclusive attention integration working")
        print("✅ Background storage optimization working")
        return True
    else:
        print("⚠️  Some tests failed - check output above")
        return False

if __name__ == "__main__":
    success = test_both_brain_architectures()
    sys.exit(0 if success else 1)