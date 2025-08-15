#!/usr/bin/env python3
"""
Cognitive Autopilot Performance Profiling
Measures the actual computational overhead of the cognitive autopilot system.
"""

import sys
import time
sys.path.append('server/src')

from server.src.utils.cognitive_autopilot import CognitiveAutopilot

def profile_autopilot_operations():
    """Profile individual operations in the cognitive autopilot."""
    print("🔍 Profiling Cognitive Autopilot Operations")
    print("=" * 50)
    
    autopilot = CognitiveAutopilot()
    
    # Simulate some history for realistic profiling
    for i in range(10):
        autopilot.confidence_history.append(0.5 + i * 0.05)
        autopilot.prediction_error_history.append(0.3 - i * 0.02)
    
    # Test data
    prediction_confidence = 0.7
    prediction_error = 0.3
    brain_state = {
        'prediction_confidence': prediction_confidence,
        'prediction_error': prediction_error,
        'total_cycles': 100
    }
    
    # Profile full update_cognitive_state call
    iterations = 100
    start_time = time.time()
    
    for i in range(iterations):
        result = autopilot.update_cognitive_state(
            prediction_confidence + (i % 10) * 0.01,  # Vary confidence slightly
            prediction_error + (i % 5) * 0.01,        # Vary error slightly
            brain_state
        )
    
    total_time = time.time() - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    print(f"📊 Full update_cognitive_state() Profile:")
    print(f"   {iterations} iterations: {total_time:.3f}s total")
    print(f"   Average per call: {avg_time_ms:.2f}ms")
    
    # Profile individual components
    print(f"\n🔬 Individual Component Profiling:")
    
    # Profile stability assessment
    start = time.time()
    for i in range(iterations):
        stability = autopilot._assess_stability()
    stability_time = ((time.time() - start) / iterations) * 1000
    print(f"   _assess_stability(): {stability_time:.2f}ms")
    
    # Profile surprise assessment
    start = time.time()
    for i in range(iterations):
        surprise = autopilot._assess_surprise_level(prediction_error)
    surprise_time = ((time.time() - start) / iterations) * 1000
    print(f"   _assess_surprise_level(): {surprise_time:.2f}ms")
    
    # Profile mode determination
    start = time.time()
    for i in range(iterations):
        mode = autopilot._determine_cognitive_mode(prediction_confidence, 0.8, 0.3)
    mode_time = ((time.time() - start) / iterations) * 1000
    print(f"   _determine_cognitive_mode(): {mode_time:.2f}ms")
    
    # Profile system recommendations
    start = time.time()
    for i in range(iterations):
        recs = autopilot._generate_system_recommendations(brain_state)
    recs_time = ((time.time() - start) / iterations) * 1000
    print(f"   _generate_system_recommendations(): {recs_time:.2f}ms")
    
    # Profile performance profile
    start = time.time()
    for i in range(iterations):
        profile = autopilot._get_performance_profile()
    profile_time = ((time.time() - start) / iterations) * 1000
    print(f"   _get_performance_profile(): {profile_time:.2f}ms")
    
    # Sum components vs total time
    component_sum = stability_time + surprise_time + mode_time + recs_time + profile_time
    overhead = avg_time_ms - component_sum
    
    print(f"\n📈 Overhead Analysis:")
    print(f"   Component sum: {component_sum:.2f}ms")
    print(f"   Total measured: {avg_time_ms:.2f}ms")
    print(f"   Overhead (history mgmt, etc): {overhead:.2f}ms")
    
    # Performance assessment
    if avg_time_ms > 10:
        print(f"\n🚨 HIGH OVERHEAD: {avg_time_ms:.1f}ms per cycle is significant!")
    elif avg_time_ms > 5:
        print(f"\n⚠️ MODERATE OVERHEAD: {avg_time_ms:.1f}ms per cycle")
    else:
        print(f"\n✅ LOW OVERHEAD: {avg_time_ms:.1f}ms per cycle is acceptable")
    
    return avg_time_ms

def estimate_behavioral_test_impact():
    """Estimate the impact on behavioral test performance."""
    print(f"\n🧪 Behavioral Test Impact Estimation")
    print("=" * 40)
    
    autopilot_overhead_ms = profile_autopilot_operations()
    
    # Estimate impact on behavioral tests
    cycles_per_test = 100  # Typical behavioral test
    tests_in_suite = 5     # Typical test suite
    
    overhead_per_test = (autopilot_overhead_ms / 1000) * cycles_per_test
    overhead_full_suite = overhead_per_test * tests_in_suite
    
    print(f"📊 Estimated Impact:")
    print(f"   Per cycle: {autopilot_overhead_ms:.1f}ms")
    print(f"   Per test (100 cycles): {overhead_per_test:.1f}s")
    print(f"   Full suite (5 tests): {overhead_full_suite:.1f}s")
    
    if overhead_full_suite > 30:  # More than 30s overhead
        print(f"🎯 RECOMMENDATION: Consider test mode without autopilot")
    elif overhead_full_suite > 10:
        print(f"⚠️ RECOMMENDATION: Autopilot adds noticeable delay")
    else:
        print(f"✅ ACCEPTABLE: Autopilot overhead is manageable")

if __name__ == "__main__":
    estimate_behavioral_test_impact()