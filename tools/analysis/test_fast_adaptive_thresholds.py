#!/usr/bin/env python3
"""
Fast Test for Adaptive Constraint Thresholds

Tests adaptive threshold functionality by manipulating the internal state
for faster validation without waiting 30+ seconds.
"""

import sys
import os
import time
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.adaptive_constraint_thresholds import create_adaptive_thresholds


def test_high_frequency_adaptation():
    """Test adaptation when constraint frequency is HIGH (above target)."""
    print("🔥 Testing High Frequency Adaptation")
    print("=" * 40)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    
    # Get initial threshold
    initial_threshold = thresholds.get_threshold(ConstraintType.PROCESSING_LOAD)
    target_freq = thresholds._threshold_configs[ConstraintType.PROCESSING_LOAD].target_frequency
    
    print(f"📊 Initial processing load threshold: {initial_threshold:.3f}")
    print(f"   Target frequency: {target_freq:.1f}/s")
    
    # Simulate high constraint frequency by directly adding to history
    # Target is 2.0/s, so let's simulate 4.0/s (double the target)
    current_time = time.time()
    constraint_history = thresholds._constraint_history[ConstraintType.PROCESSING_LOAD]
    
    # Add 120 events over the last 30 seconds (4.0/s rate)
    num_events = 120
    time_span = 30.0
    
    print(f"🚀 Simulating {num_events} events over {time_span}s = {num_events/time_span:.1f}/s")
    
    for i in range(num_events):
        event_time = current_time - (time_span * (num_events - i) / num_events)
        constraint_history.append({
            'timestamp': event_time,
            'intensity': 0.8
        })
    
    # Add some load measurements
    for i in range(10):
        thresholds.update_load_measurement(
            StreamType.SENSORY,
            processing_time=20.0,
            budget_ms=20.0,
            attention_used=0.3,
            energy_used=0.4
        )
    
    # Force adaptation by calling _adapt_thresholds directly
    thresholds._update_system_statistics()
    thresholds._adapt_thresholds()
    
    # Check results
    final_threshold = thresholds.get_threshold(ConstraintType.PROCESSING_LOAD)
    threshold_increased = final_threshold > initial_threshold
    
    stats = thresholds.get_adaptation_stats()
    actual_frequency = stats['constraint_frequencies'].get('processing_load', 0.0)
    
    print(f"📈 Results:")
    print(f"   Initial threshold: {initial_threshold:.3f}")
    print(f"   Final threshold: {final_threshold:.3f}")
    print(f"   Change: {final_threshold - initial_threshold:+.3f}")
    print(f"   Actual frequency: {actual_frequency:.2f}/s")
    print(f"   Threshold increased: {'✅' if threshold_increased else '❌'}")
    print(f"   Expected: Threshold should INCREASE when frequency > target")
    
    return threshold_increased


def test_low_frequency_adaptation():
    """Test adaptation when constraint frequency is LOW (below target)."""
    print("\n❄️ Testing Low Frequency Adaptation")
    print("=" * 38)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    
    # Get initial threshold  
    initial_threshold = thresholds.get_threshold(ConstraintType.PROCESSING_LOAD)
    target_freq = thresholds._threshold_configs[ConstraintType.PROCESSING_LOAD].target_frequency
    
    print(f"📊 Initial processing load threshold: {initial_threshold:.3f}")
    print(f"   Target frequency: {target_freq:.1f}/s")
    
    # Simulate low constraint frequency
    # Target is 2.0/s, so let's simulate 0.5/s (quarter of the target)
    current_time = time.time()
    constraint_history = thresholds._constraint_history[ConstraintType.PROCESSING_LOAD]
    
    # Add only 15 events over the last 30 seconds (0.5/s rate)
    num_events = 15
    time_span = 30.0
    
    print(f"🐌 Simulating {num_events} events over {time_span}s = {num_events/time_span:.1f}/s")
    
    for i in range(num_events):
        event_time = current_time - (time_span * (num_events - i) / num_events)
        constraint_history.append({
            'timestamp': event_time,
            'intensity': 0.8
        })
    
    # Add some load measurements
    for i in range(10):
        thresholds.update_load_measurement(
            StreamType.SENSORY,
            processing_time=10.0,  # Lower processing time
            budget_ms=20.0,
            attention_used=0.5,    # Better resources
            energy_used=0.5
        )
    
    # Force adaptation
    thresholds._update_system_statistics()
    thresholds._adapt_thresholds()
    
    # Check results
    final_threshold = thresholds.get_threshold(ConstraintType.PROCESSING_LOAD)
    threshold_decreased = final_threshold < initial_threshold
    
    stats = thresholds.get_adaptation_stats()
    actual_frequency = stats['constraint_frequencies'].get('processing_load', 0.0)
    
    print(f"📉 Results:")
    print(f"   Initial threshold: {initial_threshold:.3f}")
    print(f"   Final threshold: {final_threshold:.3f}")
    print(f"   Change: {final_threshold - initial_threshold:+.3f}")
    print(f"   Actual frequency: {actual_frequency:.2f}/s")
    print(f"   Threshold decreased: {'✅' if threshold_decreased else '❌'}")
    print(f"   Expected: Threshold should DECREASE when frequency < target")
    
    return threshold_decreased


def test_multiple_constraint_adaptation():
    """Test that different constraint types adapt independently."""
    print("\n🌐 Testing Multiple Constraint Types")
    print("=" * 38)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    
    # Get initial thresholds
    initial_thresholds = {}
    for constraint_type in [ConstraintType.PROCESSING_LOAD, ConstraintType.RESOURCE_SCARCITY, ConstraintType.URGENCY_SIGNAL]:
        initial_thresholds[constraint_type] = thresholds.get_threshold(constraint_type)
    
    print(f"📊 Initial thresholds:")
    for constraint_type, threshold in initial_thresholds.items():
        target = thresholds._threshold_configs[constraint_type].target_frequency
        print(f"   {constraint_type.value}: {threshold:.3f} (target: {target:.1f}/s)")
    
    # Simulate different frequencies for different constraint types
    current_time = time.time()
    
    # High frequency for PROCESSING_LOAD (above target of 2.0/s)
    load_history = thresholds._constraint_history[ConstraintType.PROCESSING_LOAD]
    for i in range(90):  # 3.0/s
        event_time = current_time - (30.0 * (90 - i) / 90)
        load_history.append({'timestamp': event_time, 'intensity': 0.8})
    
    # Low frequency for RESOURCE_SCARCITY (below target of 1.5/s)
    scarcity_history = thresholds._constraint_history[ConstraintType.RESOURCE_SCARCITY]
    for i in range(15):  # 0.5/s
        event_time = current_time - (30.0 * (15 - i) / 15)
        scarcity_history.append({'timestamp': event_time, 'intensity': 0.7})
    
    # Target frequency for URGENCY_SIGNAL (exactly target of 0.5/s)
    urgency_history = thresholds._constraint_history[ConstraintType.URGENCY_SIGNAL]
    for i in range(15):  # 0.5/s (matches target)
        event_time = current_time - (30.0 * (15 - i) / 15)
        urgency_history.append({'timestamp': event_time, 'intensity': 0.9})
    
    # Force adaptation
    thresholds._update_system_statistics()
    thresholds._adapt_thresholds()
    
    # Check results
    print(f"\n📈 Adaptation results:")
    adaptations = []
    
    for constraint_type in initial_thresholds:
        initial = initial_thresholds[constraint_type]
        final = thresholds.get_threshold(constraint_type)
        change = final - initial
        
        stats = thresholds.get_adaptation_stats()
        actual_freq = stats['constraint_frequencies'].get(constraint_type.value, 0.0)
        target_freq = thresholds._threshold_configs[constraint_type].target_frequency
        
        expected_direction = "↑" if actual_freq > target_freq else "↓" if actual_freq < target_freq else "="
        actual_direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "="
        correct_direction = expected_direction == actual_direction
        
        adaptations.append(correct_direction)
        
        print(f"   {constraint_type.value}:")
        print(f"     Frequency: {actual_freq:.2f}/s (target: {target_freq:.2f}/s)")
        print(f"     Threshold: {initial:.3f} → {final:.3f} ({change:+.3f})")
        print(f"     Direction: Expected {expected_direction}, Got {actual_direction} {'✅' if correct_direction else '❌'}")
    
    all_correct = all(adaptations)
    print(f"\n🎯 All constraint types adapted correctly: {'✅' if all_correct else '❌'}")
    
    return all_correct


if __name__ == "__main__":
    print("🚀 Fast Adaptive Constraint Thresholds Test")
    print("=" * 55)
    
    # Run tests
    test1_success = test_high_frequency_adaptation()
    test2_success = test_low_frequency_adaptation()
    test3_success = test_multiple_constraint_adaptation()
    
    # Results
    all_passed = test1_success and test2_success and test3_success
    
    print(f"\n🏁 Test Results:")
    print(f"  High Frequency → Higher Thresholds: {'✅' if test1_success else '❌'}")
    print(f"  Low Frequency → Lower Thresholds: {'✅' if test2_success else '❌'}")
    print(f"  Multiple Types Independent: {'✅' if test3_success else '❌'}")
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Dynamic Constraint Adaptation is working correctly!")
        print("   ✅ Thresholds increase when constraint frequency is too high")
        print("   ✅ Thresholds decrease when constraint frequency is too low")
        print("   ✅ Different constraint types adapt independently")
        print("   ✅ System maintains target constraint frequencies")
    else:
        print("\n⚠️ Some adaptation logic needs review")
    
    exit(0 if all_passed else 1)