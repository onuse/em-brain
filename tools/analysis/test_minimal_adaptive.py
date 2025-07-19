#!/usr/bin/env python3
"""
Minimal Test for Adaptive Constraint Thresholds

Direct unit test of the adaptation logic with known constraint frequencies.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.adaptive_constraint_thresholds import create_adaptive_thresholds


def test_direct_adaptation_logic():
    """Test the adaptation logic directly by manipulating internal state."""
    print("🔧 Testing Direct Adaptation Logic")
    print("=" * 40)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    constraint_type = ConstraintType.PROCESSING_LOAD
    
    # Get initial state
    initial_threshold = thresholds.get_threshold(constraint_type)
    config = thresholds._threshold_configs[constraint_type]
    
    print(f"📊 Initial state:")
    print(f"   Threshold: {initial_threshold:.3f}")
    print(f"   Target frequency: {config.target_frequency:.1f}/s")
    print(f"   Adaptation rate: {config.adaptation_rate:.3f}")
    
    # Test Case 1: High frequency (above target)
    print(f"\n🔥 Test Case 1: High frequency")
    
    # Manually set constraint history to simulate high frequency
    current_time = time.time()
    constraint_history = thresholds._constraint_history[constraint_type]
    constraint_history.clear()
    
    # Add exactly 90 events (3.0/s over 30s window)
    num_events = 90
    print(f"   Generating {num_events} events...")
    for i in range(num_events):
        # Distribute events evenly over 30 seconds
        event_time = current_time - (29.0 * i / (num_events - 1))  # Fix: use (num_events - 1)
        constraint_history.append({
            'timestamp': event_time,
            'intensity': 0.8
        })
    
    print(f"   Added {len(constraint_history)} constraint events")
    
    # Calculate expected frequency
    recent_events = [
        event for event in constraint_history
        if current_time - event['timestamp'] < 30.0
    ]
    expected_frequency = len(recent_events) / 30.0
    print(f"   Expected frequency: {expected_frequency:.2f}/s")
    
    # Run adaptation
    thresholds._update_system_statistics()
    thresholds._adapt_thresholds()
    
    # Check result
    new_threshold = thresholds.get_threshold(constraint_type)
    stats = thresholds.get_adaptation_stats()
    actual_frequency = stats['constraint_frequencies'].get('processing_load', 0.0)
    
    print(f"   Actual frequency calculated: {actual_frequency:.2f}/s")
    print(f"   Threshold changed: {initial_threshold:.3f} → {new_threshold:.3f}")
    print(f"   Change: {new_threshold - initial_threshold:+.3f}")
    
    frequency_above_target = actual_frequency > config.target_frequency
    threshold_increased = new_threshold > initial_threshold
    
    print(f"   Frequency above target: {'✅' if frequency_above_target else '❌'}")
    print(f"   Threshold increased: {'✅' if threshold_increased else '❌'}")
    
    case1_correct = frequency_above_target and threshold_increased
    
    # Test Case 2: Low frequency (below target)
    print(f"\n❄️ Test Case 2: Low frequency")
    
    # Create a fresh thresholds system for low frequency test
    thresholds2 = create_adaptive_thresholds(quiet_mode=True)
    constraint_history = thresholds2._constraint_history[constraint_type]
    
    # Add only 15 events (0.5/s over 30s window)
    for i in range(15):
        event_time = current_time - (29.0 * i / 14)
        constraint_history.append({
            'timestamp': event_time,
            'intensity': 0.8
        })
    
    print(f"   Added {len(constraint_history)} constraint events")
    
    # Calculate expected frequency
    recent_events = [
        event for event in constraint_history
        if current_time - event['timestamp'] < 30.0
    ]
    expected_frequency = len(recent_events) / 30.0
    print(f"   Expected frequency: {expected_frequency:.2f}/s")
    
    # Run adaptation
    initial_threshold = thresholds2.get_threshold(constraint_type)  # Get fresh baseline
    thresholds2._update_system_statistics()
    thresholds2._adapt_thresholds()
    
    # Check result
    new_threshold = thresholds2.get_threshold(constraint_type)
    stats = thresholds2.get_adaptation_stats()
    actual_frequency = stats['constraint_frequencies'].get('processing_load', 0.0)
    
    print(f"   Actual frequency calculated: {actual_frequency:.2f}/s")
    print(f"   Threshold changed: {initial_threshold:.3f} → {new_threshold:.3f}")
    print(f"   Change: {new_threshold - initial_threshold:+.3f}")
    
    frequency_below_target = actual_frequency < config.target_frequency
    threshold_decreased = new_threshold < initial_threshold
    
    print(f"   Frequency below target: {'✅' if frequency_below_target else '❌'}")
    print(f"   Threshold decreased: {'✅' if threshold_decreased else '❌'}")
    
    case2_correct = frequency_below_target and threshold_decreased
    
    # Test Case 3: Target frequency (should stay stable)
    print(f"\n🎯 Test Case 3: Target frequency")
    
    # Create a fresh thresholds system for target frequency test
    thresholds3 = create_adaptive_thresholds(quiet_mode=True)
    constraint_history = thresholds3._constraint_history[constraint_type]
    
    # Add exactly target frequency events (2.0/s * 30s = 60 events)
    for i in range(60):
        event_time = current_time - (29.0 * i / 59)
        constraint_history.append({
            'timestamp': event_time,
            'intensity': 0.8
        })
    
    print(f"   Added {len(constraint_history)} constraint events")
    
    # Calculate expected frequency
    recent_events = [
        event for event in constraint_history
        if current_time - event['timestamp'] < 30.0
    ]
    expected_frequency = len(recent_events) / 30.0
    print(f"   Expected frequency: {expected_frequency:.2f}/s")
    
    # Run adaptation
    initial_threshold = thresholds3.get_threshold(constraint_type)
    thresholds3._update_system_statistics()
    thresholds3._adapt_thresholds()
    
    # Check result
    new_threshold = thresholds3.get_threshold(constraint_type)
    stats = thresholds3.get_adaptation_stats()
    actual_frequency = stats['constraint_frequencies'].get('processing_load', 0.0)
    
    print(f"   Actual frequency calculated: {actual_frequency:.2f}/s")
    print(f"   Threshold changed: {initial_threshold:.3f} → {new_threshold:.3f}")
    print(f"   Change: {new_threshold - initial_threshold:+.3f}")
    
    frequency_at_target = abs(actual_frequency - config.target_frequency) < 0.1
    threshold_stable = abs(new_threshold - initial_threshold) < 0.05  # Small changes ok
    
    print(f"   Frequency at target: {'✅' if frequency_at_target else '❌'}")
    print(f"   Threshold stable: {'✅' if threshold_stable else '❌'}")
    
    case3_correct = frequency_at_target and threshold_stable
    
    print(f"\n🎯 Test Results:")
    print(f"   High frequency → Higher threshold: {'✅' if case1_correct else '❌'}")
    print(f"   Low frequency → Lower threshold: {'✅' if case2_correct else '❌'}")
    print(f"   Target frequency → Stable threshold: {'✅' if case3_correct else '❌'}")
    
    all_correct = case1_correct and case2_correct and case3_correct
    
    return all_correct


if __name__ == "__main__":
    print("🚀 Minimal Adaptive Constraint Thresholds Test")
    print("=" * 60)
    
    success = test_direct_adaptation_logic()
    
    print(f"\n🏁 Overall Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Dynamic Constraint Adaptation works correctly!")
        print("   ✅ High frequency increases thresholds (less sensitive)")
        print("   ✅ Low frequency decreases thresholds (more sensitive)")
        print("   ✅ Target frequency maintains stable thresholds")
    else:
        print("\n⚠️ Adaptation logic needs debugging")
    
    exit(0 if success else 1)