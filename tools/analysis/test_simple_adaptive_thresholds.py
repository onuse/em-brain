#!/usr/bin/env python3
"""
Simple Test for Adaptive Constraint Thresholds

Tests the core adaptive threshold functionality without complex brain integration.
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.adaptive_constraint_thresholds import create_adaptive_thresholds


def test_core_adaptive_functionality():
    """Test core adaptive threshold learning behavior."""
    print("🔧 Testing Core Adaptive Threshold Functionality")
    print("=" * 55)
    
    # Create adaptive thresholds system
    thresholds = create_adaptive_thresholds(quiet_mode=False)
    
    # Get initial thresholds
    initial_thresholds = thresholds.get_all_thresholds()
    print(f"\n📊 Initial thresholds:")
    for constraint_type, threshold in initial_thresholds.items():
        target_freq = thresholds._threshold_configs[constraint_type].target_frequency
        print(f"  {constraint_type.value}: {threshold:.3f} (target freq: {target_freq:.1f}/s)")
    
    print(f"\n🔥 Scenario 1: High constraint frequency (should increase thresholds)")
    
    # Simulate high constraint frequency for PROCESSING_LOAD (above target of 2.0/s)
    constraint_type = ConstraintType.PROCESSING_LOAD
    
    # Generate high frequency events over the full 30-second measurement window
    # Target: 2.0/s, so we need >60 events in 30s. Let's generate 90 events (3.0/s)
    events_to_generate = 90
    time_span = 30.0  # seconds
    delay_between_events = time_span / events_to_generate
    
    print(f"  Generating {events_to_generate} events over {time_span}s = {events_to_generate/time_span:.1f}/s rate")
    
    for i in range(events_to_generate):
        thresholds.log_constraint_event(constraint_type, 0.8)
        if i % 10 == 0:  # Only update load measurements occasionally
            thresholds.update_load_measurement(
                StreamType.SENSORY,
                processing_time=25.0,  # High processing time
                budget_ms=20.0,        # Low budget
                attention_used=0.3,
                energy_used=0.4
            )
        time.sleep(delay_between_events)
    
    # Force adaptation
    time.sleep(6.0)
    thresholds.update_load_measurement(StreamType.SENSORY, 15.0, 20.0, 0.3, 0.4)
    
    # Check adaptation
    scenario1_thresholds = thresholds.get_all_thresholds()
    load_threshold_increased = (scenario1_thresholds[ConstraintType.PROCESSING_LOAD] > 
                               initial_thresholds[ConstraintType.PROCESSING_LOAD])
    
    print(f"  Processing load threshold: {initial_thresholds[ConstraintType.PROCESSING_LOAD]:.3f} → {scenario1_thresholds[ConstraintType.PROCESSING_LOAD]:.3f}")
    print(f"  Threshold increased: {'✅' if load_threshold_increased else '❌'}")
    
    stats = thresholds.get_adaptation_stats()
    load_freq = stats['constraint_frequencies'].get('processing_load', 0.0)
    print(f"  Actual frequency: {load_freq:.2f}/s (target: 2.0/s)")
    
    print(f"\n❄️ Scenario 2: Low constraint frequency (should decrease thresholds)")
    
    # Reset and test opposite scenario - no constraints for a while
    thresholds.reset_adaptations()
    
    # Simulate low load measurements (no constraints triggered)
    for i in range(10):
        thresholds.update_load_measurement(
            StreamType.SENSORY,
            processing_time=5.0,   # Low processing time
            budget_ms=20.0,        # Normal budget
            attention_used=0.5,    # Good resources
            energy_used=0.5
        )
        time.sleep(0.2)  # Slow measurements, no constraint events
    
    # Force adaptation
    time.sleep(6.0)
    thresholds.update_load_measurement(StreamType.SENSORY, 5.0, 20.0, 0.5, 0.5)
    
    # Check adaptation
    scenario2_thresholds = thresholds.get_all_thresholds()
    load_threshold_decreased = (scenario2_thresholds[ConstraintType.PROCESSING_LOAD] < 
                               initial_thresholds[ConstraintType.PROCESSING_LOAD])
    
    print(f"  Processing load threshold: {initial_thresholds[ConstraintType.PROCESSING_LOAD]:.3f} → {scenario2_thresholds[ConstraintType.PROCESSING_LOAD]:.3f}")
    print(f"  Threshold decreased: {'✅' if load_threshold_decreased else '❌'}")
    
    stats = thresholds.get_adaptation_stats()
    load_freq = stats['constraint_frequencies'].get('processing_load', 0.0)
    print(f"  Actual frequency: {load_freq:.2f}/s (target: 2.0/s)")
    
    print(f"\n🎯 Core Functionality Validation:")
    print(f"  High frequency → Higher thresholds: {'✅' if load_threshold_increased else '❌'}")
    print(f"  Low frequency → Lower thresholds: {'✅' if load_threshold_decreased else '❌'}")
    
    core_functionality_works = load_threshold_increased and load_threshold_decreased
    
    return core_functionality_works


def test_multiple_constraint_types():
    """Test adaptation across multiple constraint types."""
    print("\n🌐 Testing Multiple Constraint Types")
    print("=" * 40)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    initial_thresholds = thresholds.get_all_thresholds()
    
    # Trigger different constraint types
    constraint_scenarios = [
        (ConstraintType.PROCESSING_LOAD, 20, 0.9),
        (ConstraintType.RESOURCE_SCARCITY, 10, 0.7),
        (ConstraintType.URGENCY_SIGNAL, 5, 0.8),
    ]
    
    print(f"📊 Triggering multiple constraint types...")
    
    for constraint_type, count, intensity in constraint_scenarios:
        print(f"  Triggering {count} {constraint_type.value} events...")
        for i in range(count):
            thresholds.log_constraint_event(constraint_type, intensity)
            thresholds.update_load_measurement(
                StreamType.SENSORY, 15.0, 20.0, 0.3, 0.3
            )
            time.sleep(0.05)
    
    # Force adaptation
    time.sleep(6.0)
    thresholds.update_load_measurement(StreamType.SENSORY, 10.0, 20.0, 0.4, 0.4)
    
    # Check adaptations
    final_thresholds = thresholds.get_all_thresholds()
    
    adaptations = []
    print(f"\n📈 Threshold adaptations:")
    for constraint_type in ConstraintType:
        initial = initial_thresholds[constraint_type]
        final = final_thresholds[constraint_type]
        change = final - initial
        change_percent = (change / initial) * 100 if initial > 0 else 0
        
        adapted = abs(change_percent) > 1.0  # At least 1% change
        adaptations.append(adapted)
        
        print(f"  {constraint_type.value}: {initial:.3f} → {final:.3f} "
              f"({change_percent:+.1f}%) {'✅' if adapted else '⚪'}")
    
    multiple_types_adapted = sum(adaptations) >= 2
    print(f"\n🎯 Multiple constraint types adapted: {'✅' if multiple_types_adapted else '❌'}")
    
    return multiple_types_adapted


def test_bounds_and_stability():
    """Test that thresholds remain stable and within bounds."""
    print("\n🛡️ Testing Bounds and Stability")
    print("=" * 35)
    
    thresholds = create_adaptive_thresholds(quiet_mode=True)
    
    # Test extreme inputs
    print(f"🔥 Testing extreme constraint frequencies...")
    
    # Extreme high frequency
    for i in range(200):  # Way above any target
        thresholds.log_constraint_event(ConstraintType.PROCESSING_LOAD, 1.0)
        if i % 20 == 0:
            thresholds.update_load_measurement(StreamType.SENSORY, 50.0, 10.0, 0.1, 0.1)
            time.sleep(0.01)
    
    # Force adaptation
    time.sleep(6.0)
    thresholds.update_load_measurement(StreamType.SENSORY, 10.0, 20.0, 0.3, 0.3)
    
    extreme_thresholds = thresholds.get_all_thresholds()
    
    # Check bounds
    configs = thresholds._threshold_configs
    bounds_respected = True
    
    print(f"📊 Bounds check after extreme inputs:")
    for constraint_type, threshold in extreme_thresholds.items():
        config = configs[constraint_type]
        within_bounds = config.min_threshold <= threshold <= config.max_threshold
        bounds_respected = bounds_respected and within_bounds
        
        print(f"  {constraint_type.value}: {threshold:.3f} "
              f"[{config.min_threshold:.3f}, {config.max_threshold:.3f}] "
              f"{'✅' if within_bounds else '❌'}")
    
    print(f"\n🎯 Bounds respected under extreme conditions: {'✅' if bounds_respected else '❌'}")
    
    return bounds_respected


if __name__ == "__main__":
    print("🚀 Simple Adaptive Constraint Thresholds Test")
    print("=" * 60)
    
    # Run tests
    test1_success = test_core_adaptive_functionality()
    test2_success = test_multiple_constraint_types()
    test3_success = test_bounds_and_stability()
    
    # Results
    all_passed = test1_success and test2_success and test3_success
    
    print(f"\n🏁 Test Results:")
    print(f"  Core Functionality: {'✅' if test1_success else '❌'}")
    print(f"  Multiple Types: {'✅' if test2_success else '❌'}")
    print(f"  Bounds & Stability: {'✅' if test3_success else '❌'}")
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Dynamic Constraint Adaptation is working correctly!")
        print("   ✅ Thresholds adapt based on constraint frequency")
        print("   ✅ Multiple constraint types are supported") 
        print("   ✅ Bounds are enforced under all conditions")
        print("   ✅ System is stable and predictable")
    
    exit(0 if all_passed else 1)