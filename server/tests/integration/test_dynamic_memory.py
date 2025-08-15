#!/usr/bin/env python3
"""
Test script to verify the dynamic memory pressure system is working
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.utils.dynamic_memory_pressure import DynamicMemoryPressure, get_dynamic_memory_pressure

def test_dynamic_adaptation():
    print("🧪 Testing Dynamic Memory Pressure System")
    print("=" * 50)
    
    # Initialize system with more realistic starting point
    dmp = DynamicMemoryPressure(target_cycle_time_ms=100.0)
    # Set a lower initial limit to see adaptation in action
    dmp.current_experience_limit = 50000
    
    print(f"📊 Initial state:")
    stats = dmp.get_performance_stats()
    print(f"   Experience limit: {stats.get('current_experience_limit', 0):,}")
    print()
    
    # Simulate good performance cycles (force timing)
    print("🚀 Simulating good performance (50ms cycles)...")
    import time
    
    for i in range(60):  # Need 50 consecutive good cycles to grow
        dmp.record_cycle_performance(cycle_time_ms=50.0, experience_count=30000 + i*100)
        if i % 20 == 19:  # Show progress every 20 cycles
            stats = dmp.get_performance_stats()
            print(f"   After {i+1} cycles: limit = {stats.get('current_experience_limit', 0):,}, "
                  f"good_cycles = {stats.get('consecutive_good_cycles', 0)}")
        
        # After 50 good cycles, force timing for adjustment
        if i == 49:
            dmp.last_adjustment_time = time.time() - 35  # Force timing constraint
    
    stats_after_good = dmp.get_performance_stats()
    print(f"✅ After good performance: limit = {stats_after_good.get('current_experience_limit', 0):,}")
    print()
    
    # Simulate degrading performance
    print("⚠️  Simulating degrading performance (300ms cycles)...")
    for i in range(5):  # Need 3 consecutive bad cycles to shrink
        dmp.record_cycle_performance(cycle_time_ms=300.0, experience_count=60000 + i*100)
        stats = dmp.get_performance_stats()
        print(f"   After {i+1} bad cycles: limit = {stats.get('current_experience_limit', 0):,}, "
              f"bad_cycles = {stats.get('consecutive_bad_cycles', 0)}")
        
        # After 3 bad cycles, force timing for adjustment
        if i == 2:
            dmp.last_adjustment_time = time.time() - 35  # Force timing constraint
    
    stats_after_bad = dmp.get_performance_stats()
    print(f"⚠️  After degrading performance: limit = {stats_after_bad.get('current_experience_limit', 0):,}")
    print()
    
    # Simulate critical performance
    print("🚨 Simulating critical performance (500ms cycles)...")
    for i in range(4):
        dmp.record_cycle_performance(cycle_time_ms=500.0, experience_count=60000 + i*100)
        stats = dmp.get_performance_stats()
        print(f"   After {i+1} critical cycles: limit = {stats.get('current_experience_limit', 0):,}")
    
    final_stats = dmp.get_performance_stats()
    print(f"🚨 After critical performance: limit = {final_stats.get('current_experience_limit', 0):,}")
    print()
    
    # Test cleanup decisions
    print("🧹 Testing cleanup decisions:")
    test_counts = [50000, 100000, 200000, 500000, 1000000]
    for count in test_counts:
        should_cleanup = dmp.should_trigger_cleanup(count)
        if should_cleanup:
            target = dmp.get_cleanup_target(count)
            print(f"   {count:,} experiences: CLEANUP to {target:,}")
        else:
            print(f"   {count:,} experiences: no cleanup needed")
    
    print()
    print("🎯 Final Performance Stats:")
    final_stats = dmp.get_performance_stats()
    for key, value in final_stats.items():
        if isinstance(value, (int, float)):
            if 'limit' in key or 'count' in key:
                print(f"   {key}: {value:,}")
            else:
                print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

if __name__ == "__main__":
    test_dynamic_adaptation()