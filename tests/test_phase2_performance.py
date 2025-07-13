#!/usr/bin/env python3
"""
Phase 2 Lite Performance Test

Compares performance between Phase 1 only vs Phase 1 + Phase 2 Lite
to ensure adaptive parameters don't hurt system speed.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.brain import MinimalBrain

def test_brain_performance(enable_phase2: bool, num_cycles: int = 50):
    """Test brain performance with/without Phase 2 adaptations."""
    
    print(f"\n{'='*60}")
    print(f"Testing {'Phase 1 + Phase 2 Lite' if enable_phase2 else 'Phase 1 Only'}")
    print(f"{'='*60}")
    
    # Initialize brain
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_phase2_adaptations=enable_phase2
    )
    
    # Test data
    sensory_input = [1.0, 2.0, 45.0, 0.5, 200.0] + [0.1] * 11  # 16D vector
    action_taken = [0.5, -0.3, 10.0, 0.0]  # 4D vector
    outcome = [1.1, 2.1, 46.0, 0.6, 198.0] + [0.12] * 11  # 16D vector
    
    # Warm up (let system stabilize)
    for i in range(10):
        predicted_action, brain_state = brain.process_sensory_input(sensory_input)
        brain.store_experience(sensory_input, action_taken, outcome, predicted_action)
    
    # Performance test
    start_time = time.time()
    total_cycle_time = 0.0
    
    for i in range(num_cycles):
        cycle_start = time.time()
        
        # Full brain cycle
        predicted_action, brain_state = brain.process_sensory_input(sensory_input)
        brain.store_experience(sensory_input, action_taken, outcome, predicted_action)
        
        cycle_time = time.time() - cycle_start
        total_cycle_time += cycle_time
        
        # Vary input slightly to simulate real environment
        sensory_input[2] += 1.0  # Change heading
        sensory_input[4] -= 0.1  # Change ultrasonic
    
    total_time = time.time() - start_time
    avg_cycle_time = total_cycle_time / num_cycles
    
    # Get final brain stats
    brain_stats = brain.get_brain_stats()
    
    print(f"📊 Performance Results:")
    print(f"   • Total time: {total_time:.3f}s")
    print(f"   • Average cycle time: {avg_cycle_time*1000:.1f}ms")
    print(f"   • Cycles per second: {num_cycles/total_time:.1f}")
    print(f"   • Total experiences: {brain_stats['brain_summary']['total_experiences']}")
    print(f"   • Working memory size: {brain_stats['activation_dynamics'].get('current_working_memory_size', 'N/A')}")
    
    if enable_phase2:
        phase2_stats = brain.get_phase2_adaptation_stats()
        print(f"   • Phase 2 adaptations: {len(phase2_stats)} systems active")
        if 'performance_monitor' in phase2_stats:
            perf_stats = phase2_stats['performance_monitor']
            print(f"   • Performance stable: {perf_stats.get('performance_stable', 'Unknown')}")
    
    return avg_cycle_time, brain_stats

def main():
    """Compare Phase 1 vs Phase 1 + Phase 2 Lite performance."""
    
    print("🔬 Phase 2 Lite Performance Validation")
    print("Testing whether adaptive parameters hurt performance...")
    
    # Test Phase 1 only
    phase1_cycle_time, phase1_stats = test_brain_performance(enable_phase2=False)
    
    # Test Phase 1 + Phase 2 Lite
    phase2_cycle_time, phase2_stats = test_brain_performance(enable_phase2=True)
    
    # Compare results
    print(f"\n{'='*60}")
    print("📈 PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    phase1_ms = phase1_cycle_time * 1000
    phase2_ms = phase2_cycle_time * 1000
    overhead = ((phase2_cycle_time - phase1_cycle_time) / phase1_cycle_time) * 100
    
    print(f"Phase 1 Only:      {phase1_ms:.1f}ms per cycle")
    print(f"Phase 1 + Phase 2: {phase2_ms:.1f}ms per cycle")
    print(f"Overhead:          {overhead:+.1f}%")
    
    # Verdict
    if overhead < 20:  # Less than 20% overhead is acceptable
        print(f"\n✅ VERDICT: Phase 2 Lite is PERFORMANT")
        print(f"   Overhead of {overhead:.1f}% is acceptable for the adaptive benefits")
    else:
        print(f"\n⚠️  VERDICT: Phase 2 Lite has HIGH OVERHEAD")
        print(f"   Overhead of {overhead:.1f}% may impact real-time performance")
    
    # Check if real-time constraints are met
    real_time_limit = 50  # 50ms for 20Hz control
    if phase2_ms < real_time_limit:
        print(f"✅ Real-time constraint met: {phase2_ms:.1f}ms < {real_time_limit}ms")
    else:
        print(f"❌ Real-time constraint violated: {phase2_ms:.1f}ms > {real_time_limit}ms")
    
    return overhead < 20

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)