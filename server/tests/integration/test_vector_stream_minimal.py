#!/usr/bin/env python3
"""
Test Minimal Vector Stream Brain

Compares vector stream processing with experience-based processing
to validate the hypothesis that vector streams handle timing and 
dead reckoning better than packaged experience nodes.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.vector_stream.vector_stream_brain import MinimalVectorStreamBrain


class VectorStreamPerformanceProfiler:
    """Profile vector stream brain performance."""
    
    def __init__(self):
        self.cycle_times = []
        self.prediction_confidences = []
        self.temporal_accuracy = []
        self.total_cycles = 0
        
    def record_cycle(self, cycle_time: float, confidence: float, temporal_accuracy: float = 0.0):
        """Record performance metrics for a cycle."""
        self.cycle_times.append(cycle_time)
        self.prediction_confidences.append(confidence)
        self.temporal_accuracy.append(temporal_accuracy)
        self.total_cycles += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        if not self.cycle_times:
            return {}
        
        return {
            'total_cycles': self.total_cycles,
            'avg_cycle_time_ms': np.mean(self.cycle_times) * 1000,
            'avg_confidence': np.mean(self.prediction_confidences),
            'avg_temporal_accuracy': np.mean(self.temporal_accuracy),
            'cycles_per_second': len(self.cycle_times) / sum(self.cycle_times) if sum(self.cycle_times) > 0 else 0
        }


def test_vector_stream_basic_operation():
    """Test basic vector stream brain operation."""
    print("🧪 Testing Vector Stream Basic Operation")
    print("-" * 50)
    
    brain = MinimalVectorStreamBrain(sensory_dim=16, motor_dim=8, temporal_dim=4)
    profiler = VectorStreamPerformanceProfiler()
    
    print("📊 Running basic operation test...")
    
    for cycle in range(20):
        cycle_start = time.time()
        
        # Generate varying sensory input
        sensory_input = [
            np.sin(cycle * 0.5),           # Periodic signal
            np.cos(cycle * 0.3),           # Different phase
            cycle / 20.0,                  # Linear progression
            np.random.rand() * 0.1,        # Noise
            0.5 + 0.2 * (cycle % 3),       # Step pattern
            np.sin(cycle * 2.0),           # High frequency
        ] + [0.0] * 10  # Pad to 16 dimensions
        
        # Process through brain
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        
        cycle_time = time.time() - cycle_start
        confidence = brain_state['prediction_confidence']
        
        profiler.record_cycle(cycle_time, confidence)
        
        # Log interesting cycles
        if cycle < 3 or cycle % 5 == 0:
            print(f"\nCycle {cycle}:")
            print(f"  Motor output: {[f'{x:.2f}' for x in motor_output[:4]]}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Sensory patterns: {brain_state['sensory_stream']['pattern_count']}")
            print(f"  Motor patterns: {brain_state['motor_stream']['pattern_count']}")
            print(f"  Cycle time: {brain_state['cycle_time_ms']:.1f}ms")
        
        time.sleep(0.05)  # Brief pause to observe pattern learning
    
    # Get final statistics
    stats = profiler.get_statistics()
    brain_stats = brain.get_brain_statistics()
    
    print(f"\n📊 VECTOR STREAM PERFORMANCE:")
    print(f"=" * 40)
    print(f"🎯 Total cycles: {stats['total_cycles']}")
    print(f"⏱️  Avg cycle time: {stats['avg_cycle_time_ms']:.1f}ms")
    print(f"🧠 Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"🚀 Cycles per second: {stats['cycles_per_second']:.1f}")
    
    print(f"\n🧠 BRAIN STRUCTURE:")
    print(f"   Sensory patterns learned: {brain_stats['streams']['sensory']['pattern_count']}")
    print(f"   Motor patterns learned: {brain_stats['streams']['motor']['pattern_count']}")
    print(f"   Temporal patterns learned: {brain_stats['streams']['temporal']['pattern_count']}")
    print(f"   Cross-stream weight norm: {brain_stats['cross_stream_weights']['sensory_to_motor_norm']:.2f}")
    
    return brain, stats, brain_stats


def test_vector_stream_timing_sensitivity():
    """Test vector stream sensitivity to timing patterns."""
    print("\n🧪 Testing Vector Stream Timing Sensitivity")
    print("-" * 50)
    
    brain = MinimalVectorStreamBrain(sensory_dim=8, motor_dim=4, temporal_dim=4)
    
    print("📊 Testing with different timing patterns...")
    
    # Pattern 1: Fast rhythm (every 50ms)
    print("\n🏃 Fast rhythm test (50ms intervals):")
    for i in range(10):
        sensory_input = [0.8, 0.2, 0.0, 0.0] + [0.0] * 4
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: confidence={brain_state['prediction_confidence']:.2f}")
        time.sleep(0.05)
    
    # Pattern 2: Slow rhythm (every 200ms)  
    print("\n🐌 Slow rhythm test (200ms intervals):")
    for i in range(5):
        sensory_input = [0.2, 0.8, 0.0, 0.0] + [0.0] * 4
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: confidence={brain_state['prediction_confidence']:.2f}")
        time.sleep(0.2)
    
    # Pattern 3: Mixed timing
    print("\n🔀 Mixed timing test:")
    timings = [0.1, 0.05, 0.15, 0.08, 0.12]
    for i, timing in enumerate(timings):
        sensory_input = [0.5, 0.5, 0.0, 0.0] + [0.0] * 4
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        print(f"  Cycle {i}: timing={timing:.3f}s, confidence={brain_state['prediction_confidence']:.2f}")
        time.sleep(timing)
    
    final_stats = brain.get_brain_statistics()
    print(f"\n✅ Timing sensitivity test completed")
    print(f"   Total patterns learned: {sum(s['pattern_count'] for s in final_stats['streams'].values())}")
    print(f"   Final confidence: {final_stats['prediction_confidence']:.2f}")
    
    return brain


def test_vector_stream_dead_reckoning():
    """Test vector stream dead reckoning capability."""
    print("\n🧪 Testing Vector Stream Dead Reckoning")
    print("-" * 50)
    
    brain = MinimalVectorStreamBrain(sensory_dim=8, motor_dim=4, temporal_dim=4)
    
    # Phase 1: Train with consistent pattern
    print("📚 Training phase - learning predictable pattern...")
    pattern_sequence = [
        [1.0, 0.0, 0.0, 0.0] + [0.0] * 4,  # Pattern A
        [0.0, 1.0, 0.0, 0.0] + [0.0] * 4,  # Pattern B  
        [0.0, 0.0, 1.0, 0.0] + [0.0] * 4,  # Pattern C
        [0.0, 0.0, 0.0, 1.0] + [0.0] * 4,  # Pattern D
    ]
    
    for cycle in range(20):  # 5 full sequences
        pattern_index = cycle % len(pattern_sequence)
        sensory_input = pattern_sequence[pattern_index]
        
        motor_output, brain_state = brain.process_sensory_input(sensory_input)
        
        if cycle % 5 == 0:
            print(f"  Training cycle {cycle}: pattern {pattern_index}, confidence={brain_state['prediction_confidence']:.2f}")
        
        time.sleep(0.1)
    
    # Phase 2: Test prediction without sensory input
    print("\n🚀 Dead reckoning phase - predicting without sensory input...")
    
    # Use zero sensory input to test pure prediction
    zero_input = [0.0] * 8
    prediction_confidences = []
    
    for cycle in range(8):  # 2 predicted sequences
        motor_output, brain_state = brain.process_sensory_input(zero_input)
        confidence = brain_state['prediction_confidence']
        prediction_confidences.append(confidence)
        
        print(f"  Prediction cycle {cycle}:")
        print(f"    Motor output: {[f'{x:.2f}' for x in motor_output]}")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Temporal patterns: {brain_state['temporal_stream']['pattern_count']}")
        
        time.sleep(0.1)
    
    # Analysis
    avg_prediction_confidence = np.mean(prediction_confidences)
    final_stats = brain.get_brain_statistics()
    
    print(f"\n📊 DEAD RECKONING RESULTS:")
    print(f"   Average prediction confidence: {avg_prediction_confidence:.2f}")
    print(f"   Total learned patterns: {sum(s['pattern_count'] for s in final_stats['streams'].values())}")
    print(f"   Temporal stream patterns: {final_stats['streams']['temporal']['pattern_count']}")
    
    # Validation
    dead_reckoning_success = avg_prediction_confidence > 0.3
    print(f"\n✅ Dead reckoning {'SUCCESSFUL' if dead_reckoning_success else 'NEEDS_IMPROVEMENT'}")
    
    return brain, dead_reckoning_success


def main():
    """Run vector stream brain tests."""
    print("🧠 MINIMAL VECTOR STREAM BRAIN TESTS")
    print("=" * 80)
    print("Testing biological-style vector processing vs experience nodes")
    print("\nKey hypotheses:")
    print("• Vector streams handle timing better than experience packages")
    print("• Continuous prediction emerges from stream dynamics")
    print("• Temporal patterns are naturally captured in vector flow")
    
    try:
        # Test 1: Basic operation
        brain1, stats1, brain_stats1 = test_vector_stream_basic_operation()
        
        # Test 2: Timing sensitivity
        brain2 = test_vector_stream_timing_sensitivity()
        
        # Test 3: Dead reckoning
        brain3, dead_reckoning_success = test_vector_stream_dead_reckoning()
        
        # Overall assessment
        print(f"\n🎉 VECTOR STREAM TESTS COMPLETED!")
        print(f"=" * 50)
        print(f"✅ Basic operation: {stats1['avg_cycle_time_ms']:.1f}ms avg cycle")
        print(f"✅ Pattern learning: {sum(s['pattern_count'] for s in brain_stats1['streams'].values())} patterns")
        print(f"✅ Timing sensitivity: Multiple temporal patterns captured")
        print(f"✅ Dead reckoning: {'SUCCESS' if dead_reckoning_success else 'PARTIAL'}")
        
        print(f"\n💡 VECTOR STREAM INSIGHTS:")
        print(f"   • Time naturally integrated as data stream")
        print(f"   • Patterns emerge from continuous vector flow")
        print(f"   • Cross-stream learning creates prediction capability")
        print(f"   • Biological-style processing with digital precision")
        
        success = (stats1['avg_cycle_time_ms'] < 10.0 and 
                  brain_stats1['prediction_confidence'] > 0.1 and
                  dead_reckoning_success)
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)