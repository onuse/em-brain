#!/usr/bin/env python3
"""
Phase 2 Optimized Test - Validates optimized Phase 2 performance with aggressive GPU usage.

This test validates the optimized Phase 2 implementation with tuned parameters
for maximum GPU utilization and performance.
"""

import sys
import time
import random
import numpy as np
from typing import Dict, List

# Add project root to path
sys.path.append('.')

from core.brain_interface import BrainInterface
from core.communication import SensoryPacket, PredictionPacket
from predictor.vectorized_triple_predictor import VectorizedTriplePredictor
from predictor.multi_drive_predictor import MultiDrivePredictor
from core.adaptive_execution_engine import AdaptiveExecutionEngine
from datetime import datetime


def create_test_sensory_packet(sequence_id: int) -> SensoryPacket:
    """Create a test sensory packet for validation."""
    # Create realistic sensory data (40x40 grid world format)
    sensor_values = []
    
    # Distance sensors (8 directions)
    for i in range(8):
        sensor_values.append(random.uniform(0.0, 1.0))
    
    # Smell sensors (8 directions)
    for i in range(8):
        sensor_values.append(random.uniform(0.0, 1.0))
    
    # Surface color sensors (RGB)
    for i in range(3):
        sensor_values.append(random.uniform(0.0, 1.0))
    
    # Internal state sensors (energy, health, etc.)
    for i in range(6):
        sensor_values.append(random.uniform(0.1, 0.9))
    
    return SensoryPacket(
        sensor_values=sensor_values,
        actuator_positions=[0.0, 0.0, 0.0],
        timestamp=datetime.now(),
        sequence_id=sequence_id,
        network_latency=0.001
    )


def test_phase2_optimized_performance():
    """Test optimized Phase 2 with aggressive GPU usage."""
    print("🚀 PHASE 2 OPTIMIZED PERFORMANCE TEST")
    print("=" * 60)
    
    # Test 1: Create optimized brain interface with aggressive GPU settings
    print("🧠 Creating optimized BrainInterface...")
    start_time = time.time()
    
    # Create vectorized predictor with optimized settings
    vectorized_predictor = VectorizedTriplePredictor(
        max_depth=10,  # Reduced depth for speed
        traversal_count=2,  # Reduced traversals for speed
        similarity_threshold=0.6,  # Lower threshold for more matches
        use_gpu=True
    )
    
    # Override adaptive execution engine with aggressive GPU settings
    vectorized_predictor.adaptive_engine = AdaptiveExecutionEngine(
        gpu_threshold_nodes=1,   # Use GPU for even 1 node
        cpu_threshold_nodes=0,   # Never prefer CPU
        learning_rate=0.2,
        performance_history_size=50
    )
    
    brain_interface = BrainInterface(
        predictor=vectorized_predictor,
        enable_persistence=False,
        use_gpu=True
    )
    
    print(f"   Brain interface created in {(time.time() - start_time)*1000:.1f}ms")
    print(f"   Predictor type: {type(brain_interface.predictor).__name__}")
    
    # Test 2: Warm up with initial experiences
    print("\n🔥 Warming up with initial experiences...")
    mental_context = [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1]
    
    # Create initial experiences for GPU warmup
    warmup_times = []
    for i in range(20):
        sensory_packet = create_test_sensory_packet(i)
        
        start_time = time.time()
        prediction = brain_interface.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        warmup_time = (time.time() - start_time) * 1000
        warmup_times.append(warmup_time)
        
        # Vary mental context for diversity
        mental_context = [min(1.0, max(0.0, x + random.uniform(-0.1, 0.1))) for x in mental_context]
    
    print(f"   Warmup complete: {len(warmup_times)} experiences")
    print(f"   Average warmup time: {np.mean(warmup_times):.1f}ms")
    
    # Test 3: Measure optimized performance
    print("\n⚡ Testing optimized prediction performance...")
    
    prediction_times = []
    for i in range(20, 40):
        sensory_packet = create_test_sensory_packet(i)
        
        start_time = time.time()
        prediction = brain_interface.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        prediction_time = (time.time() - start_time) * 1000
        
        prediction_times.append(prediction_time)
        print(f"   Prediction {i-19}: {prediction_time:.1f}ms")
        
        # Vary mental context
        mental_context = [min(1.0, max(0.0, x + random.uniform(-0.1, 0.1))) for x in mental_context]
    
    # Calculate optimized performance metrics
    avg_time = np.mean(prediction_times)
    min_time = np.min(prediction_times)
    max_time = np.max(prediction_times)
    fps = 1000 / avg_time
    
    print(f"\n📊 Optimized Performance Metrics:")
    print(f"   Average prediction time: {avg_time:.1f}ms")
    print(f"   Min prediction time: {min_time:.1f}ms")
    print(f"   Max prediction time: {max_time:.1f}ms")
    print(f"   Effective FPS: {fps:.1f}")
    
    # Test 4: Check GPU utilization
    print("\n🔧 Testing GPU utilization...")
    
    if hasattr(brain_interface.predictor, 'adaptive_engine'):
        engine = brain_interface.predictor.adaptive_engine
        stats = engine.get_performance_stats()
        
        gpu_stats = stats.get('gpu', {})
        cpu_stats = stats.get('cpu', {})
        
        print(f"   GPU calls: {gpu_stats.get('total_calls', 0)}")
        print(f"   CPU calls: {cpu_stats.get('total_calls', 0)}")
        
        total_calls = gpu_stats.get('total_calls', 0) + cpu_stats.get('total_calls', 0)
        if total_calls > 0:
            gpu_utilization = gpu_stats.get('total_calls', 0) / total_calls * 100
            print(f"   GPU utilization: {gpu_utilization:.1f}%")
    
    # Test 5: Performance target validation
    print("\n🎯 Phase 2 Optimized Performance Target Validation:")
    print(f"   Target: 10-20ms prediction time")
    print(f"   Achieved: {avg_time:.1f}ms")
    
    if avg_time <= 20:
        print("   ✅ Phase 2 prediction target MET!")
    elif avg_time <= 50:
        print("   🔸 Phase 2 prediction target close (within 50ms)")
    else:
        print("   ⚠️  Phase 2 prediction target missed")
    
    print(f"   Target: 50-100 FPS")
    print(f"   Achieved: {fps:.1f} FPS")
    
    if fps >= 50:
        print("   ✅ Phase 2 FPS target MET!")
    elif fps >= 20:
        print("   🔸 Phase 2 FPS target close (≥20 FPS)")
    else:
        print("   ⚠️  Phase 2 FPS target missed")
    
    # Test 6: Comparison with standard configuration
    print("\n🔄 Comparing with standard configuration...")
    
    # Create standard brain interface
    standard_brain = BrainInterface(enable_persistence=False, use_gpu=True)
    
    standard_times = []
    for i in range(5):
        sensory_packet = create_test_sensory_packet(i + 100)
        
        start_time = time.time()
        prediction = standard_brain.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        standard_time = (time.time() - start_time) * 1000
        standard_times.append(standard_time)
    
    avg_standard_time = np.mean(standard_times)
    improvement = avg_standard_time / avg_time
    
    print(f"   Standard configuration: {avg_standard_time:.1f}ms")
    print(f"   Optimized configuration: {avg_time:.1f}ms")
    print(f"   Improvement: {improvement:.1f}x")
    
    if improvement >= 1.5:
        print("   ✅ Significant optimization achieved!")
    else:
        print("   ⚠️  Limited optimization benefit")
    
    # Test 7: Memory usage validation
    print("\n💾 Memory usage validation...")
    
    total_experiences = brain_interface.world_graph.node_count()
    vectorized_experiences = brain_interface.world_graph.vectorized_backend.get_size()
    
    print(f"   Total experiences: {total_experiences}")
    print(f"   Vectorized experiences: {vectorized_experiences}")
    print(f"   Vectorization ratio: {vectorized_experiences/max(total_experiences, 1)*100:.1f}%")
    
    print("\n🎉 PHASE 2 OPTIMIZED PERFORMANCE TEST COMPLETE!")
    
    return {
        'avg_prediction_time': avg_time,
        'fps': fps,
        'improvement': improvement,
        'total_experiences': total_experiences,
        'targets_met': avg_time <= 20 and fps >= 50,
        'close_to_targets': avg_time <= 50 and fps >= 20
    }


if __name__ == "__main__":
    test_phase2_optimized_performance()