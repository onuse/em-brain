#!/usr/bin/env python3
"""
Phase 2 Complete Integration Test - Validates full Phase 2 GPU vectorization.

This test validates the complete Phase 2 implementation with all vectorized
components integrated into the main brain system.
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
from predictor.multi_drive_predictor import MultiDrivePredictor
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


def test_phase2_complete_integration():
    """Test complete Phase 2 integration with performance validation."""
    print("🚀 PHASE 2 COMPLETE INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Default BrainInterface with GPU acceleration
    print("🧠 Testing default BrainInterface with GPU acceleration...")
    start_time = time.time()
    
    # Create brain interface with default GPU acceleration
    brain_interface = BrainInterface(enable_persistence=False, use_gpu=True)
    
    print(f"   Brain interface created in {(time.time() - start_time)*1000:.1f}ms")
    print(f"   Predictor type: {type(brain_interface.predictor).__name__}")
    print(f"   Novelty detector type: {type(brain_interface.novelty_detector).__name__}")
    print(f"   World graph type: {type(brain_interface.world_graph).__name__}")
    
    # Test 2: Performance validation with multiple predictions
    print("\n⚡ Testing prediction performance...")
    
    prediction_times = []
    mental_context = [0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.1]
    
    for i in range(10):
        sensory_packet = create_test_sensory_packet(i)
        
        start_time = time.time()
        prediction = brain_interface.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        prediction_time = (time.time() - start_time) * 1000
        
        prediction_times.append(prediction_time)
        print(f"   Prediction {i+1}: {prediction_time:.1f}ms")
    
    # Calculate performance metrics
    avg_time = np.mean(prediction_times)
    min_time = np.min(prediction_times)
    max_time = np.max(prediction_times)
    fps = 1000 / avg_time
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Average prediction time: {avg_time:.1f}ms")
    print(f"   Min prediction time: {min_time:.1f}ms")
    print(f"   Max prediction time: {max_time:.1f}ms")
    print(f"   Effective FPS: {fps:.1f}")
    
    # Test 3: GPU utilization validation
    print("\n🔧 Testing GPU utilization...")
    
    # Check if vectorized components are being used
    if hasattr(brain_interface.predictor, 'vectorized_engine'):
        vectorized_engine = brain_interface.predictor.vectorized_engine
        if vectorized_engine:
            print(f"   Vectorized engine device: {vectorized_engine.device}")
            print(f"   Vectorized engine initialized: {vectorized_engine is not None}")
    
    # Check world graph vectorization
    if hasattr(brain_interface.world_graph, 'vectorized_backend'):
        backend = brain_interface.world_graph.vectorized_backend
        print(f"   World graph backend device: {backend.device}")
        print(f"   World graph experiences: {backend.get_size()}")
    
    # Test 4: Memory and experience validation
    print("\n🧪 Testing experience creation and memory...")
    
    # Process several sensory inputs to create experiences
    for i in range(5):
        sensory_packet = create_test_sensory_packet(i + 10)
        prediction = brain_interface.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        
        # Vary mental context for next prediction
        mental_context = [min(1.0, max(0.0, x + random.uniform(-0.1, 0.1))) for x in mental_context]
    
    total_experiences = brain_interface.world_graph.node_count()
    print(f"   Total experiences created: {total_experiences}")
    
    # Test 5: Performance target validation
    print("\n🎯 Phase 2 Performance Target Validation:")
    print(f"   Target: 10-20ms prediction time")
    print(f"   Achieved: {avg_time:.1f}ms")
    
    if avg_time <= 20:
        print("   ✅ Phase 2 prediction target MET!")
    else:
        print("   ⚠️  Phase 2 prediction target missed")
    
    print(f"   Target: 50-100 FPS")
    print(f"   Achieved: {fps:.1f} FPS")
    
    if fps >= 50:
        print("   ✅ Phase 2 FPS target MET!")
    else:
        print("   ⚠️  Phase 2 FPS target missed")
    
    # Test 6: Comparison with CPU-only version
    print("\n🔄 Comparing with CPU-only version...")
    
    brain_interface_cpu = BrainInterface(enable_persistence=False, use_gpu=False)
    
    cpu_times = []
    for i in range(5):
        sensory_packet = create_test_sensory_packet(i + 20)
        
        start_time = time.time()
        prediction = brain_interface_cpu.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        cpu_time = (time.time() - start_time) * 1000
        cpu_times.append(cpu_time)
    
    avg_cpu_time = np.mean(cpu_times)
    speedup = avg_cpu_time / avg_time
    
    print(f"   CPU-only average time: {avg_cpu_time:.1f}ms")
    print(f"   GPU-accelerated time: {avg_time:.1f}ms")
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup >= 2.0:
        print("   ✅ Significant GPU acceleration achieved!")
    else:
        print("   ⚠️  GPU acceleration less than expected")
    
    print("\n🎉 PHASE 2 COMPLETE INTEGRATION TEST FINISHED!")
    
    return {
        'avg_prediction_time': avg_time,
        'fps': fps,
        'speedup': speedup,
        'total_experiences': total_experiences,
        'targets_met': avg_time <= 20 and fps >= 50
    }


if __name__ == "__main__":
    test_phase2_complete_integration()