#!/usr/bin/env python3
"""
Measure actual brain FPS in different configurations
"""

import sys
sys.path.append('.')

import time
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket
from datetime import datetime


def measure_brain_prediction_fps():
    """Measure pure brain prediction FPS without visualization"""
    print("🧠 BRAIN PREDICTION FPS MEASUREMENT")
    print("=" * 50)
    
    # Test 1: Fresh brain performance
    print("\\n📊 Test 1: Fresh Brain (No Persistence)")
    brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
    
    # Don't start memory session to avoid persistence
    print(f"   Brain nodes: {len(brainstem.brain_client.brain_interface.world_graph.nodes)}")
    
    # Measure prediction speed
    def single_prediction():
        state = brainstem.simulation.get_state()
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        mental_context = state['sensors'][:8]
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        if prediction:
            brainstem.simulation.apply_action(prediction.motor_action)
        return prediction is not None
    
    # Warmup
    for _ in range(3):
        single_prediction()
    
    # Measure fresh brain
    n_predictions = 50
    start_time = time.time()
    
    successful_predictions = 0
    for i in range(n_predictions):
        if single_prediction():
            successful_predictions += 1
    
    fresh_time = time.time() - start_time
    fresh_fps = n_predictions / fresh_time
    
    fresh_nodes = len(brainstem.brain_client.brain_interface.world_graph.nodes)
    
    print(f"   Results: {n_predictions} predictions in {fresh_time:.3f}s")
    print(f"   Fresh brain FPS: {fresh_fps:.1f}")
    print(f"   Brain grew to: {fresh_nodes} nodes")
    print(f"   Success rate: {successful_predictions}/{n_predictions}")
    
    # Test 2: Loaded brain performance 
    print("\\n📊 Test 2: Loaded Brain (With Persistence)")
    brainstem2 = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
    
    # Start memory session to load existing brain
    session_id = brainstem2.brain_client.start_memory_session("FPS Test")
    loaded_nodes = len(brainstem2.brain_client.brain_interface.world_graph.nodes)
    print(f"   Loaded brain nodes: {loaded_nodes}")
    
    # Measure loaded brain speed
    def single_prediction_loaded():
        state = brainstem2.simulation.get_state()
        sensory_packet = SensoryPacket(
            sequence_id=brainstem2.sequence_counter,
            sensor_values=state['sensors'],
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        mental_context = state['sensors'][:8]
        prediction = brainstem2.brain_client.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        if prediction:
            brainstem2.simulation.apply_action(prediction.motor_action)
        return prediction is not None
    
    # Warmup
    for _ in range(3):
        single_prediction_loaded()
    
    # Measure loaded brain
    start_time = time.time()
    
    successful_predictions_loaded = 0
    for i in range(n_predictions):
        if single_prediction_loaded():
            successful_predictions_loaded += 1
    
    loaded_time = time.time() - start_time
    loaded_fps = n_predictions / loaded_time
    
    print(f"   Results: {n_predictions} predictions in {loaded_time:.3f}s")
    print(f"   Loaded brain FPS: {loaded_fps:.1f}")
    print(f"   Success rate: {successful_predictions_loaded}/{n_predictions}")
    
    # Analysis
    print(f"\\n📈 PERFORMANCE ANALYSIS")
    print(f"   Fresh brain (0 nodes): {fresh_fps:.1f} FPS")
    print(f"   Loaded brain ({loaded_nodes} nodes): {loaded_fps:.1f} FPS")
    print(f"   Performance impact: {(fresh_fps/loaded_fps):.1f}x slower with {loaded_nodes} nodes")
    
    # Theoretical demo FPS
    print(f"\\n🎮 DEMO IMPLICATIONS")
    print(f"   Pure brain FPS: {loaded_fps:.1f}")
    print(f"   Demo FPS (observed): 1.3")
    print(f"   Visualization overhead: {loaded_fps/1.3:.1f}x")
    print(f"   Brain operations: {(1/loaded_fps):.4f}s per frame")
    print(f"   Demo frame time: {1/1.3:.4f}s per frame")
    print(f"   Visualization time: {(1/1.3) - (1/loaded_fps):.4f}s per frame")
    
    if loaded_fps > 10:
        print(f"   ✅ Brain is fast enough - visualization is the bottleneck")
    else:
        print(f"   ⚠️  Brain itself may be a bottleneck")
    
    # End session
    brainstem2.brain_client.end_memory_session()
    
    return fresh_fps, loaded_fps, loaded_nodes


if __name__ == "__main__":
    fresh_fps, loaded_fps, nodes = measure_brain_prediction_fps()
    
    print(f"\\n🎯 SUMMARY")
    print(f"   Your brain with {nodes} nodes runs at {loaded_fps:.1f} FPS")
    print(f"   Demo visualization adds {(1/1.3) - (1/loaded_fps):.3f}s overhead per frame")
    
    if loaded_fps > 5:
        print(f"   🚀 Recommendation: Optimize visualization, brain is fast enough")
    else:
        print(f"   🧠 Recommendation: Consider brain optimizations")