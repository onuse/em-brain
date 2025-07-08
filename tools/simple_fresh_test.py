#!/usr/bin/env python3
"""
Simple test of fresh brain performance without complex monitoring
"""

import sys
sys.path.append('.')

import time
from simulation.brainstem_sim import GridWorldBrainstem
from core.communication import SensoryPacket
from datetime import datetime


def test_fresh_brain():
    """Test fresh brain performance"""
    print("🆕 SIMPLE FRESH BRAIN TEST")
    print("=" * 40)
    
    # Create fresh brain
    brainstem = GridWorldBrainstem(12, 12, seed=42, use_sockets=False)
    
    # Start session (creates fresh memory)
    session_id = brainstem.brain_client.start_memory_session("Fresh Test")
    print(f"Session: {session_id}")
    
    # Check brain size
    brain_interface = brainstem.brain_client
    if hasattr(brain_interface, 'world_graph'):
        brain_nodes = len(brain_interface.world_graph.nodes)
    else:
        brain_nodes = 0
    
    print(f"Initial brain nodes: {brain_nodes}")
    
    # Test prediction performance
    print("Testing prediction speed...")
    
    def single_prediction():
        # Get sensor readings directly
        sensor_values = brainstem.simulation.get_sensor_readings()
        sensory_packet = SensoryPacket(
            sequence_id=brainstem.sequence_counter,
            sensor_values=sensor_values,
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        mental_context = sensor_values[:8]
        
        pred_start = time.time()
        prediction = brainstem.brain_client.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        pred_time = time.time() - pred_start
        
        if prediction:
            brainstem.simulation.apply_action(prediction.motor_action)
        
        return pred_time
    
    # Warmup
    for _ in range(3):
        single_prediction()
    
    # Measure 20 predictions
    n_tests = 20
    prediction_times = []
    
    start_time = time.time()
    
    for i in range(n_tests):
        pred_time = single_prediction()
        prediction_times.append(pred_time)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{n_tests}")
    
    total_time = time.time() - start_time
    
    # Final brain size
    if hasattr(brain_interface, 'world_graph'):
        final_nodes = len(brain_interface.world_graph.nodes)
    else:
        final_nodes = 0
    
    # Results
    avg_pred_time = sum(prediction_times) / len(prediction_times)
    fps = 1.0 / avg_pred_time
    actual_fps = n_tests / total_time
    
    print(f"\\n📊 RESULTS:")
    print(f"Total test time: {total_time:.3f}s")
    print(f"Average prediction time: {avg_pred_time:.6f}s")
    print(f"Theoretical FPS: {fps:.1f}")
    print(f"Actual FPS: {actual_fps:.1f}")
    print(f"Brain growth: {brain_nodes} → {final_nodes} nodes")
    
    # Compare to demo performance
    demo_fps = 1.3
    demo_frame_time = 1.0 / demo_fps
    
    print(f"\\n🎮 DEMO COMPARISON:")
    print(f"Demo FPS: {demo_fps}")
    print(f"Demo frame time: {demo_frame_time:.3f}s")
    print(f"Brain prediction time: {avg_pred_time:.6f}s")
    print(f"Visualization overhead: {demo_frame_time - avg_pred_time:.3f}s ({(demo_frame_time - avg_pred_time)/demo_frame_time*100:.1f}%)")
    
    if avg_pred_time < 0.1:
        print("✅ Brain is very fast - visualization is the bottleneck")
    elif avg_pred_time < 0.5:
        print("✅ Brain is reasonably fast - some optimization possible")
    else:
        print("⚠️  Brain itself may need optimization")
    
    # End session
    brainstem.brain_client.end_memory_session()
    
    return {
        'avg_prediction_time': avg_pred_time,
        'theoretical_fps': fps,
        'actual_fps': actual_fps,
        'brain_growth': final_nodes - brain_nodes
    }


if __name__ == "__main__":
    test_fresh_brain()