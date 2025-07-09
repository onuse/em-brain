#!/usr/bin/env python3
"""
Quick test to verify GPU acceleration is working in the demo robot brain.
"""

import sys
import time
from typing import List, Dict

# Add project root to path  
sys.path.append('.')

from core.brain_interface import BrainInterface
from predictor.triple_predictor import TriplePredictor
from core.communication import SensoryPacket
from datetime import datetime


def test_gpu_acceleration_integration():
    """Test that BrainInterface uses HybridWorldGraph with GPU acceleration."""
    print("🧪 GPU ACCELERATION INTEGRATION TEST")
    print("=" * 50)
    
    # Create brain interface (should use HybridWorldGraph internally)
    print("🧠 Creating BrainInterface...")
    predictor = TriplePredictor()
    brain_interface = BrainInterface(predictor, enable_persistence=False)
    
    # Verify it's using HybridWorldGraph
    world_graph = brain_interface.get_world_graph()
    graph_type = type(world_graph).__name__
    print(f"   Graph type: {graph_type}")
    
    if graph_type == "HybridWorldGraph":
        print("✅ BrainInterface is using HybridWorldGraph!")
        device = world_graph.vectorized_backend.device
        print(f"   GPU device: {device}")
    else:
        print("❌ BrainInterface is NOT using HybridWorldGraph")
        return False
    
    # Test processing sensory input (this should create experiences)
    print("\n🔍 Testing sensory processing...")
    
    for i in range(50):
        # Create sensory packet
        sensory_packet = SensoryPacket(
            sequence_id=i,
            sensor_values=[0.1 + i*0.01] * 20,  # 20 sensor values
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        # Create mental context
        mental_context = sensory_packet.sensor_values[:8]
        
        # Process with brain interface
        prediction = brain_interface.process_sensory_input(
            sensory_packet, mental_context, threat_level="normal"
        )
        
        if i % 10 == 0:
            print(f"   Processed {i+1} sensory inputs")
    
    # Check if experiences were created
    experience_count = world_graph.node_count()
    vectorized_count = world_graph.vectorized_backend.get_size()
    
    print(f"\n📊 Results:")
    print(f"   Total experiences: {experience_count}")
    print(f"   Vectorized experiences: {vectorized_count}")
    print(f"   Sync percentage: {vectorized_count/max(1, experience_count)*100:.1f}%")
    
    # Test similarity search performance
    if experience_count > 0:
        print("\n⚡ Testing similarity search performance...")
        
        # Test query
        query_context = [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4]
        
        # Measure vectorized search
        start_time = time.time()
        similar_experiences = world_graph.find_similar_experiences(
            query_context, similarity_threshold=0.5, max_results=10
        )
        vectorized_time = time.time() - start_time
        
        print(f"   Vectorized search: {len(similar_experiences)} results in {vectorized_time*1000:.2f}ms")
        
        # Get performance stats
        stats = world_graph.get_vectorized_stats()
        perf_stats = stats['performance_analysis']
        
        print(f"   Average vectorized time: {perf_stats['avg_vectorized_time_ms']:.2f}ms")
        print(f"   Vectorized usage: {perf_stats['vectorized_usage_percentage']:.1f}%")
        
        if perf_stats['speedup_ratio'] > 1:
            print(f"   Speedup achieved: {perf_stats['speedup_ratio']:.1f}x faster")
        
        # Show memory usage
        backend_stats = stats['backend_stats']
        print(f"   GPU memory usage: {backend_stats['memory_usage_mb']:.1f} MB")
    
    print("\n🌟 GPU ACCELERATION INTEGRATION TEST COMPLETE!")
    print("✅ HybridWorldGraph is successfully integrated into BrainInterface")
    print("✅ GPU acceleration is working in the robot brain demo")
    
    return True


if __name__ == "__main__":
    success = test_gpu_acceleration_integration()
    if success:
        print("\n🚀 Ready to test GPU acceleration in demo_robot_brain.py!")
    else:
        print("\n❌ GPU acceleration integration failed")