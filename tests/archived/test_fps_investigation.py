#!/usr/bin/env python3
"""
Investigation script to understand the FPS performance issue.
"""

import time
import sys
sys.path.append('.')

from core.brain_interface import BrainInterface
from predictor.triple_predictor import TriplePredictor
from core.communication import SensoryPacket
from datetime import datetime
from brain_prediction_profiler import get_brain_profiler

def test_fps_performance():
    """Test where the bottleneck is in the brain pipeline."""
    print("🔍 FPS PERFORMANCE INVESTIGATION")
    print("=" * 50)
    
    # Create brain interface
    predictor = TriplePredictor()
    brain_interface = BrainInterface(predictor, enable_persistence=False)
    
    # Check if GPU acceleration is enabled
    world_graph = brain_interface.get_world_graph()
    if hasattr(world_graph, 'vectorized_backend'):
        print(f"✅ GPU Backend: {world_graph.vectorized_backend.device}")
        print(f"✅ Vectorized similarity enabled: {world_graph.use_vectorized_similarity}")
    else:
        print("❌ No GPU acceleration detected")
    
    # Time individual operations
    print("\n⏱️ Testing individual operations:")
    
    # Test 1: Brain prediction timing
    profiler = get_brain_profiler()
    
    total_predictions = 100
    start_time = time.time()
    
    for i in range(total_predictions):
        # Create sensory packet
        sensory_packet = SensoryPacket(
            sequence_id=i,
            sensor_values=[0.1 + i*0.01] * 20,
            actuator_positions=[0.0, 0.0, 0.0],
            timestamp=datetime.now()
        )
        
        mental_context = sensory_packet.sensor_values[:8]
        
        # Time brain prediction
        prediction_start = time.time()
        try:
            prediction = brain_interface.process_sensory_input(
                sensory_packet, mental_context, threat_level="normal"
            )
        except Exception as e:
            print(f"   Error in prediction {i}: {e}")
            continue
        prediction_time = time.time() - prediction_start
        
        if i % 20 == 0:
            print(f"   Prediction {i}: {prediction_time:.6f}s")
    
    total_time = time.time() - start_time
    avg_prediction_time = total_time / total_predictions
    theoretical_fps = 1.0 / avg_prediction_time
    
    print(f"\n📊 Results:")
    print(f"   Total predictions: {total_predictions}")
    print(f"   Total time: {total_time:.3f}s")
    print(f"   Average prediction time: {avg_prediction_time:.6f}s")
    print(f"   Theoretical FPS: {theoretical_fps:.1f}")
    
    # Print detailed profiling
    print("\n📋 Detailed profiling:")
    profiler.print_timing_report(top_n=15)
    
    # Test similarity search specifically
    if hasattr(world_graph, 'vectorized_backend') and world_graph.vectorized_backend.get_size() > 0:
        print("\n🚀 Testing similarity search performance:")
        
        query_context = [0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4]
        
        # Test multiple similarity searches
        search_times = []
        for _ in range(50):
            start = time.time()
            results = world_graph.find_similar_experiences(query_context, 0.5, 10)
            search_times.append(time.time() - start)
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"   Average similarity search time: {avg_search_time:.6f}s")
        print(f"   Search results per query: {len(results) if 'results' in locals() else 0}")
        
        # Get vectorized stats
        stats = world_graph.get_vectorized_stats()
        print(f"   Vectorized experiences: {stats['storage_comparison']['vectorized_experiences']}")
        print(f"   Vectorized usage: {stats['performance_analysis']['vectorized_usage_percentage']:.1f}%")
        
        if stats['performance_analysis']['speedup_ratio'] > 1:
            print(f"   GPU speedup: {stats['performance_analysis']['speedup_ratio']:.1f}x")
    
    print("\n✅ Investigation complete!")

if __name__ == "__main__":
    test_fps_performance()