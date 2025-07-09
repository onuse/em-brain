#!/usr/bin/env python3
"""
Quick Phase 2 Validation - Fast test of core vectorized functionality.
"""

import sys
import time
import random

sys.path.append('.')

from core.hybrid_world_graph import HybridWorldGraph
from predictor.vectorized_triple_predictor import VectorizedTriplePredictor


def test_core_functionality():
    """Test core Phase 2 functionality quickly."""
    print("🚀 PHASE 2 QUICK VALIDATION")
    print("=" * 40)
    
    # Create small test dataset
    graph = HybridWorldGraph()
    
    for i in range(100):
        mental_context = [random.gauss(0, 1) for _ in range(8)]
        action_taken = {'forward_motor': random.uniform(-1, 1)}
        predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
        actual_sensory = [p + random.gauss(0, 0.1) for p in predicted_sensory]
        prediction_error = random.uniform(0, 0.3)
        
        graph.add_experience(mental_context, action_taken, predicted_sensory, 
                           actual_sensory, prediction_error)
    
    print(f"✅ Created graph with {graph.node_count()} experiences")
    
    # Test vectorized prediction
    predictor = VectorizedTriplePredictor(max_depth=5, traversal_count=3, use_gpu=True)
    test_context = [random.gauss(0, 1) for _ in range(8)]
    
    # Test with different sizes
    for i in range(3):
        start_time = time.time()
        result = predictor.generate_prediction(test_context, graph, i, "normal")
        elapsed = time.time() - start_time
        
        print(f"   Prediction {i+1}: {elapsed*1000:.1f}ms, success: {result.prediction is not None}")
    
    # Show stats
    stats = predictor.get_vectorized_stats()
    adaptive_stats = stats['adaptive_engine_stats']
    
    print(f"✅ Adaptive predictions: {stats['adaptive_predictions']}")
    print(f"✅ CPU utilization: {adaptive_stats['utilization']['cpu_percentage']:.1f}%")
    print(f"✅ GPU utilization: {adaptive_stats['utilization']['gpu_percentage']:.1f}%")
    
    return True


if __name__ == "__main__":
    try:
        success = test_core_functionality()
        if success:
            print("\n🎉 Phase 2 core functionality validated!")
        else:
            print("\n⚠️  Phase 2 validation failed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()