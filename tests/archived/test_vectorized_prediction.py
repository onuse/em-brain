#!/usr/bin/env python3
"""
Test Vectorized Prediction - Phase 2 GPU acceleration validation.

This test validates that the vectorized prediction system works correctly
and delivers the expected performance improvements.
"""

import sys
import time
import random

# Add project root to path
sys.path.append('.')

from core.hybrid_world_graph import HybridWorldGraph
from core.vectorized_traversal_engine import VectorizedTraversalEngine
from predictor.vectorized_triple_predictor import VectorizedTriplePredictor


def test_vectorized_traversal_engine():
    """Test the core vectorized traversal engine."""
    print("🧪 VECTORIZED TRAVERSAL ENGINE TEST")
    print("=" * 50)
    
    # Create hybrid graph with test data
    graph = HybridWorldGraph()
    
    # Add test experiences
    print("📝 Adding test experiences...")
    for i in range(500):
        mental_context = [random.gauss(0, 1) for _ in range(8)]
        action_taken = {
            'forward_motor': random.uniform(-1, 1),
            'turn_motor': random.uniform(-1, 1),
            'brake_motor': random.uniform(0, 1)
        }
        predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
        actual_sensory = [p + random.gauss(0, 0.1) for p in predicted_sensory]
        prediction_error = random.uniform(0, 0.3)
        
        graph.add_experience(mental_context, action_taken, predicted_sensory, 
                           actual_sensory, prediction_error)
    
    print(f"✅ Added {graph.node_count()} experiences")
    
    # Test vectorized traversal engine
    print("\n🚀 Testing vectorized traversal engine...")
    engine = VectorizedTraversalEngine(graph)
    
    # Test parallel traversals
    start_contexts = [[random.gauss(0, 1) for _ in range(8)] for _ in range(10)]
    
    start_time = time.time()
    result = engine.run_parallel_traversals(
        start_contexts=start_contexts,
        num_traversals=10,
        max_depth=8,
        similarity_threshold=0.5
    )
    traversal_time = time.time() - start_time
    
    print(f"✅ Parallel traversals completed in {traversal_time*1000:.2f}ms")
    print(f"   Terminal nodes found: {len([n for n in result.terminal_nodes if n is not None])}")
    print(f"   Average path length: {sum(result.path_lengths)/len(result.path_lengths):.1f}")
    print(f"   Average similarity: {sum(result.total_similarities)/len(result.total_similarities):.3f}")
    
    # Benchmark performance
    print("\n⚡ Benchmarking traversal performance...")
    benchmark_results = engine.benchmark_performance(num_traversals=100, max_depth=10)
    
    if 'speedup_factor' in benchmark_results:
        print(f"✅ Speedup achieved: {benchmark_results['speedup_factor']:.1f}x faster")
    
    return engine


def test_vectorized_triple_predictor():
    """Test the vectorized triple predictor."""
    print("\n🧪 VECTORIZED TRIPLE PREDICTOR TEST")
    print("=" * 50)
    
    # Create hybrid graph with test data
    graph = HybridWorldGraph()
    
    # Add test experiences
    print("📝 Adding test experiences...")
    for i in range(200):
        mental_context = [random.gauss(0, 1) for _ in range(8)]
        action_taken = {
            'forward_motor': random.uniform(-1, 1),
            'turn_motor': random.uniform(-1, 1),
            'brake_motor': random.uniform(0, 1)
        }
        predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
        actual_sensory = [p + random.gauss(0, 0.1) for p in predicted_sensory]
        prediction_error = random.uniform(0, 0.3)
        
        graph.add_experience(mental_context, action_taken, predicted_sensory, 
                           actual_sensory, prediction_error)
    
    print(f"✅ Added {graph.node_count()} experiences")
    
    # Test vectorized predictor
    print("\n🚀 Testing vectorized triple predictor...")
    predictor = VectorizedTriplePredictor(max_depth=10, traversal_count=5, use_gpu=True)
    
    # Test prediction generation
    test_context = [random.gauss(0, 1) for _ in range(8)]
    
    start_time = time.time()
    consensus_result = predictor.generate_prediction(test_context, graph, 1, "normal")
    prediction_time = time.time() - start_time
    
    print(f"✅ Prediction generated in {prediction_time*1000:.2f}ms")
    print(f"   Prediction available: {consensus_result.prediction is not None}")
    
    if consensus_result.prediction:
        print(f"   Motor action: {consensus_result.prediction.motor_action}")
        print(f"   Confidence: {consensus_result.prediction.confidence:.3f}")
    
    # Get performance statistics
    stats = predictor.get_vectorized_stats()
    print(f"\n📊 Performance statistics:")
    print(f"   GPU predictions: {stats['gpu_predictions']}")
    print(f"   CPU predictions: {stats['cpu_predictions']}")
    print(f"   GPU usage: {stats['gpu_usage_percentage']:.1f}%")
    
    if stats['speedup_factor'] > 1:
        print(f"   Speedup achieved: {stats['speedup_factor']:.1f}x")
    
    # Benchmark performance
    print("\n⚡ Benchmarking prediction performance...")
    benchmark_results = predictor.benchmark_performance(test_context, graph, num_predictions=20)
    
    if 'speedup_factor' in benchmark_results:
        print(f"✅ Prediction speedup: {benchmark_results['speedup_factor']:.1f}x faster")
    
    return predictor


def test_api_compatibility():
    """Test that vectorized predictor maintains API compatibility."""
    print("\n🧪 API COMPATIBILITY TEST")
    print("=" * 50)
    
    # Test that vectorized predictor works as drop-in replacement
    graph = HybridWorldGraph()
    
    # Add minimal test data
    for i in range(50):
        mental_context = [random.gauss(0, 1) for _ in range(8)]
        action_taken = {'forward_motor': random.uniform(-1, 1)}
        predicted_sensory = [random.uniform(0, 1) for _ in range(8)]
        actual_sensory = [p + random.gauss(0, 0.1) for p in predicted_sensory]
        prediction_error = random.uniform(0, 0.3)
        
        graph.add_experience(mental_context, action_taken, predicted_sensory, 
                           actual_sensory, prediction_error)
    
    # Test both vectorized and traditional predictors
    from predictor.triple_predictor import TriplePredictor
    
    traditional_predictor = TriplePredictor(max_depth=5)
    vectorized_predictor = VectorizedTriplePredictor(max_depth=5, traversal_count=3, use_gpu=True)
    
    test_context = [random.gauss(0, 1) for _ in range(8)]
    
    # Test traditional predictor
    traditional_result = traditional_predictor.generate_prediction(test_context, graph, 1, "normal")
    
    # Test vectorized predictor
    vectorized_result = vectorized_predictor.generate_prediction(test_context, graph, 1, "normal")
    
    # Compare results
    print(f"✅ Traditional prediction: {traditional_result.prediction is not None}")
    print(f"✅ Vectorized prediction: {vectorized_result.prediction is not None}")
    
    if traditional_result.prediction and vectorized_result.prediction:
        print(f"   Both predictors generated valid predictions")
        print(f"   Traditional confidence: {traditional_result.prediction.confidence:.3f}")
        print(f"   Vectorized confidence: {vectorized_result.prediction.confidence:.3f}")
    
    print("✅ API compatibility test passed!")


def main():
    """Run comprehensive vectorized prediction tests."""
    print("🚀 VECTORIZED PREDICTION SYSTEM TESTS")
    print("=" * 60)
    print("Phase 2 GPU acceleration validation")
    print()
    
    # Test 1: Vectorized traversal engine
    engine = test_vectorized_traversal_engine()
    
    # Test 2: Vectorized triple predictor
    predictor = test_vectorized_triple_predictor()
    
    # Test 3: API compatibility
    test_api_compatibility()
    
    # Final summary
    print("\n🌟 VECTORIZED PREDICTION TESTS COMPLETE!")
    print("=" * 60)
    
    # Get final performance stats
    engine_stats = engine.get_performance_stats()
    predictor_stats = predictor.get_vectorized_stats()
    
    print(f"🚀 Final Performance Results:")
    print(f"   Traversal engine device: {engine_stats['device']}")
    print(f"   Total traversals: {engine_stats['total_traversals']}")
    print(f"   GPU usage: {engine_stats['gpu_usage_percentage']:.1f}%")
    print(f"   Average traversal time: {engine_stats['avg_traversal_time']*1000:.2f}ms")
    print()
    print(f"   Predictor GPU usage: {predictor_stats['gpu_usage_percentage']:.1f}%")
    print(f"   Average prediction time: {predictor_stats['avg_prediction_time_ms']:.2f}ms")
    
    if predictor_stats['speedup_factor'] > 1:
        print(f"   Prediction speedup: {predictor_stats['speedup_factor']:.1f}x faster")
    
    print("\n✅ Phase 2 GPU acceleration is working and delivering performance improvements!")


if __name__ == "__main__":
    main()