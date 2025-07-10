#!/usr/bin/env python3
"""
Test GPU-accelerated sensory prediction system.
"""

import sys
import time
import torch
from typing import List, Dict

# Add current directory to path
sys.path.append('.')

from prediction.sensory.gpu_sensory_predictor import GPUSensoryPredictor, BatchSensoryPrediction
from prediction.sensory.sensory_predictor import SensoryPredictor
from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def test_gpu_sensory_predictor_basic():
    """Test basic GPU sensory predictor functionality."""
    print("🔮 Testing GPU Sensory Predictor - Basic")
    print("=" * 50)
    
    # Create hybrid world graph
    graph = HybridWorldGraph()
    
    # Add some experiences
    print("📊 Adding experiences...")
    for i in range(50):
        experience = ExperienceNode(
            mental_context=[0.1 * i, 0.2 * i, 0.3 * i],
            action_taken={
                'forward_motor': 0.5 + 0.01 * i,
                'turn_motor': 0.1 * i,
                'brake_motor': 0.05 * i
            },
            predicted_sensory=[0.4 * i, 0.5 * i, 0.6 * i],
            actual_sensory=[0.4 * i + 0.01, 0.5 * i + 0.01, 0.6 * i + 0.01],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    print(f"   Added {graph.vectorized_backend.size} experiences")
    
    # Create GPU predictor
    predictor = GPUSensoryPredictor(graph)
    
    # Test single prediction
    test_action = {
        'forward_motor': 0.6,
        'turn_motor': 0.3,
        'brake_motor': 0.1
    }
    
    print(f"\n🔍 Testing single prediction...")
    start_time = time.time()
    
    prediction = predictor.predict_sensory_outcome(test_action)
    
    prediction_time = time.time() - start_time
    print(f"   Prediction time: {prediction_time*1000:.2f}ms")
    print(f"   Confidence: {prediction.confidence:.3f}")
    print(f"   Method: {prediction.prediction_method}")
    print(f"   Basis experiences: {len(prediction.prediction_basis)}")
    
    # Test with context
    print(f"\n🧠 Testing with context...")
    context = [0.1, 0.2, 0.3]
    
    start_time = time.time()
    prediction_with_context = predictor.predict_sensory_outcome(test_action, context)
    context_time = time.time() - start_time
    
    print(f"   Context prediction time: {context_time*1000:.2f}ms")
    print(f"   Context confidence: {prediction_with_context.confidence:.3f}")
    
    # Test performance stats
    stats = predictor.get_performance_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Total predictions: {stats['prediction_count']}")
    print(f"   Average time: {stats['avg_prediction_time']*1000:.2f}ms")
    print(f"   GPU percentage: {stats['gpu_percentage']:.1f}%")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    print(f"   Device: {stats['device']}")
    
    print("✅ Basic GPU sensory predictor test completed!")
    return True


def test_batch_prediction():
    """Test batch prediction functionality."""
    print("\n🚀 Testing Batch Prediction")
    print("=" * 35)
    
    # Create larger graph for meaningful batch test
    graph = HybridWorldGraph()
    
    # Add more experiences
    for i in range(100):
        experience = ExperienceNode(
            mental_context=[0.01 * i, 0.02 * i, 0.03 * i],
            action_taken={
                'forward_motor': 0.5 + 0.001 * i,
                'turn_motor': 0.1 + 0.001 * i,
                'brake_motor': 0.05 + 0.001 * i
            },
            predicted_sensory=[0.1 * i, 0.2 * i, 0.3 * i],
            actual_sensory=[0.1 * i + 0.001, 0.2 * i + 0.001, 0.3 * i + 0.001],
            prediction_error=0.001
        )
        graph.add_node(experience)
    
    predictor = GPUSensoryPredictor(graph)
    
    # Create batch of actions
    actions = []
    for i in range(20):
        action = {
            'forward_motor': 0.5 + 0.01 * i,
            'turn_motor': 0.1 + 0.01 * i,
            'brake_motor': 0.05 + 0.01 * i
        }
        actions.append(action)
    
    print(f"📊 Testing batch prediction with {len(actions)} actions...")
    
    # Test batch prediction
    start_time = time.time()
    batch_result = predictor.batch_predict_sensory_outcomes(actions)
    batch_time = time.time() - start_time
    
    print(f"   Batch prediction time: {batch_time*1000:.2f}ms")
    print(f"   Batch confidence: {batch_result.batch_confidence:.3f}")
    print(f"   GPU utilization: {batch_result.gpu_utilization:.1f}")
    print(f"   Predictions generated: {len(batch_result.predictions)}")
    
    # Test individual predictions for comparison
    print(f"\n⚡ Comparing batch vs individual predictions...")
    start_time = time.time()
    
    individual_predictions = []
    for action in actions:
        pred = predictor.predict_sensory_outcome(action)
        individual_predictions.append(pred)
    
    individual_time = time.time() - start_time
    
    print(f"   Individual predictions time: {individual_time*1000:.2f}ms")
    print(f"   Batch speedup: {individual_time/batch_time:.1f}x")
    print(f"   Batch actions per second: {len(actions)/batch_time:.1f}")
    
    # Test best prediction
    best_prediction = batch_result.get_best_prediction()
    if best_prediction:
        print(f"   Best prediction confidence: {best_prediction.confidence:.3f}")
    
    print("✅ Batch prediction test completed!")
    return True


def test_gpu_vs_cpu_performance():
    """Test GPU vs CPU performance comparison."""
    print("\n⚡ Testing GPU vs CPU Performance")
    print("=" * 40)
    
    # Create graph with substantial data
    graph = HybridWorldGraph()
    
    print("📊 Creating substantial dataset...")
    for i in range(200):
        experience = ExperienceNode(
            mental_context=[0.005 * i, 0.01 * i, 0.015 * i],
            action_taken={
                'forward_motor': 0.5 + 0.0005 * i,
                'turn_motor': 0.1 + 0.0005 * i,
                'brake_motor': 0.05 + 0.0005 * i
            },
            predicted_sensory=[0.05 * i, 0.1 * i, 0.15 * i],
            actual_sensory=[0.05 * i + 0.0005, 0.1 * i + 0.0005, 0.15 * i + 0.0005],
            prediction_error=0.0005
        )
        graph.add_node(experience)
    
    print(f"   Dataset size: {graph.vectorized_backend.size} experiences")
    
    # Test GPU predictor
    gpu_predictor = GPUSensoryPredictor(graph, device='auto')
    
    # Test CPU predictor (force CPU)
    cpu_predictor = GPUSensoryPredictor(graph, device='cpu')
    
    # Test actions
    test_actions = [
        {'forward_motor': 0.6, 'turn_motor': 0.3, 'brake_motor': 0.1},
        {'forward_motor': 0.7, 'turn_motor': 0.2, 'brake_motor': 0.1},
        {'forward_motor': 0.5, 'turn_motor': 0.4, 'brake_motor': 0.1},
        {'forward_motor': 0.8, 'turn_motor': 0.1, 'brake_motor': 0.1},
        {'forward_motor': 0.4, 'turn_motor': 0.5, 'brake_motor': 0.1}
    ]
    
    # Benchmark GPU
    print(f"\n🚀 Benchmarking GPU predictor...")
    start_time = time.time()
    
    for _ in range(10):  # Run multiple times for stability
        for action in test_actions:
            gpu_predictor.predict_sensory_outcome(action)
    
    gpu_time = time.time() - start_time
    
    # Benchmark CPU
    print(f"💻 Benchmarking CPU predictor...")
    start_time = time.time()
    
    for _ in range(10):  # Run multiple times for stability
        for action in test_actions:
            cpu_predictor.predict_sensory_outcome(action)
    
    cpu_time = time.time() - start_time
    
    # Compare results
    print(f"\n📊 Performance Comparison:")
    print(f"   GPU time: {gpu_time*1000:.2f}ms")
    print(f"   CPU time: {cpu_time*1000:.2f}ms")
    print(f"   GPU speedup: {cpu_time/gpu_time:.1f}x")
    print(f"   GPU predictions/sec: {(len(test_actions)*10)/gpu_time:.1f}")
    print(f"   CPU predictions/sec: {(len(test_actions)*10)/cpu_time:.1f}")
    
    # Get detailed stats
    gpu_stats = gpu_predictor.get_performance_stats()
    cpu_stats = cpu_predictor.get_performance_stats()
    
    print(f"\n🚀 GPU Stats:")
    print(f"   Device: {gpu_stats['device']}")
    print(f"   GPU percentage: {gpu_stats['gpu_percentage']:.1f}%")
    print(f"   Cache hit rate: {gpu_stats['cache_hit_rate']:.1f}%")
    
    print(f"\n💻 CPU Stats:")
    print(f"   Device: {cpu_stats['device']}")
    print(f"   Cache hit rate: {cpu_stats['cache_hit_rate']:.1f}%")
    
    print("✅ GPU vs CPU performance test completed!")
    return True


def test_prediction_accuracy():
    """Test prediction accuracy and quality."""
    print("\n🎯 Testing Prediction Accuracy")
    print("=" * 35)
    
    # Create graph with predictable patterns
    graph = HybridWorldGraph()
    
    # Add experiences with clear patterns
    print("📊 Creating predictable patterns...")
    for i in range(50):
        # Forward motion pattern
        if i < 25:
            experience = ExperienceNode(
                mental_context=[1.0, 0.0, 0.0],
                action_taken={'forward_motor': 0.8, 'turn_motor': 0.0, 'brake_motor': 0.0},
                predicted_sensory=[0.1, 0.2, 0.3],
                actual_sensory=[0.1 + i * 0.01, 0.2, 0.3],  # Moving forward
                prediction_error=0.01
            )
        else:
            # Turn pattern
            experience = ExperienceNode(
                mental_context=[0.0, 1.0, 0.0],
                action_taken={'forward_motor': 0.0, 'turn_motor': 0.8, 'brake_motor': 0.0},
                predicted_sensory=[0.4, 0.5, 0.6],
                actual_sensory=[0.4, 0.5 + (i-25) * 0.01, 0.6],  # Turning
                prediction_error=0.01
            )
        
        graph.add_node(experience)
    
    predictor = GPUSensoryPredictor(graph)
    
    # Test forward motion prediction
    print(f"\n🔍 Testing forward motion prediction...")
    forward_action = {'forward_motor': 0.8, 'turn_motor': 0.0, 'brake_motor': 0.0}
    forward_context = [1.0, 0.0, 0.0]
    
    forward_prediction = predictor.predict_sensory_outcome(forward_action, forward_context)
    
    print(f"   Forward prediction confidence: {forward_prediction.confidence:.3f}")
    print(f"   Forward prediction method: {forward_prediction.prediction_method}")
    print(f"   Forward basis experiences: {len(forward_prediction.prediction_basis)}")
    
    # Test turn prediction
    print(f"\n🔄 Testing turn prediction...")
    turn_action = {'forward_motor': 0.0, 'turn_motor': 0.8, 'brake_motor': 0.0}
    turn_context = [0.0, 1.0, 0.0]
    
    turn_prediction = predictor.predict_sensory_outcome(turn_action, turn_context)
    
    print(f"   Turn prediction confidence: {turn_prediction.confidence:.3f}")
    print(f"   Turn prediction method: {turn_prediction.prediction_method}")
    print(f"   Turn basis experiences: {len(turn_prediction.prediction_basis)}")
    
    # Test unknown action
    print(f"\n❓ Testing unknown action...")
    unknown_action = {'forward_motor': 0.5, 'turn_motor': 0.5, 'brake_motor': 0.5}
    unknown_context = [0.0, 0.0, 1.0]
    
    unknown_prediction = predictor.predict_sensory_outcome(unknown_action, unknown_context)
    
    print(f"   Unknown prediction confidence: {unknown_prediction.confidence:.3f}")
    print(f"   Unknown prediction method: {unknown_prediction.prediction_method}")
    print(f"   Unknown basis experiences: {len(unknown_prediction.prediction_basis)}")
    
    # Test prediction quality
    print(f"\n📊 Prediction Quality Analysis:")
    print(f"   Forward quality: {forward_prediction.get_prediction_quality():.3f}")
    print(f"   Turn quality: {turn_prediction.get_prediction_quality():.3f}")
    print(f"   Unknown quality: {unknown_prediction.get_prediction_quality():.3f}")
    
    print("✅ Prediction accuracy test completed!")
    return True


def benchmark_sensory_predictor():
    """Comprehensive benchmark of sensory predictor."""
    print("\n📊 Comprehensive Sensory Predictor Benchmark")
    print("=" * 55)
    
    # Create large graph
    graph = HybridWorldGraph()
    
    print("📊 Creating large dataset for benchmark...")
    for i in range(500):
        experience = ExperienceNode(
            mental_context=[0.002 * i, 0.004 * i, 0.006 * i],
            action_taken={
                'forward_motor': 0.5 + 0.0002 * i,
                'turn_motor': 0.1 + 0.0002 * i,
                'brake_motor': 0.05 + 0.0002 * i
            },
            predicted_sensory=[0.02 * i, 0.04 * i, 0.06 * i],
            actual_sensory=[0.02 * i + 0.0002, 0.04 * i + 0.0002, 0.06 * i + 0.0002],
            prediction_error=0.0002
        )
        graph.add_node(experience)
    
    print(f"   Dataset size: {graph.vectorized_backend.size} experiences")
    
    # Create predictor
    predictor = GPUSensoryPredictor(graph)
    
    # Run comprehensive benchmark
    print(f"\n🚀 Running comprehensive benchmark...")
    benchmark_results = predictor.benchmark_prediction_performance(num_predictions=100)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Single predictions/sec: {benchmark_results['single_predictions_per_second']:.1f}")
    print(f"   Batch predictions/sec: {benchmark_results['batch_predictions_per_second']:.1f}")
    print(f"   Batch speedup: {benchmark_results['batch_speedup']:.1f}x")
    print(f"   Batch confidence: {benchmark_results['batch_confidence']:.3f}")
    print(f"   GPU utilization: {benchmark_results['gpu_utilization']:.1f}")
    
    # Test with different batch sizes
    print(f"\n🔄 Testing different batch sizes...")
    batch_sizes = [10, 25, 50, 100]
    
    for batch_size in batch_sizes:
        actions = []
        for i in range(batch_size):
            action = {
                'forward_motor': 0.5 + 0.001 * i,
                'turn_motor': 0.1 + 0.001 * i,
                'brake_motor': 0.05 + 0.001 * i
            }
            actions.append(action)
        
        start_time = time.time()
        batch_result = predictor.batch_predict_sensory_outcomes(actions)
        batch_time = time.time() - start_time
        
        print(f"   Batch size {batch_size}: {batch_time*1000:.2f}ms ({batch_size/batch_time:.1f} pred/sec)")
    
    # Final performance stats
    final_stats = predictor.get_performance_stats()
    print(f"\n🎯 Final Performance Stats:")
    print(f"   Total predictions: {final_stats['prediction_count']}")
    print(f"   Average prediction time: {final_stats['avg_prediction_time']*1000:.2f}ms")
    print(f"   GPU percentage: {final_stats['gpu_percentage']:.1f}%")
    print(f"   Cache hit rate: {final_stats['cache_hit_rate']:.1f}%")
    print(f"   Predictions per second: {final_stats['predictions_per_second']:.1f}")
    
    print("✅ Comprehensive benchmark completed!")
    return True


def main():
    """Run all GPU sensory predictor tests."""
    print("🔮 GPU SENSORY PREDICTOR TESTS")
    print("=" * 80)
    
    try:
        # Test basic functionality
        test_gpu_sensory_predictor_basic()
        
        # Test batch prediction
        test_batch_prediction()
        
        # Test GPU vs CPU performance
        test_gpu_vs_cpu_performance()
        
        # Test prediction accuracy
        test_prediction_accuracy()
        
        # Comprehensive benchmark
        benchmark_sensory_predictor()
        
        print("\n🎉 All GPU sensory predictor tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)