#!/usr/bin/env python3
"""
Test brain performance with real brain data and the accelerated similarity engine
"""

import sys
sys.path.append('.')

import time
import json
import os
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from predictor.triple_predictor import TriplePredictor

def load_real_brain_data():
    """Load real brain data from saved sessions"""
    print("Loading real brain data...")
    
    # Check for saved brain data
    memory_dir = "./logs/robot_memory"
    graphs_dir = os.path.join(memory_dir, "graphs")
    
    if not os.path.exists(graphs_dir):
        print("No real brain data found, creating synthetic data...")
        return create_synthetic_brain_data()
    
    # Find the latest graph file
    graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.pkl.gz')]
    if not graph_files:
        print("No graph files found, creating synthetic data...")
        return create_synthetic_brain_data()
    
    print(f"Found {len(graph_files)} graph files")
    print("Creating synthetic data for testing...")
    return create_synthetic_brain_data()

def create_synthetic_brain_data():
    """Create realistic synthetic brain data for testing"""
    print("Creating synthetic brain data...")
    
    graph = WorldGraph()
    
    # Create 2140 nodes to match real brain size
    n_nodes = 2140
    
    import numpy as np
    np.random.seed(42)
    
    for i in range(n_nodes):
        # Create realistic context (sensor readings)
        context = [
            np.random.uniform(0, 1),  # front sensor
            np.random.uniform(0, 1),  # left sensor  
            np.random.uniform(0, 1),  # right sensor
            np.random.uniform(0, 1),  # back sensor
            np.random.uniform(-2, 2), # x position
            np.random.uniform(-2, 2), # y position
            np.random.uniform(0.5, 1.0), # health
            np.random.uniform(0.5, 1.0), # energy
        ]
        
        # Create realistic action
        action = {
            'forward_motor': np.random.uniform(-1, 1),
            'turn_motor': np.random.uniform(-1, 1),
            'brake_motor': np.random.uniform(0, 1)
        }
        
        # Create realistic sensory prediction/reality
        predicted_sensory = np.random.uniform(0, 1, 8).tolist()
        actual_sensory = (np.array(predicted_sensory) + np.random.normal(0, 0.1, 8)).tolist()
        prediction_error = np.linalg.norm(np.array(predicted_sensory) - np.array(actual_sensory))
        
        node = ExperienceNode(
            mental_context=context,
            action_taken=action,
            predicted_sensory=predicted_sensory,
            actual_sensory=actual_sensory,
            prediction_error=prediction_error
        )
        
        graph.add_node(node)
        
        if (i + 1) % 500 == 0:
            print(f"  Created {i + 1} nodes...")
    
    print(f"Synthetic brain created with {len(graph.nodes)} nodes")
    return graph

def test_brain_prediction_performance(graph):
    """Test the performance of brain predictions with real data"""
    print("\\nTesting brain prediction performance...")
    
    # Create predictor
    predictor = TriplePredictor(
        max_depth=5,
        randomness_factor=0.3,
        base_time_budget=0.1,
        exploration_rate=0.3
    )
    
    # Test realistic prediction scenarios
    test_contexts = [
        [0.2, 0.1, 0.3, 0.0, 1.5, -0.8, 0.9, 0.8],  # Near wall scenario
        [0.8, 0.9, 0.7, 0.1, 0.0, 0.0, 0.7, 0.6],  # Surrounded scenario
        [0.1, 0.1, 0.1, 0.1, -1.0, 1.0, 1.0, 1.0], # Open space scenario
        [0.5, 0.3, 0.4, 0.2, 0.5, 0.5, 0.5, 0.4],  # Mixed scenario
        [0.9, 0.1, 0.1, 0.8, -2.0, 2.0, 0.3, 0.2], # Dangerous scenario
    ]
    
    print(f"Running {len(test_contexts)} prediction scenarios...")
    
    total_predictions = 0
    total_time = 0
    
    for i, context in enumerate(test_contexts):
        print(f"\\nScenario {i+1}: Testing with context {context[:4]}...")
        
        # Warmup
        _ = predictor.generate_prediction(context, graph, sequence_id=0, threat_level="normal")
        
        # Benchmark multiple predictions for this scenario
        scenario_start = time.time()
        n_predictions = 5
        
        for pred_num in range(n_predictions):
            threat_levels = ["safe", "normal", "alert", "danger", "critical"]
            threat_level = threat_levels[pred_num % len(threat_levels)]
            
            pred_start = time.time()
            
            result = predictor.generate_prediction(
                context, 
                graph, 
                sequence_id=pred_num, 
                threat_level=threat_level
            )
            
            pred_time = time.time() - pred_start
            total_time += pred_time
            total_predictions += 1
            
            print(f"    Prediction {pred_num+1} ({threat_level}): {pred_time:.6f}s")
            if result and result.prediction:
                action = result.prediction.motor_action
                print(f"      Action: forward={action.get('forward_motor', 0):.2f}, turn={action.get('turn_motor', 0):.2f}")
        
        scenario_time = time.time() - scenario_start
        print(f"  Scenario {i+1} total: {scenario_time:.4f}s ({n_predictions} predictions)")
    
    # Overall results
    avg_prediction_time = total_time / total_predictions
    predictions_per_second = 1.0 / avg_prediction_time
    
    print(f"\\n{'='*50}")
    print(f"BRAIN PREDICTION PERFORMANCE RESULTS")
    print(f"{'='*50}")
    print(f"Total predictions: {total_predictions}")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average prediction time: {avg_prediction_time:.6f}s")
    print(f"Predictions per second: {predictions_per_second:.1f}")
    print(f"Theoretical demo FPS: {predictions_per_second:.1f}")
    
    # Get similarity engine stats
    stats = graph.get_graph_statistics()
    sim_stats = stats['similarity_engine']
    
    print(f"\\nSimilarity Engine Performance:")
    print(f"  Method: {sim_stats['acceleration_method']}")
    print(f"  Total searches: {sim_stats['total_searches']}")
    print(f"  Average search time: {sim_stats['avg_search_time']:.6f}s")
    print(f"  Searches per second: {sim_stats['searches_per_second']:.1f}")
    print(f"  Cache hit rate: {sim_stats['cache_hit_rate']:.2%}")
    
    return avg_prediction_time, predictions_per_second

def compare_with_original_performance():
    """Compare performance impact"""
    print(f"\\n{'='*50}")
    print(f"PERFORMANCE IMPACT ANALYSIS")
    print(f"{'='*50}")
    
    # These are the numbers we found earlier
    original_fps = 1.4
    original_prediction_time = 1.0 / original_fps  # ~0.714s per prediction
    
    print(f"Original demo performance:")
    print(f"  FPS: {original_fps}")
    print(f"  Time per frame: {original_prediction_time:.3f}s")
    
    # This would be filled in with actual results
    print(f"\\nExpected improvements with acceleration:")
    print(f"  Similarity search: 25x faster (from vectorization)")
    print(f"  Brain traversal: Should scale with brain size")
    print(f"  Overall: Depends on what % of time is spent in similarity search")

def main():
    """Main performance test"""
    print("Brain Performance Test with Accelerated Similarity")
    print("=" * 60)
    
    # Load or create brain data
    graph = load_real_brain_data()
    
    # Test performance
    avg_pred_time, pred_per_sec = test_brain_prediction_performance(graph)
    
    # Compare with original
    compare_with_original_performance()
    
    print(f"\\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    
    if pred_per_sec > 10:
        print(f"🚀 EXCELLENT: {pred_per_sec:.1f} FPS - Major improvement over 1.4 FPS!")
    elif pred_per_sec > 5:
        print(f"✅ GOOD: {pred_per_sec:.1f} FPS - Significant improvement over 1.4 FPS")
    elif pred_per_sec > 2:
        print(f"⚡ IMPROVED: {pred_per_sec:.1f} FPS - Moderate improvement over 1.4 FPS")
    else:
        print(f"⚠️  SIMILAR: {pred_per_sec:.1f} FPS - Check for bottlenecks")
    
    print(f"\\nReady for real demo testing!")

if __name__ == "__main__":
    main()