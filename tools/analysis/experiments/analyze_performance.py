#!/usr/bin/env python3
"""
Performance Analysis Script for Brain System

Analyzes current similarity search and GPU utilization patterns
to understand performance bottlenecks and optimization opportunities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'src'))

import time
import random
import numpy as np
from typing import List, Dict, Any

# Import brain systems
try:
    from similarity.engine import SimilarityEngine
    from experience.models import Experience
    from activation.dynamics import ActivationDynamics
    from utils.hardware_adaptation import get_hardware_adaptation
    print("✅ All brain imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_test_experience(i: int) -> Experience:
    """Create a test experience for analysis."""
    return Experience(
        sensory_input=[random.random() for _ in range(16)],
        action_taken=[random.random() for _ in range(4)],
        outcome=[random.random() for _ in range(16)],
        prediction_error=random.random(),
        timestamp=time.time() + i
    )

def analyze_similarity_quality(engine: SimilarityEngine, experiences: List[Experience]) -> Dict[str, Any]:
    """Analyze whether similarity search is producing meaningful diversity."""
    print("\n=== SIMILARITY QUALITY ANALYSIS ===")
    
    if len(experiences) < 10:
        return {"error": "Need at least 10 experiences for quality analysis"}
    
    # Test with multiple query vectors
    results_analysis = []
    
    for test_run in range(5):
        # Create a test query
        target = [random.random() for _ in range(20)]
        
        # Extract experience vectors
        exp_vectors = []
        exp_ids = []
        for i, exp in enumerate(experiences):
            vector = exp.sensory_input + exp.action_taken
            exp_vectors.append(vector)
            exp_ids.append(f'exp_{i}')
        
        # Get similarity results
        start_time = time.time()
        results = engine.find_similar_experiences(
            target, exp_vectors, exp_ids, max_results=10
        )
        search_time = time.time() - start_time
        
        if results:
            similarities = [score for _, score in results]
            results_analysis.append({
                'search_time_ms': search_time * 1000,
                'num_results': len(results),
                'max_similarity': max(similarities),
                'min_similarity': min(similarities),
                'similarity_range': max(similarities) - min(similarities),
                'avg_similarity': np.mean(similarities),
                'similarity_std': np.std(similarities)
            })
    
    if not results_analysis:
        return {"error": "No results found"}
    
    # Analyze diversity
    avg_range = np.mean([r['similarity_range'] for r in results_analysis])
    avg_std = np.mean([r['similarity_std'] for r in results_analysis])
    avg_search_time = np.mean([r['search_time_ms'] for r in results_analysis])
    avg_max_sim = np.mean([r['max_similarity'] for r in results_analysis])
    
    # Assess quality
    diversity_quality = "good" if avg_range > 0.2 else "poor" if avg_range < 0.05 else "moderate"
    speed_quality = "good" if avg_search_time < 50 else "poor" if avg_search_time > 150 else "moderate"
    
    return {
        'diversity_quality': diversity_quality,
        'speed_quality': speed_quality,
        'avg_similarity_range': avg_range,
        'avg_similarity_std': avg_std,
        'avg_search_time_ms': avg_search_time,
        'avg_max_similarity': avg_max_sim,
        'analysis': 'genuine_diversity' if avg_range > 0.2 and avg_std > 0.1 else 'high_similarity_cluster'
    }

def analyze_gpu_utilization() -> Dict[str, Any]:
    """Analyze current GPU usage patterns."""
    print("\n=== GPU UTILIZATION ANALYSIS ===")
    
    hardware_adapter = get_hardware_adaptation()
    hw_profile = hardware_adapter.get_hardware_profile()
    
    # Test GPU decision making for different operation sizes
    gpu_decisions = {}
    test_sizes = [10, 20, 50, 100, 200, 500, 1000]
    
    for size in test_sizes:
        similarity_decision = hardware_adapter.should_use_gpu_for_operation(size, 'similarity')
        activation_decision = hardware_adapter.should_use_gpu_for_operation(size, 'activation')
        pattern_decision = hardware_adapter.should_use_gpu_for_operation(size, 'pattern')
        
        gpu_decisions[size] = {
            'similarity': similarity_decision,
            'activation': activation_decision,
            'pattern': pattern_decision
        }
    
    return {
        'hardware_profile': hw_profile,
        'gpu_thresholds': gpu_decisions,
        'recommendations': analyze_gpu_strategy(gpu_decisions, hw_profile)
    }

def analyze_gpu_strategy(decisions: Dict, hw_profile: Dict) -> List[str]:
    """Analyze GPU strategy and provide recommendations."""
    recommendations = []
    
    # Check if GPU is available
    if not hw_profile.get('gpu_available', False):
        recommendations.append("❌ No GPU available - all operations will use CPU")
        return recommendations
    
    # Find GPU activation thresholds
    similarity_threshold = None
    activation_threshold = None
    pattern_threshold = None
    
    for size, decision in decisions.items():
        if decision['similarity'] and similarity_threshold is None:
            similarity_threshold = size
        if decision['activation'] and activation_threshold is None:
            activation_threshold = size
        if decision['pattern'] and pattern_threshold is None:
            pattern_threshold = size
    
    if similarity_threshold:
        recommendations.append(f"✅ Similarity GPU threshold: {similarity_threshold} experiences")
    else:
        recommendations.append("⚠️ Similarity never uses GPU - threshold too high")
    
    if activation_threshold:
        recommendations.append(f"✅ Activation GPU threshold: {activation_threshold} experiences")
    else:
        recommendations.append("⚠️ Activation never uses GPU - threshold too high")
    
    if pattern_threshold:
        recommendations.append(f"✅ Pattern GPU threshold: {pattern_threshold} experiences")
    else:
        recommendations.append("⚠️ Pattern never uses GPU - threshold too high")
    
    # Check for efficient sparse matrix operations
    batch_threshold = hw_profile.get('batch_processing_threshold', 50)
    if batch_threshold > 100:
        recommendations.append("⚠️ Batch threshold high - may not benefit from sparse operations")
    else:
        recommendations.append(f"✅ Batch threshold reasonable: {batch_threshold}")
    
    return recommendations

def analyze_sparse_connectivity_needs(experiences: List[Experience]) -> Dict[str, Any]:
    """Analyze what sparse connectivity algorithm would be best."""
    print("\n=== SPARSE CONNECTIVITY ANALYSIS ===")
    
    if len(experiences) < 20:
        return {"error": "Need at least 20 experiences for connectivity analysis"}
    
    # Analyze current connection patterns
    total_connections = 0
    connection_strengths = []
    
    for exp in experiences:
        total_connections += len(exp.similar_experiences)
        connection_strengths.extend(exp.similar_experiences.values())
    
    if not connection_strengths:
        return {"error": "No similarity connections found"}
    
    avg_connections_per_exp = total_connections / len(experiences)
    avg_connection_strength = np.mean(connection_strengths)
    connection_variance = np.var(connection_strengths)
    
    # Recommend sparse connection strategy
    if avg_connections_per_exp > len(experiences) * 0.5:
        strategy_rec = "threshold_based"
        reason = "High connectivity density suggests threshold-based filtering"
    elif connection_variance > 0.1:
        strategy_rec = "k_nearest_neighbors"
        reason = "High connection strength variance suggests k-NN approach"
    else:
        strategy_rec = "hierarchical_clustering"
        reason = "Uniform connectivity suggests hierarchical clustering"
    
    return {
        'avg_connections_per_experience': avg_connections_per_exp,
        'total_connections': total_connections,
        'avg_connection_strength': avg_connection_strength,
        'connection_variance': connection_variance,
        'connectivity_density': avg_connections_per_exp / len(experiences),
        'recommended_strategy': strategy_rec,
        'reasoning': reason,
        'biological_realism': analyze_biological_realism(avg_connections_per_exp, len(experiences))
    }

def analyze_biological_realism(avg_connections: float, total_experiences: int) -> str:
    """Assess biological realism of connectivity patterns."""
    # Real neurons typically connect to 1-10% of other neurons
    bio_ratio = avg_connections / total_experiences
    
    if bio_ratio > 0.3:
        return "unrealistic - too highly connected"
    elif bio_ratio > 0.1:
        return "moderate - higher than biological but acceptable"
    elif bio_ratio > 0.01:
        return "good - within biological range"
    else:
        return "sparse - lower than typical but acceptable"

def analyze_data_structures() -> Dict[str, Any]:
    """Analyze current data structure efficiency for sparse operations."""
    print("\n=== DATA STRUCTURE EFFICIENCY ANALYSIS ===")
    
    # Test memory efficiency of different representations
    test_sizes = [100, 500, 1000]
    efficiency_results = {}
    
    for size in test_sizes:
        # Simulate dense vs sparse representations
        dense_memory = size * size * 4  # float32 full matrix
        sparse_memory = size * 10 * 12  # COO format: 10 connections avg, 3 values per connection
        
        efficiency_results[size] = {
            'dense_memory_kb': dense_memory / 1024,
            'sparse_memory_kb': sparse_memory / 1024,
            'memory_ratio': sparse_memory / dense_memory,
            'recommended': 'sparse' if sparse_memory < dense_memory * 0.5 else 'dense'
        }
    
    return {
        'memory_efficiency': efficiency_results,
        'recommendations': [
            "✅ Use COO sparse matrices for connection storage",
            "✅ Use CSR format for fast row-based operations", 
            "✅ Batch sparse operations on GPU when possible",
            "⚠️ Consider hybrid dense/sparse based on connectivity density"
        ]
    }

def main():
    """Run comprehensive performance analysis."""
    print("🔬 BRAIN PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Initialize systems
    print("Initializing brain systems...")
    engine = SimilarityEngine(use_gpu=True, use_learnable_similarity=True)
    
    # Create test dataset
    print("Creating test dataset...")
    experiences = [create_test_experience(i) for i in range(200)]
    
    # Run analyses
    similarity_analysis = analyze_similarity_quality(engine, experiences)
    gpu_analysis = analyze_gpu_utilization()
    connectivity_analysis = analyze_sparse_connectivity_needs(experiences)
    data_structure_analysis = analyze_data_structures()
    
    # Generate comprehensive report
    print("\n" + "=" * 50)
    print("📊 PERFORMANCE ANALYSIS REPORT")
    print("=" * 50)
    
    print(f"\n🔍 SIMILARITY QUALITY: {similarity_analysis.get('diversity_quality', 'unknown').upper()}")
    print(f"   Speed Quality: {similarity_analysis.get('speed_quality', 'unknown')}")
    print(f"   Avg Search Time: {similarity_analysis.get('avg_search_time_ms', 0):.1f}ms")
    print(f"   Similarity Diversity: {similarity_analysis.get('avg_similarity_range', 0):.3f}")
    print(f"   Analysis: {similarity_analysis.get('analysis', 'unknown')}")
    
    print(f"\n🚀 GPU UTILIZATION:")
    gpu_recs = gpu_analysis.get('recommendations', [])
    for rec in gpu_recs:
        print(f"   {rec}")
    
    print(f"\n🕸️ SPARSE CONNECTIVITY:")
    if 'recommended_strategy' in connectivity_analysis:
        print(f"   Recommended Strategy: {connectivity_analysis['recommended_strategy']}")
        print(f"   Reasoning: {connectivity_analysis['reasoning']}")
        print(f"   Biological Realism: {connectivity_analysis['biological_realism']}")
        print(f"   Connectivity Density: {connectivity_analysis['connectivity_density']:.3f}")
    
    print(f"\n💾 DATA STRUCTURE EFFICIENCY:")
    ds_recs = data_structure_analysis.get('recommendations', [])
    for rec in ds_recs:
        print(f"   {rec}")
    
    # Final recommendations
    print(f"\n🎯 KEY FINDINGS & RECOMMENDATIONS:")
    
    # Similarity issues
    if similarity_analysis.get('diversity_quality') == 'poor':
        print("   ❌ CRITICAL: Similarity search shows poor diversity")
        print("      → All experiences appearing too similar")
        print("      → Learnable similarity may need more training data")
        print("      → Consider adjusting similarity learning parameters")
    
    # Performance issues  
    if similarity_analysis.get('speed_quality') == 'poor':
        print("   ⚠️ PERFORMANCE: Similarity search is slow")
        print("      → Consider optimizing GPU operations")
        print("      → Implement sparse matrix operations")
        print("      → Use more aggressive caching")
    
    # GPU optimization opportunities
    hw_profile = gpu_analysis.get('hardware_profile', {})
    if hw_profile.get('gpu_available') and similarity_analysis.get('speed_quality') != 'good':
        print("   🚀 GPU OPTIMIZATION NEEDED:")
        print("      → Implement sparse tensor operations")
        print("      → Use PyTorch sparse.FloatTensor for connections")
        print("      → Batch similarity computations more efficiently")
        print("      → Consider mixed precision (FP16) for speed")
    
    # Sparse connectivity strategy
    if 'recommended_strategy' in connectivity_analysis:
        strategy = connectivity_analysis['recommended_strategy']
        print(f"   🕸️ SPARSE CONNECTIVITY: Implement {strategy}")
        if strategy == 'k_nearest_neighbors':
            print("      → Use k=5-10 nearest neighbors per experience")
            print("      → Implement GPU-accelerated k-NN search")
        elif strategy == 'threshold_based':
            print("      → Set similarity threshold around 0.3-0.5")
            print("      → Dynamically adjust based on connectivity density")
        elif strategy == 'hierarchical_clustering':
            print("      → Group similar experiences into clusters")
            print("      → Connect within clusters, sparse between clusters")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()