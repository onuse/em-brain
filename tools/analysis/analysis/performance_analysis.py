#!/usr/bin/env python3
"""
Performance Analysis for Brain System

Investigates similarity quality, GPU utilization, and sparse connectivity needs.
"""

import sys
import os
import time
import random
import numpy as np
import json
from typing import List, Dict, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))

def analyze_log_data():
    """Analyze recent brain session logs to understand performance."""
    print("🔬 ANALYZING RECENT BRAIN SESSION LOGS")
    print("=" * 50)
    
    logs_dir = os.path.join(brain_root, 'logs')
    if not os.path.exists(logs_dir):
        print("❌ No logs directory found")
        return
    
    # Find most recent brain state log
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('_brain_state.jsonl')]
    if not log_files:
        print("❌ No brain state logs found")
        return
    
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(logs_dir, latest_log)
    
    print(f"📋 Analyzing: {latest_log}")
    
    # Read and analyze log entries
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    
    if not entries:
        print("❌ No valid log entries found")
        return
    
    print(f"📊 Found {len(entries)} log entries")
    
    # Analyze similarity performance over time
    analyze_similarity_trends(entries)
    analyze_gpu_usage_patterns(entries)
    analyze_activation_patterns(entries)
    provide_recommendations(entries)

def analyze_similarity_trends(entries: List[Dict]) -> None:
    """Analyze similarity search performance trends."""
    print(f"\n🔍 SIMILARITY SEARCH ANALYSIS")
    print("-" * 30)
    
    similarity_data = []
    for entry in entries:
        sim_stats = entry.get('system_stats', {}).get('similarity_engine', {})
        if sim_stats:
            similarity_data.append({
                'experience_count': entry.get('experience_count', 0),
                'total_searches': sim_stats.get('total_searches', 0),
                'avg_search_time': sim_stats.get('avg_search_time', 0),
                'searches_per_second': sim_stats.get('searches_per_second', 0),
                'cache_hit_rate': sim_stats.get('cache_stats', {}).get('hit_rate', 0),
                'gpu_enabled': sim_stats.get('gpu_enabled', False),
                'similarity_learning': sim_stats.get('similarity_learning', {})
            })
    
    if not similarity_data:
        print("❌ No similarity data found")
        return
    
    # Analyze trends
    final_data = similarity_data[-1]
    initial_data = similarity_data[0] if len(similarity_data) > 1 else final_data
    
    print(f"Experience Growth: {initial_data['experience_count']} → {final_data['experience_count']}")
    print(f"Search Performance:")
    print(f"  - Average search time: {final_data['avg_search_time']*1000:.2f}ms")
    print(f"  - Searches per second: {final_data['searches_per_second']:.1f}")
    print(f"  - Cache hit rate: {final_data['cache_hit_rate']*100:.1f}%")
    print(f"  - GPU enabled: {final_data['gpu_enabled']}")
    
    # Analyze similarity learning
    sim_learning = final_data['similarity_learning']
    print(f"Similarity Learning:")
    print(f"  - Adaptations performed: {sim_learning.get('adaptations_performed', 0)}")
    print(f"  - Learning rate: {sim_learning.get('learning_rate', 0)}")
    print(f"  - Success correlation: {sim_learning.get('similarity_success_correlation', 0):.3f}")
    
    # Performance trend analysis
    if len(similarity_data) > 1:
        search_times = [d['avg_search_time'] for d in similarity_data[-5:]]  # Last 5 entries
        if len(search_times) > 1:
            trend = "improving" if search_times[-1] < search_times[0] else "degrading"
            print(f"  - Performance trend: {trend}")

def analyze_gpu_usage_patterns(entries: List[Dict]) -> None:
    """Analyze GPU utilization patterns."""
    print(f"\n🚀 GPU UTILIZATION ANALYSIS")
    print("-" * 30)
    
    for entry in entries[-3:]:  # Look at last 3 entries
        exp_count = entry.get('experience_count', 0)
        similarity_stats = entry.get('system_stats', {}).get('similarity_engine', {})
        activation_stats = entry.get('system_stats', {}).get('activation_dynamics', {})
        
        print(f"\nAt {exp_count} experiences:")
        
        # Similarity engine GPU usage
        sim_gpu = similarity_stats.get('gpu_enabled', False)
        sim_device = similarity_stats.get('device', 'unknown')
        print(f"  Similarity: GPU={sim_gpu}, Device={sim_device}")
        
        # Check activation dynamics (if available)
        if 'system_type' in activation_stats:
            act_type = activation_stats.get('system_type', 'unknown')
            print(f"  Activation: Type={act_type}")
        
        # Performance metrics
        sim_time = similarity_stats.get('avg_search_time', 0) * 1000
        print(f"  Performance: {sim_time:.1f}ms avg search")

def analyze_activation_patterns(entries: List[Dict]) -> None:
    """Analyze activation dynamics patterns."""
    print(f"\n🧠 ACTIVATION DYNAMICS ANALYSIS")
    print("-" * 30)
    
    activation_data = []
    for entry in entries:
        act_stats = entry.get('system_stats', {}).get('activation_dynamics', {})
        if act_stats:
            activation_data.append({
                'experience_count': entry.get('experience_count', 0),
                'working_memory_size': act_stats.get('current_working_memory_size', 0),
                'total_activations': act_stats.get('total_activations', 0),
                'avg_utility_score': act_stats.get('avg_utility_score', 0),
                'system_type': act_stats.get('system_type', 'unknown')
            })
    
    if not activation_data:
        print("❌ No activation data found")
        return
    
    final_data = activation_data[-1]
    print(f"Current State:")
    print(f"  - Working memory size: {final_data['working_memory_size']}")
    print(f"  - Total activations: {final_data['total_activations']}")
    print(f"  - System type: {final_data['system_type']}")
    print(f"  - Avg utility score: {final_data['avg_utility_score']:.3f}")
    
    # Working memory efficiency
    exp_count = final_data['experience_count']
    wm_size = final_data['working_memory_size']
    if exp_count > 0:
        wm_ratio = wm_size / exp_count
        print(f"  - WM efficiency: {wm_ratio:.3f} (WM/total experiences)")

def provide_recommendations(entries: List[Dict]) -> None:
    """Provide concrete recommendations based on analysis."""
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 30)
    
    if not entries:
        return
    
    final_entry = entries[-1]
    exp_count = final_entry.get('experience_count', 0)
    sim_stats = final_entry.get('system_stats', {}).get('similarity_engine', {})
    
    # Performance recommendations
    avg_search_time_ms = sim_stats.get('avg_search_time', 0) * 1000
    
    print("📊 SIMILARITY QUALITY ISSUES:")
    sim_learning = sim_stats.get('similarity_learning', {})
    adaptations = sim_learning.get('adaptations_performed', 0)
    correlation = sim_learning.get('similarity_success_correlation', 0)
    
    if adaptations == 0:
        print("  ❌ CRITICAL: No similarity learning adaptations performed")
        print("     → Similarity function may not be learning effectively")
        print("     → All experiences may appear equally similar (poor diversity)")
        print("     → Need to investigate prediction feedback loop")
    
    if correlation < 0.3:
        print(f"  ⚠️ Low similarity-success correlation: {correlation:.3f}")
        print("     → Similar experiences not helping with predictions")
        print("     → May need different similarity features or weights")
    
    print("\n⚡ PERFORMANCE OPTIMIZATION:")
    if avg_search_time_ms > 100:
        print(f"  ❌ Slow similarity search: {avg_search_time_ms:.1f}ms")
        print("     → Should be <50ms for real-time performance")
        print("     → Consider sparse matrix operations")
        print("     → Implement more aggressive GPU batching")
    
    if exp_count > 50 and not sim_stats.get('gpu_enabled', False):
        print("  ⚠️ GPU not being used despite sufficient data size")
        print("     → Check GPU thresholds in hardware adaptation")
    
    cache_hit_rate = sim_stats.get('cache_stats', {}).get('hit_rate', 0)
    if cache_hit_rate < 0.1:
        print(f"  ⚠️ Low cache hit rate: {cache_hit_rate*100:.1f}%")
        print("     → Consider longer cache lifetimes")
        print("     → Implement smarter cache eviction policies")
    
    print("\n🚀 GPU ACCELERATION OPPORTUNITIES:")
    print("  ✅ Implement PyTorch sparse tensors for similarity matrices")
    print("  ✅ Use mixed precision (FP16) for memory efficiency") 
    print("  ✅ Batch similar operations together on GPU")
    print("  ✅ Implement sparse matrix multiplication (SpMM) operations")
    
    print("\n🕸️ SPARSE CONNECTIVITY STRATEGY:")
    if exp_count > 100:
        print("  ✅ Implement k-nearest neighbors (k=5-10)")
        print("  ✅ Use threshold-based filtering (similarity > 0.3)")
        print("  ✅ Consider hierarchical clustering for large datasets")
        print("  ✅ Store connections in COO/CSR sparse matrix format")
    
    print("\n💾 DATA STRUCTURE OPTIMIZATIONS:")
    print("  ✅ Use scipy.sparse matrices for connection storage")
    print("  ✅ Implement PyTorch sparse.FloatTensor for GPU operations")
    print("  ✅ Consider blocked sparse matrices for cache efficiency")
    print("  ✅ Use compressed representations for inactive connections")

def main():
    """Run the performance analysis."""
    print("🔬 BRAIN PERFORMANCE INVESTIGATION")
    print("=" * 50)
    print("Investigating similarity quality, GPU utilization, and sparse connectivity needs...\n")
    
    analyze_log_data()
    
    print(f"\n✅ Analysis complete!")
    print(f"\nKey findings:")
    print(f"1. Similarity learning may not be adapting (check prediction feedback)")
    print(f"2. GPU operations need sparse matrix optimizations")
    print(f"3. Sparse connectivity strategy needed for biological realism")
    print(f"4. Data structures should use sparse matrix formats")

if __name__ == "__main__":
    main()