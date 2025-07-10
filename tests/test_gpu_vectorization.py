#!/usr/bin/env python3
"""
Test GPU vectorization performance improvement.
"""

import time
import sys
sys.path.append('.')

from core.async_brain_server import AsyncBrainServer
from simulation.brainstem_sim import GridWorldBrainstem
from core.gpu_similarity_engine import get_gpu_similarity_engine


def test_gpu_vectorization_performance():
    """Test the performance improvement from GPU vectorization."""
    print("🚀 TESTING GPU VECTORIZATION PERFORMANCE")
    print("=" * 60)
    
    # Initialize brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    brain_server = AsyncBrainServer(brainstem, brainstem.brain_client)
    
    # Get initial performance stats
    print("\n📊 Testing brain performance with GPU vectorization...")
    
    # Start brain server
    brain_server.start()
    
    # Let it run for 10 seconds
    time.sleep(10)
    
    # Get performance stats
    stats = brain_server.get_performance_stats()
    
    print(f"\n🚀 GPU Vectorized Brain Performance:")
    print(f"   Brain FPS: {stats['brain_fps']:.1f}")
    print(f"   Total steps: {stats['step_count']}")
    print(f"   Average step time: {stats['average_frame_time']*1000:.2f}ms")
    
    # Get GPU similarity engine stats
    gpu_engine = get_gpu_similarity_engine()
    gpu_stats = gpu_engine.get_performance_stats()
    
    print(f"\n🎯 GPU Similarity Engine Performance:")
    print(f"   Total queries: {gpu_stats['total_queries']}")
    print(f"   Average query time: {gpu_stats['avg_query_time']*1000:.2f}ms")
    print(f"   GPU time: {gpu_stats['gpu_time']:.2f}s ({gpu_stats['gpu_percentage']:.1f}%)")
    print(f"   CPU time: {gpu_stats['cpu_time']:.2f}s")
    print(f"   Average batch size: {gpu_stats['avg_batch_size']:.1f}")
    print(f"   Cache hit rate: {gpu_stats['cache_hit_rate']:.1f}%")
    print(f"   Device: {gpu_stats['device']}")
    
    # Stop brain server
    brain_server.stop()
    
    return stats['brain_fps'], gpu_stats


def test_similarity_engine_directly():
    """Test the GPU similarity engine directly."""
    print("\n🔬 TESTING GPU SIMILARITY ENGINE DIRECTLY")
    print("=" * 60)
    
    gpu_engine = get_gpu_similarity_engine()
    
    # Test with different batch sizes
    batch_sizes = [100, 500, 1000, 1871]  # 1871 is the current brain size
    
    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")
        
        # Create test data
        target_context = [0.1 * i for i in range(8)]
        candidate_contexts = [
            [0.1 * i + 0.01 * j for i in range(8)]
            for j in range(batch_size)
        ]
        candidate_strengths = [1.0 + j * 0.1 for j in range(batch_size)]
        
        # Time the operation
        start_time = time.perf_counter()
        
        # Run multiple queries
        num_queries = 10
        for _ in range(num_queries):
            best_idx, best_similarity = gpu_engine.find_most_similar_batch(
                target_context, candidate_contexts, candidate_strengths
            )
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_queries
        
        print(f"   Average time: {avg_time*1000:.2f}ms")
        print(f"   Best match: idx={best_idx}, similarity={best_similarity:.3f}")
        print(f"   Effective FPS: {1.0/avg_time:.1f}")


def compare_cpu_vs_gpu():
    """Compare CPU vs GPU performance."""
    print("\n⚡ COMPARING CPU vs GPU PERFORMANCE")
    print("=" * 60)
    
    # Test parameters
    batch_size = 1871  # Current brain size
    num_tests = 20
    
    # Create test data
    target_context = [0.1 * i for i in range(8)]
    candidate_contexts = [
        [0.1 * i + 0.01 * j for i in range(8)]
        for j in range(batch_size)
    ]
    candidate_strengths = [1.0 + j * 0.1 for j in range(batch_size)]
    
    # Test GPU
    gpu_engine = get_gpu_similarity_engine()
    
    gpu_times = []
    for _ in range(num_tests):
        start_time = time.perf_counter()
        gpu_engine.find_most_similar_batch(target_context, candidate_contexts, candidate_strengths)
        end_time = time.perf_counter()
        gpu_times.append(end_time - start_time)
    
    # Test CPU (force CPU mode)
    cpu_engine = get_gpu_similarity_engine()
    cpu_engine.use_gpu = False
    cpu_engine.device = 'cpu'
    
    cpu_times = []
    for _ in range(num_tests):
        start_time = time.perf_counter()
        cpu_engine.find_most_similar_batch(target_context, candidate_contexts, candidate_strengths)
        end_time = time.perf_counter()
        cpu_times.append(end_time - start_time)
    
    # Calculate statistics
    avg_gpu_time = sum(gpu_times) / len(gpu_times)
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
    
    print(f"📊 Performance Comparison ({batch_size} contexts):")
    print(f"   GPU average: {avg_gpu_time*1000:.2f}ms ({1.0/avg_gpu_time:.1f} FPS)")
    print(f"   CPU average: {avg_cpu_time*1000:.2f}ms ({1.0/avg_cpu_time:.1f} FPS)")
    print(f"   Speedup: {speedup:.1f}x faster")
    
    return speedup


def main():
    """Run all GPU vectorization tests."""
    print("🧠 GPU VECTORIZATION PERFORMANCE TESTS")
    print("=" * 80)
    
    # Test 1: Direct similarity engine performance
    test_similarity_engine_directly()
    
    # Test 2: CPU vs GPU comparison
    speedup = compare_cpu_vs_gpu()
    
    # Test 3: Full brain system with GPU vectorization
    brain_fps, gpu_stats = test_gpu_vectorization_performance()
    
    # Final summary
    print("\n🎯 FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Brain FPS (with GPU vectorization): {brain_fps:.1f}")
    print(f"GPU vs CPU speedup: {speedup:.1f}x")
    print(f"GPU similarity queries: {gpu_stats['total_queries']}")
    print(f"GPU percentage of time: {gpu_stats['gpu_percentage']:.1f}%")
    
    # Success criteria
    print(f"\n🏆 SUCCESS CRITERIA:")
    
    if brain_fps > 50:
        print("✅ EXCELLENT: Brain FPS > 50 (massive improvement)")
    elif brain_fps > 30:
        print("✅ GOOD: Brain FPS > 30 (significant improvement)")
    elif brain_fps > 20:
        print("✅ MODERATE: Brain FPS > 20 (some improvement)")
    else:
        print("❌ POOR: Brain FPS still low - needs investigation")
    
    if speedup > 10:
        print("✅ EXCELLENT: GPU speedup > 10x")
    elif speedup > 5:
        print("✅ GOOD: GPU speedup > 5x")
    elif speedup > 2:
        print("✅ MODERATE: GPU speedup > 2x")
    else:
        print("❌ POOR: GPU speedup insufficient")
    
    if gpu_stats['gpu_percentage'] > 80:
        print("✅ EXCELLENT: GPU handling most similarity calculations")
    elif gpu_stats['gpu_percentage'] > 50:
        print("✅ GOOD: GPU handling majority of calculations")
    else:
        print("⚠️  WARNING: GPU not being used effectively")


if __name__ == "__main__":
    main()