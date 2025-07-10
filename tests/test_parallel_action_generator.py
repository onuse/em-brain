#!/usr/bin/env python3
"""
Test Massive Parallel Action Generation System.

This tests the breakthrough algorithmic improvement where the robot can generate
action candidates from its entire life experience using GPU-accelerated parallel search.
"""

import sys
import time
import numpy as np
from typing import List, Dict

# Add current directory to path
sys.path.append('.')

from core.parallel_action_generator import ParallelActionGenerator, ActionCandidate, ActionGenerationResult
from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def test_parallel_action_generator_basic():
    """Test basic parallel action generation functionality."""
    print("🚀 Testing Parallel Action Generator - Basic")
    print("=" * 50)
    
    # Create hybrid world graph
    graph = HybridWorldGraph()
    
    # Add diverse experiences with different action patterns
    print("📊 Adding diverse experiences...")
    
    # Forward motion experiences
    for i in range(30):
        experience = ExperienceNode(
            mental_context=[1.0, 0.0, 0.0],  # Forward context
            action_taken={
                'forward_motor': 0.8 + 0.01 * i,
                'turn_motor': 0.0,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.1, 0.2, 0.3],
            actual_sensory=[0.1 + i * 0.01, 0.2, 0.3],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    # Turn experiences
    for i in range(30):
        experience = ExperienceNode(
            mental_context=[0.0, 1.0, 0.0],  # Turn context
            action_taken={
                'forward_motor': 0.0,
                'turn_motor': 0.8 + 0.01 * i,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.4, 0.5, 0.6],
            actual_sensory=[0.4, 0.5 + i * 0.01, 0.6],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    # Brake experiences
    for i in range(20):
        experience = ExperienceNode(
            mental_context=[0.0, 0.0, 1.0],  # Brake context
            action_taken={
                'forward_motor': 0.0,
                'turn_motor': 0.0,
                'brake_motor': 0.8 + 0.01 * i
            },
            predicted_sensory=[0.7, 0.8, 0.9],
            actual_sensory=[0.7, 0.8, 0.9 + i * 0.01],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    print(f"   Added {graph.vectorized_backend.size} experiences")
    
    # Create parallel action generator
    generator = ParallelActionGenerator(graph)
    
    # Test forward motion context
    print(f"\n🔍 Testing forward motion context...")
    forward_context = [1.0, 0.0, 0.0]
    
    start_time = time.time()
    result = generator.generate_massive_action_candidates(forward_context, max_candidates=50)
    generation_time = time.time() - start_time
    
    print(f"   Generation time: {generation_time*1000:.2f}ms")
    print(f"   Candidates generated: {len(result.candidates)}")
    print(f"   Experiences searched: {result.total_experiences_searched}")
    print(f"   GPU computation time: {result.gpu_computation_time*1000:.2f}ms")
    print(f"   Candidate diversity: {result.candidate_diversity:.3f}")
    print(f"   Context coverage: {result.context_coverage:.3f}")
    
    # Check quality of candidates
    if result.candidates:
        best_candidates = result.get_best_candidates(5)
        print(f"\n   Top 5 candidates:")
        for i, candidate in enumerate(best_candidates):
            print(f"   {i+1}. Action: {candidate.action}")
            print(f"      Quality: {candidate.get_quality_score():.3f}")
            print(f"      Similarity: {candidate.similarity_score:.3f}")
            print(f"      Confidence: {candidate.confidence:.3f}")
    
    # Test turn context
    print(f"\n🔄 Testing turn context...")
    turn_context = [0.0, 1.0, 0.0]
    
    turn_result = generator.generate_massive_action_candidates(turn_context, max_candidates=50)
    print(f"   Turn candidates: {len(turn_result.candidates)}")
    
    if turn_result.candidates:
        best_turn = turn_result.get_best_candidates(3)
        print(f"   Best turn actions:")
        for i, candidate in enumerate(best_turn):
            print(f"   {i+1}. Turn motor: {candidate.action.get('turn_motor', 0):.3f}")
    
    print("✅ Basic parallel action generator test completed!")
    return True


def test_massive_scale_generation():
    """Test massive scale action generation performance."""
    print("\n🚀 Testing Massive Scale Generation")
    print("=" * 40)
    
    # Create large graph
    graph = HybridWorldGraph()
    
    print("📊 Creating large experience dataset...")
    
    # Add many diverse experiences
    for i in range(500):
        # Random context
        context = [np.random.uniform(-1, 1) for _ in range(8)]
        
        # Random action
        action = {
            'forward_motor': np.random.uniform(0, 1),
            'turn_motor': np.random.uniform(-1, 1),
            'brake_motor': np.random.uniform(0, 1)
        }
        
        # Random sensory
        sensory = [np.random.uniform(0, 1) for _ in range(6)]
        
        experience = ExperienceNode(
            mental_context=context,
            action_taken=action,
            predicted_sensory=sensory,
            actual_sensory=[s + np.random.uniform(-0.1, 0.1) for s in sensory],
            prediction_error=np.random.uniform(0, 0.1)
        )
        graph.add_node(experience)
    
    print(f"   Dataset size: {graph.vectorized_backend.size} experiences")
    
    # Create generator
    generator = ParallelActionGenerator(graph)
    
    # Test massive generation
    print(f"\n⚡ Testing massive candidate generation...")
    test_context = [0.5, 0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.6]
    
    start_time = time.time()
    result = generator.generate_massive_action_candidates(test_context, max_candidates=200)
    generation_time = time.time() - start_time
    
    print(f"   Generation time: {generation_time*1000:.2f}ms")
    print(f"   Candidates generated: {len(result.candidates)}")
    print(f"   Candidates per second: {len(result.candidates)/generation_time:.1f}")
    print(f"   Experiences per second: {result.total_experiences_searched/generation_time:.1f}")
    
    # Test diversity
    diverse_candidates = result.get_diverse_candidates(50)
    print(f"   Diverse candidates: {len(diverse_candidates)}")
    
    # Analyze action distribution
    action_stats = analyze_action_distribution(result.candidates)
    print(f"\n📊 Action Distribution Analysis:")
    for action_type, stats in action_stats.items():
        print(f"   {action_type}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    print("✅ Massive scale generation test completed!")
    return True


def test_performance_comparison():
    """Test performance comparison with traditional methods."""
    print("\n⚡ Testing Performance Comparison")
    print("=" * 40)
    
    # Create medium-sized graph
    graph = HybridWorldGraph()
    
    # Add structured experiences
    for i in range(200):
        base_context = [0.1 * i, 0.2 * i, 0.3 * i]
        while len(base_context) < 8:
            base_context.append(np.random.uniform(0, 1))
        
        action = {
            'forward_motor': 0.5 + 0.01 * i,
            'turn_motor': 0.1 + 0.01 * i,
            'brake_motor': 0.05 + 0.01 * i
        }
        
        experience = ExperienceNode(
            mental_context=base_context,
            action_taken=action,
            predicted_sensory=[0.1 * i, 0.2 * i, 0.3 * i],
            actual_sensory=[0.1 * i + 0.01, 0.2 * i + 0.01, 0.3 * i + 0.01],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    print(f"   Test dataset: {graph.vectorized_backend.size} experiences")
    
    # Test parallel generation
    generator = ParallelActionGenerator(graph)
    test_context = [0.5, 0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.6]
    
    # Benchmark massive parallel generation
    print(f"\n🚀 Benchmarking massive parallel generation...")
    parallel_start = time.time()
    
    for _ in range(10):  # Multiple runs for stability
        result = generator.generate_massive_action_candidates(test_context, max_candidates=100)
    
    parallel_time = time.time() - parallel_start
    
    # Benchmark traditional method (if available)
    print(f"💻 Benchmarking traditional method...")
    traditional_start = time.time()
    
    try:
        from core.experience_based_actions import ExperienceBasedActionSystem
        exp_system = ExperienceBasedActionSystem(graph)
        
        traditional_candidates = 0
        for _ in range(10):
            actions = exp_system.generate_experience_based_actions(test_context, max_candidates=20)
            traditional_candidates += len(actions)
        
        traditional_time = time.time() - traditional_start
        
        # Compare results
        print(f"\n📊 Performance Comparison:")
        print(f"   Parallel generation: {parallel_time*1000:.2f}ms (avg: {parallel_time*100:.2f}ms)")
        print(f"   Traditional method: {traditional_time*1000:.2f}ms (avg: {traditional_time*100:.2f}ms)")
        print(f"   Speedup: {traditional_time/parallel_time:.1f}x")
        print(f"   Parallel candidates: {len(result.candidates)} per generation")
        print(f"   Traditional candidates: {traditional_candidates/10:.1f} per generation")
        print(f"   Candidate increase: {len(result.candidates)/(traditional_candidates/10):.1f}x")
        
    except ImportError:
        print("   Traditional method not available for comparison")
    
    # Performance statistics
    stats = generator.get_performance_stats()
    print(f"\n🎯 Generator Performance Stats:")
    print(f"   Total generations: {stats['generation_count']}")
    print(f"   Average time: {stats['avg_generation_time']*1000:.2f}ms")
    print(f"   Average candidates: {stats['avg_candidates_per_generation']:.1f}")
    print(f"   Candidates per second: {stats['candidates_per_second']:.1f}")
    print(f"   Device: {stats['device']}")
    
    print("✅ Performance comparison test completed!")
    return True


def test_action_quality_analysis():
    """Test quality and relevance of generated actions."""
    print("\n🎯 Testing Action Quality Analysis")
    print("=" * 40)
    
    # Create graph with known patterns
    graph = HybridWorldGraph()
    
    # Add specific patterns
    print("📊 Adding patterned experiences...")
    
    # Pattern 1: High forward speed in open space
    for i in range(50):
        experience = ExperienceNode(
            mental_context=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Open space
            action_taken={
                'forward_motor': 0.8 + 0.01 * i,
                'turn_motor': 0.0,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            actual_sensory=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            prediction_error=0.01  # Good predictions
        )
        graph.add_node(experience)
    
    # Pattern 2: Braking near obstacles
    for i in range(30):
        experience = ExperienceNode(
            mental_context=[0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Obstacle detected
            action_taken={
                'forward_motor': 0.0,
                'turn_motor': 0.0,
                'brake_motor': 0.8 + 0.01 * i
            },
            predicted_sensory=[0.7, 0.8, 0.9, 0.5, 0.4, 0.3],
            actual_sensory=[0.7, 0.8, 0.9, 0.5, 0.4, 0.3],
            prediction_error=0.01  # Good predictions
        )
        graph.add_node(experience)
    
    print(f"   Added {graph.vectorized_backend.size} patterned experiences")
    
    # Test action generation
    generator = ParallelActionGenerator(graph)
    
    # Test in open space context
    print(f"\n🔍 Testing open space context...")
    open_context = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    open_result = generator.generate_massive_action_candidates(open_context, max_candidates=30)
    
    print(f"   Candidates generated: {len(open_result.candidates)}")
    
    # Analyze action relevance
    forward_actions = [c for c in open_result.candidates if c.action.get('forward_motor', 0) > 0.5]
    brake_actions = [c for c in open_result.candidates if c.action.get('brake_motor', 0) > 0.5]
    
    print(f"   Forward actions: {len(forward_actions)} (expected: high)")
    print(f"   Brake actions: {len(brake_actions)} (expected: low)")
    
    # Test in obstacle context
    print(f"\n🚧 Testing obstacle context...")
    obstacle_context = [0.1, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    obstacle_result = generator.generate_massive_action_candidates(obstacle_context, max_candidates=30)
    
    forward_actions_obs = [c for c in obstacle_result.candidates if c.action.get('forward_motor', 0) > 0.5]
    brake_actions_obs = [c for c in obstacle_result.candidates if c.action.get('brake_motor', 0) > 0.5]
    
    print(f"   Forward actions: {len(forward_actions_obs)} (expected: low)")
    print(f"   Brake actions: {len(brake_actions_obs)} (expected: high)")
    
    # Quality analysis
    print(f"\n📊 Quality Analysis:")
    if open_result.candidates:
        best_open = open_result.get_best_candidates(5)
        avg_quality_open = np.mean([c.get_quality_score() for c in best_open])
        print(f"   Open space avg quality: {avg_quality_open:.3f}")
    
    if obstacle_result.candidates:
        best_obstacle = obstacle_result.get_best_candidates(5)
        avg_quality_obstacle = np.mean([c.get_quality_score() for c in best_obstacle])
        print(f"   Obstacle avg quality: {avg_quality_obstacle:.3f}")
    
    print("✅ Action quality analysis test completed!")
    return True


def analyze_action_distribution(candidates: List[ActionCandidate]) -> Dict[str, Dict[str, float]]:
    """Analyze the distribution of actions in candidates."""
    action_stats = {}
    
    # Collect all action values by type
    action_values = {}
    for candidate in candidates:
        for action_type, value in candidate.action.items():
            if action_type not in action_values:
                action_values[action_type] = []
            action_values[action_type].append(value)
    
    # Calculate statistics
    for action_type, values in action_values.items():
        if values:
            action_stats[action_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return action_stats


def benchmark_generator():
    """Comprehensive benchmark of parallel action generator."""
    print("\n📊 Comprehensive Parallel Action Generator Benchmark")
    print("=" * 60)
    
    # Create large graph
    graph = HybridWorldGraph()
    
    print("📊 Creating benchmark dataset...")
    for i in range(1000):
        context = [np.random.uniform(-1, 1) for _ in range(8)]
        action = {
            'forward_motor': np.random.uniform(0, 1),
            'turn_motor': np.random.uniform(-1, 1),
            'brake_motor': np.random.uniform(0, 1)
        }
        sensory = [np.random.uniform(0, 1) for _ in range(6)]
        
        experience = ExperienceNode(
            mental_context=context,
            action_taken=action,
            predicted_sensory=sensory,
            actual_sensory=[s + np.random.uniform(-0.05, 0.05) for s in sensory],
            prediction_error=np.random.uniform(0, 0.05)
        )
        graph.add_node(experience)
    
    print(f"   Dataset size: {graph.vectorized_backend.size} experiences")
    
    # Create generator
    generator = ParallelActionGenerator(graph)
    
    # Run comprehensive benchmark
    print(f"\n🚀 Running comprehensive benchmark...")
    benchmark_results = generator.benchmark_generation_performance(num_tests=20)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Tests run: {benchmark_results['num_tests']}")
    print(f"   Total time: {benchmark_results['total_time']:.3f}s")
    print(f"   Average time per generation: {benchmark_results['avg_time_per_generation']*1000:.2f}ms")
    print(f"   Total candidates: {benchmark_results['total_candidates']}")
    print(f"   Average candidates per generation: {benchmark_results['avg_candidates_per_generation']:.1f}")
    print(f"   Candidates per second: {benchmark_results['candidates_per_second']:.1f}")
    print(f"   Experiences searched per test: {benchmark_results['experiences_searched_per_test']}")
    
    # Test different candidate limits
    print(f"\n🔄 Testing different candidate limits...")
    candidate_limits = [50, 100, 200, 500]
    
    for limit in candidate_limits:
        test_context = [np.random.uniform(-1, 1) for _ in range(8)]
        
        start_time = time.time()
        result = generator.generate_massive_action_candidates(test_context, max_candidates=limit)
        generation_time = time.time() - start_time
        
        print(f"   Limit {limit}: {generation_time*1000:.2f}ms, {len(result.candidates)} candidates")
    
    # Final performance stats
    final_stats = generator.get_performance_stats()
    print(f"\n🎯 Final Performance Stats:")
    print(f"   Total generations: {final_stats['generation_count']}")
    print(f"   Total GPU time: {final_stats['total_gpu_time']:.3f}s")
    print(f"   Average generation time: {final_stats['avg_generation_time']*1000:.2f}ms")
    print(f"   Average candidates per generation: {final_stats['avg_candidates_per_generation']:.1f}")
    print(f"   Total candidates generated: {final_stats['total_candidates_generated']}")
    print(f"   Device: {final_stats['device']}")
    
    print("✅ Comprehensive benchmark completed!")
    return True


def main():
    """Run all parallel action generator tests."""
    print("🚀 PARALLEL ACTION GENERATOR TESTS")
    print("=" * 80)
    
    try:
        # Test basic functionality
        test_parallel_action_generator_basic()
        
        # Test massive scale
        test_massive_scale_generation()
        
        # Test performance comparison
        test_performance_comparison()
        
        # Test action quality
        test_action_quality_analysis()
        
        # Comprehensive benchmark
        benchmark_generator()
        
        print("\n🎉 All parallel action generator tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)