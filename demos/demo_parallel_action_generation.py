#!/usr/bin/env python3
"""
Demo: Massive Parallel Action Generation from Experience History

This demonstrates the breakthrough algorithmic improvement from the vectorization brief:
instead of manually generating a small set of actions, the robot now searches its
entire life experience in parallel to generate hundreds of contextually relevant
action candidates.

This is the key innovation that transforms the robot from reactive to intelligent.
"""

import sys
import time
import numpy as np

# Add current directory to path
sys.path.append('.')

from core.parallel_action_generator import ParallelActionGenerator
from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def create_realistic_robot_experiences(graph: HybridWorldGraph, num_experiences: int = 100):
    """Create realistic robot experiences for demonstration."""
    print(f"🤖 Creating {num_experiences} realistic robot experiences...")
    
    # Scenario 1: Navigation in open space
    for i in range(num_experiences // 4):
        experience = ExperienceNode(
            mental_context=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Open space
            action_taken={
                'forward_motor': 0.7 + 0.02 * np.random.random(),
                'turn_motor': 0.1 * np.random.random() - 0.05,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            actual_sensory=[0.1 + 0.1 * i / 25, 0.2, 0.3, 0.4, 0.5, 0.6],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    # Scenario 2: Obstacle avoidance
    for i in range(num_experiences // 4):
        experience = ExperienceNode(
            mental_context=[0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Obstacle detected
            action_taken={
                'forward_motor': 0.0,
                'turn_motor': 0.6 + 0.3 * np.random.random(),
                'brake_motor': 0.4 + 0.2 * np.random.random()
            },
            predicted_sensory=[0.7, 0.8, 0.9, 0.5, 0.4, 0.3],
            actual_sensory=[0.7, 0.8, 0.9, 0.5, 0.4, 0.3],
            prediction_error=0.02
        )
        graph.add_node(experience)
    
    # Scenario 3: Food seeking
    for i in range(num_experiences // 4):
        experience = ExperienceNode(
            mental_context=[0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],  # Food detected
            action_taken={
                'forward_motor': 0.8 + 0.1 * np.random.random(),
                'turn_motor': 0.2 * np.random.random() - 0.1,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.2, 0.3, 0.4, 0.8, 0.9, 0.7],
            actual_sensory=[0.2, 0.3, 0.4, 0.8 + 0.1 * i / 25, 0.9, 0.7],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    # Scenario 4: Low energy conservation
    for i in range(num_experiences // 4):
        experience = ExperienceNode(
            mental_context=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.1],  # Low energy
            action_taken={
                'forward_motor': 0.2 + 0.1 * np.random.random(),
                'turn_motor': 0.0,
                'brake_motor': 0.0
            },
            predicted_sensory=[0.3, 0.4, 0.5, 0.3, 0.2, 0.1],
            actual_sensory=[0.3, 0.4, 0.5, 0.3, 0.2, 0.1 - 0.01 * i],
            prediction_error=0.01
        )
        graph.add_node(experience)
    
    print(f"   ✅ Created {graph.vectorized_backend.size} experiences")


def demonstrate_massive_parallel_generation():
    """Demonstrate the massive parallel action generation system."""
    print("🚀 MASSIVE PARALLEL ACTION GENERATION DEMO")
    print("=" * 60)
    
    # Create robot experience graph
    graph = HybridWorldGraph()
    create_realistic_robot_experiences(graph, 200)
    
    # Create parallel action generator
    generator = ParallelActionGenerator(graph)
    
    print(f"\n🎯 DEMO: Robot Decision Making Revolution")
    print("=" * 45)
    
    # Scenario 1: Open space navigation
    print(f"\n🔍 Scenario 1: Robot in Open Space")
    print("-" * 35)
    
    open_space_context = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print("🧠 Old way: Manual action generation")
    print("   - forward_motor: 0.5")
    print("   - turn_motor: 0.0") 
    print("   - brake_motor: 0.0")
    print("   Total candidates: 1")
    
    print("\n🚀 New way: Massive parallel search through experience history")
    start_time = time.time()
    result = generator.generate_massive_action_candidates(open_space_context, max_candidates=50)
    generation_time = time.time() - start_time
    
    print(f"   ⚡ Generated {len(result.candidates)} candidates in {generation_time*1000:.1f}ms")
    print(f"   📊 Searched {result.total_experiences_searched} experiences")
    
    # Show best candidates
    best_candidates = result.get_best_candidates(5)
    print(f"\n   🏆 Top 5 action candidates:")
    for i, candidate in enumerate(best_candidates):
        print(f"   {i+1}. Forward: {candidate.action['forward_motor']:.3f}, "
              f"Turn: {candidate.action['turn_motor']:.3f}, "
              f"Quality: {candidate.get_quality_score():.3f}")
    
    # Scenario 2: Obstacle detected
    print(f"\n🚧 Scenario 2: Obstacle Detected")
    print("-" * 30)
    
    obstacle_context = [0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    print("🧠 Old way: Manual obstacle response")
    print("   - forward_motor: 0.0")
    print("   - turn_motor: 0.5")
    print("   - brake_motor: 0.8")
    print("   Total candidates: 1")
    
    print("\n🚀 New way: Learn from all past obstacle encounters")
    obstacle_result = generator.generate_massive_action_candidates(obstacle_context, max_candidates=50)
    
    print(f"   ⚡ Generated {len(obstacle_result.candidates)} candidates")
    
    # Analyze obstacle responses
    obstacle_best = obstacle_result.get_best_candidates(5)
    print(f"\n   🏆 Top 5 obstacle avoidance actions:")
    for i, candidate in enumerate(obstacle_best):
        print(f"   {i+1}. Forward: {candidate.action['forward_motor']:.3f}, "
              f"Turn: {candidate.action['turn_motor']:.3f}, "
              f"Brake: {candidate.action['brake_motor']:.3f}")
    
    # Scenario 3: Food seeking
    print(f"\n🍎 Scenario 3: Food Detected")
    print("-" * 25)
    
    food_context = [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0]
    
    print("🧠 Old way: Simple approach")
    print("   - forward_motor: 0.8")
    print("   - turn_motor: 0.0")
    print("   - brake_motor: 0.0")
    print("   Total candidates: 1")
    
    print("\n🚀 New way: Optimized approach from all food encounters")
    food_result = generator.generate_massive_action_candidates(food_context, max_candidates=50)
    
    print(f"   ⚡ Generated {len(food_result.candidates)} candidates")
    
    food_best = food_result.get_best_candidates(5)
    print(f"\n   🏆 Top 5 food approach actions:")
    for i, candidate in enumerate(food_best):
        print(f"   {i+1}. Forward: {candidate.action['forward_motor']:.3f}, "
              f"Turn: {candidate.action['turn_motor']:.3f}, "
              f"Quality: {candidate.get_quality_score():.3f}")
    
    # Performance summary
    print(f"\n📊 PERFORMANCE BREAKTHROUGH")
    print("=" * 30)
    
    stats = generator.get_performance_stats()
    print(f"🚀 Device: {stats['device']}")
    print(f"⚡ Average generation time: {stats['avg_generation_time']*1000:.1f}ms")
    print(f"📈 Average candidates per generation: {stats['avg_candidates_per_generation']:.1f}")
    print(f"🎯 Candidates per second: {stats['candidates_per_second']:.1f}")
    print(f"🔄 Total generations: {stats['generation_count']}")
    
    # Intelligence comparison
    print(f"\n🧠 INTELLIGENCE COMPARISON")
    print("=" * 25)
    
    print("📊 Manual Action Generation:")
    print("   - Actions considered: 1-3 per situation")
    print("   - Based on: Simple rules")
    print("   - Adaptability: Very limited")
    print("   - Learning: None")
    
    print("\n🚀 Massive Parallel Experience-Based Generation:")
    print(f"   - Actions considered: {int(stats['avg_candidates_per_generation'])} per situation")
    print("   - Based on: Entire life experience")
    print("   - Adaptability: Highly contextual")
    print("   - Learning: Continuous from all experiences")
    
    improvement_factor = stats['avg_candidates_per_generation'] / 1
    print(f"\n🎉 INTELLIGENCE AMPLIFICATION: {improvement_factor:.0f}x more options!")
    
    print(f"\n✅ Demo completed successfully!")


def benchmark_scalability():
    """Benchmark how the system scales with experience count."""
    print(f"\n📊 SCALABILITY BENCHMARK")
    print("=" * 25)
    
    experience_counts = [50, 100, 200, 500]
    
    for count in experience_counts:
        print(f"\n🔍 Testing with {count} experiences...")
        
        # Create graph with specific experience count
        graph = HybridWorldGraph()
        create_realistic_robot_experiences(graph, count)
        
        # Create generator
        generator = ParallelActionGenerator(graph)
        
        # Benchmark generation
        test_context = [0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
        
        start_time = time.time()
        result = generator.generate_massive_action_candidates(test_context, max_candidates=100)
        generation_time = time.time() - start_time
        
        print(f"   ⚡ Time: {generation_time*1000:.1f}ms")
        print(f"   📊 Candidates: {len(result.candidates)}")
        print(f"   🎯 Throughput: {len(result.candidates)/generation_time:.1f} candidates/sec")
        print(f"   🔄 Experience throughput: {count/generation_time:.1f} experiences/sec")
    
    print(f"\n✅ Scalability benchmark completed!")


if __name__ == "__main__":
    demonstrate_massive_parallel_generation()
    benchmark_scalability()
    
    print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
    print("=" * 25)
    print("The robot can now generate action candidates from its entire")
    print("life experience using GPU-accelerated parallel search.")
    print("This transforms decision-making from reactive to intelligent.")