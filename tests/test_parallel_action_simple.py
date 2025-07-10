#!/usr/bin/env python3
"""
Simple test for parallel action generation.
"""

import sys
import time
import numpy as np

# Add current directory to path
sys.path.append('.')

from core.parallel_action_generator import ParallelActionGenerator
from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode


def test_simple_parallel_action_generation():
    """Test basic parallel action generation with minimal setup."""
    print("🚀 Testing Simple Parallel Action Generation")
    print("=" * 50)
    
    # Create hybrid world graph
    graph = HybridWorldGraph()
    
    # Add just a few experiences
    print("📊 Adding simple experiences...")
    
    for i in range(10):
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
    
    # Create parallel action generator
    generator = ParallelActionGenerator(graph)
    
    # Test simple generation
    print(f"\n🔍 Testing simple generation...")
    test_context = [0.1, 0.2, 0.3]
    
    start_time = time.time()
    try:
        result = generator.generate_massive_action_candidates(test_context, max_candidates=5)
        generation_time = time.time() - start_time
        
        print(f"   Generation time: {generation_time*1000:.2f}ms")
        print(f"   Candidates generated: {len(result.candidates)}")
        print(f"   Experiences searched: {result.total_experiences_searched}")
        
        # Show first few candidates
        if result.candidates:
            print(f"   First candidate: {result.candidates[0].action}")
            print(f"   Similarity: {result.candidates[0].similarity_score:.3f}")
            print(f"   Quality: {result.candidates[0].get_quality_score():.3f}")
        
        print("✅ Simple parallel action generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_parallel_action_generation()
    sys.exit(0 if success else 1)