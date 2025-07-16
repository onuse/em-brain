#!/usr/bin/env python3
"""
Ten Thousand Node Challenge

Tests what happens when we scale to 10k+ experiences and identifies the
specific bottlenecks that prevent efficient scaling.
"""

import sys
import os
import time
import math
import numpy as np
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

class TenThousandNodeChallenge:
    """Test scaling to 10k+ experiences and identify bottlenecks."""
    
    def __init__(self):
        self.results = {}
        
    def run_challenge(self):
        """Run the 10k node scaling challenge."""
        print("🚀 TEN THOUSAND NODE CHALLENGE")
        print("=" * 60)
        print("Testing brain scaling to 10k+ experiences")
        print("Target: 10k experiences processable in <100ms")
        print()
        
        # Test 1: Profile current scaling behavior
        print("1. PROFILING CURRENT SCALING BEHAVIOR")
        print("-" * 40)
        self.profile_scaling_behavior()
        
        # Test 2: Identify specific bottlenecks
        print("\n2. IDENTIFYING SCALING BOTTLENECKS")
        print("-" * 40)
        self.identify_scaling_bottlenecks()
        
        # Test 3: Test sparse connectivity effectiveness
        print("\n3. TESTING SPARSE CONNECTIVITY")
        print("-" * 40)
        self.test_sparse_connectivity()
        
        # Test 4: Memory usage analysis
        print("\n4. MEMORY USAGE ANALYSIS")
        print("-" * 40)
        self.analyze_memory_usage()
        
        # Generate scaling recommendations
        print("\n5. SCALING RECOMMENDATIONS")
        print("-" * 40)
        self.generate_scaling_recommendations()
    
    def profile_scaling_behavior(self):
        """Profile how performance scales with experience count."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=False,
            quiet_mode=True
        )
        
        # Test scaling at different sizes
        test_sizes = [100, 500, 1000, 2000, 5000]
        scaling_data = []
        
        for target_size in test_sizes:
            print(f"   Testing {target_size} experiences...")
            
            # Add experiences up to target size
            start_time = time.time()
            for i in range(len(brain.experience_storage._experiences), target_size):
                # Create diverse experiences
                angle = i * 0.1
                sensory = [
                    0.5 + 0.3 * math.sin(angle),
                    0.5 + 0.3 * math.cos(angle * 1.1),
                    0.5 + 0.2 * math.sin(angle * 0.7),
                    0.5 + 0.2 * math.cos(angle * 1.3)
                ]
                predicted_action, _ = brain.process_sensory_input(sensory)
                outcome = [a * 0.9 + 0.05 for a in predicted_action]
                brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            
            setup_time = time.time() - start_time
            
            # Test processing time
            test_sensory = [0.5, 0.4, 0.6, 0.3]
            
            # Time just the similarity search
            similarity_start = time.time()
            experience_vectors = [exp.get_context_vector() for exp in brain.experience_storage._experiences.values()]
            experience_ids = list(brain.experience_storage._experiences.keys())
            similar_experiences = brain.similarity_engine.find_similar_experiences(
                test_sensory, experience_vectors, experience_ids, max_results=10
            )
            similarity_time = (time.time() - similarity_start) * 1000
            
            # Time full cycle
            cycle_start = time.time()
            predicted_action, brain_state = brain.process_sensory_input(test_sensory)
            cycle_time = (time.time() - cycle_start) * 1000
            
            # Get memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            scaling_data.append({
                'size': target_size,
                'similarity_time': similarity_time,
                'cycle_time': cycle_time,
                'memory_mb': memory_mb,
                'similar_found': len(similar_experiences)
            })
            
            print(f"     Similarity search: {similarity_time:.1f}ms")
            print(f"     Full cycle: {cycle_time:.1f}ms")
            print(f"     Memory usage: {memory_mb:.1f}MB")
            print(f"     Similar experiences found: {len(similar_experiences)}")
            
            # Stop if too slow
            if cycle_time > 500:
                print(f"     ❌ STOPPING: Too slow at {target_size} experiences")
                break
        
        self.results['scaling_data'] = scaling_data
        brain.finalize_session()
        
        # Analyze scaling behavior
        if len(scaling_data) >= 2:
            first = scaling_data[0]
            last = scaling_data[-1]
            
            size_ratio = last['size'] / first['size']
            time_ratio = last['cycle_time'] / first['cycle_time']
            
            print(f"\n   📊 SCALING ANALYSIS:")
            print(f"   Experience ratio: {size_ratio:.1f}x")
            print(f"   Time ratio: {time_ratio:.1f}x")
            print(f"   Scaling: {'✅ Sub-linear' if time_ratio < size_ratio else '❌ Linear or worse'}")
            
            # Project to 10k
            projected_10k = first['cycle_time'] * (10000 / first['size']) * (time_ratio / size_ratio)
            print(f"   📈 Projected 10k performance: {projected_10k:.1f}ms")
    
    def identify_scaling_bottlenecks(self):
        """Identify specific bottlenecks that prevent scaling."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=False,
            quiet_mode=True
        )
        
        # Create a large dataset for profiling
        print("   Creating 2000 experiences for bottleneck analysis...")
        for i in range(2000):
            angle = i * 0.1
            sensory = [
                0.5 + 0.3 * math.sin(angle),
                0.5 + 0.3 * math.cos(angle * 1.1),
                0.5 + 0.2 * math.sin(angle * 0.7),
                0.5 + 0.2 * math.cos(angle * 1.3)
            ]
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Profile individual components
        test_sensory = [0.5, 0.4, 0.6, 0.3]
        
        # Time similarity search
        start_time = time.time()
        experience_vectors = [exp.get_context_vector() for exp in brain.experience_storage._experiences.values()]
        experience_ids = list(brain.experience_storage._experiences.keys())
        vector_creation_time = (time.time() - start_time) * 1000
        
        start_time = time.time()
        similar_experiences = brain.similarity_engine.find_similar_experiences(
            test_sensory, experience_vectors, experience_ids, max_results=10
        )
        similarity_search_time = (time.time() - start_time) * 1000
        
        # Time activation system
        start_time = time.time()
        brain._activate_by_utility(test_sensory)
        activation_time = (time.time() - start_time) * 1000
        
        # Time prediction
        start_time = time.time()
        brain_state = {'prediction_confidence': 0.5, 'num_experiences': len(brain.experience_storage._experiences)}
        predicted_action, confidence, details = brain.prediction_engine.predict_action(
            test_sensory, brain.similarity_engine, brain.activation_dynamics,
            brain.experience_storage._experiences, 4, brain_state
        )
        prediction_time = (time.time() - start_time) * 1000
        
        print(f"   🔍 BOTTLENECK ANALYSIS (2000 experiences):")
        print(f"   Vector creation: {vector_creation_time:.1f}ms")
        print(f"   Similarity search: {similarity_search_time:.1f}ms")
        print(f"   Activation system: {activation_time:.1f}ms")
        print(f"   Prediction engine: {prediction_time:.1f}ms")
        
        total_time = vector_creation_time + similarity_search_time + activation_time + prediction_time
        print(f"   Total measured: {total_time:.1f}ms")
        
        # Identify worst bottleneck
        bottlenecks = [
            ('vector_creation', vector_creation_time),
            ('similarity_search', similarity_search_time),
            ('activation_system', activation_time),
            ('prediction_engine', prediction_time)
        ]
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   🎯 WORST BOTTLENECK: {bottlenecks[0][0]} ({bottlenecks[0][1]:.1f}ms)")
        
        brain.finalize_session()
    
    def test_sparse_connectivity(self):
        """Test if sparse connectivity is actually working."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=False,
            quiet_mode=True
        )
        
        # Add experiences
        for i in range(1000):
            angle = i * 0.1
            sensory = [
                0.5 + 0.3 * math.sin(angle),
                0.5 + 0.3 * math.cos(angle * 1.1),
                0.5 + 0.2 * math.sin(angle * 0.7),
                0.5 + 0.2 * math.cos(angle * 1.3)
            ]
            predicted_action, _ = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Test similarity search results
        test_sensory = [0.5, 0.4, 0.6, 0.3]
        experience_vectors = [exp.get_context_vector() for exp in brain.experience_storage._experiences.values()]
        experience_ids = list(brain.experience_storage._experiences.keys())
        
        similar_experiences = brain.similarity_engine.find_similar_experiences(
            test_sensory, experience_vectors, experience_ids, max_results=10
        )
        
        print(f"   📊 SPARSE CONNECTIVITY TEST (1000 experiences):")
        print(f"   Total experiences: {len(brain.experience_storage._experiences)}")
        print(f"   Similar experiences found: {len(similar_experiences)}")
        print(f"   Connectivity ratio: {len(similar_experiences) / len(brain.experience_storage._experiences) * 100:.1f}%")
        
        if len(similar_experiences) > 0:
            similarities = [sim for _, sim in similar_experiences]
            print(f"   Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
        
        brain.finalize_session()
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns for scaling."""
        import psutil
        
        # Test memory usage at different scales
        sizes = [1000, 2000, 5000]
        memory_data = []
        
        for size in sizes:
            brain = MinimalBrain(
                enable_logging=False,
                enable_persistence=False,
                enable_storage_optimization=True,
                use_utility_based_activation=True,
                enable_phase2_adaptations=False,
                quiet_mode=True
            )
            
            # Add experiences
            for i in range(size):
                angle = i * 0.1
                sensory = [
                    0.5 + 0.3 * math.sin(angle),
                    0.5 + 0.3 * math.cos(angle * 1.1),
                    0.5 + 0.2 * math.sin(angle * 0.7),
                    0.5 + 0.2 * math.cos(angle * 1.3)
                ]
                predicted_action, _ = brain.process_sensory_input(sensory)
                outcome = [a * 0.9 + 0.05 for a in predicted_action]
                brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            
            # Measure memory
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            memory_data.append({
                'size': size,
                'memory_mb': memory_mb,
                'mb_per_experience': memory_mb / size
            })
            
            print(f"   {size} experiences: {memory_mb:.1f}MB ({memory_mb/size:.3f}MB/exp)")
            
            brain.finalize_session()
        
        # Project to 10k
        if len(memory_data) >= 2:
            avg_mb_per_exp = sum(d['mb_per_experience'] for d in memory_data) / len(memory_data)
            projected_10k_mb = avg_mb_per_exp * 10000
            print(f"   📈 Projected 10k memory: {projected_10k_mb:.1f}MB")
    
    def generate_scaling_recommendations(self):
        """Generate recommendations for scaling to 10k+ experiences."""
        print("   🎯 SCALING RECOMMENDATIONS FOR 10K+ EXPERIENCES:")
        print()
        
        print("   📈 ALGORITHMIC CHANGES NEEDED:")
        print("   1. Hierarchical Indexing:")
        print("      - Use k-d trees or LSH for similarity search")
        print("      - Cluster experiences into regions")
        print("      - Only search relevant clusters")
        print()
        
        print("   2. Approximate Similarity Search:")
        print("      - Use approximate nearest neighbor (ANN) algorithms")
        print("      - Facebook's Faiss library for GPU-accelerated search")
        print("      - Trade small accuracy for massive speed gains")
        print()
        
        print("   3. Lazy Loading:")
        print("      - Don't load all experiences into memory")
        print("      - Use memory-mapped storage")
        print("      - Load only active working set")
        print()
        
        print("   4. Time-Bounded Processing:")
        print("      - Implement your 400ms time budget concept")
        print("      - Use anytime algorithms that improve with more time")
        print("      - Return best answer found within time limit")
        print()
        
        print("   💾 STORAGE OPTIMIZATIONS:")
        print("   - Use compressed vector storage")
        print("   - Implement experience clustering/compression")
        print("   - Use sparse vector representations")
        print()
        
        print("   🚀 GPU ACCELERATION:")
        print("   - Batch all similarity computations")
        print("   - Use GPU-optimized similarity metrics")
        print("   - Implement parallel activation updates")
        print()
        
        print("   ⚡ IMMEDIATE NEXT STEPS:")
        print("   1. Implement k-NN indexing for similarity search")
        print("   2. Add time-bounded processing framework")
        print("   3. Test with approximate similarity algorithms")
        print("   4. Implement hierarchical experience clustering")

def main():
    """Run the ten thousand node challenge."""
    challenge = TenThousandNodeChallenge()
    challenge.run_challenge()

if __name__ == "__main__":
    main()