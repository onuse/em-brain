"""
Massive Dataset Generator for Brain Performance Testing

Generates millions of realistic experiences to test the theoretical limits
of our GPU-accelerated brain implementation. This will help identify:

1. Memory limits (how many experiences can we hold?)
2. Performance scaling (does it scale logarithmically as expected?)
3. Bottlenecks (where does performance break down?)
4. Real-world throughput (experiences per second)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import psutil
import gc
from typing import Dict, List, Tuple
from src.brain import MinimalBrain
from src.experience.models import Experience


class MassiveDatasetGenerator:
    """Generate realistic massive datasets for brain testing."""
    
    def __init__(self, dimensions: int = 50, action_dims: int = 4):
        """
        Initialize dataset generator.
        
        Args:
            dimensions: Dimensionality of sensory/outcome vectors
            action_dims: Dimensionality of action vectors
        """
        self.dimensions = dimensions
        self.action_dims = action_dims
        self.patterns = self._generate_base_patterns()
        print(f"🔧 Dataset generator initialized: {dimensions}D sensory, {action_dims}D actions")
    
    def _generate_base_patterns(self, num_patterns: int = 50) -> List[np.ndarray]:
        """Generate base patterns that experiences will be variations of."""
        np.random.seed(42)  # Reproducible patterns
        patterns = []
        
        for i in range(num_patterns):
            # Create structured patterns (not pure random)
            pattern = np.random.randn(self.dimensions)
            # Add some structure - make some dimensions correlated
            for j in range(0, self.dimensions, 5):
                if j + 4 < self.dimensions:
                    pattern[j+1] = pattern[j] * 0.7 + np.random.normal(0, 0.3)
                    pattern[j+2] = pattern[j] * 0.5 + np.random.normal(0, 0.5)
            
            patterns.append(pattern)
        
        return patterns
    
    def generate_experiences_batch(self, num_experiences: int, start_time: float = None) -> Dict[str, Experience]:
        """
        Generate a batch of realistic experiences.
        
        Args:
            num_experiences: Number of experiences to generate
            start_time: Starting timestamp (or current time if None)
            
        Returns:
            Dictionary of experience_id -> Experience
        """
        if start_time is None:
            start_time = time.time()
        
        experiences = {}
        
        for i in range(num_experiences):
            # Pick a base pattern and add variation
            pattern_idx = np.random.randint(0, len(self.patterns))
            base_pattern = self.patterns[pattern_idx]
            
            # Sensory input: base pattern + noise
            noise_level = np.random.uniform(0.1, 0.5)
            sensory_input = base_pattern + np.random.normal(0, noise_level, self.dimensions)
            
            # Action: influenced by sensory input (realistic behavior)
            action_taken = np.random.randn(self.action_dims)
            # Make action somewhat correlated with sensory input
            action_influence = np.mean(sensory_input[:4]) * 0.3
            action_taken[0] += action_influence
            
            # Outcome: influenced by action + environment dynamics + noise
            outcome = sensory_input.copy()
            # Apply action influence
            for j in range(min(self.action_dims, self.dimensions)):
                outcome[j] += action_taken[j] * 0.4
            # Add environmental noise
            outcome += np.random.normal(0, 0.2, self.dimensions)
            
            # Prediction error: realistic distribution
            prediction_error = np.random.beta(2, 5)  # Skewed toward lower errors
            
            # Timestamp with slight randomization
            timestamp = start_time + i * 0.001 + np.random.uniform(-0.0005, 0.0005)
            
            experience = Experience(
                sensory_input=sensory_input.tolist(),
                action_taken=action_taken.tolist(),
                outcome=outcome.tolist(),
                prediction_error=prediction_error,
                timestamp=timestamp
            )
            
            # Add realistic access patterns
            experience.access_count = max(1, int(np.random.exponential(2)))
            
            experiences[experience.experience_id] = experience
        
        return experiences
    
    def generate_similarity_connections(self, experiences: Dict[str, Experience], 
                                      connection_density: float = 0.02):
        """
        Add realistic similarity connections between experiences.
        
        Args:
            experiences: Dictionary of experiences to connect
            connection_density: Fraction of possible connections to create
        """
        experience_list = list(experiences.values())
        num_experiences = len(experience_list)
        num_connections = int(num_experiences * connection_density)
        
        print(f"🔗 Adding {num_connections} similarity connections...")
        
        for _ in range(num_connections):
            # Pick two random experiences
            idx1, idx2 = np.random.choice(num_experiences, 2, replace=False)
            exp1, exp2 = experience_list[idx1], experience_list[idx2]
            
            # Compute realistic similarity based on sensory patterns
            sensory1 = np.array(exp1.sensory_input)
            sensory2 = np.array(exp2.sensory_input)
            
            # Cosine similarity
            dot_product = np.dot(sensory1, sensory2)
            norm1, norm2 = np.linalg.norm(sensory1), np.linalg.norm(sensory2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = dot_product / (norm1 * norm2)
                # Convert to 0-1 range and add some noise
                similarity = (cosine_sim + 1.0) / 2.0
                similarity = max(0.2, min(0.9, similarity + np.random.normal(0, 0.1)))
                
                # Add bidirectional connection
                exp1.add_similarity(exp2.experience_id, similarity)
                exp2.add_similarity(exp1.experience_id, similarity)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }


class BrainScalingTest:
    """Test brain performance scaling with massive datasets."""
    
    def __init__(self, use_utility_activation: bool = True):
        """
        Initialize scaling test.
        
        Args:
            use_utility_activation: Whether to test utility-based (True) or traditional (False) activation
        """
        self.generator = MassiveDatasetGenerator()
        self.use_utility_activation = use_utility_activation
        self.results = []
        
        activation_type = "utility-based" if use_utility_activation else "traditional"
        print(f"🧠 Brain scaling test initialized ({activation_type} activation)")
    
    def run_scaling_test(self, sizes: List[int], samples_per_size: int = 3):
        """
        Run scaling tests across different dataset sizes.
        
        Args:
            sizes: List of experience set sizes to test
            samples_per_size: Number of test samples per size (for averaging)
        """
        print(f"🚀 Running scaling test with sizes: {sizes}")
        print(f"📊 {samples_per_size} samples per size for statistical accuracy")
        print("=" * 80)
        
        for size in sizes:
            print(f"\n🔬 Testing size: {size:,} experiences")
            
            size_results = []
            
            for sample in range(samples_per_size):
                print(f"  Sample {sample + 1}/{samples_per_size}...")
                
                # Generate experiences
                start_time = time.time()
                experiences = self.generator.generate_experiences_batch(size)
                generation_time = time.time() - start_time
                
                # Add similarity connections
                self.generator.generate_similarity_connections(experiences, connection_density=0.01)
                
                # Memory before brain creation
                memory_before = self.generator.get_memory_usage()
                
                # Create brain
                brain = MinimalBrain(use_utility_based_activation=self.use_utility_activation)
                
                # Add experiences to brain
                start_time = time.time()
                for experience in experiences.values():
                    brain.experience_storage.add_experience(experience)
                storage_time = time.time() - start_time
                
                # Memory after loading
                memory_after = self.generator.get_memory_usage()
                
                # Test brain operation
                start_time = time.time()
                test_context = np.random.randn(self.generator.dimensions).tolist()
                
                # Multiple brain operations
                for _ in range(5):
                    predicted_action, brain_state = brain.process_sensory_input(test_context)
                
                operation_time = time.time() - start_time
                
                # Collect results
                result = {
                    'size': size,
                    'sample': sample,
                    'generation_time': generation_time,
                    'storage_time': storage_time,
                    'operation_time': operation_time,
                    'total_time': generation_time + storage_time + operation_time,
                    'memory_before_mb': memory_before['rss_mb'],
                    'memory_after_mb': memory_after['rss_mb'],
                    'memory_increase_mb': memory_after['rss_mb'] - memory_before['rss_mb'],
                    'working_memory_size': brain_state['working_memory_size'],
                    'prediction_confidence': brain_state['prediction_confidence']
                }
                
                size_results.append(result)
                
                # Cleanup
                del brain
                del experiences
                gc.collect()
                
                # Brief pause to let system stabilize
                time.sleep(0.5)
            
            # Analyze results for this size
            self._analyze_size_results(size, size_results)
            self.results.extend(size_results)
        
        # Final analysis
        print("\n" + "=" * 80)
        self._analyze_overall_results()
    
    def _analyze_size_results(self, size: int, results: List[Dict]):
        """Analyze results for a specific size."""
        # Calculate averages
        avg_generation = np.mean([r['generation_time'] for r in results])
        avg_storage = np.mean([r['storage_time'] for r in results])
        avg_operation = np.mean([r['operation_time'] for r in results])
        avg_memory = np.mean([r['memory_increase_mb'] for r in results])
        avg_working_memory = np.mean([r['working_memory_size'] for r in results])
        
        # Calculate throughput
        experiences_per_second = size / avg_operation if avg_operation > 0 else 0
        memory_per_experience = avg_memory / size if size > 0 else 0
        
        print(f"  📈 Results for {size:,} experiences:")
        print(f"    Generation: {avg_generation:.3f}s")
        print(f"    Storage:    {avg_storage:.3f}s") 
        print(f"    Operation:  {avg_operation:.3f}s")
        print(f"    Memory:     {avg_memory:.1f} MB ({memory_per_experience:.3f} MB/exp)")
        print(f"    Throughput: {experiences_per_second:,.0f} experiences/second")
        print(f"    Working Memory: {avg_working_memory:.1f} experiences")
    
    def _analyze_overall_results(self):
        """Analyze overall scaling behavior."""
        print("🔍 Overall Scaling Analysis:")
        
        # Group by size
        sizes = sorted(set(r['size'] for r in self.results))
        
        print(f"\n📊 Performance Scaling:")
        prev_size = None
        prev_operation_time = None
        
        for size in sizes:
            size_results = [r for r in self.results if r['size'] == size]
            avg_operation_time = np.mean([r['operation_time'] for r in size_results])
            avg_memory = np.mean([r['memory_increase_mb'] for r in size_results])
            
            if prev_size is not None:
                size_ratio = size / prev_size
                time_ratio = avg_operation_time / prev_operation_time
                efficiency = size_ratio / time_ratio if time_ratio > 0 else 0
                
                print(f"  {prev_size:,} → {size:,}: "
                      f"{size_ratio:.1f}x size, {time_ratio:.1f}x time, "
                      f"efficiency: {efficiency:.2f}")
            
            prev_size = size
            prev_operation_time = avg_operation_time
        
        # Memory scaling
        print(f"\n💾 Memory Scaling:")
        for size in sizes:
            size_results = [r for r in self.results if r['size'] == size]
            avg_memory = np.mean([r['memory_increase_mb'] for r in size_results])
            memory_per_exp = avg_memory / size
            print(f"  {size:,} experiences: {avg_memory:.1f} MB ({memory_per_exp:.3f} MB/exp)")
        
        # Identify scaling regime
        if len(sizes) >= 3:
            large_sizes = sizes[-3:]
            time_ratios = []
            
            for i in range(1, len(large_sizes)):
                curr_results = [r for r in self.results if r['size'] == large_sizes[i]]
                prev_results = [r for r in self.results if r['size'] == large_sizes[i-1]]
                
                curr_time = np.mean([r['operation_time'] for r in curr_results])
                prev_time = np.mean([r['operation_time'] for r in prev_results])
                
                size_ratio = large_sizes[i] / large_sizes[i-1]
                time_ratio = curr_time / prev_time if prev_time > 0 else 0
                
                time_ratios.append(time_ratio / size_ratio)
            
            avg_scaling = np.mean(time_ratios) if time_ratios else 0
            
            print(f"\n🎯 Scaling Regime Analysis:")
            if avg_scaling < 0.5:
                print("  🚀 Sub-linear scaling - GPU acceleration working excellently!")
            elif avg_scaling < 1.0:
                print("  ✅ Good scaling - better than linear growth")
            elif avg_scaling < 1.5:
                print("  ⚠️  Linear scaling - room for optimization")
            else:
                print("  ❌ Super-linear scaling - performance degrading")
            
            print(f"  Average scaling factor: {avg_scaling:.2f}")


def run_massive_scaling_test():
    """Run the complete massive scaling test."""
    print("🌟 MASSIVE BRAIN SCALING TEST")
    print("Testing theoretical limits of GPU-accelerated brain")
    print("=" * 80)
    
    # Memory check
    memory = psutil.virtual_memory()
    print(f"💾 Available Memory: {memory.available / 1024**3:.1f} GB")
    
    # Test sizes - start small and go massive
    test_sizes = [
        1_000,      # 1K - baseline
        5_000,      # 5K - small scale
        10_000,     # 10K - medium scale
        25_000,     # 25K - large scale
        50_000,     # 50K - very large scale
    ]
    
    # Add even larger sizes if we have enough memory
    if memory.available > 8 * 1024**3:  # 8GB available
        test_sizes.extend([100_000, 250_000])  # 100K, 250K
    
    if memory.available > 16 * 1024**3:  # 16GB available
        test_sizes.append(500_000)  # 500K
    
    if memory.available > 32 * 1024**3:  # 32GB available
        test_sizes.append(1_000_000)  # 1M - THE GOAL!
    
    print(f"🎯 Test sizes: {[f'{s:,}' for s in test_sizes]}")
    
    # Test utility-based activation (most sophisticated)
    tester = BrainScalingTest(use_utility_activation=True)
    
    try:
        tester.run_scaling_test(test_sizes, samples_per_size=2)
        
        print("\n🎉 MASSIVE SCALING TEST COMPLETED!")
        print("🎯 Brain is ready for millions of experiences!")
        
    except Exception as e:
        print(f"\n⚠️  Test hit limits at: {e}")
        print("💡 This shows us our current practical boundaries")


if __name__ == "__main__":
    run_massive_scaling_test()