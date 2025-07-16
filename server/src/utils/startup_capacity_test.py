"""
Startup Hardware Capacity Test

Discovers the actual hardware limits at startup by stress-testing
the brain systems with increasing experience counts until performance
degrades to unacceptable levels.

Much simpler and more reliable than dynamic adaptation.
"""

import time
import psutil
import gc
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import numpy as np


class StartupCapacityTest:
    """
    Tests hardware capacity at startup to find optimal experience limits.
    """
    
    def __init__(self, target_cycle_time_ms: float = 100.0, max_acceptable_ms: float = 400.0):
        """
        Initialize capacity test.
        
        Args:
            target_cycle_time_ms: Ideal cycle time
            max_acceptable_ms: Maximum acceptable cycle time for robotics
        """
        self.target_cycle_time_ms = target_cycle_time_ms
        self.max_acceptable_ms = max_acceptable_ms
        self.test_results = []
        
        print(f"🧪 Startup Capacity Test initialized")
        print(f"   Target: {target_cycle_time_ms}ms cycles")
        print(f"   Maximum acceptable: {max_acceptable_ms}ms cycles")
    
    def run_capacity_test(self) -> Dict[str, int]:
        """
        Run the complete capacity test to find optimal limits.
        
        Returns:
            Dictionary with discovered limits
        """
        print(f"\n🚀 Running startup capacity test...")
        
        # Test experience storage capacity
        max_experiences = self._test_experience_storage_capacity()
        
        # Test similarity search capacity  
        max_similarity_search = self._test_similarity_search_capacity()
        
        # Test activation dynamics capacity
        max_activation_dynamics = self._test_activation_dynamics_capacity()
        
        # Calculate final limits based on bottlenecks
        limits = self._calculate_final_limits(
            max_experiences, max_similarity_search, max_activation_dynamics
        )
        
        print(f"\n✅ Capacity test complete!")
        print(f"   Discovered experience limit: {limits['max_experiences']:,}")
        print(f"   Similarity search limit: {limits['max_similarity_search']:,}")
        print(f"   Activation dynamics limit: {limits['max_activation_dynamics']:,}")
        print(f"   Recommended cleanup threshold: {limits['cleanup_threshold']:,}")
        
        return limits
    
    def _test_experience_storage_capacity(self) -> int:
        """Test how many experiences we can store before memory becomes an issue."""
        print(f"\n📦 Testing experience storage capacity...")
        
        # Create mock experiences to test memory usage
        test_experiences = {}
        test_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 200000, 500000]
        
        max_safe_experiences = 1000
        
        for size in test_sizes:
            print(f"   Testing {size:,} experiences...", end=" ")
            
            # Clear previous test
            test_experiences.clear()
            gc.collect()
            
            # Measure memory before
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            
            # Create mock experiences
            start_time = time.time()
            try:
                for i in range(size):
                    # Mock experience: sensory(16) + action(4) + metadata
                    experience_data = {
                        'sensory_input': [0.1] * 16,
                        'action_taken': [0.1] * 4,
                        'outcome': [0.1] * 8,
                        'timestamp': time.time(),
                        'prediction_error': 0.5,
                        'activation_level': 0.3
                    }
                    test_experiences[f"exp_{i}"] = experience_data
                
                creation_time = (time.time() - start_time) * 1000
                
                # Measure memory after
                memory_after = psutil.virtual_memory().used / (1024 * 1024)
                memory_used = memory_after - memory_before
                
                # Test access time (simulate brain cycle)
                start_time = time.time()
                for _ in range(100):  # Simulate 100 accesses
                    random_key = f"exp_{np.random.randint(0, size)}"
                    _ = test_experiences.get(random_key)
                access_time = (time.time() - start_time) * 10  # Per access in ms
                
                # Check if performance is acceptable
                if memory_used > 1000:  # More than 1GB is getting large
                    print(f"MEMORY LIMIT ({memory_used:.1f}MB)")
                    break
                elif access_time > 1.0:  # Access time too slow
                    print(f"ACCESS LIMIT ({access_time:.2f}ms)")
                    break
                else:
                    max_safe_experiences = size
                    print(f"OK ({memory_used:.1f}MB, {access_time:.2f}ms access)")
                    
            except MemoryError:
                print(f"MEMORY ERROR")
                break
        
        # Clean up
        test_experiences.clear()
        gc.collect()
        
        return max_safe_experiences
    
    def _test_similarity_search_capacity(self) -> int:
        """Test similarity search performance with varying dataset sizes."""
        print(f"\n🔍 Testing similarity search capacity...")
        
        test_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        max_safe_similarity = 1000
        
        for size in test_sizes:
            print(f"   Testing similarity search on {size:,} vectors...", end=" ")
            
            # Create test vectors (16D sensory input)
            test_vectors = np.random.random((size, 16)).astype(np.float32)
            query_vector = np.random.random(16).astype(np.float32)
            
            # Test similarity computation time
            start_time = time.time()
            
            try:
                # Compute similarities (dot product)
                similarities = np.dot(test_vectors, query_vector)
                
                # Find top 20 similar (typical brain operation)
                top_indices = np.argsort(similarities)[-20:]
                
                search_time = (time.time() - start_time) * 1000
                
                if search_time > self.max_acceptable_ms / 2:  # Use half the cycle budget
                    print(f"TOO SLOW ({search_time:.1f}ms)")
                    break
                else:
                    max_safe_similarity = size
                    print(f"OK ({search_time:.1f}ms)")
                    
            except Exception as e:
                print(f"ERROR ({e})")
                break
        
        return max_safe_similarity
    
    def _test_activation_dynamics_capacity(self) -> int:
        """Test activation dynamics performance."""
        print(f"\n⚡ Testing activation dynamics capacity...")
        
        test_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        max_safe_activation = 1000
        
        for size in test_sizes:
            print(f"   Testing activation update on {size:,} experiences...", end=" ")
            
            # Create mock activation levels
            activations = np.random.random(size).astype(np.float32)
            
            # Test activation update time (decay + spreading)
            start_time = time.time()
            
            try:
                # Simulate decay
                activations *= 0.99
                
                # Simulate spreading (simplified)
                for _ in range(10):  # 10 spread operations
                    spread_source = np.random.randint(0, size)
                    spread_target = np.random.randint(0, size)
                    spread_amount = activations[spread_source] * 0.1
                    activations[spread_target] = min(1.0, activations[spread_target] + spread_amount)
                
                update_time = (time.time() - start_time) * 1000
                
                if update_time > self.max_acceptable_ms / 4:  # Use quarter of cycle budget
                    print(f"TOO SLOW ({update_time:.1f}ms)")
                    break
                else:
                    max_safe_activation = size
                    print(f"OK ({update_time:.1f}ms)")
                    
            except Exception as e:
                print(f"ERROR ({e})")
                break
        
        return max_safe_activation
    
    def _calculate_final_limits(self, max_experiences: int, max_similarity: int, max_activation: int) -> Dict[str, int]:
        """Calculate final limits based on bottlenecks."""
        
        # Use the most conservative limit as the bottleneck
        bottleneck = min(max_experiences, max_similarity, max_activation)
        
        # Add some safety margin (80% of bottleneck)
        safe_limit = int(bottleneck * 0.8)
        
        # But ensure minimum functionality
        safe_limit = max(10000, safe_limit)
        
        return {
            'max_experiences': safe_limit,
            'max_similarity_search': max_similarity,
            'max_activation_dynamics': max_activation,
            'cleanup_threshold': int(safe_limit * 1.1),  # 10% buffer before cleanup
            'bottleneck': 'experiences' if bottleneck == max_experiences else 
                         'similarity' if bottleneck == max_similarity else 'activation'
        }


# Global instance
_startup_capacity_test = None
_discovered_limits = None


def get_startup_limits() -> Optional[Dict[str, int]]:
    """Get the limits discovered at startup."""
    return _discovered_limits


def run_startup_capacity_test() -> Dict[str, int]:
    """Run the startup capacity test and cache results."""
    global _startup_capacity_test, _discovered_limits
    
    if _discovered_limits is not None:
        return _discovered_limits
    
    _startup_capacity_test = StartupCapacityTest()
    _discovered_limits = _startup_capacity_test.run_capacity_test()
    
    return _discovered_limits


def get_experience_limit() -> int:
    """Get the discovered experience limit."""
    limits = get_startup_limits()
    if limits:
        return limits['max_experiences']
    return 50000  # Fallback


def should_trigger_cleanup(current_count: int) -> bool:
    """Check if cleanup should be triggered based on startup limits."""
    limits = get_startup_limits()
    if limits:
        return current_count > limits['cleanup_threshold']
    return current_count > 55000  # Fallback


def get_cleanup_target(current_count: int) -> int:
    """Get cleanup target based on startup limits."""
    limits = get_startup_limits()
    if limits:
        return limits['max_experiences']
    return 45000  # Fallback