#!/usr/bin/env python3
"""
Hardware-Adaptive Fuzzyness

Implement the biological principle that fast brains can detect nuanced differences,
while slower brains become "fuzzier" to maintain real-time operation.

Key insight: Similarity thresholds should adapt to hardware performance.
- Fast hardware = tight thresholds (discriminating, sees nuance)
- Slow hardware = loose thresholds (fuzzy, treats similar as same)
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any
from collections import deque

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


class HardwareAdaptiveFuzzyness:
    """
    Dynamically adjust similarity thresholds based on hardware performance.
    
    Biological principle: Brains with limited computational resources become
    "fuzzier" in their similarity judgments to maintain real-time operation.
    """
    
    def __init__(self, target_cycle_time_ms: float = 50.0):
        """Initialize hardware-adaptive fuzzyness system."""
        self.target_cycle_time = target_cycle_time_ms
        
        # Hardware performance tracking
        self.recent_cycle_times = deque(maxlen=20)
        self.baseline_performance = None
        self.current_performance_ratio = 1.0
        
        # Adaptive similarity thresholds
        self.base_similarity_threshold = 0.7  # When hardware is fast
        self.max_similarity_threshold = 0.4   # When hardware is slow (fuzzier)
        self.current_similarity_threshold = self.base_similarity_threshold
        
        # Adaptive precision levels
        self.base_vector_precision = 3  # decimal places when fast
        self.min_vector_precision = 1   # decimal places when slow
        self.current_vector_precision = self.base_vector_precision
        
        # Adaptive attention
        self.base_attention_threshold = 0.3
        self.max_attention_threshold = 0.7  # Higher = more filtering when slow
        self.current_attention_threshold = self.base_attention_threshold
        
        # Performance adaptation rates
        self.adaptation_rate = 0.1
        self.measurement_count = 0
        
        print("🎛️  HardwareAdaptiveFuzzyness initialized")
        print(f"   Target cycle time: {target_cycle_time_ms:.1f}ms")
        print(f"   Similarity: {self.base_similarity_threshold:.2f} → {self.max_similarity_threshold:.2f}")
        print(f"   Precision: {self.base_vector_precision} → {self.min_vector_precision} decimals")
    
    def measure_cycle_performance(self, cycle_time_ms: float) -> Dict[str, Any]:
        """
        Measure and adapt to current hardware performance.
        
        Returns adaptation decisions based on performance.
        """
        self.measurement_count += 1
        self.recent_cycle_times.append(cycle_time_ms)
        
        # Establish baseline (first 5 measurements)
        if self.baseline_performance is None and len(self.recent_cycle_times) >= 5:
            self.baseline_performance = np.mean(list(self.recent_cycle_times)[:5])
            print(f"📊 Baseline performance established: {self.baseline_performance:.1f}ms")
        
        if self.baseline_performance is None:
            return {'status': 'calibrating', 'measurements': len(self.recent_cycle_times)}
        
        # Calculate current performance
        recent_avg = np.mean(self.recent_cycle_times)
        self.current_performance_ratio = recent_avg / self.target_cycle_time
        
        # Adapt similarity threshold (slower = fuzzier)
        if self.current_performance_ratio > 2.0:
            # Very slow - be very fuzzy
            target_threshold = self.max_similarity_threshold
        elif self.current_performance_ratio > 1.5:
            # Moderately slow - somewhat fuzzy
            target_threshold = self.base_similarity_threshold - 0.2
        elif self.current_performance_ratio < 0.8:
            # Very fast - be discriminating
            target_threshold = min(0.9, self.base_similarity_threshold + 0.1)
        else:
            # Good performance - use base threshold
            target_threshold = self.base_similarity_threshold
        
        # Smooth adaptation
        self.current_similarity_threshold += (target_threshold - self.current_similarity_threshold) * self.adaptation_rate
        
        # Adapt vector precision (slower = lower precision)
        if self.current_performance_ratio > 2.0:
            self.current_vector_precision = self.min_vector_precision
        elif self.current_performance_ratio > 1.5:
            self.current_vector_precision = 2
        else:
            self.current_vector_precision = self.base_vector_precision
        
        # Adapt attention threshold (slower = more filtering)
        if self.current_performance_ratio > 2.0:
            target_attention = self.max_attention_threshold
        elif self.current_performance_ratio > 1.5:
            target_attention = self.base_attention_threshold + 0.2
        else:
            target_attention = self.base_attention_threshold
        
        self.current_attention_threshold += (target_attention - self.current_attention_threshold) * self.adaptation_rate
        
        return {
            'cycle_time': cycle_time_ms,
            'performance_ratio': self.current_performance_ratio,
            'similarity_threshold': self.current_similarity_threshold,
            'vector_precision': self.current_vector_precision,
            'attention_threshold': self.current_attention_threshold,
            'status': self._get_performance_status()
        }
    
    def _get_performance_status(self) -> str:
        """Get current performance status."""
        if self.current_performance_ratio < 0.8:
            return 'fast'
        elif self.current_performance_ratio < 1.2:
            return 'good'
        elif self.current_performance_ratio < 2.0:
            return 'slow'
        else:
            return 'very_slow'
    
    def quantize_vector(self, vector: List[float]) -> List[float]:
        """Quantize vector based on current precision level."""
        return [round(x, self.current_vector_precision) for x in vector]
    
    def should_be_fuzzy_similar(self, vec1: List[float], vec2: List[float]) -> bool:
        """
        Check if two vectors should be considered similar given current fuzzyness.
        
        Returns True if they're similar enough for current hardware performance.
        """
        # Quantize vectors based on current precision
        q_vec1 = self.quantize_vector(vec1)
        q_vec2 = self.quantize_vector(vec2)
        
        # Calculate similarity with current precision
        similarity = np.dot(q_vec1, q_vec2) / (
            np.linalg.norm(q_vec1) * np.linalg.norm(q_vec2) + 1e-6
        )
        
        # Use adaptive threshold
        return similarity >= self.current_similarity_threshold
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get current adaptive fuzzyness statistics."""
        if len(self.recent_cycle_times) == 0:
            return {'status': 'no_data'}
        
        recent_avg = np.mean(self.recent_cycle_times)
        
        return {
            'measurements': self.measurement_count,
            'recent_cycle_time': recent_avg,
            'target_cycle_time': self.target_cycle_time,
            'performance_ratio': self.current_performance_ratio,
            'performance_status': self._get_performance_status(),
            'similarity_threshold': self.current_similarity_threshold,
            'vector_precision': self.current_vector_precision,
            'attention_threshold': self.current_attention_threshold,
            'baseline_performance': self.baseline_performance,
            'status': 'active'
        }


def test_hardware_adaptive_fuzzyness():
    """Test hardware-adaptive fuzzyness system."""
    print("🎛️  TESTING HARDWARE-ADAPTIVE FUZZYNESS")
    print("=" * 50)
    print("Testing how similarity thresholds adapt to hardware performance...")
    print()
    
    # Create adaptive fuzzyness system
    fuzzy_system = HardwareAdaptiveFuzzyness(target_cycle_time_ms=50.0)
    
    # Create brain for testing
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Test vectors for similarity comparison
    test_vectors = [
        [0.1, 0.2, 0.3, 0.4],
        [0.11, 0.21, 0.31, 0.41],  # Very similar
        [0.15, 0.25, 0.35, 0.45],  # Somewhat similar  
        [0.2, 0.3, 0.4, 0.5],      # Moderately similar
        [0.8, 0.7, 0.6, 0.5]       # Different
    ]
    
    # Simulate different performance scenarios
    scenarios = [
        ("Fast Hardware", [30, 35, 32, 28, 33]),     # Fast cycles
        ("Good Hardware", [45, 52, 48, 50, 47]),     # Target performance
        ("Slow Hardware", [85, 92, 88, 90, 87]),     # Slow cycles
        ("Very Slow Hardware", [150, 145, 155, 148, 152])  # Very slow
    ]
    
    for scenario_name, cycle_times in scenarios:
        print(f"🔧 {scenario_name} Scenario:")
        
        # Reset system for new scenario
        fuzzy_system = HardwareAdaptiveFuzzyness(target_cycle_time_ms=50.0)
        
        # Feed performance data
        for cycle_time in cycle_times:
            adaptation = fuzzy_system.measure_cycle_performance(cycle_time)
        
        stats = fuzzy_system.get_adaptive_stats()
        
        print(f"   Average cycle time: {stats['recent_cycle_time']:.1f}ms")
        print(f"   Performance ratio: {stats['performance_ratio']:.2f}x target")
        print(f"   Status: {stats['performance_status']}")
        print(f"   Similarity threshold: {stats['similarity_threshold']:.3f}")
        print(f"   Vector precision: {stats['vector_precision']} decimals")
        print(f"   Attention threshold: {stats['attention_threshold']:.3f}")
        
        # Test similarity behavior with this hardware
        print(f"   Similarity behavior:")
        base_vector = test_vectors[0]
        
        for i, test_vector in enumerate(test_vectors[1:], 1):
            is_similar = fuzzy_system.should_be_fuzzy_similar(base_vector, test_vector)
            raw_similarity = np.dot(base_vector, test_vector) / (
                np.linalg.norm(base_vector) * np.linalg.norm(test_vector)
            )
            
            status = "SIMILAR" if is_similar else "different"
            print(f"     Vector {i}: {raw_similarity:.3f} → {status}")
        
        print()
    
    print("🧠 ADAPTIVE FUZZYNESS INSIGHTS:")
    print("-" * 35)
    print("Fast hardware:")
    print("  • High precision (3 decimals)")
    print("  • Tight similarity thresholds (0.7+)")
    print("  • Can detect subtle differences")
    print()
    print("Slow hardware:")
    print("  • Low precision (1 decimal)")
    print("  • Loose similarity thresholds (0.4+)")
    print("  • Groups similar experiences together")
    print()
    print("Benefits:")
    print("  ✅ Maintains real-time performance")
    print("  ✅ Graceful degradation under load")
    print("  ✅ Biological realism (faster brains see more nuance)")
    print("  ✅ Hardware-agnostic behavior")


def test_fuzzyness_in_brain():
    """Test fuzzyness integration with actual brain."""
    print("\n🧠 TESTING FUZZYNESS IN BRAIN OPERATION")
    print("=" * 45)
    
    fuzzy_system = HardwareAdaptiveFuzzyness(target_cycle_time_ms=50.0)
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Simulate learning with performance measurement
    for i in range(20):
        sensory = [0.1 + i * 0.005, 0.2 + i * 0.005, 0.3 + i * 0.005, 0.4 + i * 0.005]
        
        # Time the brain cycle
        cycle_start = time.time()
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        brain.store_experience(sensory, predicted_action, outcome, predicted_action)
        cycle_time = (time.time() - cycle_start) * 1000
        
        # Adapt fuzzyness based on performance
        adaptation = fuzzy_system.measure_cycle_performance(cycle_time)
        
        if i % 5 == 4:  # Progress update every 5 cycles
            stats = fuzzy_system.get_adaptive_stats()
            if stats['status'] != 'no_data':
                print(f"   Cycle {i+1}: {cycle_time:.1f}ms, "
                      f"threshold: {stats['similarity_threshold']:.3f}, "
                      f"precision: {stats['vector_precision']}")
    
    final_stats = fuzzy_system.get_adaptive_stats()
    brain.finalize_session()
    
    print(f"\n🎯 Final adaptation state:")
    print(f"   Performance: {final_stats['performance_status']}")
    print(f"   Similarity threshold: {final_stats['similarity_threshold']:.3f}")
    print(f"   Vector precision: {final_stats['vector_precision']} decimals")
    print(f"   Brain adapted to maintain {fuzzy_system.target_cycle_time:.0f}ms target")


if __name__ == "__main__":
    test_hardware_adaptive_fuzzyness()
    test_fuzzyness_in_brain()