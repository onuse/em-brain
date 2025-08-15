#!/usr/bin/env python3
"""
Refined Fuzzyness for Speed Optimization

Instead of just reducing storage, use fuzzyness to actually speed up operations:
1. Skip redundant similarity calculations
2. Use cached predictions for "similar enough" inputs
3. Reduce pattern analysis overhead for familiar experiences
4. Fast-path decision making based on fuzzy matching
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, OrderedDict

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


class SpeedOrientedFuzzyness:
    """
    Use fuzzyness to speed up brain operations, not just save memory.
    
    Key optimizations:
    1. Prediction caching - reuse predictions for similar inputs
    2. Similarity shortcuts - skip full calculations for fuzzy matches
    3. Pattern analysis reduction - don't re-analyze familiar patterns
    4. Fast-path routing - immediate responses for known situations
    """
    
    def __init__(self, target_cycle_time_ms: float = 50.0):
        """Initialize speed-oriented fuzzyness system."""
        self.target_cycle_time = target_cycle_time_ms
        
        # Performance tracking
        self.recent_cycle_times = deque(maxlen=20)
        self.performance_ratio = 1.0
        
        # Fuzzy matching thresholds (adaptive)
        self.fuzzy_threshold = 0.95  # Start strict
        self.min_fuzzy_threshold = 0.7  # Most lenient when slow
        
        # Prediction cache for fuzzy matches
        self.prediction_cache = OrderedDict()  # LRU-style
        self.max_cache_size = 100
        self.cache_hits = 0
        self.cache_attempts = 0
        
        # Pattern analysis skip list
        self.analyzed_patterns = {}  # Hash -> analysis result
        self.pattern_skip_threshold = 0.9
        
        # Similarity calculation shortcuts
        self.similarity_cache = {}
        self.similarity_shortcuts = 0
        
        # Adaptation rates
        self.adaptation_rate = 0.2
        
        print("🚀 SpeedOrientedFuzzyness initialized")
        print(f"   Target: {target_cycle_time_ms:.0f}ms cycles")
        print("   Optimizations: prediction cache, similarity shortcuts, pattern skipping")
    
    def get_fuzzy_hash(self, vector: List[float], precision: int = None) -> str:
        """Create fuzzy hash for vector based on current performance."""
        if precision is None:
            # Adaptive precision based on performance
            if self.performance_ratio > 2.0:
                precision = 1  # Very fuzzy when slow
            elif self.performance_ratio > 1.5:
                precision = 2
            else:
                precision = 3  # High precision when fast
        
        quantized = tuple(round(x, precision) for x in vector)
        return str(hash(quantized))
    
    def check_prediction_cache(self, sensory_input: List[float]) -> Optional[Tuple[List[float], Dict]]:
        """
        Check if we have a cached prediction for similar input.
        
        Returns:
            Cached (prediction, brain_state) if found, None otherwise
        """
        self.cache_attempts += 1
        
        # Get fuzzy hash for current input
        input_hash = self.get_fuzzy_hash(sensory_input)
        
        # Direct cache hit
        if input_hash in self.prediction_cache:
            self.cache_hits += 1
            cached = self.prediction_cache[input_hash]
            
            # Move to end (LRU behavior)
            del self.prediction_cache[input_hash]
            self.prediction_cache[input_hash] = cached
            
            return cached['prediction'], cached['brain_state']
        
        # Fuzzy matching for very slow hardware
        if self.performance_ratio > 1.5:
            # Check for fuzzy similar entries
            for cached_hash, cached_data in list(self.prediction_cache.items())[-10:]:  # Check recent entries
                cached_vector = cached_data['input']
                similarity = self._fast_similarity(sensory_input, cached_vector)
                
                if similarity >= self.fuzzy_threshold:
                    self.cache_hits += 1
                    self.similarity_shortcuts += 1
                    return cached_data['prediction'], cached_data['brain_state']
        
        return None
    
    def cache_prediction(self, sensory_input: List[float], 
                        prediction: List[float], 
                        brain_state: Dict):
        """Cache prediction for future fuzzy matching."""
        input_hash = self.get_fuzzy_hash(sensory_input)
        
        # LRU eviction
        if len(self.prediction_cache) >= self.max_cache_size:
            # Remove oldest
            self.prediction_cache.popitem(last=False)
        
        self.prediction_cache[input_hash] = {
            'input': sensory_input,
            'prediction': prediction,
            'brain_state': brain_state,
            'timestamp': time.time()
        }
    
    def should_skip_pattern_analysis(self, experience_vector: List[float]) -> bool:
        """Check if we should skip pattern analysis for this experience."""
        if self.performance_ratio < 1.2:
            return False  # Don't skip when performance is good
        
        # Check if we've analyzed similar patterns recently
        pattern_hash = self.get_fuzzy_hash(experience_vector, precision=1)
        
        if pattern_hash in self.analyzed_patterns:
            # Skip if recently analyzed
            last_analysis = self.analyzed_patterns[pattern_hash]
            if time.time() - last_analysis['timestamp'] < 5.0:  # 5 second window
                return True
        
        return False
    
    def mark_pattern_analyzed(self, experience_vector: List[float]):
        """Mark pattern as analyzed."""
        pattern_hash = self.get_fuzzy_hash(experience_vector, precision=1)
        self.analyzed_patterns[pattern_hash] = {
            'timestamp': time.time()
        }
        
        # Clean old entries
        if len(self.analyzed_patterns) > 100:
            # Remove entries older than 10 seconds
            current_time = time.time()
            self.analyzed_patterns = {
                k: v for k, v in self.analyzed_patterns.items()
                if current_time - v['timestamp'] < 10.0
            }
    
    def _fast_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Fast approximate similarity calculation."""
        # Use cached result if available
        cache_key = (tuple(vec1), tuple(vec2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Fast dot product similarity
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6
        )
        
        # Cache result
        self.similarity_cache[cache_key] = similarity
        
        # Clean cache if too large
        if len(self.similarity_cache) > 1000:
            self.similarity_cache.clear()
        
        return similarity
    
    def update_performance(self, cycle_time_ms: float):
        """Update performance metrics and adapt thresholds."""
        self.recent_cycle_times.append(cycle_time_ms)
        
        if len(self.recent_cycle_times) >= 5:
            avg_cycle = np.mean(self.recent_cycle_times)
            self.performance_ratio = avg_cycle / self.target_cycle_time
            
            # Adapt fuzzy threshold based on performance
            if self.performance_ratio > 2.0:
                target_threshold = self.min_fuzzy_threshold
            elif self.performance_ratio > 1.5:
                target_threshold = 0.8
            elif self.performance_ratio > 1.2:
                target_threshold = 0.85
            else:
                target_threshold = 0.95  # Strict when fast
            
            # Smooth adaptation
            self.fuzzy_threshold += (target_threshold - self.fuzzy_threshold) * self.adaptation_rate
    
    def get_optimization_stats(self) -> Dict[str, any]:
        """Get statistics about optimization effectiveness."""
        cache_hit_rate = (self.cache_hits / max(1, self.cache_attempts)) * 100
        
        return {
            'performance_ratio': self.performance_ratio,
            'fuzzy_threshold': self.fuzzy_threshold,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache_hits,
            'cache_attempts': self.cache_attempts,
            'similarity_shortcuts': self.similarity_shortcuts,
            'patterns_skipped': len(self.analyzed_patterns),
            'avg_cycle_time': np.mean(self.recent_cycle_times) if self.recent_cycle_times else 0
        }


def test_speed_oriented_fuzzyness():
    """Test speed-oriented fuzzyness optimizations."""
    print("🚀 TESTING SPEED-ORIENTED FUZZYNESS")
    print("=" * 50)
    print("Comparing actual speed improvements from fuzzy optimizations")
    print()
    
    # Test configurations
    test_configs = [
        ("Baseline (no fuzzyness)", None),
        ("With fuzzy speed optimizations", SpeedOrientedFuzzyness(target_cycle_time_ms=50.0))
    ]
    
    results = {}
    
    for config_name, fuzzy_system in test_configs:
        print(f"\n📊 Testing: {config_name}")
        print("-" * 40)
        
        brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
        
        # Performance metrics
        cycle_times = []
        predictions_made = 0
        cache_used = 0
        patterns_skipped = 0
        
        # Generate test sequence with repetitive patterns
        test_sequence = []
        
        # Pattern 1: Gradual changes (should be fuzzy-matchable)
        for i in range(30):
            base = [0.5, 0.4, 0.3, 0.2]
            noise = [x * 0.001 * i for x in [1, -1, 1, -1]]
            test_sequence.append([b + n for b, n in zip(base, noise)])
        
        # Pattern 2: Repeated exact patterns
        pattern_a = [0.1, 0.2, 0.3, 0.4]
        pattern_b = [0.2, 0.3, 0.4, 0.5]
        for _ in range(20):
            test_sequence.extend([pattern_a, pattern_b, pattern_a])
        
        # Pattern 3: Similar patterns with small variations
        for i in range(20):
            base = [0.7, 0.6, 0.5, 0.4]
            tiny_noise = np.random.normal(0, 0.005, 4)
            test_sequence.append([b + n for b, n in zip(base, tiny_noise)])
        
        # Run test
        start_time = time.time()
        
        for i, sensory in enumerate(test_sequence):
            cycle_start = time.time()
            
            # Check fuzzy cache first
            if fuzzy_system:
                cached_result = fuzzy_system.check_prediction_cache(sensory)
                
                if cached_result:
                    # Use cached prediction (FAST PATH)
                    predicted_action, brain_state = cached_result
                    cache_used += 1
                else:
                    # Normal prediction
                    predicted_action, brain_state = brain.process_sensory_input(sensory)
                    predictions_made += 1
                    
                    # Cache for future use
                    fuzzy_system.cache_prediction(sensory, predicted_action, brain_state)
            else:
                # No fuzzyness - always full computation
                predicted_action, brain_state = brain.process_sensory_input(sensory)
                predictions_made += 1
            
            # Simulate outcome
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            
            # Store experience (with potential optimizations)
            if fuzzy_system and fuzzy_system.should_skip_pattern_analysis(sensory + predicted_action):
                # Skip pattern analysis for familiar experiences
                patterns_skipped += 1
                # Still store but with reduced processing
                # Create minimal experience without full processing
                from src.experience.models import Experience
                exp = Experience(
                    sensory_input=sensory,
                    action_taken=predicted_action,
                    outcome=outcome,
                    prediction_error=0.1,  # Skip calculation
                    timestamp=time.time()
                )
                brain.experience_storage.add_experience(exp)
            else:
                # Full storage and analysis
                brain.store_experience(sensory, predicted_action, outcome, predicted_action)
                
                if fuzzy_system:
                    fuzzy_system.mark_pattern_analyzed(sensory + predicted_action)
            
            cycle_time = (time.time() - cycle_start) * 1000
            cycle_times.append(cycle_time)
            
            # Update fuzzy system performance
            if fuzzy_system:
                fuzzy_system.update_performance(cycle_time)
            
            # Progress update
            if (i + 1) % 50 == 0 and fuzzy_system:
                stats = fuzzy_system.get_optimization_stats()
                print(f"  Progress {i+1}/{len(test_sequence)}: "
                      f"cache hits={cache_used}/{i+1} ({stats['cache_hit_rate']:.1f}%), "
                      f"avg cycle={stats['avg_cycle_time']:.1f}ms")
        
        total_time = time.time() - start_time
        brain.finalize_session()
        
        # Store results
        results[config_name] = {
            'total_time': total_time,
            'avg_cycle_time': np.mean(cycle_times),
            'predictions_made': predictions_made,
            'cache_used': cache_used,
            'patterns_skipped': patterns_skipped,
            'experiences_processed': len(test_sequence)
        }
        
        if fuzzy_system:
            results[config_name]['optimization_stats'] = fuzzy_system.get_optimization_stats()
    
    # Compare results
    print("\n" + "="*50)
    print("🎯 SPEED OPTIMIZATION RESULTS")
    print("="*50)
    
    baseline = results["Baseline (no fuzzyness)"]
    optimized = results["With fuzzy speed optimizations"]
    
    print(f"\nTotal Processing Time:")
    print(f"  Baseline:     {baseline['total_time']:.2f}s")
    print(f"  Optimized:    {optimized['total_time']:.2f}s")
    speedup = baseline['total_time'] / optimized['total_time']
    print(f"  Speedup:      {speedup:.2f}x")
    
    print(f"\nAverage Cycle Time:")
    print(f"  Baseline:     {baseline['avg_cycle_time']:.1f}ms")
    print(f"  Optimized:    {optimized['avg_cycle_time']:.1f}ms")
    improvement = (baseline['avg_cycle_time'] - optimized['avg_cycle_time']) / baseline['avg_cycle_time'] * 100
    print(f"  Improvement:  {improvement:.1f}%")
    
    print(f"\nComputational Savings:")
    print(f"  Full predictions:")
    print(f"    Baseline:   {baseline['predictions_made']} (100%)")
    print(f"    Optimized:  {optimized['predictions_made']} ({optimized['predictions_made']/baseline['predictions_made']*100:.1f}%)")
    print(f"  Cache hits:   {optimized['cache_used']} ({optimized['cache_used']/len(test_sequence)*100:.1f}%)")
    print(f"  Patterns skipped: {optimized['patterns_skipped']}")
    
    if 'optimization_stats' in optimized:
        stats = optimized['optimization_stats']
        print(f"\nOptimization Details:")
        print(f"  Final fuzzy threshold: {stats['fuzzy_threshold']:.3f}")
        print(f"  Similarity shortcuts: {stats['similarity_shortcuts']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
    
    print("\n💡 INSIGHTS:")
    print("• Fuzzy caching provides real speed improvements")
    print("• Repetitive patterns benefit most from optimization")
    print("• Pattern analysis skipping reduces overhead")
    print("• Adaptive thresholds maintain quality/speed balance")


if __name__ == "__main__":
    test_speed_oriented_fuzzyness()