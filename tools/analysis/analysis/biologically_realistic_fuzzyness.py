#!/usr/bin/env python3
"""
Biologically Realistic Fuzzyness

Performance tiers based on biological reaction times:
- Below 150ms: Fast reflexive responses (high precision)
- 150-400ms: Normal cognitive processing (balanced fuzzyness)
- 400ms+: Slow deliberative thinking (high fuzzyness for speed)

These align with human cognitive timescales:
- 100-150ms: Visual recognition, reflexes
- 200-300ms: Conscious awareness, basic decisions
- 400ms+: Complex reasoning, deliberation
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque, OrderedDict
from enum import Enum

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


class PerformanceTier(Enum):
    """Biologically realistic performance tiers."""
    FAST = "fast"           # <150ms - reflexive
    NORMAL = "normal"       # 150-400ms - cognitive
    SLOW = "slow"           # 400ms+ - deliberative


class BiologicallyRealisticFuzzyness:
    """
    Fuzzyness system with biologically realistic performance tiers.
    
    Key principles:
    - Fast (<150ms): High precision, discriminating (like reflexes)
    - Normal (150-400ms): Balanced fuzzyness (like normal thinking)
    - Slow (400ms+): High fuzzyness to maintain function (like tired thinking)
    """
    
    def __init__(self):
        """Initialize biologically realistic fuzzyness."""
        
        # Performance tier thresholds (milliseconds)
        self.FAST_THRESHOLD = 150
        self.NORMAL_THRESHOLD = 400
        
        # Current performance tracking
        self.recent_cycle_times = deque(maxlen=10)
        self.current_tier = PerformanceTier.FAST
        self.tier_history = deque(maxlen=20)
        
        # Tier-specific parameters
        self.tier_configs = {
            PerformanceTier.FAST: {
                'similarity_threshold': 0.95,      # Very discriminating
                'vector_precision': 3,             # High precision
                'cache_threshold': 0.98,           # Only exact matches cached
                'pattern_skip_probability': 0.0,  # Analyze everything
                'attention_threshold': 0.2         # Pay attention to most things
            },
            PerformanceTier.NORMAL: {
                'similarity_threshold': 0.85,      # Moderately discriminating
                'vector_precision': 2,             # Medium precision
                'cache_threshold': 0.90,           # Cache similar experiences
                'pattern_skip_probability': 0.3,  # Skip some patterns
                'attention_threshold': 0.4         # Moderate attention filtering
            },
            PerformanceTier.SLOW: {
                'similarity_threshold': 0.70,      # Fuzzy matching
                'vector_precision': 1,             # Low precision
                'cache_threshold': 0.75,           # Aggressive caching
                'pattern_skip_probability': 0.7,  # Skip most pattern analysis
                'attention_threshold': 0.6         # High attention filtering
            }
        }
        
        # Adaptive caches
        self.prediction_cache = OrderedDict()
        self.similarity_cache = {}
        self.pattern_cache = {}
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_attempts': 0,
            'patterns_skipped': 0,
            'attention_filtered': 0,
            'tier_changes': 0,
            'time_in_tier': {tier: 0 for tier in PerformanceTier}
        }
        
        print("🧠 BiologicallyRealisticFuzzyness initialized")
        print(f"   Performance tiers:")
        print(f"   • FAST: <{self.FAST_THRESHOLD}ms (reflexive)")
        print(f"   • NORMAL: {self.FAST_THRESHOLD}-{self.NORMAL_THRESHOLD}ms (cognitive)")
        print(f"   • SLOW: >{self.NORMAL_THRESHOLD}ms (deliberative)")
    
    def update_performance_tier(self, cycle_time_ms: float) -> PerformanceTier:
        """Update performance tier based on recent cycle times."""
        self.recent_cycle_times.append(cycle_time_ms)
        
        if len(self.recent_cycle_times) < 3:
            return self.current_tier
        
        # Use median of recent times for stability
        median_time = np.median(self.recent_cycle_times)
        
        # Determine new tier
        if median_time < self.FAST_THRESHOLD:
            new_tier = PerformanceTier.FAST
        elif median_time < self.NORMAL_THRESHOLD:
            new_tier = PerformanceTier.NORMAL
        else:
            new_tier = PerformanceTier.SLOW
        
        # Track tier changes
        if new_tier != self.current_tier:
            self.stats['tier_changes'] += 1
            print(f"⚡ Performance tier: {self.current_tier.value} → {new_tier.value} "
                  f"(median cycle: {median_time:.0f}ms)")
            self.current_tier = new_tier
        
        # Track time in tier
        self.tier_history.append(new_tier)
        
        return new_tier
    
    def get_current_config(self) -> Dict:
        """Get configuration for current performance tier."""
        return self.tier_configs[self.current_tier]
    
    def should_use_cache(self, sensory_input: List[float]) -> Optional[Tuple[List[float], Dict]]:
        """Check if we should use cached prediction based on current tier."""
        self.stats['cache_attempts'] += 1
        config = self.get_current_config()
        
        # Quantize input based on tier precision
        quantized = self._quantize_vector(sensory_input, config['vector_precision'])
        cache_key = tuple(quantized)
        
        # Direct cache hit
        if cache_key in self.prediction_cache:
            self.stats['cache_hits'] += 1
            cached = self.prediction_cache[cache_key]
            
            # LRU update
            del self.prediction_cache[cache_key]
            self.prediction_cache[cache_key] = cached
            
            return cached['prediction'], cached['brain_state']
        
        # Fuzzy matching for slower tiers
        if self.current_tier != PerformanceTier.FAST:
            threshold = config['cache_threshold']
            
            # Check recent cache entries
            for cached_key, cached_data in list(self.prediction_cache.items())[-10:]:
                similarity = self._calculate_similarity(sensory_input, cached_data['input'])
                
                if similarity >= threshold:
                    self.stats['cache_hits'] += 1
                    return cached_data['prediction'], cached_data['brain_state']
        
        return None
    
    def cache_prediction(self, sensory_input: List[float], 
                        prediction: List[float], 
                        brain_state: Dict):
        """Cache prediction based on current tier."""
        config = self.get_current_config()
        quantized = self._quantize_vector(sensory_input, config['vector_precision'])
        cache_key = tuple(quantized)
        
        # LRU eviction
        max_cache_size = {
            PerformanceTier.FAST: 50,     # Small cache when fast
            PerformanceTier.NORMAL: 100,   # Medium cache
            PerformanceTier.SLOW: 200      # Large cache when slow
        }[self.current_tier]
        
        if len(self.prediction_cache) >= max_cache_size:
            self.prediction_cache.popitem(last=False)
        
        self.prediction_cache[cache_key] = {
            'input': sensory_input,
            'prediction': prediction,
            'brain_state': brain_state,
            'timestamp': time.time()
        }
    
    def should_skip_pattern_analysis(self) -> bool:
        """Decide if pattern analysis should be skipped based on tier."""
        config = self.get_current_config()
        skip_prob = config['pattern_skip_probability']
        
        if np.random.random() < skip_prob:
            self.stats['patterns_skipped'] += 1
            return True
        
        return False
    
    def should_filter_experience(self, sensory_input: List[float], 
                               prediction_error: float) -> bool:
        """Decide if experience should be filtered out based on attention threshold."""
        config = self.get_current_config()
        attention_threshold = config['attention_threshold']
        
        # Calculate attention score (higher error = more attention)
        attention_score = min(1.0, prediction_error * 2.0)
        
        # Add some randomness for biological realism
        attention_score += np.random.normal(0, 0.1)
        
        if attention_score < attention_threshold:
            self.stats['attention_filtered'] += 1
            return True
        
        return False
    
    def _quantize_vector(self, vector: List[float], precision: int) -> List[float]:
        """Quantize vector to given precision."""
        return [round(x, precision) for x in vector]
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        cache_key = (tuple(vec1), tuple(vec2))
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarity = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6
        )
        
        self.similarity_cache[cache_key] = similarity
        
        # Clean cache if too large
        if len(self.similarity_cache) > 1000:
            self.similarity_cache.clear()
        
        return similarity
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        total_cycles = len(self.tier_history)
        
        if total_cycles > 0:
            tier_distribution = {
                tier: self.tier_history.count(tier) / total_cycles * 100
                for tier in PerformanceTier
            }
        else:
            tier_distribution = {tier: 0 for tier in PerformanceTier}
        
        cache_hit_rate = (self.stats['cache_hits'] / 
                         max(1, self.stats['cache_attempts']) * 100)
        
        return {
            'current_tier': self.current_tier.value,
            'median_cycle_time': np.median(self.recent_cycle_times) if self.recent_cycle_times else 0,
            'tier_distribution': tier_distribution,
            'cache_hit_rate': cache_hit_rate,
            'patterns_skipped': self.stats['patterns_skipped'],
            'attention_filtered': self.stats['attention_filtered'],
            'tier_changes': self.stats['tier_changes'],
            'current_config': self.get_current_config()
        }


def test_biologically_realistic_fuzzyness():
    """Test fuzzyness with realistic performance tiers."""
    print("🧪 TESTING BIOLOGICALLY REALISTIC FUZZYNESS")
    print("=" * 60)
    print("Testing how fuzzyness adapts to realistic performance tiers")
    print()
    
    fuzzy_system = BiologicallyRealisticFuzzyness()
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Test scenarios with different performance characteristics
    scenarios = [
        ("Fast Reflexive (<150ms)", 0.05, 50),      # 50ms base + noise
        ("Normal Cognitive (150-400ms)", 0.2, 50),   # 200ms base + noise
        ("Slow Deliberative (400ms+)", 0.5, 50),     # 500ms base + noise
        ("Variable Performance", "variable", 100)     # Changing performance
    ]
    
    for scenario_name, base_delay, num_cycles in scenarios:
        print(f"\n📊 Scenario: {scenario_name}")
        print("-" * 40)
        
        # Reset statistics
        fuzzy_system = BiologicallyRealisticFuzzyness()
        
        # Metrics
        actual_predictions = 0
        cached_predictions = 0
        filtered_experiences = 0
        
        for i in range(num_cycles):
            # Create sensory input with some patterns
            if i % 10 < 3:  # Repeated pattern
                sensory = [0.5, 0.4, 0.3, 0.2]
            else:  # Variations
                sensory = [0.5 + np.random.normal(0, 0.05) for _ in range(4)]
            
            # Simulate performance delay
            if base_delay == "variable":
                # Variable performance - cycles through tiers
                if i < 30:
                    delay = 0.08  # Fast
                elif i < 60:
                    delay = 0.25  # Normal
                else:
                    delay = 0.45  # Slow
            else:
                # Fixed tier with noise
                delay = base_delay + np.random.normal(0, base_delay * 0.2)
            
            # Add artificial delay
            time.sleep(max(0, delay))
            
            cycle_start = time.time()
            
            # Check cache first
            cached_result = fuzzy_system.should_use_cache(sensory)
            
            if cached_result:
                predicted_action, brain_state = cached_result
                cached_predictions += 1
            else:
                # Full prediction
                predicted_action, brain_state = brain.process_sensory_input(sensory)
                actual_predictions += 1
                
                # Cache for future
                fuzzy_system.cache_prediction(sensory, predicted_action, brain_state)
            
            # Calculate cycle time
            cycle_time = (time.time() - cycle_start + delay) * 1000
            
            # Update performance tier
            fuzzy_system.update_performance_tier(cycle_time)
            
            # Simulate outcome and error
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            prediction_error = np.random.random() * 0.5  # Simulated error
            
            # Check if should filter
            if fuzzy_system.should_filter_experience(sensory, prediction_error):
                filtered_experiences += 1
                continue
            
            # Store experience (maybe skip pattern analysis)
            if fuzzy_system.should_skip_pattern_analysis():
                # Minimal storage
                pass
            else:
                # Full storage
                brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            
            # Progress update
            if (i + 1) % 20 == 0:
                stats = fuzzy_system.get_statistics()
                print(f"  Cycle {i+1}: tier={stats['current_tier']}, "
                      f"cache={stats['cache_hit_rate']:.0f}%, "
                      f"filtered={filtered_experiences}")
        
        # Final statistics
        stats = fuzzy_system.get_statistics()
        
        print(f"\n  Final Statistics:")
        print(f"    Median cycle time: {stats['median_cycle_time']:.0f}ms")
        print(f"    Performance tier: {stats['current_tier']}")
        print(f"    Tier distribution:")
        for tier, pct in stats['tier_distribution'].items():
            print(f"      {tier.value}: {pct:.1f}%")
        print(f"    Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"    Actual predictions: {actual_predictions}")
        print(f"    Cached predictions: {cached_predictions}")
        print(f"    Patterns skipped: {stats['patterns_skipped']}")
        print(f"    Experiences filtered: {stats['attention_filtered']}")
        print(f"    Tier changes: {stats['tier_changes']}")
    
    brain.finalize_session()
    
    print("\n" + "="*60)
    print("💡 BIOLOGICAL INSIGHTS:")
    print("="*60)
    print("\nFAST tier (<150ms) - Reflexive:")
    print("  • High precision (3 decimals)")
    print("  • Minimal caching (98% threshold)")
    print("  • No pattern skipping")
    print("  • Like visual recognition or reflexes")
    print("\nNORMAL tier (150-400ms) - Cognitive:")
    print("  • Balanced precision (2 decimals)")
    print("  • Moderate caching (90% threshold)")
    print("  • Some pattern skipping (30%)")
    print("  • Like normal thinking and decisions")
    print("\nSLOW tier (400ms+) - Deliberative:")
    print("  • Low precision (1 decimal)")
    print("  • Aggressive caching (75% threshold)")
    print("  • Heavy pattern skipping (70%)")
    print("  • Like tired or complex reasoning")


if __name__ == "__main__":
    test_biologically_realistic_fuzzyness()