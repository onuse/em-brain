#!/usr/bin/env python3
"""
Biological Laziness Strategies

Implement biologically-inspired adaptive laziness to reduce learning overhead.
Real brains are incredibly selective about what they learn and when.

Key biological principles:
1. Attention filtering - Most sensory input is ignored
2. Novelty detection - Only surprising things trigger learning
3. Consolidation thresholds - Multiple exposures needed for long-term storage
4. Cognitive load adaptation - Less learning when overloaded
5. Forgetting curves - Unused knowledge naturally decays
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque, defaultdict

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


class BiologicalLazinessManager:
    """
    Manages adaptive laziness in learning using biological principles.
    
    The brain becomes lazier when:
    - Cognitive load is high
    - Recent learning has been unsuccessful
    - Current situation is familiar/predictable
    - Energy resources are low
    
    The brain becomes more eager when:
    - Novel/surprising situations occur
    - Recent predictions failed
    - Learning has been successful
    - Cognitive load is manageable
    """
    
    def __init__(self):
        """Initialize biological laziness manager."""
        
        # 1. ATTENTION FILTERING (Cocktail Party Effect)
        self.attention_threshold = 0.3  # How interesting must something be?
        self.attention_adaptation_rate = 0.02
        
        # 2. NOVELTY DETECTION (Hippocampal Surprise)
        self.novelty_threshold = 0.4  # How different must something be?
        self.recent_experiences = deque(maxlen=20)  # Context for novelty
        
        # 3. CONSOLIDATION REQUIREMENTS (Spaced Repetition)
        self.consolidation_threshold = 3  # Need 3+ exposures for long-term storage
        self.experience_exposure_count = defaultdict(int)  # Track exposures
        self.pending_consolidation = {}  # Temporary storage
        
        # 4. COGNITIVE LOAD ADAPTATION (Stress Response)
        self.cognitive_load = 0.5  # Current mental effort (0-1)
        self.load_threshold = 0.7  # When to become lazy
        self.recent_learning_times = deque(maxlen=10)
        
        # 5. ENERGY MANAGEMENT (Metabolic Conservation)
        self.cognitive_energy = 1.0  # Available mental energy
        self.energy_depletion_rate = 0.01  # Energy cost per learning event
        self.energy_recovery_rate = 0.02  # Energy recovery during rest
        
        # 6. SUCCESS-BASED MOTIVATION (Dopamine Learning)
        self.recent_learning_success = deque(maxlen=20)
        self.motivation_level = 0.5  # How eager to learn (0-1)
        
        # Learning statistics
        self.experiences_filtered = 0
        self.experiences_delayed = 0
        self.experiences_learned = 0
        self.total_experiences = 0
        
        print("🧠 BiologicalLazinessManager initialized")
        print("   Attention filtering, novelty detection, consolidation")
        print("   Cognitive load adaptation, energy management")
    
    def should_process_experience(self, sensory_input: List[float], 
                                current_prediction_error: float,
                                current_confidence: float) -> Dict[str, Any]:
        """
        Decide whether to process this experience based on biological laziness.
        
        Returns:
            dict with 'action' and reasoning
        """
        self.total_experiences += 1
        
        # Update cognitive state
        self._update_cognitive_state()
        
        # 1. ATTENTION FILTERING - Is this worth paying attention to?
        attention_score = self._calculate_attention_score(sensory_input, current_prediction_error)
        
        if attention_score < self.attention_threshold:
            self.experiences_filtered += 1
            return {
                'action': 'ignore',
                'reason': f'attention_filter (score: {attention_score:.2f} < {self.attention_threshold:.2f})',
                'cognitive_load': self.cognitive_load,
                'energy': self.cognitive_energy
            }
        
        # 2. NOVELTY DETECTION - Is this genuinely new/surprising?
        novelty_score = self._calculate_novelty_score(sensory_input)
        
        if novelty_score < self.novelty_threshold and current_confidence > 0.8:
            self.experiences_filtered += 1
            return {
                'action': 'ignore',
                'reason': f'familiar_experience (novelty: {novelty_score:.2f}, confidence: {current_confidence:.2f})',
                'cognitive_load': self.cognitive_load,
                'energy': self.cognitive_energy
            }
        
        # 3. COGNITIVE LOAD CHECK - Are we overloaded?
        if self.cognitive_load > self.load_threshold:
            # Under high load, be much more selective
            if attention_score < self.attention_threshold * 1.5:
                self.experiences_filtered += 1
                return {
                    'action': 'ignore',
                    'reason': f'cognitive_overload (load: {self.cognitive_load:.2f} > {self.load_threshold:.2f})',
                    'cognitive_load': self.cognitive_load,
                    'energy': self.cognitive_energy
                }
        
        # 4. ENERGY CHECK - Do we have mental energy?
        if self.cognitive_energy < 0.3:
            # Low energy = very lazy
            if attention_score < self.attention_threshold * 2.0:
                self.experiences_filtered += 1
                return {
                    'action': 'ignore',
                    'reason': f'low_energy (energy: {self.cognitive_energy:.2f} < 0.3)',
                    'cognitive_load': self.cognitive_load,
                    'energy': self.cognitive_energy
                }
        
        # 5. CONSOLIDATION DECISION - Learn immediately or delay?
        experience_signature = self._get_experience_signature(sensory_input)
        exposure_count = self.experience_exposure_count[experience_signature]
        
        if exposure_count < self.consolidation_threshold:
            # First few exposures - store in temporary buffer
            self.experience_exposure_count[experience_signature] += 1
            self.experiences_delayed += 1
            
            return {
                'action': 'buffer',
                'reason': f'insufficient_exposure (count: {exposure_count + 1}/{self.consolidation_threshold})',
                'experience_signature': experience_signature,
                'cognitive_load': self.cognitive_load,
                'energy': self.cognitive_energy
            }
        
        # 6. COMMIT TO LEARNING - This is worth full processing
        self.experiences_learned += 1
        self._consume_cognitive_resources()
        
        return {
            'action': 'learn',
            'reason': f'worthy_of_learning (attention: {attention_score:.2f}, novelty: {novelty_score:.2f})',
            'cognitive_load': self.cognitive_load,
            'energy': self.cognitive_energy
        }
    
    def _calculate_attention_score(self, sensory_input: List[float], prediction_error: float) -> float:
        """Calculate how attention-worthy this experience is."""
        
        # Base attention from prediction error (surprising = attention-grabbing)
        error_attention = min(1.0, prediction_error * 2.0)
        
        # Sensory salience (how different from recent experiences)
        if len(self.recent_experiences) > 0:
            recent_avg = np.mean([exp['sensory'] for exp in self.recent_experiences], axis=0)
            sensory_diff = np.linalg.norm(np.array(sensory_input) - recent_avg)
            sensory_attention = min(1.0, sensory_diff * 2.0)
        else:
            sensory_attention = 0.5
        
        # Motivation modulation (more motivated = lower threshold)
        motivation_boost = self.motivation_level * 0.3
        
        # Combined attention score
        attention_score = (error_attention * 0.6 + sensory_attention * 0.4) + motivation_boost
        
        return min(1.0, attention_score)
    
    def _calculate_novelty_score(self, sensory_input: List[float]) -> float:
        """Calculate how novel this experience is."""
        
        if len(self.recent_experiences) < 5:
            return 1.0  # Everything is novel early on
        
        # Compare to recent experiences
        similarities = []
        for exp in self.recent_experiences:
            similarity = np.dot(sensory_input, exp['sensory']) / (
                np.linalg.norm(sensory_input) * np.linalg.norm(exp['sensory']) + 1e-6
            )
            similarities.append(similarity)
        
        # Novelty = 1 - maximum similarity
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        
        return max(0.0, novelty)
    
    def _get_experience_signature(self, sensory_input: List[float]) -> str:
        """Create a signature for experience to track exposures."""
        # Quantize sensory input to create signatures for similar experiences
        quantized = [round(x, 1) for x in sensory_input]  # Round to 1 decimal
        return str(hash(tuple(quantized)))
    
    def _update_cognitive_state(self):
        """Update cognitive load, energy, and motivation."""
        
        # Update cognitive load based on recent learning times
        if len(self.recent_learning_times) > 0:
            avg_time = np.mean(self.recent_learning_times)
            target_time = 2.0  # Target 2ms per learning event
            
            if avg_time > target_time:
                # Slower learning = higher cognitive load
                self.cognitive_load = min(1.0, self.cognitive_load + 0.05)
            else:
                # Faster learning = lower cognitive load
                self.cognitive_load = max(0.2, self.cognitive_load - 0.02)
        
        # Recover energy during low activity
        self.cognitive_energy = min(1.0, self.cognitive_energy + self.energy_recovery_rate)
        
        # Update motivation based on recent learning success
        if len(self.recent_learning_success) > 0:
            success_rate = np.mean(self.recent_learning_success)
            self.motivation_level = 0.3 + success_rate * 0.7  # 0.3 to 1.0 range
        
        # Adapt attention threshold based on cognitive load
        if self.cognitive_load > 0.7:
            # High load = higher attention threshold (more filtering)
            self.attention_threshold = min(0.8, self.attention_threshold + self.attention_adaptation_rate)
        else:
            # Low load = lower attention threshold (less filtering)
            self.attention_threshold = max(0.1, self.attention_threshold - self.attention_adaptation_rate)
    
    def _consume_cognitive_resources(self):
        """Consume energy and increase load when learning."""
        self.cognitive_energy = max(0.0, self.cognitive_energy - self.energy_depletion_rate)
        self.cognitive_load = min(1.0, self.cognitive_load + 0.02)
    
    def record_learning_outcome(self, learning_time: float, prediction_success: float):
        """Record the outcome of a learning event."""
        self.recent_learning_times.append(learning_time)
        self.recent_learning_success.append(prediction_success)
    
    def add_experience_to_context(self, sensory_input: List[float], prediction_error: float):
        """Add experience to recent context for novelty detection."""
        self.recent_experiences.append({
            'sensory': sensory_input,
            'error': prediction_error,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get biological laziness statistics."""
        total = max(1, self.total_experiences)
        
        return {
            'total_experiences': self.total_experiences,
            'experiences_filtered': self.experiences_filtered,
            'experiences_delayed': self.experiences_delayed,
            'experiences_learned': self.experiences_learned,
            'filter_rate': (self.experiences_filtered / total) * 100,
            'delay_rate': (self.experiences_delayed / total) * 100,
            'learning_rate': (self.experiences_learned / total) * 100,
            'cognitive_load': self.cognitive_load,
            'cognitive_energy': self.cognitive_energy,
            'motivation_level': self.motivation_level,
            'attention_threshold': self.attention_threshold,
            'novelty_threshold': self.novelty_threshold
        }


def test_biological_laziness():
    """Test the biological laziness system."""
    print("🧪 TESTING BIOLOGICAL LAZINESS STRATEGIES")
    print("=" * 50)
    print("Simulating 100 experiences with adaptive laziness...")
    print()
    
    # Create laziness manager
    laziness = BiologicalLazinessManager()
    
    # Create simple brain for testing
    brain = MinimalBrain(enable_logging=False, enable_persistence=False, quiet_mode=True)
    
    # Track performance
    learning_times = []
    total_start = time.time()
    
    for i in range(100):
        # Create varied experiences (some repetitive, some novel)
        if i < 20:
            # Early experiences - more novel
            sensory = [0.1 + i * 0.02, 0.2 + i * 0.03, 0.3 + i * 0.01, 0.4 + i * 0.02]
        elif i < 60:
            # Middle experiences - more repetitive
            base_pattern = i % 5
            sensory = [0.1 + base_pattern * 0.1, 0.2 + base_pattern * 0.1, 
                      0.3 + base_pattern * 0.1, 0.4 + base_pattern * 0.1]
        else:
            # Later experiences - mix of familiar and novel
            if i % 3 == 0:  # Novel
                sensory = [0.8 + (i-60) * 0.01, 0.7 + (i-60) * 0.01, 
                          0.6 + (i-60) * 0.01, 0.5 + (i-60) * 0.01]
            else:  # Familiar
                base_pattern = i % 3
                sensory = [0.1 + base_pattern * 0.1, 0.2 + base_pattern * 0.1,
                          0.3 + base_pattern * 0.1, 0.4 + base_pattern * 0.1]
        
        # Get prediction
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        confidence = brain_state.get('prediction_confidence', 0.5)
        
        # Simulate outcome and prediction error
        outcome = [a * 0.9 + 0.05 + np.random.normal(0, 0.1) for a in predicted_action]
        prediction_error = np.mean([(p - o) ** 2 for p, o in zip(predicted_action, outcome)])
        
        # Apply biological laziness
        decision = laziness.should_process_experience(sensory, prediction_error, confidence)
        
        if decision['action'] == 'learn':
            # Full learning (expensive)
            learn_start = time.time()
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            learn_time = (time.time() - learn_start) * 1000
            learning_times.append(learn_time)
            
            # Record learning outcome
            prediction_success = 1.0 - min(1.0, prediction_error)
            laziness.record_learning_outcome(learn_time, prediction_success)
            
        elif decision['action'] == 'buffer':
            # Temporary storage (cheap)
            learn_time = 0.1  # Minimal time for buffering
            learning_times.append(learn_time)
            
        else:  # ignore
            # No learning (free)
            learn_time = 0.0
            learning_times.append(learn_time)
        
        # Update experience context
        laziness.add_experience_to_context(sensory, prediction_error)
        
        # Progress update
        if (i + 1) % 20 == 0:
            stats = laziness.get_statistics()
            print(f"   Progress {i+1}/100: {stats['filter_rate']:.1f}% filtered, "
                  f"{stats['delay_rate']:.1f}% delayed, {stats['learning_rate']:.1f}% learned")
    
    total_time = time.time() - total_start
    brain.finalize_session()
    
    # Final analysis
    stats = laziness.get_statistics()
    avg_learning_time = np.mean(learning_times) if learning_times else 0
    
    print(f"\n🎯 BIOLOGICAL LAZINESS RESULTS:")
    print("-" * 35)
    print(f"Total simulation time: {total_time:.2f}s")
    print(f"Average learning time: {avg_learning_time:.2f}ms")
    print()
    print(f"📊 EXPERIENCE PROCESSING:")
    print(f"   Filtered (ignored):  {stats['experiences_filtered']:3d} ({stats['filter_rate']:5.1f}%)")
    print(f"   Delayed (buffered):  {stats['experiences_delayed']:3d} ({stats['delay_rate']:5.1f}%)")
    print(f"   Learned (full):      {stats['experiences_learned']:3d} ({stats['learning_rate']:5.1f}%)")
    print(f"   Total experiences:   {stats['total_experiences']:3d}")
    print()
    print(f"🧠 COGNITIVE STATE:")
    print(f"   Cognitive load:      {stats['cognitive_load']:.2f}")
    print(f"   Cognitive energy:    {stats['cognitive_energy']:.2f}")
    print(f"   Motivation level:    {stats['motivation_level']:.2f}")
    print(f"   Attention threshold: {stats['attention_threshold']:.2f}")
    print()
    print(f"💡 EFFICIENCY GAINS:")
    computational_savings = (stats['filter_rate'] + stats['delay_rate'] * 0.8) / 100
    print(f"   Computational savings: ~{computational_savings:.1%}")
    print(f"   Learning selectivity:  {stats['learning_rate']:.1f}% (vs 100% naive)")
    
    if stats['filter_rate'] > 30:
        print(f"   ✅ EXCELLENT filtering - brain is appropriately lazy")
    elif stats['filter_rate'] > 15:
        print(f"   📊 GOOD filtering - reasonable selectivity")
    else:
        print(f"   ⚠️  LOW filtering - brain may be too eager")


if __name__ == "__main__":
    test_biological_laziness()