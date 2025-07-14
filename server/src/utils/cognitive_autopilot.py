"""
Cognitive Autopilot System

Adaptive computational intensity based on prediction confidence and environmental familiarity.
Implements biological "mental autopilot" - coast when confident, engage when surprised.

Integrates with existing adaptive systems:
- AdaptiveAttentionScorer (attention baseline)
- AdaptiveThresholds (cognitive load monitoring)  
- NaturalAttentionSimilarity (retrieval modes)
- Hardware Adaptation (performance limits)
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional
from collections import deque
from enum import Enum


class CognitiveMode(Enum):
    """Cognitive processing intensity modes."""
    AUTOPILOT = "autopilot"      # 90%+ confidence - minimal analysis
    FOCUSED = "focused"          # 70-90% confidence - moderate analysis  
    DEEP_THINK = "deep_think"    # <70% confidence - full analysis


class CognitiveAutopilot:
    """
    Adaptive computational intensity system.
    
    Biological inspiration: Real brains vary computational effort based on:
    - Prediction confidence (familiar vs novel situations)
    - Recent performance (successful vs struggling)
    - Environmental stability (predictable vs chaotic)
    - Cognitive resources (energy conservation)
    """
    
    def __init__(self, 
                 autopilot_confidence_threshold: float = 0.90,
                 focused_confidence_threshold: float = 0.70,
                 stability_window: int = 10):
        """
        Initialize cognitive autopilot.
        
        Args:
            autopilot_confidence_threshold: Confidence level for autopilot mode
            focused_confidence_threshold: Confidence level for focused mode
            stability_window: Number of cycles to assess stability
        """
        self.autopilot_threshold = autopilot_confidence_threshold
        self.focused_threshold = focused_confidence_threshold
        self.stability_window = stability_window
        
        # State tracking
        self.current_mode = CognitiveMode.DEEP_THINK
        self.mode_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=stability_window)
        self.prediction_error_history = deque(maxlen=stability_window)
        self.surprise_events = deque(maxlen=20)
        
        # Performance tracking
        self.mode_switch_count = 0
        self.time_in_modes = {mode: 0.0 for mode in CognitiveMode}
        self.last_mode_switch = time.time()
        
        # Integration with existing systems
        self.attention_mode_mapping = {
            CognitiveMode.AUTOPILOT: 'normal',      # Standard attention
            CognitiveMode.FOCUSED: 'hybrid',        # Boosted attention
            CognitiveMode.DEEP_THINK: 'utility_focused'  # Maximum attention
        }
        
        print(f"🧠 CognitiveAutopilot initialized")
        print(f"   Autopilot: >{self.autopilot_threshold:.0%} confidence")
        print(f"   Focused: {self.focused_threshold:.0%}-{self.autopilot_threshold:.0%} confidence")
        print(f"   Deep Think: <{self.focused_threshold:.0%} confidence")
    
    def update_cognitive_state(self, 
                             prediction_confidence: float,
                             prediction_error: float,
                             brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update cognitive mode based on current brain state.
        
        Args:
            prediction_confidence: Current prediction confidence (0.0-1.0)
            prediction_error: Current prediction error (0.0+)
            brain_state: Current brain state information
            
        Returns:
            Cognitive autopilot recommendations for system adaptation
        """
        # Track state
        self.confidence_history.append(prediction_confidence)
        self.prediction_error_history.append(prediction_error)
        
        # Assess environmental stability
        stability_score = self._assess_stability()
        surprise_level = self._assess_surprise_level(prediction_error)
        
        # Determine appropriate cognitive mode
        new_mode = self._determine_cognitive_mode(
            prediction_confidence, stability_score, surprise_level
        )
        
        # Update mode if changed
        mode_changed = False
        if new_mode != self.current_mode:
            self._switch_cognitive_mode(new_mode)
            mode_changed = True
        
        # Generate system recommendations
        recommendations = self._generate_system_recommendations(brain_state)
        
        return {
            'cognitive_mode': self.current_mode.value,
            'mode_changed': mode_changed,
            'confidence': prediction_confidence,
            'stability_score': stability_score,
            'surprise_level': surprise_level,
            'recommendations': recommendations,
            'performance_profile': self._get_performance_profile()
        }
    
    def _assess_stability(self) -> float:
        """Assess environmental/performance stability."""
        if len(self.confidence_history) < 5:
            return 0.5  # Neutral stability
        
        # Stability = low variance in recent confidence
        recent_confidences = list(self.confidence_history)[-5:]
        confidence_stability = 1.0 - np.std(recent_confidences)
        
        # Stability = low variance in recent prediction errors  
        if len(self.prediction_error_history) >= 5:
            recent_errors = list(self.prediction_error_history)[-5:]
            error_stability = 1.0 - min(1.0, np.std(recent_errors))
        else:
            error_stability = 0.5
        
        # Combined stability score
        return (confidence_stability + error_stability) / 2.0
    
    def _assess_surprise_level(self, current_error: float) -> float:
        """Assess current surprise level relative to recent experience."""
        if len(self.prediction_error_history) < 3:
            return 0.5  # Neutral surprise
        
        recent_errors = list(self.prediction_error_history)[:-1]  # Exclude current
        avg_recent_error = np.mean(recent_errors)
        
        # Surprise = how much current error exceeds recent average
        if avg_recent_error > 0:
            surprise = min(2.0, current_error / avg_recent_error) / 2.0
        else:
            surprise = 0.5 if current_error < 0.1 else 1.0
        
        # Track surprise events
        if surprise > 0.8:
            self.surprise_events.append(time.time())
        
        return surprise
    
    def _determine_cognitive_mode(self, 
                                confidence: float, 
                                stability: float, 
                                surprise: float) -> CognitiveMode:
        """Determine appropriate cognitive mode."""
        
        # Primary decision based on confidence thresholds
        if confidence >= self.autopilot_threshold:
            # High confidence - but check for instability
            if stability > 0.5 and surprise < 0.5:
                return CognitiveMode.AUTOPILOT
            else:
                return CognitiveMode.FOCUSED  # High confidence but unstable
        
        elif confidence >= self.focused_threshold:
            # Medium confidence - focused mode unless very unstable
            if surprise > 0.8:
                return CognitiveMode.DEEP_THINK
            else:
                return CognitiveMode.FOCUSED
        
        else:
            # Low confidence - deep thinking needed
            return CognitiveMode.DEEP_THINK
    
    def _switch_cognitive_mode(self, new_mode: CognitiveMode):
        """Switch to new cognitive mode."""
        current_time = time.time()
        
        # Track time in previous mode
        time_in_mode = current_time - self.last_mode_switch
        self.time_in_modes[self.current_mode] += time_in_mode
        
        # Switch mode
        old_mode = self.current_mode
        self.current_mode = new_mode
        self.mode_history.append((current_time, old_mode.value, new_mode.value))
        self.mode_switch_count += 1
        self.last_mode_switch = current_time
        
        print(f"🧠 Cognitive mode: {old_mode.value} → {new_mode.value}")
    
    def _generate_system_recommendations(self, brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for other brain systems."""
        
        recommendations = {
            'attention_retrieval_mode': self.attention_mode_mapping[self.current_mode],
            'pattern_analysis_intensity': self._get_pattern_analysis_intensity(),
            'working_memory_adjustment': self._get_working_memory_adjustment(),
            'similarity_search_depth': self._get_similarity_search_depth()
        }
        
        return recommendations
    
    def _get_pattern_analysis_intensity(self) -> str:
        """Get recommended pattern analysis intensity."""
        return {
            CognitiveMode.AUTOPILOT: 'minimal',    # Skip or cache-only
            CognitiveMode.FOCUSED: 'selective',    # Analyze only novel patterns
            CognitiveMode.DEEP_THINK: 'full'       # Complete analysis
        }[self.current_mode]
    
    def _get_working_memory_adjustment(self) -> float:
        """Get working memory size adjustment factor."""
        return {
            CognitiveMode.AUTOPILOT: 0.8,     # Smaller working memory
            CognitiveMode.FOCUSED: 1.0,       # Normal working memory
            CognitiveMode.DEEP_THINK: 1.2     # Expanded working memory
        }[self.current_mode]
    
    def _get_similarity_search_depth(self) -> int:
        """Get recommended similarity search depth."""
        base_depth = 10
        return {
            CognitiveMode.AUTOPILOT: max(5, int(base_depth * 0.6)),
            CognitiveMode.FOCUSED: base_depth,
            CognitiveMode.DEEP_THINK: int(base_depth * 1.5)
        }[self.current_mode]
    
    def _get_performance_profile(self) -> Dict[str, Any]:
        """Get cognitive autopilot performance statistics."""
        total_time = sum(self.time_in_modes.values())
        
        if total_time == 0:
            mode_percentages = {mode.value: 0.0 for mode in CognitiveMode}
        else:
            mode_percentages = {
                mode.value: (time_spent / total_time) * 100
                for mode, time_spent in self.time_in_modes.items()
            }
        
        # Recent surprise frequency
        recent_surprises = [t for t in self.surprise_events if time.time() - t < 60]
        surprise_frequency = len(recent_surprises) / 60.0  # Per second
        
        return {
            'current_mode': self.current_mode.value,
            'mode_switches': self.mode_switch_count,
            'time_distribution': mode_percentages,
            'recent_surprise_frequency': surprise_frequency,
            'stability_score': self._assess_stability() if self.confidence_history else 0.5
        }
    
    def should_skip_pattern_analysis(self) -> bool:
        """Should pattern analysis be skipped this cycle?"""
        return self.current_mode == CognitiveMode.AUTOPILOT
    
    def should_use_cached_patterns(self) -> bool:
        """Should we use cached patterns instead of fresh analysis?"""
        return self.current_mode in [CognitiveMode.AUTOPILOT, CognitiveMode.FOCUSED]
    
    def get_recommended_attention_mode(self) -> str:
        """Get recommended attention retrieval mode."""
        return self.attention_mode_mapping[self.current_mode]


# Integration helper functions
def create_cognitive_autopilot(config: Optional[Dict[str, Any]] = None) -> CognitiveAutopilot:
    """Create cognitive autopilot with optional configuration."""
    if config is None:
        config = {}
    
    return CognitiveAutopilot(
        autopilot_confidence_threshold=config.get('autopilot_threshold', 0.90),
        focused_confidence_threshold=config.get('focused_threshold', 0.70),
        stability_window=config.get('stability_window', 10)
    )


def integrate_autopilot_with_brain(brain, autopilot: CognitiveAutopilot):
    """
    Integration helper to connect autopilot with existing brain systems.
    
    This would modify:
    - Pattern analyzer to respect intensity recommendations
    - Attention system to use recommended retrieval modes
    - Working memory to adjust size based on cognitive load
    """
    # Store reference for brain to use autopilot recommendations
    brain.cognitive_autopilot = autopilot
    
    print("🔗 Cognitive autopilot integrated with brain systems")
    print("   Pattern analysis will adapt to cognitive demands")
    print("   Attention system will use mode-appropriate retrieval")
    print("   Working memory will scale with cognitive load")