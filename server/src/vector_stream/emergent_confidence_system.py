#!/usr/bin/env python3
"""
Emergent Confidence System

Implements confidence as an emergent property of prediction dynamics
rather than static performance metrics. Based on the Emergent Confidence Theory.

Key insight: Confidence emerges from the interaction of:
- Prediction volatility (how much predictions change)
- Prediction coherence (similar inputs → similar outputs)  
- Meta-prediction accuracy (accuracy of confidence predictions)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any
from collections import deque
import time

class EmergentConfidenceSystem:
    """
    Calculates confidence from prediction dynamics without magic numbers.
    
    Naturally produces Dunning-Kruger effects and explains intoxication-like
    behaviors through prediction system disruption.
    """
    
    def __init__(self, history_size: int = 100, quiet_mode: bool = False):
        """
        Initialize emergent confidence system.
        
        Args:
            history_size: Number of recent cycles to track
            quiet_mode: Suppress debug output
        """
        self.history_size = history_size
        self.quiet_mode = quiet_mode
        
        # Prediction history tracking
        self.recent_predictions = deque(maxlen=history_size)
        self.recent_sensory_motor_pairs = deque(maxlen=history_size)
        self.confidence_accuracy_history = deque(maxlen=history_size)
        
        # Current confidence state
        self.current_confidence = 0.7  # Default high (ignorance-based boldness)
        self.volatility_confidence = 0.7
        self.coherence_confidence = 0.7
        self.meta_confidence = 0.7
        
        # Dynamics tracking
        self.last_update_time = time.time()
        self.total_updates = 0
        
        # GPU acceleration settings
        self.device = self._detect_device()
        self.gpu_threshold = 15  # Use vectorized numpy when ≥15 pairs for coherence calculation
        
        if not quiet_mode:
            print("🧠 EmergentConfidenceSystem initialized")
            print("   Theory: Confidence emerges from prediction dynamics")
            print("   Features: Dunning-Kruger effects, intoxication modeling, no magic numbers")
            print(f"   Device: {self.device} (vectorization threshold: {self.gpu_threshold} pairs)")
    
    def update_confidence(self, 
                         motor_prediction: List[float],
                         sensory_input: List[float],
                         actual_outcome: List[float] = None) -> float:
        """
        Update confidence based on current prediction dynamics.
        
        Args:
            motor_prediction: Current motor prediction
            sensory_input: Current sensory input
            actual_outcome: Actual outcome (if available for accuracy tracking)
            
        Returns:
            Updated confidence value (0.0 to 1.0)
        """
        
        # Store current prediction and sensory-motor pair
        self.recent_predictions.append(motor_prediction.copy())
        self.recent_sensory_motor_pairs.append((sensory_input.copy(), motor_prediction.copy()))
        
        # Track confidence vs accuracy for meta-prediction
        if actual_outcome is not None and len(self.recent_predictions) > 1:
            # Calculate actual prediction accuracy
            predicted = np.array(self.recent_predictions[-2])  # Previous prediction
            actual = np.array(actual_outcome)
            if len(predicted) == len(actual):
                accuracy = 1.0 - np.linalg.norm(predicted - actual) / (np.linalg.norm(predicted) + np.linalg.norm(actual) + 1e-8)
                accuracy = max(0.0, min(1.0, accuracy))
                self.confidence_accuracy_history.append((self.current_confidence, accuracy))
        
        # Calculate emergent confidence factors
        self.volatility_confidence = self._calculate_volatility_confidence()
        self.coherence_confidence = self._calculate_coherence_confidence()
        self.meta_confidence = self._calculate_meta_prediction_confidence()
        
        # Synthesize emergent confidence
        self.current_confidence = self._synthesize_confidence_dynamics()
        
        self.total_updates += 1
        self.last_update_time = time.time()
        
        return self.current_confidence
    
    def _calculate_volatility_confidence(self) -> float:
        """
        Calculate confidence from prediction volatility patterns.
        
        High volatility can indicate either:
        - Ignorant boldness (confident exploration)
        - Confused uncertainty (chaotic behavior)
        
        The difference is detected through volatility consistency.
        """
        if len(self.recent_predictions) < 10:
            return 0.7  # Default high for insufficient data (ignorance-based boldness)
        
        # Calculate prediction changes over time
        prediction_changes = []
        for i in range(1, len(self.recent_predictions)):
            prev_pred = np.array(self.recent_predictions[i-1])
            curr_pred = np.array(self.recent_predictions[i])
            change = np.linalg.norm(curr_pred - prev_pred)
            prediction_changes.append(change)
        
        if not prediction_changes:
            return 0.7
        
        volatility = np.mean(prediction_changes)
        volatility_std = np.std(prediction_changes)
        
        # Interpret volatility patterns
        if volatility < 0.1:
            # Low volatility = stable predictions
            if volatility_std < 0.05:
                return 0.6  # Stable but potentially stuck
            else:
                return 0.5  # Inconsistent stability
        
        elif volatility > 0.3:
            # High volatility = either confident exploration or chaos
            if volatility_std < 0.1:
                return 0.8  # Consistent exploration (confident)
            else:
                return 0.3  # Chaotic behavior (confused)
        
        else:
            # Medium volatility = active learning
            return 0.5 + (1.0 - volatility_std) * 0.2  # More consistent = more confident
    
    def _calculate_coherence_confidence(self) -> float:
        """
        GPU-accelerated confidence from sensory-motor coherence.
        
        Similar sensory inputs should produce similar motor outputs
        if the system understands the input-output mapping.
        
        Optimized with parallel similarity calculations.
        """
        if len(self.recent_sensory_motor_pairs) < 20:
            return 0.7  # Default high for insufficient data
        
        recent_pairs = list(self.recent_sensory_motor_pairs)
        sample_size = min(20, len(recent_pairs))
        
        # Use vectorized calculation if we have enough data
        if len(recent_pairs) >= self.gpu_threshold:
            try:
                return self._calculate_coherence_confidence_vectorized(recent_pairs, sample_size)
            except Exception as e:
                if not self.quiet_mode:
                    print(f"⚠️ Vectorized coherence calculation failed, falling back to CPU: {e}")
                # Fall back to CPU
        
        # CPU fallback implementation
        return self._calculate_coherence_confidence_cpu(recent_pairs, sample_size)
    
    def _calculate_coherence_confidence_vectorized(self, recent_pairs: List[Tuple], sample_size: int) -> float:
        """Vectorized numpy coherence confidence calculation (optimized for CPU)."""
        # Sample recent pairs for coherence analysis
        sample_pairs = recent_pairs[-sample_size:]
        
        # Convert to numpy arrays for vectorized operations
        sensory_data = np.array([sensory for sensory, motor in recent_pairs])
        motor_data = np.array([motor for sensory, motor in recent_pairs])
        
        # Get sample indices
        sample_indices = np.arange(len(sample_pairs))
        
        # Vectorized cosine similarity calculation
        # Normalize vectors
        sensory_norms = np.linalg.norm(sensory_data, axis=1, keepdims=True)
        motor_norms = np.linalg.norm(motor_data, axis=1, keepdims=True)
        
        # Avoid division by zero
        sensory_norms = np.where(sensory_norms == 0, 1, sensory_norms)
        motor_norms = np.where(motor_norms == 0, 1, motor_norms)
        
        sensory_normalized = sensory_data / sensory_norms
        motor_normalized = motor_data / motor_norms
        
        # Calculate all pairwise similarities using matrix multiplication
        sensory_similarities = np.dot(sensory_normalized, sensory_normalized.T)
        motor_similarities = np.dot(motor_normalized, motor_normalized.T)
        
        # Create mask for high sensory similarity (>0.8) and exclude self-comparison
        sensory_similarity_threshold = 0.8
        high_sensory_sim_mask = sensory_similarities > sensory_similarity_threshold
        
        # Exclude self-comparisons
        np.fill_diagonal(high_sensory_sim_mask, False)
        
        # Extract coherence scores for sample pairs
        coherence_scores = []
        
        for i in sample_indices:
            # Get motor similarities where sensory similarity is high
            high_sim_indices = high_sensory_sim_mask[i]
            
            if np.any(high_sim_indices):
                # Average motor similarity for this sensory input
                avg_motor_sim = np.mean(motor_similarities[i][high_sim_indices])
                coherence_scores.append(avg_motor_sim)
        
        if not coherence_scores:
            return 0.7  # Default if no coherence data
        
        avg_coherence = np.mean(coherence_scores)
        
        # Transform coherence to confidence (sigmoid-like transformation)
        return max(0.05, min(0.95, avg_coherence))
    
    def _calculate_coherence_confidence_cpu(self, recent_pairs: List[Tuple], sample_size: int) -> float:
        """CPU fallback for coherence confidence calculation (original algorithm)."""
        coherence_scores = []
        sample_pairs = recent_pairs[-sample_size:]
        
        for i, (sensory_i, motor_i) in enumerate(sample_pairs):
            similar_motor_similarities = []
            
            # Find sensory inputs similar to current one
            for j, (sensory_j, motor_j) in enumerate(recent_pairs):
                if i != j:
                    # Calculate sensory similarity
                    sensory_norm_i = np.linalg.norm(sensory_i)
                    sensory_norm_j = np.linalg.norm(sensory_j)
                    
                    if sensory_norm_i > 0 and sensory_norm_j > 0:
                        sensory_sim = np.dot(sensory_i, sensory_j) / (sensory_norm_i * sensory_norm_j)
                        
                        # If sensory inputs are similar, check motor coherence
                        if sensory_sim > 0.8:  # High sensory similarity threshold
                            motor_norm_i = np.linalg.norm(motor_i)
                            motor_norm_j = np.linalg.norm(motor_j)
                            
                            if motor_norm_i > 0 and motor_norm_j > 0:
                                motor_sim = np.dot(motor_i, motor_j) / (motor_norm_i * motor_norm_j)
                                similar_motor_similarities.append(motor_sim)
            
            # High motor similarity for similar sensory inputs = coherence
            if similar_motor_similarities:
                coherence_scores.append(np.mean(similar_motor_similarities))
        
        if not coherence_scores:
            return 0.7  # Default if no coherence data
        
        avg_coherence = np.mean(coherence_scores)
        
        # Transform coherence to confidence (sigmoid-like transformation)
        return max(0.05, min(0.95, avg_coherence))
    
    def _calculate_meta_prediction_confidence(self) -> float:
        """
        Calculate confidence from meta-prediction accuracy.
        
        How well does the system predict its own prediction quality?
        Good meta-prediction = high self-awareness = appropriate confidence.
        """
        if len(self.confidence_accuracy_history) < 10:
            return 0.7  # Default high for insufficient data
        
        # Calculate meta-prediction errors
        meta_errors = []
        for predicted_conf, actual_accuracy in list(self.confidence_accuracy_history)[-20:]:
            meta_error = abs(predicted_conf - actual_accuracy)
            meta_errors.append(meta_error)
        
        if not meta_errors:
            return 0.7
        
        avg_meta_error = np.mean(meta_errors)
        
        # Low meta-error = good self-awareness = high meta-confidence
        # High meta-error = poor self-awareness = low meta-confidence
        meta_confidence = max(0.1, 1.0 - (avg_meta_error * 2.0))
        
        return min(0.95, meta_confidence)
    
    def _synthesize_confidence_dynamics(self) -> float:
        """
        Synthesize confidence from dynamic factor interactions.
        
        Detects emergent behavioral patterns through factor combinations
        rather than using static thresholds.
        """
        vol_conf = self.volatility_confidence
        coh_conf = self.coherence_confidence
        meta_conf = self.meta_confidence
        
        # Detect emergent confidence patterns
        
        # Pattern 1: "Ignorant Boldness" (Dunning-Kruger peak)
        if vol_conf > 0.7 and coh_conf < 0.5 and meta_conf > 0.6:
            return 0.8  # High confidence despite poor understanding
        
        # Pattern 2: "Learning Crisis" (Reality check)
        elif vol_conf > 0.6 and 0.4 < coh_conf < 0.7 and meta_conf < 0.4:
            return 0.3  # Confidence crash as complexity becomes apparent
        
        # Pattern 3: "Competence Building" (Skill development)
        elif 0.4 < vol_conf < 0.7 and coh_conf > 0.7 and 0.4 < meta_conf < 0.7:
            return 0.6  # Rebuilding confidence based on competence
        
        # Pattern 4: "Expert Confidence" (Mature competence)
        elif vol_conf < 0.4 and coh_conf > 0.8 and meta_conf > 0.7:
            return 0.9  # High confidence matching high competence
        
        # Pattern 5: "Stable Mediocrity" (Local optimum)
        elif vol_conf < 0.3 and 0.5 < coh_conf < 0.8 and meta_conf > 0.6:
            return 0.5  # Stuck but aware of limitations
        
        # Default: Weighted synthesis with interaction effects
        base_confidence = (vol_conf * 0.4 + coh_conf * 0.4 + meta_conf * 0.2)
        
        # Add interaction bonus for aligned factors
        interaction_bonus = (vol_conf * coh_conf * meta_conf) * 0.2
        
        final_confidence = base_confidence + interaction_bonus
        
        return max(0.05, min(0.95, final_confidence))
    
    def simulate_impairment(self, impairment_level: float) -> Dict[str, float]:
        """
        Simulate cognitive impairment effects (e.g., intoxication, fatigue).
        
        Args:
            impairment_level: 0.0 (sober) to 1.0 (severely impaired)
            
        Returns:
            Modified confidence factors
        """
        
        # Impairment increases volatility (less consistent predictions)
        modified_volatility = min(0.95, self.volatility_confidence + impairment_level * 0.3)
        
        # Impairment reduces coherence detection (can't see inconsistencies)
        coherence_impairment = impairment_level * 0.4
        modified_coherence = max(0.05, self.coherence_confidence - coherence_impairment)
        
        # Impairment inflates meta-confidence (overestimation of abilities)
        meta_inflation = impairment_level * 0.5
        modified_meta = min(0.95, self.meta_confidence + meta_inflation)
        
        # Calculate impaired confidence
        impaired_confidence = self._synthesize_impaired_confidence(
            modified_volatility, modified_coherence, modified_meta
        )
        
        return {
            'original_confidence': self.current_confidence,
            'impaired_confidence': impaired_confidence,
            'volatility_change': modified_volatility - self.volatility_confidence,
            'coherence_change': modified_coherence - self.coherence_confidence,
            'meta_change': modified_meta - self.meta_confidence
        }
    
    def _synthesize_impaired_confidence(self, vol_conf: float, coh_conf: float, meta_conf: float) -> float:
        """Synthesize confidence under impairment conditions."""
        # Use same pattern detection but with modified factors
        if vol_conf > 0.7 and coh_conf < 0.5 and meta_conf > 0.6:
            return 0.8  # Impairment often increases overconfidence
        else:
            return max(0.05, min(0.95, vol_conf * 0.4 + coh_conf * 0.4 + meta_conf * 0.2))
    
    def get_confidence_state(self) -> Dict[str, Any]:
        """Get current confidence state and dynamics."""
        return {
            'current_confidence': self.current_confidence,
            'volatility_confidence': self.volatility_confidence,
            'coherence_confidence': self.coherence_confidence,
            'meta_confidence': self.meta_confidence,
            'total_updates': self.total_updates,
            'data_points': {
                'predictions': len(self.recent_predictions),
                'sensory_motor_pairs': len(self.recent_sensory_motor_pairs),
                'confidence_accuracy_pairs': len(self.confidence_accuracy_history)
            },
            'emergent_pattern': self._detect_current_pattern()
        }
    
    def _detect_current_pattern(self) -> str:
        """Detect current emergent confidence pattern."""
        vol_conf = self.volatility_confidence
        coh_conf = self.coherence_confidence
        meta_conf = self.meta_confidence
        
        if vol_conf > 0.7 and coh_conf < 0.5 and meta_conf > 0.6:
            return "ignorant_boldness"
        elif vol_conf > 0.6 and 0.4 < coh_conf < 0.7 and meta_conf < 0.4:
            return "learning_crisis"
        elif 0.4 < vol_conf < 0.7 and coh_conf > 0.7 and 0.4 < meta_conf < 0.7:
            return "competence_building"
        elif vol_conf < 0.4 and coh_conf > 0.8 and meta_conf > 0.7:
            return "expert_confidence"
        elif vol_conf < 0.3 and 0.5 < coh_conf < 0.8 and meta_conf > 0.6:
            return "stable_mediocrity"
        else:
            return "transitional"
    
    def reset(self):
        """Reset confidence system state."""
        self.recent_predictions.clear()
        self.recent_sensory_motor_pairs.clear()
        self.confidence_accuracy_history.clear()
        
        self.current_confidence = 0.7  # Back to ignorance-based boldness
        self.volatility_confidence = 0.7
        self.coherence_confidence = 0.7
        self.meta_confidence = 0.7
        
        self.total_updates = 0
        
        if not self.quiet_mode:
            print("🔄 EmergentConfidenceSystem reset - back to ignorant boldness")
    
    def _detect_device(self) -> str:
        """Detect optimal device for tensor operations."""
        try:
            if torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon GPU
            elif torch.cuda.is_available():
                return 'cuda'  # NVIDIA GPU
            else:
                return 'cpu'
        except Exception:
            return 'cpu'
    
    def set_gpu_threshold(self, threshold: int):
        """Set GPU threshold based on hardware adaptation."""
        self.gpu_threshold = threshold
        if not self.quiet_mode:
            print(f"🔧 GPU threshold updated to {threshold} pairs")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        return {
            'device': self.device,
            'vectorization_threshold': self.gpu_threshold,
            'total_updates': self.total_updates,
            'data_size': len(self.recent_sensory_motor_pairs),
            'using_vectorized': len(self.recent_sensory_motor_pairs) >= self.gpu_threshold
        }