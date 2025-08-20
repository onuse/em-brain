#!/usr/bin/env python3
"""
Enhanced Critical Mass Brain with Brilliant Magic Dust
The four critical additions that create genuine emergent intelligence:
1. Predictive Resonance Chains (causal learning)
2. Semantic Grounding (meaning through outcomes)
3. Temporal Working Memory (coherence across time)
4. Surprise-Driven Curiosity (intrinsic motivation)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import time
import logging

from .critical_mass_field_brain import CriticalMassFieldBrain, EmergenceConfig

logger = logging.getLogger(__name__)


class PredictiveResonanceChains:
    """
    Learn temporal sequences of resonances.
    When resonance A consistently precedes B, strengthen A→B coupling.
    This creates causal understanding without programming it.
    """
    
    def __init__(self, device='cuda', max_chains=100):
        self.device = device
        self.max_chains = max_chains
        
        # Temporal couplings: (before_signature, after_signature) -> strength
        self.temporal_couplings = {}
        
        # Track prediction accuracy for each chain
        self.prediction_accuracy = {}
        
        # Recent history for learning
        self.resonance_history = deque(maxlen=10)
        
    def observe_sequence(self, current_resonances: torch.Tensor):
        """
        Learn which patterns predict which by observing sequences.
        """
        # Convert resonances to signatures (simplified - use magnitude peaks)
        current_signatures = self._extract_signatures(current_resonances)
        
        # Learn from history
        if len(self.resonance_history) > 0:
            previous_signatures = self.resonance_history[-1]
            
            # Strengthen couplings between consecutive patterns
            for prev_sig in previous_signatures:
                for curr_sig in current_signatures:
                    key = (prev_sig, curr_sig)
                    
                    # Strengthen this temporal coupling
                    if key not in self.temporal_couplings:
                        self.temporal_couplings[key] = 0.0
                    
                    # Learning rate based on consistency
                    self.temporal_couplings[key] = (
                        0.9 * self.temporal_couplings[key] + 0.1
                    )
                    
                    # Prune weak couplings
                    if self.temporal_couplings[key] < 0.01:
                        del self.temporal_couplings[key]
        
        # Add to history
        self.resonance_history.append(current_signatures)
        
        # Limit number of chains
        if len(self.temporal_couplings) > self.max_chains:
            # Keep only strongest chains
            sorted_couplings = sorted(
                self.temporal_couplings.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.temporal_couplings = dict(sorted_couplings[:self.max_chains])
    
    def predict_next(self, current_resonances: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Use learned couplings to predict what patterns come next.
        """
        current_signatures = self._extract_signatures(current_resonances)
        predictions = []
        
        for curr_sig in current_signatures:
            # Find all patterns this predicts
            for (before, after), strength in self.temporal_couplings.items():
                if before == curr_sig and strength > 0.3:  # Threshold for confidence
                    predictions.append((after, strength))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def _extract_signatures(self, resonances: torch.Tensor) -> List[int]:
        """
        Convert resonance patterns to hashable signatures.
        """
        signatures = []
        
        # Handle different input shapes
        if resonances.dim() == 4:
            # Shape: (N, D, H, W) - multiple resonances
            # Find peaks in resonance magnitudes
            for i, resonance in enumerate(resonances):
                magnitude = resonance.abs().mean()
                if magnitude.item() > 0.01:  # Threshold for significance
                    # Create simple signature from index and rough magnitude
                    signature = int(i * 1000 + int(magnitude.item() * 100))
                    signatures.append(signature)
        else:
            # Single resonance or different shape
            magnitude = resonances.abs().mean()
            if magnitude.item() > 0.01:
                signatures.append(int(magnitude.item() * 1000))
        
        return signatures
    
    def get_causal_graph(self) -> Dict:
        """
        Return the learned causal relationships.
        """
        return {
            'chains': len(self.temporal_couplings),
            'strongest': sorted(
                self.temporal_couplings.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class SemanticGrounding:
    """
    Bind resonances to their real-world outcomes.
    Frequencies gain meaning through their effects.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Map resonance signatures to their outcomes
        self.resonance_outcomes = {}  # signature -> list of outcomes
        
        # Cluster resonances with similar effects (shared semantics)
        self.semantic_clusters = {}
        
        # Track success rates
        self.success_rates = {}
        
    def ground_resonance(self, resonance_signature: int,
                        before_sensors: np.ndarray,
                        after_sensors: np.ndarray,
                        motor_action: np.ndarray):
        """
        Bind a resonance to its real-world effect.
        """
        # Calculate what changed in the world
        sensor_delta = after_sensors - before_sensors
        
        # Evaluate if this was a successful outcome
        success = self._evaluate_outcome(sensor_delta, motor_action)
        
        # Store this outcome
        if resonance_signature not in self.resonance_outcomes:
            self.resonance_outcomes[resonance_signature] = []
        
        self.resonance_outcomes[resonance_signature].append({
            'sensor_change': sensor_delta,
            'motor_action': motor_action,
            'success': success,
            'timestamp': time.time()
        })
        
        # Update success rate
        outcomes = self.resonance_outcomes[resonance_signature]
        successes = sum(1 for o in outcomes if o['success'])
        self.success_rates[resonance_signature] = successes / len(outcomes)
        
        # Limit history per resonance
        if len(outcomes) > 100:
            self.resonance_outcomes[resonance_signature] = outcomes[-100:]
        
        # Update semantic clusters periodically
        if len(self.resonance_outcomes) % 10 == 0:
            self._update_semantic_clusters()
    
    def get_meaning(self, resonance_signature: int) -> Optional[np.ndarray]:
        """
        A resonance's meaning is its typical effect on the world.
        """
        if resonance_signature in self.resonance_outcomes:
            outcomes = self.resonance_outcomes[resonance_signature]
            if outcomes:
                # Average effect = meaning
                sensor_changes = [o['sensor_change'] for o in outcomes]
                return np.mean(sensor_changes, axis=0)
        return None
    
    def get_success_rate(self, resonance_signature: int) -> float:
        """
        How successful is this resonance at achieving goals?
        """
        return self.success_rates.get(resonance_signature, 0.5)
    
    def _evaluate_outcome(self, sensor_delta: np.ndarray, 
                         motor_action: np.ndarray) -> bool:
        """
        Evaluate if an outcome was successful.
        Success = movement without collision.
        """
        # Simple heuristic: success if we moved and didn't hit something
        moved = np.abs(motor_action).sum() > 0.1
        
        # Check if ultrasonic got closer (potential collision)
        if len(sensor_delta) > 0:
            ultrasonic_closer = sensor_delta[0] < -5  # Got 5cm closer to obstacle
            
            # Success = moved without collision
            return moved and not ultrasonic_closer
        
        return moved
    
    def _update_semantic_clusters(self):
        """
        Group resonances with similar effects (shared meaning).
        """
        if len(self.resonance_outcomes) < 2:
            return
        
        # Simple clustering based on effect similarity
        self.semantic_clusters = {}
        
        for sig1, outcomes1 in self.resonance_outcomes.items():
            if not outcomes1:
                continue
                
            meaning1 = np.mean([o['sensor_change'] for o in outcomes1], axis=0)
            
            # Find similar resonances
            similar = []
            for sig2, outcomes2 in self.resonance_outcomes.items():
                if sig1 != sig2 and outcomes2:
                    meaning2 = np.mean([o['sensor_change'] for o in outcomes2], axis=0)
                    
                    # Cosine similarity
                    similarity = np.dot(meaning1, meaning2) / (
                        np.linalg.norm(meaning1) * np.linalg.norm(meaning2) + 1e-8
                    )
                    
                    if similarity > 0.7:  # Threshold for semantic similarity
                        similar.append(sig2)
            
            if similar:
                self.semantic_clusters[sig1] = similar


class TemporalWorkingMemory:
    """
    Maintain context across time.
    Creates coherent behavior instead of reactive twitching.
    """
    
    def __init__(self, capacity=10, device='cuda'):
        self.capacity = capacity
        self.device = device
        
        # Rolling buffer of recent states
        self.memory_buffer = deque(maxlen=capacity)
        
        # Compressed history representation
        self.context_vector = None
        
        # Track temporal patterns
        self.temporal_patterns = []
        
    def update(self, field_state: torch.Tensor, 
              resonances: torch.Tensor,
              motor_action: np.ndarray):
        """
        Add current state to working memory.
        """
        # Store state with metadata
        self.memory_buffer.append({
            'field': field_state.clone().detach(),
            'resonances': resonances.clone().detach(),
            'motor': motor_action.copy(),
            'timestamp': time.time()
        })
        
        # Compress history into context
        self.context_vector = self._compress_history()
        
        # Detect temporal patterns
        if len(self.memory_buffer) >= 3:
            self._detect_temporal_patterns()
    
    def _compress_history(self) -> Optional[torch.Tensor]:
        """
        Create a single tensor representing recent history.
        """
        if not self.memory_buffer:
            return None
        
        # Weight recent states more heavily (exponential decay)
        weights = np.exp(np.linspace(-2, 0, len(self.memory_buffer)))
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Weighted average of field states
        field_sum = None
        for i, memory in enumerate(self.memory_buffer):
            weighted_field = memory['field'] * weights[i].reshape(1, 1, 1, 1)
            
            if field_sum is None:
                field_sum = weighted_field
            else:
                field_sum += weighted_field
        
        # Normalize
        context = field_sum / weights.sum()
        
        return context
    
    def get_context(self) -> Optional[torch.Tensor]:
        """
        Get compressed temporal context for injection into current processing.
        """
        return self.context_vector
    
    def _detect_temporal_patterns(self):
        """
        Look for recurring patterns in the temporal sequence.
        """
        # Simple pattern: repeating motor sequences
        # Convert deque to list for slicing
        buffer_list = list(self.memory_buffer)
        if len(buffer_list) >= 3:
            recent_motors = [m['motor'] for m in buffer_list[-3:]]
        else:
            return
        
        # Check if motor pattern is repeating
        if len(recent_motors) == 3:
            similarity_01 = np.corrcoef(recent_motors[0], recent_motors[1])[0, 1]
            similarity_12 = np.corrcoef(recent_motors[1], recent_motors[2])[0, 1]
            
            if similarity_01 > 0.8 and similarity_12 > 0.8:
                # Detected repeating pattern
                self.temporal_patterns.append({
                    'type': 'motor_loop',
                    'pattern': recent_motors,
                    'timestamp': time.time()
                })
    
    def has_repeating_pattern(self) -> bool:
        """
        Check if behavior is stuck in a loop.
        """
        # Check recent patterns
        recent = [p for p in self.temporal_patterns 
                 if time.time() - p['timestamp'] < 10]
        
        return len(recent) > 2  # Multiple recent repetitions


class SurpriseDrivenCuriosity:
    """
    The brain seeks states that violate its predictions.
    This is intrinsic motivation for learning.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Track prediction errors
        self.prediction_errors = deque(maxlen=100)
        
        # States that surprised us (high learning value)
        self.curiosity_targets = deque(maxlen=20)
        
        # Baseline error for comparison
        self.baseline_error = 0.5
        
        # Exploration bonus field
        self.exploration_field = None
        
    def measure_surprise(self, predicted: List[Tuple[int, float]], 
                        actual_signatures: List[int]) -> float:
        """
        How wrong were our predictions?
        """
        if not predicted:
            return 0.0
        
        # Check if predictions matched reality
        predicted_sigs = [p[0] for p in predicted]
        
        # Intersection over union
        correct = len(set(predicted_sigs) & set(actual_signatures))
        total = len(set(predicted_sigs) | set(actual_signatures))
        
        if total > 0:
            accuracy = correct / total
            error = 1.0 - accuracy
        else:
            error = 0.5  # Neutral if no predictions
        
        self.prediction_errors.append(error)
        
        # Update baseline
        if len(self.prediction_errors) > 10:
            self.baseline_error = np.mean(list(self.prediction_errors))
        
        # High error = surprising = interesting
        if error > self.baseline_error * 1.5:
            # This state violated our model - explore more like this!
            self.curiosity_targets.append({
                'error': error,
                'timestamp': time.time()
            })
        
        return error
    
    def get_curiosity_drive(self, field_shape) -> Optional[torch.Tensor]:
        """
        Create field disturbance toward surprising/novel states.
        """
        if not self.curiosity_targets:
            return None
        
        # Recent surprises drive exploration
        recent_surprises = [
            t for t in self.curiosity_targets
            if time.time() - t['timestamp'] < 30
        ]
        
        if recent_surprises:
            # Create exploration bonus proportional to surprise
            avg_surprise = np.mean([t['error'] for t in recent_surprises])
            
            # Random exploration weighted by surprise level
            if self.exploration_field is None or np.random.random() < 0.1:
                self.exploration_field = torch.randn(
                    *field_shape, 
                    device=self.device
                ) * 0.01
            
            # Scale by surprise level
            return self.exploration_field * avg_surprise
        
        return None
    
    def get_exploration_score(self) -> float:
        """
        How much is the brain exploring vs exploiting?
        """
        if not self.prediction_errors:
            return 0.5
        
        recent_errors = list(self.prediction_errors)[-10:]
        avg_error = np.mean(recent_errors)
        
        # High error = exploring, low error = exploiting
        return min(1.0, avg_error * 2)


class EnhancedCriticalMassBrain(CriticalMassFieldBrain):
    """
    The Critical Mass Brain enhanced with the four critical additions
    that create genuine emergent intelligence.
    """
    
    def __init__(self, config: Optional[EmergenceConfig] = None):
        """Initialize enhanced brain with learning systems."""
        super().__init__(config)
        
        logger.info("Initializing Enhanced Critical Mass Brain with learning systems")
        
        # Add the brilliant magic dust
        self.predictive_chains = PredictiveResonanceChains(device=self.device)
        self.semantic_grounding = SemanticGrounding(device=self.device)
        self.temporal_memory = TemporalWorkingMemory(device=self.device)
        self.curiosity = SurpriseDrivenCuriosity(device=self.device)
        
        # Track previous state for learning
        self.previous_resonances = None
        self.previous_sensors = None
        self.previous_signatures = []
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'causal_chains_learned': 0,
            'semantic_meanings': 0,
            'temporal_coherence': 0.0,
            'exploration_rate': 0.5,
            'prediction_accuracy': 0.0
        }
        
        logger.info("Enhanced brain initialized with full learning capabilities")
    
    def process(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        Enhanced processing with causal learning, semantic grounding,
        temporal memory, and curiosity-driven exploration.
        """
        # Convert sensor dict to array
        sensory_input = np.array([
            sensor_data.get('ultrasonic', 50.0) / 100.0,
            sensor_data.get('vision_detected', 0.0),
            sensor_data.get('audio_level', 0.0),
            sensor_data.get('battery', 1.0),
            (sensor_data.get('temperature', 25.0) - 25.0) / 10.0
        ])
        
        # TEMPORAL CONTEXT: Inject memory into current processing
        context = self.temporal_memory.get_context()
        if context is not None:
            # Blend current field with temporal context
            self.field = 0.8 * self.field + 0.2 * context
            
            # Check for stuck patterns
            if self.temporal_memory.has_repeating_pattern():
                logger.info("Detected repeating pattern - injecting novelty")
                # Break out of loops with random perturbation
                self.field += torch.randn_like(self.field) * 0.1
        
        # Standard processing cycle
        action, emergence_indicators = self._process_cycle(sensory_input)
        
        # CAUSAL LEARNING: Learn temporal sequences
        current_resonances = self.resonance_buffer[:self.metrics['concepts_formed']]
        if self.metrics['concepts_formed'] > 0:
            self.predictive_chains.observe_sequence(current_resonances)
            
            # Get predictions for next state
            predictions = self.predictive_chains.predict_next(current_resonances)
            
            # Extract current signatures for surprise measurement
            current_signatures = self.predictive_chains._extract_signatures(current_resonances)
            
            # CURIOSITY: Measure surprise if we had predictions
            if self.previous_signatures and predictions:
                surprise = self.curiosity.measure_surprise(
                    predictions,
                    current_signatures
                )
                self.enhanced_metrics['prediction_accuracy'] = 1.0 - surprise
                
                # Add curiosity drive to encourage exploration
                curiosity_drive = self.curiosity.get_curiosity_drive(self.field.shape)
                if curiosity_drive is not None:
                    self.field += curiosity_drive
                    
                    if self.metrics['cycles'] % 50 == 0:
                        exploration = self.curiosity.get_exploration_score()
                        logger.info(f"Exploration rate: {exploration:.1%}")
            
            # SEMANTIC GROUNDING: Bind patterns to outcomes
            if self.previous_sensors is not None:
                for i, resonance in enumerate(current_resonances):
                    if resonance.abs().mean() > 0.01:
                        # Create signature
                        signature = int(i * 1000 + int(resonance.abs().mean().item() * 100))
                        
                        # Ground this resonance
                        self.semantic_grounding.ground_resonance(
                            signature,
                            self.previous_sensors,
                            sensory_input,
                            action
                        )
                        
                        # Use success rate to modulate resonance strength
                        success_rate = self.semantic_grounding.get_success_rate(signature)
                        if success_rate < 0.3:
                            # Weaken unsuccessful patterns
                            self.resonance_coupling[i, i] *= 0.95
                        elif success_rate > 0.7:
                            # Strengthen successful patterns
                            self.resonance_coupling[i, i] *= 1.05
            
            # Store signatures for next cycle
            self.previous_signatures = current_signatures
        
        # UPDATE TEMPORAL MEMORY
        self.temporal_memory.update(self.field, current_resonances, action)
        
        # Update enhanced metrics
        self.enhanced_metrics['causal_chains_learned'] = len(
            self.predictive_chains.temporal_couplings
        )
        self.enhanced_metrics['semantic_meanings'] = len(
            self.semantic_grounding.resonance_outcomes
        )
        self.enhanced_metrics['temporal_coherence'] = (
            1.0 if context is not None else 0.0
        )
        self.enhanced_metrics['exploration_rate'] = (
            self.curiosity.get_exploration_score()
        )
        
        # Store for next cycle
        self.previous_sensors = sensory_input
        self.previous_resonances = current_resonances
        
        # Convert action to motor dict
        motor_commands = {
            'pan': float(np.tanh(action[0])),
            'tilt': float(np.tanh(action[1])),
            'motor1': float(np.tanh(action[2])),
            'motor2': float(np.tanh(action[3])),
            'motor3': float(np.tanh(action[4])),
            'motor4': float(np.tanh(action[5]))
        }
        
        # Log learning progress periodically
        if self.metrics['cycles'] % 100 == 0:
            self._log_learning_progress()
        
        return motor_commands
    
    def _log_learning_progress(self):
        """Log the brain's learning progress."""
        logger.info("=" * 60)
        logger.info("LEARNING PROGRESS REPORT")
        logger.info("-" * 60)
        
        # Causal understanding
        causal_graph = self.predictive_chains.get_causal_graph()
        logger.info(f"Causal chains learned: {causal_graph['chains']}")
        if causal_graph['strongest']:
            logger.info("Strongest predictions:")
            for (before, after), strength in causal_graph['strongest'][:3]:
                logger.info(f"  {before} → {after}: {strength:.2f}")
        
        # Semantic grounding
        logger.info(f"Patterns with meaning: {self.enhanced_metrics['semantic_meanings']}")
        
        # Success rates
        if self.semantic_grounding.success_rates:
            successful = sum(1 for r in self.semantic_grounding.success_rates.values() if r > 0.6)
            logger.info(f"Successful behaviors: {successful}")
        
        # Exploration vs exploitation
        exploration = self.enhanced_metrics['exploration_rate']
        logger.info(f"Exploration rate: {exploration:.1%}")
        logger.info(f"Prediction accuracy: {self.enhanced_metrics['prediction_accuracy']:.1%}")
        
        # Temporal coherence
        logger.info(f"Temporal coherence: {'Active' if self.enhanced_metrics['temporal_coherence'] > 0 else 'Building'}")
        
        logger.info("=" * 60)
    
    def get_telemetry(self) -> Dict:
        """Get enhanced telemetry including learning metrics."""
        base_telemetry = super().get_telemetry()
        
        # Add enhanced metrics
        base_telemetry.update({
            'causal_chains': self.enhanced_metrics['causal_chains_learned'],
            'semantic_meanings': self.enhanced_metrics['semantic_meanings'],
            'temporal_coherence': self.enhanced_metrics['temporal_coherence'],
            'exploration_rate': self.enhanced_metrics['exploration_rate'],
            'prediction_accuracy': self.enhanced_metrics['prediction_accuracy'],
            'learning_score': self._calculate_learning_score()
        })
        
        return base_telemetry
    
    def _calculate_learning_score(self) -> float:
        """
        Calculate overall learning progress score.
        """
        scores = [
            min(1.0, self.enhanced_metrics['causal_chains_learned'] / 50),  # 50 chains = good
            min(1.0, self.enhanced_metrics['semantic_meanings'] / 30),  # 30 meanings = good
            self.enhanced_metrics['temporal_coherence'],
            self.enhanced_metrics['prediction_accuracy'],
            1.0 - abs(self.enhanced_metrics['exploration_rate'] - 0.3)  # 30% exploration is optimal
        ]
        
        return np.mean(scores)