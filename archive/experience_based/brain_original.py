"""
Minimal Brain Coordinator - Vector Stream Architecture

The central orchestrator that coordinates vector stream processing:
1. Modular Streams - continuous sensory, motor, and temporal vectors
2. Cross-Stream Learning - associations between stream patterns  
3. Temporal Integration - time as data stream with organic metronome
4. Continuous Prediction - emergent from vector flow dynamics

Vector streams replace discrete experience packages with biologically-realistic
continuous processing that handles timing and dead reckoning naturally.
"""

import time
from typing import List, Dict, Tuple, Optional, Any

from .vector_stream.minimal_brain import MinimalVectorStreamBrain
from .utils.cognitive_autopilot import CognitiveAutopilot
from .utils.brain_logger import BrainLogger
from .utils.hardware_adaptation import get_hardware_adaptation, record_brain_cycle_performance


class MinimalBrain:
    """
    The complete minimal brain - vector stream processing for biological realism.
    
    Everything emerges from continuous vector flow:
    - Spatial navigation emerges from sensory stream patterns
    - Motor skills emerge from sensory-motor stream associations
    - Temporal adaptation emerges from organic metronome integration
    - Dead reckoning emerges from cross-stream prediction dynamics
    """
    
    def __init__(self, config=None, enable_logging=True, log_session_name=None, quiet_mode=False, 
                 sensory_dim=16, motor_dim=8, temporal_dim=4):
        """Initialize the minimal brain with vector stream architecture."""
        
        # Store config for use in other methods
        self.config = config
        self.quiet_mode = quiet_mode
        
        # Hardware adaptation system
        self.hardware_adaptation = get_hardware_adaptation()
        
        # Core vector stream brain
        self.vector_brain = MinimalVectorStreamBrain(
            sensory_dim=sensory_dim,
            motor_dim=motor_dim, 
            temporal_dim=temporal_dim
        )
        
        # Initialize cognitive autopilot for adaptive intensity control
        self.cognitive_autopilot = CognitiveAutopilot()
        
        # Brain state tracking
        self.total_cycles = 0
        self.brain_start_time = time.time()
        
        # Logging system for emergence analysis
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = BrainLogger(session_name=log_session_name, config=config)
        else:
            self.logger = None
        
        if not quiet_mode:
            print(f"🧠 MinimalBrain initialized - Vector Stream Architecture")
            print(f"   Sensory stream: {sensory_dim}D")
            print(f"   Motor stream: {motor_dim}D") 
            print(f"   Temporal stream: {temporal_dim}D")
            print(f"   Continuous prediction and dead reckoning enabled")
        else:
            # Show minimal essential summary
            print(f"🧠 Vector Brain ready: {sensory_dim}D→{motor_dim}D processing")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = 4) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input and return predicted action using vector streams.
        
        This is the complete brain cycle:
        1. Update vector streams with current input
        2. Generate action through cross-stream prediction
        3. Return action with brain state info
        
        Args:
            sensory_input: Current sensory observation vector
            action_dimensions: Number of action dimensions to output (ignored - motor stream determines size)
            
        Returns:
            Tuple of (predicted_action, brain_state_info)
        """
        process_start_time = time.time()
        
        # Process through vector stream brain
        predicted_action, vector_brain_state = self.vector_brain.process_sensory_input(sensory_input)
        
        # Update cognitive autopilot with vector stream confidence
        confidence = vector_brain_state['prediction_confidence']
        prediction_error = 1.0 - confidence
        
        initial_brain_state = {
            'prediction_confidence': confidence,
            'total_cycles': self.total_cycles
        }
        
        autopilot_state = self.cognitive_autopilot.update_cognitive_state(
            confidence, prediction_error, initial_brain_state
        )
        
        self.total_cycles += 1
        
        # Log prediction outcome if logging enabled
        if self.logger and predicted_action:
            self.logger.log_prediction_outcome(
                predicted_action, sensory_input, confidence, 0  # Vector streams don't use "similar experiences"
            )
        
        # Performance monitoring
        cycle_time = time.time() - process_start_time
        cycle_time_ms = cycle_time * 1000
        
        # Record performance for hardware adaptation
        memory_usage_mb = 50.0  # Vector streams use much less memory than experience storage
        record_brain_cycle_performance(cycle_time_ms, memory_usage_mb)
        
        # Compile brain state
        brain_state = {
            'total_cycles': self.total_cycles,
            'prediction_confidence': confidence,
            'cycle_time': cycle_time,
            'cycle_time_ms': cycle_time_ms,
            'hardware_adaptive_limits': self.hardware_adaptation.get_cognitive_limits(),
            'cognitive_autopilot': autopilot_state,
            'brain_uptime': time.time() - self.brain_start_time,
            'architecture': 'vector_stream',
            **vector_brain_state  # Include vector stream specific state
        }
        
        return predicted_action, brain_state
    
    def store_experience(self, sensory_input: List[float], action_taken: List[float], 
                        outcome: List[float], predicted_action: List[float] = None) -> str:
        """
        Store experience in vector streams (no discrete experience objects).
        
        Args:
            sensory_input: What was sensed
            action_taken: What action was taken  
            outcome: What actually happened
            predicted_action: What was predicted (for computing error)
            
        Returns:
            The experience ID (timestamp-based for vector streams)
        """
        # In vector streams, we don't store discrete experiences
        # Instead, the vector brain continuously learns from the stream flow
        
        # Generate experience ID for compatibility
        experience_id = f"vector_stream_{int(time.time() * 1000)}"
        
        # Log learning outcome if logging enabled
        if self.logger:
            # Compute simple prediction error for logging
            prediction_error = 0.5
            if predicted_action:
                import numpy as np
                predicted = np.array(predicted_action[:len(outcome)])
                actual = np.array(outcome[:len(predicted_action)])
                if len(predicted) > 0:
                    error = np.linalg.norm(predicted - actual)
                    max_error = np.linalg.norm(predicted) + np.linalg.norm(actual)
                    prediction_error = min(1.0, error / max_error) if max_error > 0 else 0.0
            
            self.logger.log_prediction_outcome(
                action_taken, sensory_input, 1.0 - prediction_error, 0
            )
        
        return experience_id
    
    def compute_intrinsic_reward(self, current_prediction_error: float = None) -> float:
        """
        The robot's DNA equivalent - minimize prediction error, but not to zero.
        
        This is the single fundamental drive that replaces biological motivations.
        The optimal prediction error adapts based on learning success rather than being hardcoded.
        
        Args:
            current_prediction_error: Current prediction error (0.0-1.0)
            
        Returns:
            Intrinsic reward (0.0-1.0, higher is better)
        """
        optimal_error = self.optimal_prediction_error  # Adaptive sweet spot for learnable patterns
        
        if current_prediction_error is None:
            # Use recent average prediction error
            recent_errors = []
            for exp in list(self.experience_storage._experiences.values())[-10:]:
                recent_errors.append(exp.prediction_error)
            
            if recent_errors:
                current_prediction_error = sum(recent_errors) / len(recent_errors)
            else:
                current_prediction_error = 0.5  # Default moderate error
        
        # Reward is maximized when prediction error is at optimal level
        # Decreases as we move away from optimal in either direction
        error_distance = abs(current_prediction_error - optimal_error)
        reward = 1.0 - (error_distance / 0.7)  # Normalize to 0-1 range
        
        return max(0.0, min(1.0, reward))
    
    def adapt_optimal_prediction_error(self):
        """
        Adapt the optimal prediction error target based on learning outcomes.
        
        This implements meta-learning: the system learns what level of prediction error
        leads to the best overall learning progress.
        """
        if len(self.recent_learning_outcomes) < 10:
            return  # Need sufficient data
        
        # Analyze learning trend
        recent_trend = sum(self.recent_learning_outcomes[-5:]) / 5
        overall_trend = sum(self.recent_learning_outcomes) / len(self.recent_learning_outcomes)
        
        # If recent learning is better than overall average, we're on the right track
        # If worse, try adjusting the optimal error target
        if recent_trend < overall_trend - 0.1:  # Learning is declining
            # Try adjusting optimal error slightly
            # If current average error is far from optimal, move optimal toward current
            recent_errors = [exp.prediction_error for exp in list(self.experience_storage._experiences.values())[-10:]]
            if recent_errors:
                current_avg_error = sum(recent_errors) / len(recent_errors)
                error_direction = current_avg_error - self.optimal_prediction_error
                # Move optimal slightly toward current to see if that improves learning
                adjustment = error_direction * self.learning_rate
                self.optimal_prediction_error = max(0.1, min(0.8, self.optimal_prediction_error + adjustment))
    
    def _record_similarity_learning_outcomes(self, sensory_input: List[float], 
                                           predicted_action: List[float], 
                                           actual_outcome: List[float]):
        """
        Record how well similar experiences helped with prediction.
        
        This provides feedback to the similarity learning system about whether
        experiences labeled as "similar" actually helped make good predictions.
        """
        # OPTIMIZATION: Single-pass similarity computation for similarity learning
        experience_vectors = []
        experience_ids = []
        similarities = []
        
        # Thread-safe copy of experiences to avoid concurrent modification
        experiences_snapshot = dict(self.experience_storage._experiences)
        
        for exp_id, exp in experiences_snapshot.items():
            exp_vector = exp.get_context_vector()
            similarity = self.similarity_engine.compute_similarity(sensory_input, exp_vector)
            
            experience_vectors.append(exp_vector)
            experience_ids.append(exp_id)
            similarities.append(similarity)
        
        if not experience_vectors:
            return
        
        # Get adaptive similarity threshold using pre-computed similarities
        adaptive_threshold = self.adaptive_thresholds.get_similarity_threshold(
            similarities, context="general"
        )
        
        # Filter similar experiences using pre-computed similarities
        similar_experiences = []
        for exp_id, similarity in zip(experience_ids, similarities):
            if similarity >= adaptive_threshold:
                similar_experiences.append((exp_id, similarity))
        
        # Sort by similarity and limit results
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        similar_experiences = similar_experiences[:5]
        
        # Compute prediction success for this outcome
        predicted_outcome = predicted_action  # In real scenario, this would be a world model prediction
        prediction_success = self._compute_prediction_success(predicted_outcome, actual_outcome)
        
        # Record the outcome for each similar experience that contributed
        for exp_id, similarity_score in similar_experiences:
            similar_experience = self.experience_storage._experiences[exp_id]
            similar_vector = similar_experience.get_context_vector()
            
            # Record prediction outcome for similarity learning
            self.similarity_engine.record_prediction_outcome(
                query_vector=sensory_input,
                similar_experience_id=exp_id,
                similar_vector=similar_vector,
                prediction_success=prediction_success
            )
    
    def _compute_prediction_success(self, predicted_outcome: List[float], 
                                  actual_outcome: List[float]) -> float:
        """
        Compute how successful a prediction was (inverse of prediction error).
        
        Returns:
            Success score from 0.0 (total failure) to 1.0 (perfect prediction)
        """
        if len(predicted_outcome) != len(actual_outcome):
            return 0.0
        
        # Compute normalized prediction error
        import numpy as np
        predicted = np.array(predicted_outcome)
        actual = np.array(actual_outcome)
        
        error = np.linalg.norm(predicted - actual)
        max_possible_error = np.linalg.norm(predicted) + np.linalg.norm(actual)
        
        if max_possible_error == 0:
            normalized_error = 0.0
        else:
            normalized_error = min(1.0, error / max_possible_error)
        
        # Convert error to success (1.0 - error)
        return 1.0 - normalized_error
    
    def _execute_triggered_adaptation(self, trigger_type: str, trigger_reason: str, evidence: Dict):
        """
        Execute an adaptation that was triggered by an information event.
        
        Args:
            trigger_type: Type of trigger that fired
            trigger_reason: Specific reason for the trigger
            evidence: Evidence supporting the adaptation need
        """
        print(f"Event-driven adaptation triggered: {trigger_type} ({trigger_reason})")
        
        if trigger_type in ["gradient_change", "poor_performance", "high_surprise"]:
            # These triggers suggest we need similarity function adaptation
            if self.logger:
                before_stats = self.similarity_engine.get_performance_stats()
                self.similarity_engine.adapt_similarity_function()
                after_stats = self.similarity_engine.get_performance_stats()
                self.logger.log_adaptation_event(
                    "similarity_function", trigger_reason,
                    before_stats.get('similarity_learning', {}), 
                    after_stats.get('similarity_learning', {}),
                    evidence
                )
                # Also log similarity evolution
                self.logger.log_similarity_evolution(
                    after_stats.get('similarity_learning', {}), self.total_experiences
                )
            else:
                self.similarity_engine.adapt_similarity_function()
        
        if trigger_type in ["poor_performance", "performance_plateau"]:
            # These triggers suggest we need activation dynamics adaptation
            recent_errors = [exp.prediction_error for exp in list(self.experience_storage._experiences.values())[-10:]]
            
            if self.use_utility_based_activation:
                # Utility-based activation adapts automatically through prediction outcomes
                # No manual parameter adaptation needed - emergence handles this
                if self.logger:
                    current_stats = self.activation_dynamics.get_utility_statistics()
                    self.logger.log_adaptation_event(
                        "utility_based_activation", trigger_reason,
                        {}, current_stats,  # No "before" stats since adaptation is continuous
                        {**evidence, "recent_errors": recent_errors, "adaptation_type": "continuous_utility_learning"}
                    )
            else:
                # Traditional activation with manual parameter adaptation
                if self.logger:
                    before_stats = self.activation_dynamics.get_activation_statistics(self.experience_storage._experiences)
                    self.activation_dynamics.adapt_parameters(recent_errors)
                    after_stats = self.activation_dynamics.get_activation_statistics(self.experience_storage._experiences)
                    self.logger.log_adaptation_event(
                        "activation_dynamics", trigger_reason,
                        before_stats, after_stats, 
                        {**evidence, "recent_errors": recent_errors}
                    )
                else:
                    self.activation_dynamics.adapt_parameters(recent_errors)
        
        if trigger_type in ["performance_plateau", "poor_performance"]:
            # These triggers suggest we might need optimal error adaptation
            if self.logger:
                before_optimal = self.optimal_prediction_error
                self.adapt_optimal_prediction_error()
                after_optimal = self.optimal_prediction_error
                self.logger.log_adaptation_event(
                    "optimal_prediction_error", trigger_reason,
                    {"optimal_error": before_optimal}, {"optimal_error": after_optimal},
                    evidence
                )
            else:
                self.adapt_optimal_prediction_error()
    
    def _compute_prediction_error(self, predicted_action: List[float], actual_outcome: List[float]) -> float:
        """Compute prediction error between predicted action and actual outcome."""
        import numpy as np
        
        # Convert to numpy arrays for computation
        predicted = np.array(predicted_action)
        actual = np.array(actual_outcome)
        
        # Compute normalized prediction error (0.0 = perfect, 1.0 = worst possible)
        if len(predicted) != len(actual):
            # If dimensions don't match, use moderate error
            return 0.5
        
        error = np.linalg.norm(predicted - actual)
        max_possible_error = np.linalg.norm(predicted) + np.linalg.norm(actual)
        
        if max_possible_error == 0:
            return 0.0
        else:
            return min(1.0, error / max_possible_error)
    
    def _activate_by_utility(self, sensory_input: List[float]):
        """Activate experiences based on their prediction utility for current context."""
        
        if len(self.experience_storage._experiences) == 0:
            return
        
        # OPTIMIZATION: Single-pass similarity computation to eliminate redundant calculations
        experience_vectors = []
        experience_ids = []
        similarities = []
        
        # Thread-safe copy of experiences to avoid concurrent modification
        experiences_snapshot = dict(self.experience_storage._experiences)
        
        for exp_id, exp in experiences_snapshot.items():
            exp_vector = exp.get_context_vector()
            similarity = self.similarity_engine.compute_similarity(sensory_input, exp_vector)
            
            experience_vectors.append(exp_vector)
            experience_ids.append(exp_id)
            similarities.append(similarity)
        
        # Get adaptive similarity threshold using pre-computed similarities
        utility_threshold = self.adaptive_thresholds.get_similarity_threshold(
            similarities, context="utility_assessment"
        )
        
        # Filter similar experiences using pre-computed similarities
        similar_experiences = []
        for i, (exp_id, similarity) in enumerate(zip(experience_ids, similarities)):
            if similarity >= utility_threshold:
                similar_experiences.append((exp_id, similarity))
        
        # Sort by similarity and limit results
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        similar_experiences = similar_experiences[:15]
        
        # Let utility-based activation system determine activation levels
        self.activation_dynamics.activate_by_prediction_utility(
            sensory_input, self.experience_storage._experiences, similar_experiences
        )
    
    def _activate_similar_experiences(self, sensory_input: List[float]):
        """Activate experiences similar to current input with attention weighting - traditional method."""
        
        if len(self.experience_storage._experiences) == 0:
            return
        
        # Get all experiences for attention-weighted similarity
        experiences = list(self.experience_storage._experiences.values())
        
        # Find similar experiences using natural attention weighting
        similar_results = self.similarity_engine.find_similar_experiences_with_natural_attention(
            sensory_input, experiences, max_results=10, min_similarity=0.4, retrieval_mode='normal'
        )
        
        # Activate similar experiences (working memory effect) 
        # Higher natural attention experiences get boosted activation
        for experience, weighted_similarity, base_similarity, natural_attention in similar_results:
            # Use natural attention weight for activation strength
            activation_strength = weighted_similarity * 0.6  # Scale by attention-weighted similarity
            
            # Boost activation for high-utility/distinctive experiences (natural attention boost)
            if natural_attention > 0.7:
                activation_strength *= 1.2  # 20% boost for naturally important memories
            
            self.activation_dynamics.activate_experience(experience, activation_strength)
    
    def _update_similarity_connections_and_clustering(self, new_experience: Experience):
        """Update similarity connections and compute natural clustering properties."""
        
        if len(self.experience_storage._experiences) <= 1:
            return
        
        new_vector = new_experience.get_context_vector()
        
        # Find all similar experiences for clustering analysis
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.experience_storage._experiences.items():
            if exp_id != new_experience.experience_id:
                experience_vectors.append(exp.get_context_vector())
                experience_ids.append(exp_id)
        
        if experience_vectors:
            # Get all similarities for clustering analysis
            all_similarities = self.similarity_engine.find_similar_experiences(
                new_vector, experience_vectors, experience_ids,
                max_results=len(experience_vectors), min_similarity=0.0
            )
            
            # Count close neighbors for cluster density (natural clustering detection)
            close_neighbors = [sim for _, sim in all_similarities if sim > 0.7]
            medium_neighbors = [sim for _, sim in all_similarities if sim > 0.5]
            
            # Update cluster density based on how many similar experiences exist
            new_experience.update_cluster_density(
                num_similar_neighbors=len(close_neighbors),
                search_radius=0.1 if close_neighbors else 0.05
            )
            
            # Store connections to most similar experiences (top 5)
            top_similar = all_similarities[:5] if len(all_similarities) >= 5 else all_similarities
            
            for exp_id, similarity in top_similar:
                if similarity > 0.3:  # Only store meaningful connections
                    new_experience.add_similarity(exp_id, similarity)
                    other_experience = self.experience_storage._experiences[exp_id]
                    other_experience.add_similarity(new_experience.experience_id, similarity)
                    
                    # Also update cluster density for the other experience
                    other_neighbors = [s for _, s in self.similarity_engine.find_similar_experiences(
                        other_experience.get_context_vector(), experience_vectors, experience_ids,
                        max_results=len(experience_vectors), min_similarity=0.7
                    )]
                    other_experience.update_cluster_density(len(other_neighbors), 0.1)
    
    def _get_learning_context(self) -> Dict[str, Any]:
        """Get learning context for adaptive attention scoring."""
        # Compute current accuracy from recent learning outcomes
        if len(self.recent_learning_outcomes) > 0:
            current_accuracy = sum(self.recent_learning_outcomes) / len(self.recent_learning_outcomes)
        else:
            current_accuracy = 0.5  # Default moderate accuracy
        
        # Compute recent prediction accuracy trend
        recent_errors = []
        for exp in list(self.experience_storage._experiences.values())[-10:]:
            recent_errors.append(exp.prediction_error)
        
        recent_accuracy = 1.0 - (sum(recent_errors) / len(recent_errors)) if recent_errors else 0.5
        
        return {
            'current_accuracy': current_accuracy,
            'recent_accuracy': recent_accuracy,
            'learning_velocity': self._compute_learning_velocity(),
            'total_experiences': len(self.experience_storage._experiences)
        }
    
    def _compute_learning_velocity(self) -> float:
        """Compute how fast learning is improving (accuracy change rate)."""
        if len(self.recent_learning_outcomes) < 10:
            return 0.0
        
        # Compare first half vs second half of recent outcomes
        outcomes = list(self.recent_learning_outcomes)
        first_half = outcomes[:len(outcomes)//2]
        second_half = outcomes[len(outcomes)//2:]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return 0.0
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        return second_avg - first_avg  # Positive = improving, negative = declining
    
    def _get_brain_state(self, prediction_details: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Get comprehensive brain state information."""
        
        # Get working memory with adaptive threshold
        current_activations = {}
        if self.use_utility_based_activation:
            current_activations = self.activation_dynamics.current_activations
            working_memory_size = len(current_activations)
        else:
            # For traditional activation, get all activations
            for exp_id, exp in self.experience_storage._experiences.items():
                if hasattr(exp, 'activation_level') and exp.activation_level > 0:
                    current_activations[exp_id] = exp.activation_level
            working_memory_size = len(current_activations)
        
        # Compute current intrinsic reward (the robot's fundamental drive state)
        current_intrinsic_reward = self.compute_intrinsic_reward()
        
        # Apply adaptive working memory threshold
        recent_performance = confidence  # Use prediction confidence as performance proxy
        adaptive_wm_threshold = self.adaptive_thresholds.get_working_memory_threshold(
            current_activations, recent_performance, 
            1.0 - current_intrinsic_reward  # Convert to prediction error proxy
        )
        
        # Count experiences above adaptive threshold
        adaptive_working_memory_size = sum(1 for activation in current_activations.values() 
                                         if activation >= adaptive_wm_threshold)
        
        # Get natural attention system stats
        attention_stats = {}
        if hasattr(self.similarity_engine, 'natural_attention_similarity') and self.similarity_engine.natural_attention_similarity:
            attention_stats = self.similarity_engine.natural_attention_similarity.get_natural_attention_stats(
                list(self.experience_storage._experiences.values())
            )
        
        return {
            'total_experiences': len(self.experience_storage._experiences),
            'working_memory_size': working_memory_size,
            'adaptive_working_memory_size': adaptive_working_memory_size,
            'adaptive_working_memory_threshold': adaptive_wm_threshold,
            'prediction_confidence': confidence,
            'prediction_method': prediction_details.get('method', 'unknown'),
            'num_similar_experiences': prediction_details.get('num_similar', 0),
            'brain_uptime': time.time() - self.brain_start_time,
            'total_predictions': self.total_predictions,
            'intrinsic_reward': current_intrinsic_reward,  # The robot's DNA-equivalent drive state
            'prediction_error_drive': f"adaptive_optimal_at_{self.optimal_prediction_error:.2f}",
            'natural_attention_system': attention_stats,  # Natural attention system status
            'adaptive_thresholds': True  # Flag that adaptive thresholds are active
        }
    
    def get_brain_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain performance statistics."""
        
        storage_stats = self.experience_storage.get_statistics()
        similarity_stats = self.similarity_engine.get_performance_stats()
        
        # Get activation statistics (method differs between systems)
        if self.use_utility_based_activation:
            activation_stats = self.activation_dynamics.get_utility_statistics()
        else:
            activation_stats = self.activation_dynamics.get_activation_statistics(
                self.experience_storage._experiences
            )
        
        prediction_stats = self.prediction_engine.get_prediction_statistics()
        
        # Compute drive statistics
        current_intrinsic_reward = self.compute_intrinsic_reward()
        
        # Analyze recent prediction errors to understand drive trends
        recent_errors = []
        for exp in list(self.experience_storage._experiences.values())[-20:]:
            recent_errors.append(exp.prediction_error)
        
        avg_recent_error = sum(recent_errors) / len(recent_errors) if recent_errors else 0.5
        drive_satisfaction = current_intrinsic_reward
        
        # Get adaptive trigger statistics
        trigger_stats = self.adaptive_trigger.get_trigger_statistics()
        
        return {
            'brain_summary': {
                'total_experiences': self.total_experiences,
                'total_predictions': self.total_predictions,
                'uptime_seconds': time.time() - self.brain_start_time,
                'experiences_per_minute': self.total_experiences / max(1, (time.time() - self.brain_start_time) / 60),
                'predictions_per_minute': self.total_predictions / max(1, (time.time() - self.brain_start_time) / 60),
                'intrinsic_drive': {
                    'current_reward': current_intrinsic_reward,
                    'optimal_prediction_error': self.optimal_prediction_error,  # Now adaptive
                    'recent_avg_error': avg_recent_error,
                    'drive_satisfaction': drive_satisfaction,
                    'drive_principle': 'adaptive_prediction_error_optimization',
                    'learning_outcomes_tracked': len(self.recent_learning_outcomes),
                    'meta_learning_active': True
                },
                'event_driven_adaptation': trigger_stats
            },
            'experience_storage': storage_stats,
            'similarity_engine': similarity_stats,
            'activation_dynamics': activation_stats,
            'prediction_engine': prediction_stats
        }
    
    def reset_brain(self):
        """Reset the brain to initial state (for testing)."""
        self.experience_storage.clear()
        self.similarity_engine.clear_cache()
        
        # Reset activation system (method differs between systems)
        if self.use_utility_based_activation:
            self.activation_dynamics.reset_activations()
        else:
            self.activation_dynamics.clear_all_activations(self.experience_storage._experiences)
            
        self.prediction_engine.reset_statistics()
        
        self.total_experiences = 0
        self.total_predictions = 0
        self.brain_start_time = time.time()
        
        # Reset logger if enabled
        if self.logger:
            self.logger.close_session()
            self.logger = BrainLogger(config=self.config) if self.enable_logging else None
        
        print("🧹 MinimalBrain reset to initial state")
    
    def close_logging_session(self):
        """Close the current logging session and generate final report."""
        if self.logger:
            return self.logger.close_session()
        return None
    
    def _load_persistent_state(self):
        """Load brain state from persistent storage if available."""
        if not self.persistence_manager:
            return
        
        try:
            checkpoint = self.persistence_manager.load_latest_checkpoint()
            if checkpoint:
                self.persistence_manager.restore_brain_state(self, checkpoint)
                print(f"🧠 Brain memory restored from {checkpoint.checkpoint_id}")
            else:
                print("🧠 Starting with fresh brain memory")
        except Exception as e:
            print(f"⚠️  Failed to load persistent state: {e}")
            print("🧠 Starting with fresh brain memory")
    
    def _save_checkpoint(self):
        """Save current brain state to persistent storage."""
        if not self.persistence_manager:
            return
        
        try:
            checkpoint_id = self.persistence_manager.save_checkpoint(self)
            return checkpoint_id
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")
            return None
    
    def save_brain_state(self) -> Optional[str]:
        """Manually save brain state (public interface)."""
        return self._save_checkpoint()
    
    def finalize_session(self):
        """Finalize brain session - call on shutdown."""
        # Shutdown storage optimization first
        if self.enable_storage_optimization and self.storage_optimizer:
            self.storage_optimizer.shutdown()
        
        # Create final checkpoint
        if self.persistence_manager:
            self._save_checkpoint()
            self.persistence_manager.finalize_session()
        
        # Close logging
        if self.logger:
            self.logger.close_session()
    
    def __str__(self) -> str:
        # Get working memory size (method differs between systems)
        if self.use_utility_based_activation:
            working_memory_size = self.activation_dynamics.get_working_memory_size()
        else:
            working_memory_size = self.activation_dynamics.get_working_memory_size(self.experience_storage._experiences)
            
        return (f"MinimalBrain({self.total_experiences} experiences, "
                f"{self.total_predictions} predictions, "
                f"working_memory={working_memory_size}, "
                f"activation={'utility-based' if self.use_utility_based_activation else 'traditional'})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def _calculate_current_memory_usage(self) -> float:
        """Calculate current memory usage for performance monitoring."""
        if self.use_utility_based_activation:
            current_activations = self.activation_dynamics.current_activations
        else:
            current_activations = {}
            for exp_id, exp in self.experience_storage._experiences.items():
                if hasattr(exp, 'activation_level') and exp.activation_level > 0:
                    current_activations[exp_id] = exp.activation_level
        
        return calculate_memory_usage(current_activations, len(self.experience_storage._experiences))
    
    # Phase 2 adaptation methods removed - they were unused optimization fluff