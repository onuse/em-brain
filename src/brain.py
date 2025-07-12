"""
Minimal Brain Coordinator

The central orchestrator that coordinates all 4 core systems:
1. Experience Storage - stores every sensory-motor moment
2. Similarity Search - finds similar past situations  
3. Activation Dynamics - creates working memory effects
4. Prediction Engine - generates actions from experience patterns

This is the complete minimal brain in <100 lines.
"""

import time
from typing import List, Dict, Tuple, Optional, Any

from .experience import ExperienceStorage, Experience
from .similarity import SimilarityEngine  
from .activation import ActivationDynamics
from .activation.utility_based_activation import UtilityBasedActivation
from .prediction import PredictionEngine
from .utils.brain_logger import BrainLogger
from .utils.adaptive_trigger import AdaptiveTrigger


class MinimalBrain:
    """
    The complete minimal brain - 4 interacting systems that create intelligence.
    
    Everything emerges from this simple coordination:
    - Spatial navigation emerges from sensory similarity clustering
    - Motor skills emerge from action pattern reinforcement  
    - Exploration emerges from prediction error seeking
    - Working memory emerges from activation dynamics
    """
    
    def __init__(self, config=None, enable_logging=True, log_session_name=None, use_utility_based_activation=True):
        """Initialize the minimal brain with all 4 core systems."""
        
        # Core systems (the only hardcoded intelligence)
        self.experience_storage = ExperienceStorage()
        self.similarity_engine = SimilarityEngine(use_gpu=True, use_learnable_similarity=True)
        
        # Choose activation system: utility-based (emergent) or traditional (engineered)
        self.use_utility_based_activation = use_utility_based_activation
        if use_utility_based_activation:
            self.activation_dynamics = UtilityBasedActivation()
        else:
            self.activation_dynamics = ActivationDynamics()
            
        self.prediction_engine = PredictionEngine()
        
        # Brain state
        self.total_experiences = 0
        self.total_predictions = 0
        self.brain_start_time = time.time()
        
        # Adaptive learning parameters  
        self.optimal_prediction_error = 0.3  # Will adapt based on learning outcomes
        self.learning_rate = 0.05  # Rate of adaptation for optimal error
        self.recent_learning_outcomes = []  # Track learning success
        
        # Logging system for emergence analysis
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = BrainLogger(session_name=log_session_name)
        else:
            self.logger = None
        
        # Event-driven adaptation system (Strategy 3)
        self.adaptive_trigger = AdaptiveTrigger()
        
        activation_type = "utility-based (emergent)" if use_utility_based_activation else "traditional (engineered)"
        print(f"MinimalBrain initialized - 4 adaptive systems ready ({activation_type} activation)")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = 4) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input and return predicted action.
        
        This is the complete brain cycle:
        1. Update activations based on current input
        2. Predict action using similar experiences
        3. Return action with brain state info
        
        Args:
            sensory_input: Current sensory observation vector
            action_dimensions: Number of action dimensions to output
            
        Returns:
            Tuple of (predicted_action, brain_state_info)
        """
        # Update activation dynamics (method differs between systems)
        if self.use_utility_based_activation:
            # Utility-based activation spreads based on prediction utility
            self._activate_by_utility(sensory_input)
        else:
            # Traditional activation with engineered spreading
            self.activation_dynamics.update_all_activations(self.experience_storage._experiences)
            self._activate_similar_experiences(sensory_input)
        
        # Predict action using all available experience
        predicted_action, confidence, prediction_details = self.prediction_engine.predict_action(
            sensory_input,
            self.similarity_engine,
            self.activation_dynamics, 
            self.experience_storage._experiences,
            action_dimensions
        )
        
        self.total_predictions += 1
        
        # Log prediction outcome if logging enabled
        if self.logger and predicted_action:
            similar_experiences_used = prediction_details.get('num_similar', 0)
            self.logger.log_prediction_outcome(
                predicted_action, sensory_input, confidence, similar_experiences_used
            )
        
        # Compile brain state
        brain_state = self._get_brain_state(prediction_details, confidence)
        
        return predicted_action, brain_state
    
    def store_experience(self, sensory_input: List[float], action_taken: List[float], 
                        outcome: List[float], predicted_action: List[float] = None) -> str:
        """
        Store a new experience and learn from prediction error.
        
        Args:
            sensory_input: What was sensed
            action_taken: What action was taken  
            outcome: What actually happened
            predicted_action: What was predicted (for computing error)
            
        Returns:
            The experience ID
        """
        # Compute prediction error
        if predicted_action:
            # For prediction error, we compare predicted vs actual sensory outcome
            # (what we predicted would happen vs what actually happened)
            prediction_error = self._compute_prediction_error(predicted_action, outcome)
        else:
            prediction_error = 0.5  # Default moderate error for first experience
        
        # Compute intrinsic reward for this experience
        intrinsic_reward = self.compute_intrinsic_reward(prediction_error)
        
        # Create and store experience
        experience = Experience(
            sensory_input=sensory_input,
            action_taken=action_taken,
            outcome=outcome,
            prediction_error=prediction_error,
            timestamp=time.time()
        )
        
        # Store the intrinsic reward in the experience (for potential future use)
        experience.intrinsic_reward = intrinsic_reward
        
        # Track learning outcome (intrinsic reward represents learning success)
        self.recent_learning_outcomes.append(intrinsic_reward)
        if len(self.recent_learning_outcomes) > 50:
            self.recent_learning_outcomes = self.recent_learning_outcomes[-25:]
        
        experience_id = self.experience_storage.add_experience(experience)
        
        # Activate the new experience (method differs between systems)
        if self.use_utility_based_activation:
            # Utility-based activation - no manual activation, all emerges from utility
            pass  # Activation will emerge from prediction utility in next process cycle
        else:
            # Traditional activation with engineered strengths
            base_activation = 0.8
            reward_modulated_activation = base_activation * (0.5 + intrinsic_reward * 0.5)
            self.activation_dynamics.activate_experience(experience, strength=reward_modulated_activation)
            
            # Boost activation for surprising experiences (traditional method)
            recent_errors = [exp.prediction_error for exp in list(self.experience_storage._experiences.values())[-20:]]
            if recent_errors:
                avg_error = sum(recent_errors) / len(recent_errors)
                surprise_threshold = avg_error + (0.5 - avg_error) * 0.4
                if prediction_error > surprise_threshold:
                    self.activation_dynamics.boost_activation_by_prediction_error(experience)
        
        # Update similarity connections to related experiences
        self._update_similarity_connections(experience)
        
        # Record prediction outcomes for similarity learning
        if predicted_action and len(self.experience_storage._experiences) > 0:
            self._record_similarity_learning_outcomes(sensory_input, predicted_action, outcome)
        
        # Record prediction outcomes for utility-based activation learning
        if self.use_utility_based_activation and predicted_action:
            # Calculate prediction success for utility learning
            prediction_success = self._compute_prediction_success(predicted_action, outcome)
            
            # Get currently activated experiences to record their utility
            working_memory = self.activation_dynamics.get_working_memory_experiences()
            activated_experience_ids = [exp_id for exp_id, _ in working_memory]
            
            # Record how well the activated experiences helped with prediction
            self.activation_dynamics.record_prediction_outcome(activated_experience_ids, prediction_success)
        
        # Event-driven adaptation (Strategy 3) - replaces fixed schedules
        self.adaptive_trigger.record_prediction_outcome(prediction_error, self.total_experiences)
        adaptation_triggers = self.adaptive_trigger.check_adaptation_triggers(
            self.total_experiences, 
            {"intrinsic_reward": intrinsic_reward, "prediction_error": prediction_error}
        )
        
        # Process triggered adaptations
        for trigger_type, trigger_reason, evidence in adaptation_triggers:
            self._execute_triggered_adaptation(trigger_type, trigger_reason, evidence)
        
        # Log comprehensive brain state periodically (less frequent than before)
        if self.logger and self.total_experiences % 50 == 0:
            self.logger.log_brain_state(self, self.total_experiences)
        
        self.total_experiences += 1
        
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
        # Find the experiences that were used for this prediction
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id)
        
        if not experience_vectors:
            return
        
        # Find which experiences were considered similar for this prediction
        similar_experiences = self.similarity_engine.find_similar_experiences(
            sensory_input, experience_vectors, experience_ids,
            max_results=5, min_similarity=0.3
        )
        
        # Compute prediction success for this outcome
        predicted_outcome = predicted_action  # In real scenario, this would be a world model prediction
        prediction_success = self._compute_prediction_success(predicted_outcome, actual_outcome)
        
        # Record the outcome for each similar experience that contributed
        for exp_id, similarity_score in similar_experiences:
            similar_experience = self.experience_storage._experiences[exp_id]
            similar_vector = similar_experience.get_context_vector()
            
            # Record how well this "similar" experience helped predict
            self.similarity_engine.record_prediction_outcome(
                sensory_input, exp_id, similar_vector, prediction_success
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
        
        # Find similar experiences for utility-based activation
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id)
        
        similar_experiences = self.similarity_engine.find_similar_experiences(
            sensory_input, experience_vectors, experience_ids,
            max_results=15, min_similarity=0.2  # Cast wider net for utility assessment
        )
        
        # Let utility-based activation system determine activation levels
        self.activation_dynamics.activate_by_prediction_utility(
            sensory_input, self.experience_storage._experiences, similar_experiences
        )
    
    def _activate_similar_experiences(self, sensory_input: List[float]):
        """Activate experiences similar to current input (brings them into working memory) - traditional method."""
        
        if len(self.experience_storage._experiences) == 0:
            return
        
        # Find similar experiences
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id)
        
        similar_experiences = self.similarity_engine.find_similar_experiences(
            sensory_input, experience_vectors, experience_ids,
            max_results=10, min_similarity=0.4
        )
        
        # Activate similar experiences (working memory effect)
        for exp_id, similarity in similar_experiences:
            experience = self.experience_storage._experiences[exp_id]
            activation_strength = similarity * 0.6  # Scale by similarity
            self.activation_dynamics.activate_experience(experience, activation_strength)
    
    def _update_similarity_connections(self, new_experience: Experience):
        """Update similarity connections between the new experience and existing ones."""
        
        if len(self.experience_storage._experiences) <= 1:
            return
        
        new_vector = new_experience.get_context_vector()
        
        # Find a few most similar experiences and cache the connections
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.experience_storage._experiences.items():
            if exp_id != new_experience.experience_id:
                experience_vectors.append(exp.get_context_vector())
                experience_ids.append(exp_id)
        
        if experience_vectors:
            similar_experiences = self.similarity_engine.find_similar_experiences(
                new_vector, experience_vectors, experience_ids,
                max_results=5, min_similarity=0.3
            )
            
            # Store bidirectional similarity connections
            for exp_id, similarity in similar_experiences:
                new_experience.add_similarity(exp_id, similarity)
                other_experience = self.experience_storage._experiences[exp_id]
                other_experience.add_similarity(new_experience.experience_id, similarity)
    
    def _get_brain_state(self, prediction_details: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Get comprehensive brain state information."""
        
        # Get working memory size (method differs between systems)
        if self.use_utility_based_activation:
            working_memory_size = self.activation_dynamics.get_working_memory_size()
        else:
            working_memory_size = self.activation_dynamics.get_working_memory_size(
                self.experience_storage._experiences
            )
        
        # Compute current intrinsic reward (the robot's fundamental drive state)
        current_intrinsic_reward = self.compute_intrinsic_reward()
        
        return {
            'total_experiences': len(self.experience_storage._experiences),
            'working_memory_size': working_memory_size,
            'prediction_confidence': confidence,
            'prediction_method': prediction_details.get('method', 'unknown'),
            'num_similar_experiences': prediction_details.get('num_similar', 0),
            'brain_uptime': time.time() - self.brain_start_time,
            'total_predictions': self.total_predictions,
            'intrinsic_reward': current_intrinsic_reward,  # The robot's DNA-equivalent drive state
            'prediction_error_drive': 'optimal_at_0.3'  # Document the fundamental drive
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
            self.logger = BrainLogger() if self.enable_logging else None
        
        print("🧹 MinimalBrain reset to initial state")
    
    def close_logging_session(self):
        """Close the current logging session and generate final report."""
        if self.logger:
            return self.logger.close_session()
        return None
    
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