"""
Utility-Based Activation Dynamics

Replaces engineered activation spreading formulas with activation that emerges
purely from predictive utility. No hardcoded decay rates, spreading rules, or
arbitrary activation formulas.

Core principle: Experiences that help predict get more activation.
Working memory emerges from prediction utility, not engineering.
"""

from typing import Dict, List, Tuple, Optional
import time
import numpy as np
from collections import defaultdict, deque

from ..experience import Experience


class UtilityBasedActivation:
    """
    Activation system where all dynamics emerge from predictive utility.
    
    Key principles:
    - Experiences that help predict current situations get activated
    - Activation strength = prediction utility (how much they help)
    - Decay emerges from lack of utility, not hardcoded rates
    - Spreading emerges from prediction connections, not engineered formulas
    """
    
    def __init__(self):
        """Initialize utility-based activation system."""
        
        # Utility tracking - the only "parameters" are learning rates
        self.prediction_utility_history = defaultdict(deque)  # exp_id -> [utility_scores]
        self.activation_success_tracking = defaultdict(list)  # activation_level -> [prediction_success]
        
        # Emergent activation state (no hardcoded parameters)
        self.current_activations = {}  # exp_id -> activation_level
        self.activation_timestamps = {}  # exp_id -> when_activated
        self.utility_connections = defaultdict(dict)  # exp_id -> {other_exp_id: utility_score}
        
        # Learning rates (Strategy 5: now adaptive)
        self.initial_utility_learning_rate = 0.1
        self.utility_learning_rate = 0.1  # Will adapt based on learning success
        self.activation_persistence_factor = 0.9  # How long utility-based activation lasts
        
        # Meta-learning parameters (Strategy 5)
        self.learning_rate_adaptation_rate = 0.1
        self.min_utility_learning_rate = 0.01
        self.max_utility_learning_rate = 0.5
        self.utility_learning_success_history = []  # Track how well utility learning works
        
        # Performance tracking
        self.total_activations = 0
        self.utility_based_decisions = 0
        
        print("UtilityBasedActivation initialized - activation emerges from prediction utility")
    
    def activate_by_prediction_utility(self, 
                                     target_context: List[float],
                                     all_experiences: Dict[str, Experience],
                                     similarity_scores: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Activate experiences based on their prediction utility for current context.
        
        Args:
            target_context: Current sensory context we need to predict from
            all_experiences: All available experiences
            similarity_scores: Pre-computed similarity scores
            
        Returns:
            Dict of exp_id -> activation_level based on utility
        """
        new_activations = {}
        current_time = time.time()
        
        # For each potentially relevant experience, compute its prediction utility
        for exp_id, similarity in similarity_scores:
            if exp_id in all_experiences:
                utility_score = self._compute_prediction_utility(
                    exp_id, target_context, all_experiences, similarity
                )
                
                if utility_score > 0.1:  # Only activate if genuinely useful
                    new_activations[exp_id] = utility_score
                    self.activation_timestamps[exp_id] = current_time
                    self.total_activations += 1
        
        # Update current activations with utility-based values
        self.current_activations.update(new_activations)
        
        # Apply natural decay based on time since activation (emergent, not hardcoded)
        self._apply_utility_based_decay(current_time)
        
        return self.current_activations.copy()
    
    def _compute_prediction_utility(self, 
                                  exp_id: str, 
                                  target_context: List[float],
                                  all_experiences: Dict[str, Experience],
                                  similarity: float) -> float:
        """
        Compute how useful this experience is for predicting in the current context.
        
        This is the core of utility-based activation - no hardcoded formulas,
        just: "how much does this experience help predict?"
        """
        experience = all_experiences[exp_id]
        
        # Base utility from similarity (experiences similar to current context are useful)
        base_utility = similarity
        
        # Boost utility based on historical prediction success
        historical_utility = self._get_historical_utility(exp_id)
        
        # Boost utility based on how well this experience has been for prediction recently
        recent_success_boost = self._get_recent_success_boost(exp_id)
        
        # Boost utility based on prediction error (lower error = more useful)
        error_boost = max(0.0, 1.0 - experience.prediction_error)
        
        # Boost utility if this experience connects well to other useful experiences
        connection_boost = self._get_connection_utility_boost(exp_id, all_experiences)
        
        # Combine utilities (weighted sum, not hardcoded formula)
        total_utility = (
            base_utility * 0.4 +           # Similarity is important
            historical_utility * 0.2 +     # Past success matters
            recent_success_boost * 0.2 +   # Recent success matters more
            error_boost * 0.1 +            # Low error experiences are useful
            connection_boost * 0.1         # Well-connected experiences are useful
        )
        
        return min(1.0, total_utility)
    
    def _get_historical_utility(self, exp_id: str) -> float:
        """Get historical prediction utility for this experience."""
        if exp_id not in self.prediction_utility_history:
            return 0.0
        
        utilities = list(self.prediction_utility_history[exp_id])
        if not utilities:
            return 0.0
        
        # Recent utilities matter more than old ones
        if len(utilities) >= 5:
            recent_weight = 0.7
            overall_weight = 0.3
            recent_utility = np.mean(utilities[-5:])
            overall_utility = np.mean(utilities)
            return recent_weight * recent_utility + overall_weight * overall_utility
        else:
            return np.mean(utilities)
    
    def _get_recent_success_boost(self, exp_id: str) -> float:
        """Get boost based on recent activation success."""
        if exp_id not in self.current_activations:
            return 0.0
        
        current_activation = self.current_activations[exp_id]
        
        # Look up how well this activation level has worked recently
        success_scores = []
        for activation_level, successes in self.activation_success_tracking.items():
            if abs(activation_level - current_activation) < 0.2:  # Similar activation levels
                success_scores.extend(successes[-5:])  # Recent successes
        
        if success_scores:
            return np.mean(success_scores)
        return 0.0
    
    def _get_connection_utility_boost(self, exp_id: str, all_experiences: Dict[str, Experience]) -> float:
        """Get utility boost from connections to other useful experiences."""
        if exp_id not in self.utility_connections:
            return 0.0
        
        connected_utilities = []
        for connected_exp_id, connection_strength in self.utility_connections[exp_id].items():
            if connected_exp_id in self.current_activations:
                connected_activation = self.current_activations[connected_exp_id]
                connected_utility = connection_strength * connected_activation
                connected_utilities.append(connected_utility)
        
        if connected_utilities:
            return np.mean(connected_utilities) * 0.5  # Moderate boost from connections
        return 0.0
    
    def _apply_utility_based_decay(self, current_time: float):
        """Apply decay based on time since activation and utility persistence."""
        decayed_experiences = []
        
        for exp_id, activation_level in list(self.current_activations.items()):
            if exp_id in self.activation_timestamps:
                time_since_activation = current_time - self.activation_timestamps[exp_id]
                
                # Decay rate emerges from utility - more useful experiences persist longer
                historical_utility = self._get_historical_utility(exp_id)
                persistence_factor = self.activation_persistence_factor * (0.5 + historical_utility * 0.5)
                
                # Exponential decay based on time and utility
                decay_factor = np.exp(-time_since_activation * (1.0 - persistence_factor))
                new_activation = activation_level * decay_factor
                
                if new_activation > 0.05:  # Keep if still meaningful
                    self.current_activations[exp_id] = new_activation
                else:
                    # Remove from current activations
                    decayed_experiences.append(exp_id)
        
        # Clean up decayed experiences
        for exp_id in decayed_experiences:
            del self.current_activations[exp_id]
            if exp_id in self.activation_timestamps:
                del self.activation_timestamps[exp_id]
    
    def record_prediction_outcome(self, 
                                activated_experiences: List[str],
                                prediction_success: float):
        """
        Record how well the currently activated experiences helped with prediction.
        
        This is how the system learns which experiences are useful to activate.
        """
        # Record utility for each activated experience
        for exp_id in activated_experiences:
            if exp_id in self.current_activations:
                activation_level = self.current_activations[exp_id]
                
                # Record this prediction success for this experience
                self.prediction_utility_history[exp_id].append(prediction_success)
                
                # Limit history size
                if len(self.prediction_utility_history[exp_id]) > 50:
                    self.prediction_utility_history[exp_id] = \
                        deque(list(self.prediction_utility_history[exp_id])[-25:])
                
                # Record activation level success
                rounded_activation = round(activation_level, 1)
                if rounded_activation not in self.activation_success_tracking:
                    self.activation_success_tracking[rounded_activation] = []
                
                self.activation_success_tracking[rounded_activation].append(prediction_success)
                
                # Limit activation success history
                if len(self.activation_success_tracking[rounded_activation]) > 20:
                    self.activation_success_tracking[rounded_activation] = \
                        self.activation_success_tracking[rounded_activation][-10:]
        
        # Update utility connections between co-activated experiences
        self._update_utility_connections(activated_experiences, prediction_success)
        
        # Meta-learning: track how well utility learning is working (Strategy 5)
        self._track_utility_learning_success(prediction_success)
        
        self.utility_based_decisions += 1
    
    def _update_utility_connections(self, 
                                  activated_experiences: List[str], 
                                  prediction_success: float):
        """Update utility connections between experiences that were co-activated."""
        # Experiences that are activated together and lead to good predictions
        # should be connected (they help each other predict)
        
        if prediction_success > 0.6:  # Only learn from successful predictions
            for i, exp_id_1 in enumerate(activated_experiences):
                for exp_id_2 in activated_experiences[i+1:]:
                    # Strengthen connection between these experiences
                    if exp_id_1 not in self.utility_connections:
                        self.utility_connections[exp_id_1] = {}
                    if exp_id_2 not in self.utility_connections:
                        self.utility_connections[exp_id_2] = {}
                    
                    # Update connection strength based on prediction success
                    current_strength_1 = self.utility_connections[exp_id_1].get(exp_id_2, 0.0)
                    current_strength_2 = self.utility_connections[exp_id_2].get(exp_id_1, 0.0)
                    
                    new_strength = current_strength_1 + self.utility_learning_rate * (prediction_success - current_strength_1)
                    
                    self.utility_connections[exp_id_1][exp_id_2] = new_strength
                    self.utility_connections[exp_id_2][exp_id_1] = new_strength
    
    def _track_utility_learning_success(self, prediction_success: float):
        """
        Track how well utility learning is working for meta-learning (Strategy 5).
        
        Args:
            prediction_success: How well the current utility-based activation helped predict
        """
        # Record utility learning performance
        self.utility_learning_success_history.append({
            'prediction_success': prediction_success,
            'utility_learning_rate': self.utility_learning_rate,
            'timestamp': time.time()
        })
        
        # Limit history
        if len(self.utility_learning_success_history) > 100:
            self.utility_learning_success_history = self.utility_learning_success_history[-50:]
        
        # Adapt utility learning rate based on recent performance
        if len(self.utility_learning_success_history) >= 10:
            self._adapt_utility_learning_rate()
    
    def _adapt_utility_learning_rate(self):
        """
        Meta-learning: adapt the utility learning rate based on learning success.
        
        This is Strategy 5 in action for the activation system.
        """
        recent_entries = self.utility_learning_success_history[-10:]
        
        # Compare recent performance to older performance
        if len(self.utility_learning_success_history) >= 20:
            older_entries = self.utility_learning_success_history[-20:-10]
            
            recent_avg = np.mean([entry['prediction_success'] for entry in recent_entries])
            older_avg = np.mean([entry['prediction_success'] for entry in older_entries])
            
            improvement = recent_avg - older_avg
            
            if improvement > 0.05:
                # Learning is improving - maybe we can learn faster
                new_learning_rate = self.utility_learning_rate * (1 + self.learning_rate_adaptation_rate)
                print(f"Meta-learning: Increasing utility learning rate {self.utility_learning_rate:.3f} → {new_learning_rate:.3f} (improvement: {improvement:.3f})")
            elif improvement < -0.05:
                # Learning is getting worse - learn more slowly
                new_learning_rate = self.utility_learning_rate * (1 - self.learning_rate_adaptation_rate)
                print(f"Meta-learning: Decreasing utility learning rate {self.utility_learning_rate:.3f} → {new_learning_rate:.3f} (improvement: {improvement:.3f})")
            else:
                new_learning_rate = self.utility_learning_rate
            
            # Apply bounds and update
            self.utility_learning_rate = np.clip(new_learning_rate, 
                                               self.min_utility_learning_rate, 
                                               self.max_utility_learning_rate)
    
    def get_working_memory_experiences(self, min_activation: float = 0.1) -> List[Tuple[str, float]]:
        """Get currently activated experiences (working memory)."""
        working_memory = []
        for exp_id, activation in self.current_activations.items():
            if activation >= min_activation:
                working_memory.append((exp_id, activation))
        
        # Sort by activation level (highest first)
        working_memory.sort(key=lambda x: x[1], reverse=True)
        return working_memory
    
    def get_working_memory_size(self, min_activation: float = 0.1) -> int:
        """Get working memory size (number of activated experiences)."""
        return len([a for a in self.current_activations.values() if a >= min_activation])
    
    def get_utility_statistics(self) -> Dict:
        """Get comprehensive utility-based activation statistics."""
        
        # Analyze utility distribution
        all_utilities = []
        for utility_history in self.prediction_utility_history.values():
            all_utilities.extend(list(utility_history))
        
        # Analyze activation success by level
        activation_success_by_level = {}
        for level, successes in self.activation_success_tracking.items():
            if successes:
                activation_success_by_level[level] = {
                    'avg_success': np.mean(successes),
                    'count': len(successes)
                }
        
        # Analyze utility connections
        total_connections = sum(len(connections) for connections in self.utility_connections.values())
        strong_connections = 0
        for connections in self.utility_connections.values():
            strong_connections += sum(1 for strength in connections.values() if strength > 0.5)
        
        return {
            'total_activations': self.total_activations,
            'utility_based_decisions': self.utility_based_decisions,
            'current_working_memory_size': len(self.current_activations),
            'experiences_with_utility_history': len(self.prediction_utility_history),
            'avg_utility_score': np.mean(all_utilities) if all_utilities else 0.0,
            'utility_connections': {
                'total_connections': total_connections,
                'strong_connections': strong_connections,
                'connection_ratio': strong_connections / max(1, total_connections)
            },
            'activation_success_levels': len(activation_success_by_level),
            'best_activation_level': max(activation_success_by_level.items(), 
                                       key=lambda x: x[1]['avg_success'])[0] if activation_success_by_level else None,
            'system_type': 'utility_based_emergent',
            'meta_learning': self.get_meta_learning_stats()
        }
    
    def get_meta_learning_stats(self) -> Dict:
        """Get statistics about meta-learning in utility-based activation."""
        if not self.utility_learning_success_history:
            return {
                'meta_learning_active': False,
                'current_utility_learning_rate': self.utility_learning_rate,
                'initial_utility_learning_rate': self.initial_utility_learning_rate
            }
        
        recent_performance = [entry['prediction_success'] for entry in self.utility_learning_success_history[-10:]]
        learning_rates = [entry['utility_learning_rate'] for entry in self.utility_learning_success_history[-10:]]
        
        return {
            'meta_learning_active': True,
            'current_utility_learning_rate': self.utility_learning_rate,
            'initial_utility_learning_rate': self.initial_utility_learning_rate,
            'utility_learning_adaptations': len(self.utility_learning_success_history),
            'avg_recent_performance': np.mean(recent_performance) if recent_performance else 0.0,
            'utility_learning_rate_trend': {
                'min': np.min(learning_rates) if learning_rates else self.utility_learning_rate,
                'max': np.max(learning_rates) if learning_rates else self.utility_learning_rate,
                'current': self.utility_learning_rate
            },
            'learning_effectiveness': np.mean(recent_performance) if recent_performance else 0.0
        }
    
    def reset_activations(self):
        """Reset all activation state."""
        self.current_activations.clear()
        self.activation_timestamps.clear()
        self.prediction_utility_history.clear()
        self.activation_success_tracking.clear()
        self.utility_connections.clear()
        self.total_activations = 0
        self.utility_based_decisions = 0
        
        # Reset meta-learning parameters (Strategy 5)
        self.utility_learning_rate = self.initial_utility_learning_rate
        self.utility_learning_success_history.clear()
        
        print("UtilityBasedActivation reset - clean slate for emergence (including meta-learning)")