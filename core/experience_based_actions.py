"""
Experience-Based Action Selection System.

This system enables the robot to learn motor patterns from its stored experiences
and reuse successful actions in similar situations, creating truly adaptive behavior.
"""

import random
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from drives.base_drive import DriveContext


@dataclass
class ExperienceBasedAction:
    """An action derived from stored experiences."""
    motor_action: Dict[str, float]
    source_experience_id: str
    similarity_to_current: float
    success_probability: float
    confidence: float
    reasoning: str


@dataclass
class MotorPattern:
    """A learned motor pattern from successful experiences."""
    pattern_id: str
    action_sequence: List[Dict[str, float]]
    success_rate: float
    contexts_used: List[List[float]]
    pain_associations: float  # Negative experiences associated with this pattern
    pleasure_associations: float  # Positive experiences associated with this pattern
    last_used: float
    usage_count: int


class ExperienceBasedActionSystem:
    """
    System for generating actions based on stored experiences and learned patterns.
    
    This bridges the memory-action gap by translating stored experiences into
    motor commands that can be executed in similar situations.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Learned motor patterns
        self.motor_patterns: Dict[str, MotorPattern] = {}
        self.pattern_id_counter = 0
        
        # Experience retrieval parameters
        self.context_similarity_threshold = 0.7
        self.max_retrieved_experiences = 10
        self.success_threshold = 0.7  # Prediction accuracy threshold for "success"
        
        # Pattern learning parameters
        self.pattern_formation_threshold = 3  # Min experiences to form a pattern
        self.pattern_confidence_threshold = 0.6
        
        # Statistics
        self.total_experience_actions = 0
        self.successful_experience_actions = 0
        self.pattern_usage_stats = defaultdict(int)
        
    def generate_experience_based_actions(self, context: DriveContext, 
                                        num_candidates: int = 3) -> List[ExperienceBasedAction]:
        """
        Generate action candidates based on stored experiences.
        
        Args:
            context: Current situation context
            num_candidates: Number of action candidates to generate
            
        Returns:
            List of experience-based actions
        """
        if not self.world_graph.has_nodes():
            return []
        
        # Get current mental context for similarity matching
        current_context = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
        
        # Retrieve similar experiences
        similar_experiences = self._retrieve_similar_experiences(current_context, context)
        
        if not similar_experiences:
            return []
        
        # Generate actions from experiences
        experience_actions = []
        
        # Method 1: Direct experience reuse
        for experience, similarity in similar_experiences[:num_candidates]:
            action = self._extract_action_from_experience(experience, similarity, context)
            if action:
                experience_actions.append(action)
        
        # Method 2: Pattern-based actions
        pattern_actions = self._generate_pattern_based_actions(current_context, context, 
                                                             num_candidates - len(experience_actions))
        experience_actions.extend(pattern_actions)
        
        # Method 3: Associative memory actions
        if len(experience_actions) < num_candidates:
            associative_actions = self._generate_associative_actions(current_context, context,
                                                                   num_candidates - len(experience_actions))
            experience_actions.extend(associative_actions)
        
        return experience_actions[:num_candidates]
    
    def _retrieve_similar_experiences(self, current_context: List[float], 
                                    context: DriveContext) -> List[Tuple[ExperienceNode, float]]:
        """Retrieve experiences similar to current context."""
        # Use world graph's similarity search
        similar_nodes = self.world_graph.find_similar_nodes(
            current_context, 
            similarity_threshold=self.context_similarity_threshold,
            max_results=self.max_retrieved_experiences
        )
        
        # Calculate similarity scores and filter by relevance
        similar_experiences = []
        for node in similar_nodes:
            similarity = self.world_graph._calculate_context_similarity(
                current_context, node.mental_context
            )
            
            # Filter by success and relevance
            if (similarity >= self.context_similarity_threshold and 
                node.prediction_error < (1.0 - self.success_threshold)):
                similar_experiences.append((node, similarity))
        
        # Sort by similarity and success (low prediction error = high success)
        similar_experiences.sort(key=lambda x: x[1] * (1.0 - x[0].prediction_error), reverse=True)
        
        return similar_experiences
    
    def _extract_action_from_experience(self, experience: ExperienceNode, 
                                      similarity: float, context: DriveContext) -> Optional[ExperienceBasedAction]:
        """Extract a motor action from a stored experience."""
        if not experience.action_taken:
            return None
        
        # Calculate success probability based on experience outcomes
        success_prob = max(0.1, 1.0 - experience.prediction_error)
        
        # Adjust for pain/pleasure associations
        pain_penalty = 0.0
        pleasure_bonus = 0.0
        
        if hasattr(experience, 'pain_signal'):
            pain_penalty = abs(experience.pain_signal) * 0.3
        if hasattr(experience, 'pleasure_signal'):
            pleasure_bonus = experience.pleasure_signal * 0.2
        
        # Calculate confidence
        confidence = similarity * success_prob * (1.0 - pain_penalty) * (1.0 + pleasure_bonus)
        confidence = max(0.1, min(1.0, confidence))
        
        # Apply context-specific adjustments
        adjusted_action = self._adapt_action_to_context(experience.action_taken, context)
        
        return ExperienceBasedAction(
            motor_action=adjusted_action,
            source_experience_id=experience.node_id,
            similarity_to_current=similarity,
            success_probability=success_prob,
            confidence=confidence,
            reasoning=f"Reusing successful action from similar experience (similarity: {similarity:.2f})"
        )
    
    def _adapt_action_to_context(self, original_action: Dict[str, float], 
                               context: DriveContext) -> Dict[str, float]:
        """Adapt a stored action to current context."""
        adapted_action = original_action.copy()
        
        # Apply threat-level adjustments
        if context.threat_level == "danger":
            # More aggressive movement in danger
            if 'forward_motor' in adapted_action:
                adapted_action['forward_motor'] *= 1.2
            if 'turn_motor' in adapted_action:
                adapted_action['turn_motor'] *= 1.1
        elif context.threat_level == "safe":
            # More conservative movement when safe
            if 'forward_motor' in adapted_action:
                adapted_action['forward_motor'] *= 0.8
        
        # Apply energy-level adjustments
        if context.robot_energy < 0.3:
            # Reduce action intensity when low energy
            for key in adapted_action:
                adapted_action[key] *= 0.7
        
        # Apply health-level adjustments
        if context.robot_health < 0.5:
            # More cautious movement when injured
            if 'forward_motor' in adapted_action:
                adapted_action['forward_motor'] *= 0.6
            # Increase braking tendency
            if 'brake_motor' in adapted_action:
                adapted_action['brake_motor'] = max(adapted_action['brake_motor'], 0.3)
        
        # Clamp values to valid range
        for key in adapted_action:
            adapted_action[key] = max(-1.0, min(1.0, adapted_action[key]))
        
        return adapted_action
    
    def _generate_pattern_based_actions(self, current_context: List[float], 
                                      context: DriveContext, num_actions: int) -> List[ExperienceBasedAction]:
        """Generate actions based on learned motor patterns."""
        if not self.motor_patterns or num_actions <= 0:
            return []
        
        pattern_actions = []
        
        # Find patterns that match current context
        matching_patterns = []
        for pattern_id, pattern in self.motor_patterns.items():
            # Check if any stored context is similar to current
            max_similarity = 0.0
            for stored_context in pattern.contexts_used:
                similarity = self.world_graph._calculate_context_similarity(
                    current_context, stored_context
                )
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= self.context_similarity_threshold:
                matching_patterns.append((pattern, max_similarity))
        
        # Sort by success rate and similarity
        matching_patterns.sort(
            key=lambda x: x[0].success_rate * x[1] * (1.0 - x[0].pain_associations), 
            reverse=True
        )
        
        # Generate actions from top patterns
        for pattern, similarity in matching_patterns[:num_actions]:
            if pattern.action_sequence:
                # Use the first action in the sequence (could be extended for multi-step planning)
                action = pattern.action_sequence[0]
                adapted_action = self._adapt_action_to_context(action, context)
                
                confidence = pattern.success_rate * similarity * (1.0 - pattern.pain_associations * 0.5)
                
                pattern_actions.append(ExperienceBasedAction(
                    motor_action=adapted_action,
                    source_experience_id=pattern.pattern_id,
                    similarity_to_current=similarity,
                    success_probability=pattern.success_rate,
                    confidence=confidence,
                    reasoning=f"Using learned motor pattern (success rate: {pattern.success_rate:.2f})"
                ))
        
        return pattern_actions
    
    def _generate_associative_actions(self, current_context: List[float], 
                                    context: DriveContext, num_actions: int) -> List[ExperienceBasedAction]:
        """Generate actions using associative memory recall."""
        if num_actions <= 0:
            return []
        
        # Use world graph's associative memory system
        activated_memories = self.world_graph.activate_memory_network(
            current_context, activation_threshold=0.3
        )
        
        associative_actions = []
        
        for memory in activated_memories[:num_actions]:
            if memory.action_taken:
                # Calculate associative strength
                associative_strength = memory.get_accessibility()
                success_prob = max(0.2, 1.0 - memory.prediction_error)
                
                # Apply pain/pleasure biases
                pain_penalty = getattr(memory, 'pain_signal', 0) * 0.2
                pleasure_bonus = getattr(memory, 'pleasure_signal', 0) * 0.1
                
                confidence = associative_strength * success_prob * (1.0 - pain_penalty) * (1.0 + pleasure_bonus)
                confidence = max(0.1, min(1.0, confidence))
                
                adapted_action = self._adapt_action_to_context(memory.action_taken, context)
                
                associative_actions.append(ExperienceBasedAction(
                    motor_action=adapted_action,
                    source_experience_id=memory.node_id,
                    similarity_to_current=associative_strength,
                    success_probability=success_prob,
                    confidence=confidence,
                    reasoning=f"Associative memory recall (activation: {associative_strength:.2f})"
                ))
        
        return associative_actions
    
    def learn_motor_patterns(self, context: DriveContext):
        """Learn motor patterns from recent successful experiences."""
        # Get recent successful experiences
        recent_nodes = self.world_graph.get_recent_nodes(20)
        successful_experiences = [
            node for node in recent_nodes 
            if node.prediction_error < (1.0 - self.success_threshold)
        ]
        
        if len(successful_experiences) < self.pattern_formation_threshold:
            return
        
        # Look for sequences of similar successful actions
        for i in range(len(successful_experiences) - self.pattern_formation_threshold + 1):
            sequence = successful_experiences[i:i + self.pattern_formation_threshold]
            
            # Check if actions in sequence are similar enough to form a pattern
            if self._actions_form_pattern(sequence):
                self._create_motor_pattern(sequence)
    
    def _actions_form_pattern(self, experiences: List[ExperienceNode]) -> bool:
        """Check if a sequence of experiences forms a coherent motor pattern."""
        if not experiences:
            return False
        
        # Check temporal proximity
        for i in range(len(experiences) - 1):
            if hasattr(experiences[i], 'timestamp') and hasattr(experiences[i+1], 'timestamp'):
                time_gap = abs(experiences[i+1].timestamp - experiences[i].timestamp)
                if time_gap > 10.0:  # More than 10 seconds apart
                    return False
        
        # Check context similarity
        contexts = [exp.mental_context for exp in experiences]
        avg_similarity = 0.0
        comparison_count = 0
        
        for i in range(len(contexts)):
            for j in range(i + 1, len(contexts)):
                similarity = self.world_graph._calculate_context_similarity(contexts[i], contexts[j])
                avg_similarity += similarity
                comparison_count += 1
        
        if comparison_count > 0:
            avg_similarity /= comparison_count
            return avg_similarity >= 0.6  # Contexts should be reasonably similar
        
        return False
    
    def _create_motor_pattern(self, experiences: List[ExperienceNode]):
        """Create a motor pattern from a sequence of experiences."""
        pattern_id = f"pattern_{self.pattern_id_counter}"
        self.pattern_id_counter += 1
        
        # Extract action sequence
        action_sequence = [exp.action_taken for exp in experiences if exp.action_taken]
        
        # Calculate success rate
        success_rate = sum(1.0 - exp.prediction_error for exp in experiences) / len(experiences)
        
        # Extract contexts
        contexts_used = [exp.mental_context for exp in experiences]
        
        # Calculate pain/pleasure associations
        pain_associations = 0.0
        pleasure_associations = 0.0
        for exp in experiences:
            if hasattr(exp, 'pain_signal'):
                pain_associations += abs(exp.pain_signal)
            if hasattr(exp, 'pleasure_signal'):
                pleasure_associations += exp.pleasure_signal
        
        pain_associations /= len(experiences)
        pleasure_associations /= len(experiences)
        
        # Create pattern
        pattern = MotorPattern(
            pattern_id=pattern_id,
            action_sequence=action_sequence,
            success_rate=success_rate,
            contexts_used=contexts_used,
            pain_associations=pain_associations,
            pleasure_associations=pleasure_associations,
            last_used=time.time(),
            usage_count=0
        )
        
        self.motor_patterns[pattern_id] = pattern
        
        print(f"📚 Learned new motor pattern: {pattern_id} (success rate: {success_rate:.2f})")
    
    def record_action_outcome(self, action: ExperienceBasedAction, success: bool):
        """Record the outcome of an experience-based action."""
        self.total_experience_actions += 1
        
        if success:
            self.successful_experience_actions += 1
        
        # Update pattern usage stats
        if action.source_experience_id in self.motor_patterns:
            pattern = self.motor_patterns[action.source_experience_id]
            pattern.usage_count += 1
            pattern.last_used = time.time()
            
            # Update success rate
            if success:
                pattern.success_rate = (pattern.success_rate * 0.9) + (1.0 * 0.1)
            else:
                pattern.success_rate = (pattern.success_rate * 0.9) + (0.0 * 0.1)
            
            self.pattern_usage_stats[action.source_experience_id] += 1
    
    def get_experience_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about experience-based action generation."""
        success_rate = 0.0
        if self.total_experience_actions > 0:
            success_rate = self.successful_experience_actions / self.total_experience_actions
        
        return {
            "total_experience_actions": self.total_experience_actions,
            "successful_experience_actions": self.successful_experience_actions,
            "experience_action_success_rate": success_rate,
            "total_motor_patterns": len(self.motor_patterns),
            "pattern_usage_stats": dict(self.pattern_usage_stats),
            "avg_pattern_success_rate": sum(p.success_rate for p in self.motor_patterns.values()) / len(self.motor_patterns) if self.motor_patterns else 0.0
        }