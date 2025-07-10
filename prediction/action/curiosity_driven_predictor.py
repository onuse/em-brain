"""
Curiosity-driven action prediction system.
Uses prediction uncertainty to drive exploration and learning.
"""

import random
import math
from typing import List, Dict, Optional, Tuple, Any
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket
from datetime import datetime


class ActionCandidate:
    """A potential action with predicted outcomes and uncertainty."""
    
    def __init__(self, action: Dict[str, float], predicted_sensory: List[float], 
                 confidence: float, curiosity_score: float, reasoning: str):
        self.action = action
        self.predicted_sensory = predicted_sensory
        self.confidence = confidence
        self.curiosity_score = curiosity_score  # Higher = more curious/uncertain
        self.reasoning = reasoning


class CuriosityDrivenPredictor:
    """
    Enhances the brain's prediction system with curiosity-driven action selection.
    
    The brain generates multiple action candidates, predicts outcomes for each,
    and chooses actions that maximize learning potential (prediction uncertainty).
    """
    
    def __init__(self, exploration_rate: float = 0.3, action_variation: float = 0.2):
        """
        Initialize curiosity-driven predictor.
        
        Args:
            exploration_rate: Balance between exploration (1.0) and exploitation (0.0)
            action_variation: How much to vary actions from past experiences
        """
        self.exploration_rate = exploration_rate
        self.action_variation = action_variation
        
        # Action space definition (robot-specific)
        # Note: forward_motor needs >0.2 or <-0.2 to trigger movement
        # turn_motor needs >0.3 or <-0.3 to trigger turning
        self.action_space = {
            "forward_motor": (-0.8, 0.8),  # Avoid extreme values, ensure >0.2 threshold
            "turn_motor": (-0.8, 0.8),     # Avoid extreme values, ensure >0.3 threshold
            "brake_motor": (0.0, 0.5)      # Moderate braking
        }
        
    def generate_curiosity_driven_prediction(self, current_context: List[float], 
                                           world_graph: WorldGraph, 
                                           terminal_node: Optional[ExperienceNode] = None,
                                           sequence_id: int = 0) -> PredictionPacket:
        """
        Generate prediction using curiosity-driven action selection.
        
        Args:
            current_context: Current mental state
            world_graph: Experience graph for predictions
            terminal_node: Node from traversal (if any)
            sequence_id: Sequence ID for prediction
            
        Returns:
            PredictionPacket with curiosity-driven action selection
        """
        # Generate multiple action candidates
        action_candidates = self._generate_action_candidates(
            current_context, world_graph, terminal_node
        )
        
        # If no candidates, use bootstrap exploration
        if not action_candidates:
            return self._create_bootstrap_prediction(sequence_id)
        
        # Select action based on curiosity (exploration vs exploitation)
        selected_candidate = self._select_action_by_curiosity(action_candidates)
        
        # Create prediction packet
        prediction = PredictionPacket(
            expected_sensory=selected_candidate.predicted_sensory,
            motor_action=selected_candidate.action,
            confidence=selected_candidate.confidence,
            timestamp=datetime.now(),
            sequence_id=sequence_id,
            thinking_depth=1
        )
        
        # Add curiosity metadata
        prediction.traversal_paths = [[selected_candidate.reasoning]]
        prediction.consensus_strength = "curiosity"
        
        return prediction
    
    def _generate_action_candidates(self, current_context: List[float], 
                                  world_graph: WorldGraph, 
                                  terminal_node: Optional[ExperienceNode]) -> List[ActionCandidate]:
        """Generate multiple action candidates with predicted outcomes."""
        candidates = []
        
        # Candidate 1: Exploitation - repeat successful past action
        if terminal_node:
            exploitation_candidate = self._create_exploitation_candidate(
                terminal_node, world_graph
            )
            candidates.append(exploitation_candidate)
        
        # Candidate 2: Exploration - novel action variation
        if terminal_node:
            exploration_candidate = self._create_exploration_candidate(
                terminal_node, world_graph
            )
            candidates.append(exploration_candidate)
        
        # Candidate 3: Random exploration - completely novel action
        random_candidate = self._create_random_exploration_candidate(
            current_context, world_graph
        )
        candidates.append(random_candidate)
        
        # Candidate 4: Curiosity-driven - action targeting high uncertainty areas
        if world_graph.has_nodes():
            curiosity_candidate = self._create_curiosity_candidate(
                current_context, world_graph
            )
            candidates.append(curiosity_candidate)
        
        return candidates
    
    def _create_exploitation_candidate(self, terminal_node: ExperienceNode, 
                                     world_graph: WorldGraph) -> ActionCandidate:
        """Create candidate that repeats a successful past action."""
        # Use the action from the terminal node
        action = terminal_node.action_taken.copy()
        
        # Predict outcome based on what happened before
        predicted_sensory = terminal_node.actual_sensory.copy()
        
        # High confidence if the node was successful (low prediction error)
        confidence = max(0.1, 1.0 - terminal_node.prediction_error)
        
        # Low curiosity score - we've done this before
        curiosity_score = 0.1 + terminal_node.prediction_error * 0.5
        
        return ActionCandidate(
            action=action,
            predicted_sensory=predicted_sensory,
            confidence=confidence,
            curiosity_score=curiosity_score,
            reasoning="exploitation_repeat"
        )
    
    def _create_exploration_candidate(self, terminal_node: ExperienceNode, 
                                    world_graph: WorldGraph) -> ActionCandidate:
        """Create candidate that varies a past action for exploration."""
        # Start with the terminal node's action
        base_action = terminal_node.action_taken.copy()
        
        # Add random variation
        varied_action = {}
        for actuator, value in base_action.items():
            if actuator in self.action_space:
                min_val, max_val = self.action_space[actuator]
                variation = random.uniform(-self.action_variation, self.action_variation)
                new_value = max(min_val, min(max_val, value + variation))
                varied_action[actuator] = new_value
            else:
                varied_action[actuator] = value
        
        # Predict outcome with more uncertainty due to variation
        predicted_sensory = terminal_node.actual_sensory.copy()
        self._add_prediction_uncertainty(predicted_sensory, 0.3)
        
        # Medium confidence - similar to past but with variation
        confidence = max(0.2, 0.7 - terminal_node.prediction_error)
        
        # Higher curiosity score - we're exploring variations
        curiosity_score = 0.5 + terminal_node.prediction_error * 0.3
        
        return ActionCandidate(
            action=varied_action,
            predicted_sensory=predicted_sensory,
            confidence=confidence,
            curiosity_score=curiosity_score,
            reasoning="exploration_variation"
        )
    
    def _create_random_exploration_candidate(self, current_context: List[float], 
                                           world_graph: WorldGraph) -> ActionCandidate:
        """Create candidate with completely random action."""
        # Generate random action within bounds
        random_action = {}
        for actuator, (min_val, max_val) in self.action_space.items():
            random_action[actuator] = random.uniform(min_val, max_val)
        
        # Can't predict outcome well - use neutral sensory prediction
        predicted_sensory = [0.5] * len(current_context) if current_context else [0.5] * 10
        
        # Low confidence - completely unknown
        confidence = 0.1
        
        # High curiosity score - completely novel
        curiosity_score = 0.9
        
        return ActionCandidate(
            action=random_action,
            predicted_sensory=predicted_sensory,
            confidence=confidence,
            curiosity_score=curiosity_score,
            reasoning="random_exploration"
        )
    
    def _create_curiosity_candidate(self, current_context: List[float], 
                                  world_graph: WorldGraph) -> ActionCandidate:
        """Create candidate targeting areas of high prediction uncertainty."""
        # Find the most uncertain/errorful experiences
        high_error_nodes = []
        for node in world_graph.all_nodes():
            if node.prediction_error > 0.5:  # High error threshold
                high_error_nodes.append(node)
        
        if not high_error_nodes:
            # No high-error nodes, fall back to random
            return self._create_random_exploration_candidate(current_context, world_graph)
        
        # Choose a high-error node and try a different action
        target_node = random.choice(high_error_nodes)
        
        # Create action that's different from what was tried before
        curious_action = {}
        for actuator, past_value in target_node.action_taken.items():
            if actuator in self.action_space:
                min_val, max_val = self.action_space[actuator]
                # Try opposite direction or random variation
                if random.random() < 0.5:
                    # Opposite direction
                    new_value = max(min_val, min(max_val, -past_value))
                else:
                    # Random variation
                    new_value = random.uniform(min_val, max_val)
                curious_action[actuator] = new_value
            else:
                curious_action[actuator] = past_value
        
        # Predict outcome based on high-error node but with uncertainty
        predicted_sensory = target_node.actual_sensory.copy()
        self._add_prediction_uncertainty(predicted_sensory, 0.6)
        
        # Medium confidence - we're targeting known uncertainty
        confidence = 0.3
        
        # Very high curiosity score - targeting uncertainty
        curiosity_score = 0.8 + target_node.prediction_error * 0.2
        
        return ActionCandidate(
            action=curious_action,
            predicted_sensory=predicted_sensory,
            confidence=confidence,
            curiosity_score=curiosity_score,
            reasoning="curiosity_targeting"
        )
    
    def _add_prediction_uncertainty(self, predicted_sensory: List[float], uncertainty_factor: float):
        """Add uncertainty noise to predicted sensory values."""
        for i in range(len(predicted_sensory)):
            noise = random.uniform(-uncertainty_factor, uncertainty_factor)
            predicted_sensory[i] = max(0.0, min(1.0, predicted_sensory[i] + noise))
    
    def _select_action_by_curiosity(self, candidates: List[ActionCandidate]) -> ActionCandidate:
        """Select action candidate balancing exploration and exploitation."""
        # Calculate weighted scores
        for candidate in candidates:
            # Exploitation score (high confidence, low curiosity)
            exploitation_score = candidate.confidence * (1.0 - candidate.curiosity_score)
            
            # Exploration score (high curiosity, medium confidence)
            exploration_score = candidate.curiosity_score * min(1.0, candidate.confidence + 0.3)
            
            # Final score balances exploration vs exploitation
            candidate.final_score = (
                exploitation_score * (1.0 - self.exploration_rate) +
                exploration_score * self.exploration_rate
            )
        
        # Select candidate with highest final score
        best_candidate = max(candidates, key=lambda c: c.final_score)
        
        return best_candidate
    
    def _create_bootstrap_prediction(self, sequence_id: int) -> PredictionPacket:
        """Create bootstrap prediction for empty graph."""
        # Choose a random action that will definitely trigger movement
        action_choice = random.random()
        
        if action_choice < 0.4:
            # Forward movement
            motor_action = {
                "forward_motor": random.uniform(0.3, 0.6),
                "turn_motor": 0.0,
                "brake_motor": 0.0
            }
        elif action_choice < 0.8:
            # Turning (left or right)
            motor_action = {
                "forward_motor": 0.0,
                "turn_motor": random.choice([-0.5, 0.5]),
                "brake_motor": 0.0
            }
        else:
            # Backward movement
            motor_action = {
                "forward_motor": random.uniform(-0.6, -0.3),
                "turn_motor": 0.0,
                "brake_motor": 0.0
            }
        
        return PredictionPacket(
            expected_sensory=[],  # Will be set by brain interface
            motor_action=motor_action,
            confidence=0.1,
            timestamp=datetime.now(),
            sequence_id=sequence_id,
            thinking_depth=0
        )
    
    def update_exploration_rate(self, new_rate: float):
        """Update exploration rate based on learning progress."""
        self.exploration_rate = max(0.1, min(0.9, new_rate))
    
    def get_curiosity_statistics(self) -> Dict[str, Any]:
        """Get statistics about curiosity-driven decisions."""
        return {
            "exploration_rate": self.exploration_rate,
            "action_variation": self.action_variation,
            "action_space": self.action_space
        }