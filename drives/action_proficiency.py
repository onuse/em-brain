"""
Action Proficiency and Maturation System.

Implements developmental stages of action competence that influence action generation.
The system progresses from chaotic exploration (infancy) through skill building 
(adolescence) to stable competent behavior (maturity).
"""

import time
import random
import statistics
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from core.world_graph import WorldGraph
from drives.base_drive import DriveContext
from .development_types import DevelopmentalStage, ActionProficiency


class ActionMaturationSystem:
    """
    Manages the maturation of action selection from chaotic exploration
    to stable competent behavior.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Proficiency tracking
        self.action_proficiencies: Dict[str, ActionProficiency] = defaultdict(
            lambda: ActionProficiency(action_type="unknown")
        )
        
        # Cognitive development system (replaces fixed thresholds)
        # Import here to avoid circular import
        from .cognitive_development import CognitiveDevelopmentSystem
        self.cognitive_development = CognitiveDevelopmentSystem(world_graph)
        
        # Maturation state
        self.total_actions_taken = 0
        self.system_age = 0.0  # Time since initialization
        self.start_time = time.time()
        
        # Current parameters (updated by cognitive development system)
        self.exploration_rate = 0.90  # Will be updated dynamically
        self.proficiency_bias_strength = 0.05  # Will be updated dynamically
        self.skill_building_rate = 0.05  # Will be updated dynamically
    
    def update_system_maturation(self, traversal_result=None, prediction_accuracy: float = None,
                               thinking_time: float = None, action_taken: Dict = None):
        """Update developmental stage based on cognitive constraints rather than fixed thresholds."""
        self.system_age = time.time() - self.start_time
        
        # Update cognitive development system with current brain state
        self.cognitive_development.update_cognitive_metrics(
            traversal_result=traversal_result,
            prediction_accuracy=prediction_accuracy,
            thinking_time=thinking_time,
            action_taken=action_taken
        )
        
        # Get current developmental parameters from cognitive constraints
        stage_params = self.cognitive_development.get_stage_progression_parameters()
        
        # Update maturation parameters based on cognitive development
        self.exploration_rate = stage_params['exploration_rate']
        self.proficiency_bias_strength = stage_params['proficiency_bias_strength']
        self.skill_building_rate = stage_params.get('skill_building_rate', 0.1)
    
    
    def record_action_outcome(self, action: Dict[str, float], 
                            prediction_accuracy: float,
                            execution_success: bool):
        """Record the outcome of an action for proficiency tracking."""
        self.total_actions_taken += 1
        
        # Update proficiency for each significant action component
        for action_type, action_value in action.items():
            if abs(action_value) > 0.1:  # Significant action
                self.action_proficiencies[action_type].action_type = action_type
                self.action_proficiencies[action_type].update_proficiency(
                    prediction_accuracy, execution_success
                )
        
        # Update system maturation
        self.update_system_maturation()
    
    def generate_maturation_influenced_actions(self, context: DriveContext,
                                             num_candidates: int = 5) -> List[Dict[str, float]]:
        """
        Generate action candidates influenced by proficiency and maturation stage.
        """
        candidates = []
        
        # Get current action proficiencies
        proficiency_scores = self._get_current_proficiency_scores()
        
        for i in range(num_candidates):
            selection_roll = random.random()
            
            # Stage-appropriate action selection
            if selection_roll < self.proficiency_bias_strength:
                # Proficiency-biased action (prefer what we're good at)
                candidate = self._generate_proficient_action(proficiency_scores, context)
                
            elif selection_roll < self.proficiency_bias_strength + 0.2:
                # Skill-building action (practice what we're learning)
                candidate = self._generate_skill_building_action(proficiency_scores, context)
                
            elif selection_roll < 1.0 - self.exploration_rate:
                # Template-based action (standard approach)
                candidate = self._generate_template_action(context)
                
            else:
                # Exploration action (try something new)
                candidate = self._generate_exploration_action(context)
            
            candidates.append(candidate)
        
        return candidates
    
    def _get_current_proficiency_scores(self) -> Dict[str, float]:
        """Get current proficiency scores for all action types."""
        scores = {}
        for action_type, proficiency in self.action_proficiencies.items():
            scores[action_type] = proficiency.mastery_level
        return scores
    
    def _generate_proficient_action(self, proficiency_scores: Dict[str, float],
                                  context: DriveContext) -> Dict[str, float]:
        """Generate an action based on what we're most proficient at."""
        if not proficiency_scores:
            return self._generate_template_action(context)
        
        # Select action type based on proficiency
        best_action_type = max(proficiency_scores, key=proficiency_scores.get)
        proficiency_level = proficiency_scores[best_action_type]
        
        # Generate refined action based on proficiency level
        action = {
            'forward_motor': 0.0,
            'turn_motor': 0.0,
            'brake_motor': 0.0
        }
        
        if best_action_type == 'forward_motor':
            # More precise forward movement based on proficiency
            base_strength = random.uniform(0.3, 0.7)
            refinement = (proficiency_level - 0.5) * 0.3  # Adjust based on skill
            action['forward_motor'] = max(-1.0, min(1.0, base_strength + refinement))
            
        elif best_action_type == 'turn_motor':
            # More precise turning based on proficiency
            turn_direction = random.choice([-1, 1])
            base_strength = random.uniform(0.4, 0.8)
            refinement = (proficiency_level - 0.5) * 0.2
            action['turn_motor'] = turn_direction * max(0.1, min(1.0, base_strength + refinement))
            
        elif best_action_type == 'brake_motor':
            # More controlled braking
            action['brake_motor'] = 0.3 + proficiency_level * 0.4
        
        return action
    
    def _generate_skill_building_action(self, proficiency_scores: Dict[str, float],
                                      context: DriveContext) -> Dict[str, float]:
        """Generate an action to practice improving skills."""
        # Find action type that needs improvement (low proficiency but some experience)
        improving_actions = {
            action_type: score for action_type, score in proficiency_scores.items()
            if 0.2 < score < 0.7 and self.action_proficiencies[action_type].total_attempts > 5
        }
        
        if improving_actions:
            practice_action_type = random.choice(list(improving_actions.keys()))
            return self._generate_practice_action(practice_action_type)
        else:
            return self._generate_template_action(context)
    
    def _generate_practice_action(self, action_type: str) -> Dict[str, float]:
        """Generate a practice action for a specific action type."""
        action = {
            'forward_motor': 0.0,
            'turn_motor': 0.0,
            'brake_motor': 0.0
        }
        
        # Generate focused practice action
        if action_type == 'forward_motor':
            action['forward_motor'] = random.uniform(-0.6, 0.6)
        elif action_type == 'turn_motor':
            action['turn_motor'] = random.uniform(-0.7, 0.7)
        elif action_type == 'brake_motor':
            action['brake_motor'] = random.uniform(0.2, 0.8)
        
        return action
    
    def _generate_template_action(self, context: DriveContext) -> Dict[str, float]:
        """Generate a standard template action."""
        # This would call the existing template generation logic
        templates = [
            {'forward_motor': random.uniform(0.2, 0.5), 'turn_motor': 0.0, 'brake_motor': 0.0},
            {'forward_motor': 0.0, 'turn_motor': random.choice([-0.6, 0.6]), 'brake_motor': 0.0},
            {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': random.uniform(0.3, 0.7)},
        ]
        return random.choice(templates)
    
    def _generate_exploration_action(self, context: DriveContext) -> Dict[str, float]:
        """Generate a random exploration action."""
        return {
            'forward_motor': random.uniform(-0.8, 0.8),
            'turn_motor': random.uniform(-0.8, 0.8),
            'brake_motor': random.uniform(0.0, 0.5)
        }
    
    def get_maturation_status(self) -> Dict[str, Any]:
        """Get comprehensive maturation status report."""
        # Get development status from cognitive development system
        dev_status = self.cognitive_development.get_development_status()
        
        return {
            'developmental_stage': dev_status['current_stage'],
            'total_actions': self.total_actions_taken,
            'system_age_seconds': self.system_age,
            'exploration_rate': self.exploration_rate,
            'proficiency_bias_strength': self.proficiency_bias_strength,
            'action_proficiencies': {
                action_type: {
                    'mastery_level': prof.mastery_level,
                    'confidence': prof.get_confidence_level(),
                    'preference_strength': prof.preference_strength,
                    'total_attempts': prof.total_attempts,
                    'consistency': prof.consistency_score
                }
                for action_type, prof in self.action_proficiencies.items()
                if prof.total_attempts > 0
            },
            'stage_transitions': dev_status['stage_transitions'],
            'cognitive_development': dev_status,
            'constraint_based_development': True
        }
    
    def get_current_stage(self) -> DevelopmentalStage:
        """Get current developmental stage from cognitive development system."""
        return self.cognitive_development.current_stage
    
    def get_stage_transition_history(self) -> List[Dict]:
        """Get stage transition history from cognitive development system."""
        return self.cognitive_development.stage_transition_history
    
    def get_active_cognitive_constraints(self) -> List[Dict]:
        """Get currently active cognitive constraints that are driving development."""
        dev_status = self.cognitive_development.get_development_status()
        return dev_status['active_constraints']