"""
Intelligent Field Brain - The Complete System

Integrates discrete dynamics, compositional binding, and temporal planning
to create a field brain capable of true intelligence.
"""

import torch
import time
from typing import List, Dict, Any, Tuple, Optional

# Import base brain
from .unified_field_brain import UnifiedFieldBrain

# Import intelligence modules
from .discrete_attractor_dynamics import DiscreteAttractorDynamics, Concept
from .compositional_binding import CompositionalBinding
from .temporal_planning import TemporalPlanning


class IntelligentFieldBrain(UnifiedFieldBrain):
    """
    Field brain enhanced with discrete dynamics for true intelligence.
    
    Adds three critical capabilities:
    1. Concept formation through discrete attractors
    2. Concept composition through phase binding
    3. Planning through temporal wave propagation
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize intelligent brain with all systems."""
        super().__init__(*args, **kwargs)
        
        # Intelligence modules
        self.attractors = DiscreteAttractorDynamics(self.field.shape, self.device)
        self.binding = CompositionalBinding(self.field.shape, self.device)
        self.planning = TemporalPlanning(self.field.shape, self.device)
        
        # Cognitive state
        self.active_concepts: List[Concept] = []
        self.current_bindings: List[int] = []
        self.current_plan = None
        self.planning_enabled = True
        
        # Intelligence metrics
        self.concept_formation_rate = 0
        self.binding_complexity = 0
        self.planning_horizon_achieved = 0
        
        if not self.quiet_mode:
            print("🧠 Intelligence modules initialized:")
            print("   ✓ Discrete attractors for concepts")
            print("   ✓ Phase binding for composition")
            print("   ✓ Temporal planning for foresight")
    
    def process(self, sensory_input: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """
        Enhanced processing with intelligence capabilities.
        
        Args:
            sensory_input: Sensor values
            
        Returns:
            motor_output: Motor commands (potentially planned)
            telemetry: Enhanced telemetry with intelligence metrics
        """
        start_time = time.perf_counter()
        
        # Run base processing first
        motors, telemetry = super().process(sensory_input)
        
        # ===== INTELLIGENCE LAYER 1: CONCEPT FORMATION =====
        # Allow coherent patterns to crystallize into discrete concepts
        pre_concepts = len(self.attractors.concepts)
        self.field = self.attractors.evolve_field_with_attractors(self.field)
        post_concepts = len(self.attractors.concepts)
        
        # Track concept formation
        if post_concepts > pre_concepts:
            self.concept_formation_rate += 1
            if not self.quiet_mode:
                print(f"  💡 New concept formed! Total: {post_concepts}")
        
        # Update active concepts
        self.active_concepts = self.attractors.get_active_concepts()
        
        # ===== INTELLIGENCE LAYER 2: COMPOSITIONAL BINDING =====
        # Bind related concepts together
        if len(self.active_concepts) >= 2:
            # Simple heuristic: bind concepts that are spatially close
            positions = [c.position for c in self.active_concepts[:3]]  # Limit binding size
            
            # Check if we should create new binding
            should_bind = self._should_bind_concepts(positions)
            
            if should_bind:
                self.field, binding_id = self.binding.bind_concepts(self.field, positions)
                if binding_id >= 0:
                    self.current_bindings.append(binding_id)
                    self.binding_complexity = max(self.binding_complexity, len(positions))
                    if not self.quiet_mode:
                        print(f"  🔗 Bound {len(positions)} concepts together")
        
        # Evolve phase dynamics for existing bindings
        self.field = self.binding.evolve_phases(self.field, dt=0.01)
        
        # ===== INTELLIGENCE LAYER 3: TEMPORAL PLANNING =====
        # Plan future actions if enabled
        if self.planning_enabled and self.cycle % 20 == 0:  # Plan every 20 cycles (reduced frequency)
            # Use comfort as planning objective
            comfort_fn = lambda field: self.tensions.get_comfort_metrics(field)['overall_comfort']
            
            # Generate plan
            self.current_plan = self.planning.plan_sequence(
                self.field, 
                motors,
                comfort_fn
            )
            
            if self.current_plan and self.current_plan.confidence > 0.3:
                # Use planned action instead of reactive action
                planned_motors = self.planning.get_immediate_action(self.current_plan)
                
                # Blend planned and reactive (for safety)
                alpha = self.current_plan.confidence
                motors = [
                    alpha * p + (1 - alpha) * r 
                    for p, r in zip(planned_motors, motors)
                ]
                
                self.planning_horizon_achieved = len(self.current_plan.steps)
                
                if not self.quiet_mode and self.current_plan.confidence > 0.5:
                    print(f"  🎯 Executing plan (confidence: {self.current_plan.confidence:.2f})")
        
        # ===== ENHANCED TELEMETRY =====
        intelligence_metrics = {
            'n_concepts': len(self.attractors.concepts),
            'n_active_concepts': len(self.active_concepts),
            'n_bindings': len(self.current_bindings),
            'concept_formation_rate': self.concept_formation_rate,
            'binding_complexity': self.binding_complexity,
            'planning_confidence': self.current_plan.confidence if self.current_plan else 0,
            'planning_horizon': self.planning_horizon_achieved,
            'intelligence_active': len(self.active_concepts) > 0
        }
        
        telemetry.update(intelligence_metrics)
        telemetry['time_ms'] = (time.perf_counter() - start_time) * 1000
        
        return motors, telemetry
    
    def _should_bind_concepts(self, positions: List[torch.Tensor]) -> bool:
        """
        Determine if concepts should be bound together.
        
        Simple heuristic based on spatial proximity and field coherence.
        """
        if len(positions) < 2:
            return False
        
        # Check spatial proximity
        max_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = torch.norm(positions[i].float() - positions[j].float())
                max_distance = max(max_distance, distance.item())
        
        # Bind if concepts are reasonably close
        return max_distance < self.spatial_size / 2
    
    def set_goal(self, goal_description: str):
        """
        Set a high-level goal for the brain.
        
        Args:
            goal_description: Natural language goal
        """
        # For now, create a simple goal pattern
        # In future, this could use language understanding
        
        if "explore" in goal_description.lower():
            # Set exploration goal (high variance)
            goal_pattern = torch.randn(self.field.shape, device=self.device) * 0.5
            self.planning.set_goal(goal_pattern)
            print(f"  🎯 Goal set: Exploration")
            
        elif "rest" in goal_description.lower():
            # Set rest goal (low energy)
            goal_pattern = torch.zeros(self.field.shape, device=self.device)
            self.planning.set_goal(goal_pattern)
            print(f"  🎯 Goal set: Rest")
            
        else:
            print(f"  ❓ Goal not understood: {goal_description}")
    
    def inspect_concepts(self) -> List[Dict[str, Any]]:
        """
        Inspect current concepts for debugging/visualization.
        
        Returns:
            List of concept information
        """
        concepts_info = []
        
        for concept in self.active_concepts:
            # Find what this concept is bound to
            bound_to = self.binding.find_bound_concepts(concept.position)
            
            info = {
                'position': concept.position.cpu().tolist(),
                'strength': concept.strength,
                'age': concept.age,
                'bound_to': len(bound_to),
                'pattern_norm': concept.pattern.norm().item()
            }
            concepts_info.append(info)
        
        return concepts_info
    
    def reason_about(self, query: str) -> str:
        """
        Simple reasoning using concepts and bindings.
        
        Args:
            query: What to reason about
            
        Returns:
            Reasoning result
        """
        # This is a simplified reasoning system
        # Real reasoning would require language grounding
        
        if "concepts" in query.lower():
            n_concepts = len(self.active_concepts)
            n_bound = sum(1 for c in self.active_concepts 
                         if self.binding.find_bound_concepts(c.position))
            return f"I have {n_concepts} active concepts, {n_bound} are bound together"
        
        elif "plan" in query.lower():
            if self.current_plan:
                return f"I have a {len(self.current_plan.steps)}-step plan with {self.current_plan.confidence:.1%} confidence"
            else:
                return "I have no current plan"
        
        elif "feeling" in query.lower() or "state" in query.lower():
            comfort = self.tensions.get_comfort_metrics(self.field)
            if comfort['overall_comfort'] > 0.7:
                return "I feel comfortable and stable"
            elif comfort['overall_comfort'] > 0.3:
                return "I feel slightly uncomfortable, exploring options"
            else:
                return "I feel very uncomfortable, actively seeking better states"
        
        return "I don't understand that query yet"
    
    def save_knowledge(self, filepath: str):
        """
        Save learned concepts and bindings.
        
        Args:
            filepath: Where to save knowledge
        """
        knowledge = {
            'concepts': [
                {
                    'position': c.position.cpu().tolist(),
                    'pattern': c.pattern.cpu().tolist(),
                    'strength': c.strength,
                    'age': c.age
                }
                for c in self.attractors.concepts
            ],
            'concept_formation_rate': self.concept_formation_rate,
            'binding_complexity': self.binding_complexity
        }
        
        torch.save(knowledge, filepath)
        print(f"  💾 Saved {len(self.attractors.concepts)} concepts to {filepath}")
    
    def load_knowledge(self, filepath: str):
        """
        Load previously learned concepts.
        
        Args:
            filepath: Knowledge file to load
        """
        knowledge = torch.load(filepath, map_location=self.device)
        
        # Restore concepts
        self.attractors.concepts = []
        for c_dict in knowledge['concepts']:
            concept = Concept(
                position=torch.tensor(c_dict['position'], device=self.device),
                pattern=torch.tensor(c_dict['pattern'], device=self.device),
                strength=c_dict['strength'],
                age=c_dict['age']
            )
            self.attractors.concepts.append(concept)
        
        self.concept_formation_rate = knowledge.get('concept_formation_rate', 0)
        self.binding_complexity = knowledge.get('binding_complexity', 0)
        
        print(f"  💾 Loaded {len(self.attractors.concepts)} concepts from {filepath}")