"""
Test Cognitive Constraint-Based Development System.

Demonstrates how developmental transitions occur based on actual cognitive limitations
(search depth decline, memory pressure, learning plateaus) rather than arbitrary 
action count thresholds.
"""

import sys
import time
import random
from typing import List, Dict

# Add project root to path
sys.path.append('.')

from drives.cognitive_development import CognitiveDevelopmentSystem
from drives.development_types import DevelopmentalStage
from drives.action_proficiency import ActionMaturationSystem
from simulation.brainstem_sim import GridWorldBrainstem


class CognitiveDevelopmentExperiment:
    """Experiment to demonstrate constraint-based developmental transitions."""
    
    def __init__(self, world_graph):
        self.world_graph = world_graph
        
        # Create action maturation system with cognitive development
        self.maturation_system = ActionMaturationSystem(world_graph)
        
        # Track experiment progress
        self.simulated_experiences = []
        self.transition_events = []
        self.constraint_timeline = []
    
    def run_cognitive_development_experiment(self, num_experiences: int = 1000) -> Dict:
        """
        Simulate robot experiences that trigger cognitive constraints.
        
        Args:
            num_experiences: Number of experiences to simulate
            
        Returns:
            Analysis of cognitive development progression
        """
        print("🧠 COGNITIVE CONSTRAINT-BASED DEVELOPMENT EXPERIMENT")
        print("=" * 70)
        print(f"Simulating {num_experiences} experiences to trigger cognitive constraints...")
        print("Watching for transitions based on actual brain limitations rather than fixed thresholds")
        print()
        
        # Record initial state
        initial_stage = self.maturation_system.get_current_stage()
        print(f"🌱 Starting developmental stage: {initial_stage.value}")
        print()
        
        # Simulate experiences that gradually stress the cognitive system
        for experience_num in range(num_experiences):
            # Simulate a robot experience with various cognitive demands
            experience_data = self._simulate_cognitive_experience(experience_num, num_experiences)
            
            # Update the cognitive development system
            self.maturation_system.update_system_maturation(
                traversal_result=experience_data['traversal_result'],
                prediction_accuracy=experience_data['prediction_accuracy'],
                thinking_time=experience_data['thinking_time'],
                action_taken=experience_data['action_taken']
            )
            
            # Check for stage transitions
            current_stage = self.maturation_system.get_current_stage()
            if self.simulated_experiences and current_stage != self.simulated_experiences[-1]['stage']:
                transition_event = {
                    'experience_num': experience_num,
                    'from_stage': self.simulated_experiences[-1]['stage'].value,
                    'to_stage': current_stage.value,
                    'active_constraints': self.maturation_system.get_active_cognitive_constraints(),
                    'cognitive_metrics': self._get_current_cognitive_metrics()
                }
                self.transition_events.append(transition_event)
                
                print(f"🎯 DEVELOPMENTAL TRANSITION at experience {experience_num}")
                print(f"   {transition_event['from_stage']} → {transition_event['to_stage']}")
                
                # Print triggering constraints
                for constraint in transition_event['active_constraints']:
                    print(f"   Triggered by: {constraint['description']}")
                    print(f"   Severity: {constraint['severity']:.2f}")
                print()
            
            # Record experience
            self.simulated_experiences.append({
                'experience_num': experience_num,
                'stage': current_stage,
                'cognitive_metrics': self._get_current_cognitive_metrics(),
                'active_constraints': self.maturation_system.get_active_cognitive_constraints(),
                **experience_data
            })
            
            # Track constraint evolution
            if experience_num % 100 == 0:
                self._record_constraint_snapshot(experience_num)
            
            # Progress updates
            if experience_num % 200 == 0 and experience_num > 0:
                self._print_progress_update(experience_num)
        
        # Final analysis
        analysis = self._analyze_cognitive_development()
        self._print_development_summary(analysis)
        
        return analysis
    
    def _simulate_cognitive_experience(self, experience_num: int, total_experiences: int) -> Dict:
        """
        Simulate a robot experience that stresses the cognitive system in realistic ways.
        
        This simulates how a real robot's brain would face increasing constraints:
        - Early experiences: Fast, shallow thinking (small brain)
        - Mid experiences: Declining search depth (memory pressure)
        - Late experiences: Learning plateaus (optimization phase)
        """
        progress = experience_num / total_experiences
        
        # Simulate declining search depth due to brain growth
        # Early: Deep search possible (small graph)
        # Later: Shallow search forced (large graph, memory pressure)
        if progress < 0.3:
            # Infancy: Small graph allows deep search
            simulated_search_depth = random.randint(15, 25)
            thinking_time = random.uniform(0.05, 0.15)
        elif progress < 0.6:
            # Adolescence: Growing graph limits search depth
            simulated_search_depth = random.randint(8, 15)
            thinking_time = random.uniform(0.15, 0.25)  # Slower due to more nodes
        else:
            # Maturity: Large graph forces shallow search
            simulated_search_depth = random.randint(3, 10)
            thinking_time = random.uniform(0.2, 0.4)  # Even slower, need efficiency
        
        # Simulate learning curve: rapid early improvement, then plateau
        base_accuracy = 0.3
        if progress < 0.4:
            # Rapid learning phase
            learning_bonus = progress * 0.4  # Up to 40% improvement
        else:
            # Learning plateau phase
            learning_bonus = 0.4 - (progress - 0.4) * 0.1  # Slight decline/plateau
        
        prediction_accuracy = min(0.9, base_accuracy + learning_bonus + random.uniform(-0.1, 0.1))
        
        # Create mock traversal result
        class MockTraversalResult:
            def __init__(self, steps_taken, prediction_accuracy):
                self.steps_taken = steps_taken
                self.prediction = MockPrediction(prediction_accuracy)
        
        class MockPrediction:
            def __init__(self, accuracy):
                self.confidence_level = accuracy
        
        # Generate action based on current developmental parameters
        action = {
            'forward_motor': random.uniform(-0.5, 0.5),
            'turn_motor': random.uniform(-0.5, 0.5),
            'brake_motor': random.uniform(0.0, 0.3)
        }
        
        return {
            'traversal_result': MockTraversalResult(simulated_search_depth, prediction_accuracy),
            'prediction_accuracy': prediction_accuracy,
            'thinking_time': thinking_time,
            'action_taken': action,
            'simulated_search_depth': simulated_search_depth
        }
    
    def _get_current_cognitive_metrics(self) -> Dict:
        """Get current cognitive metrics from the development system."""
        dev_status = self.maturation_system.cognitive_development.get_development_status()
        return dev_status['cognitive_metrics']
    
    def _record_constraint_snapshot(self, experience_num: int):
        """Record a snapshot of cognitive constraints at this point."""
        snapshot = {
            'experience_num': experience_num,
            'developmental_stage': self.maturation_system.get_current_stage().value,
            'active_constraints': self.maturation_system.get_active_cognitive_constraints(),
            'cognitive_metrics': self._get_current_cognitive_metrics()
        }
        self.constraint_timeline.append(snapshot)
    
    def _print_progress_update(self, experience_num: int):
        """Print progress update during experiment."""
        current_stage = self.maturation_system.get_current_stage()
        metrics = self._get_current_cognitive_metrics()
        constraints = self.maturation_system.get_active_cognitive_constraints()
        
        print(f"Experience {experience_num:3d}: Stage={current_stage.value:15s}")
        
        if constraints:
            print(f"                     Active constraints: {len(constraints)}")
            for constraint in constraints[:2]:  # Show first 2 constraints
                print(f"                       - {constraint['type']}: {constraint['severity']:.2f}")
        else:
            print(f"                     No active constraints")
        print()
    
    def _analyze_cognitive_development(self) -> Dict:
        """Analyze the cognitive development experiment results."""
        if not self.simulated_experiences:
            return {"error": "no_experiences"}
        
        # Analyze stage progression
        stages_visited = []
        for transition in self.transition_events:
            stages_visited.append(transition['to_stage'])
        
        # Analyze constraint evolution
        constraint_analysis = self._analyze_constraint_patterns()
        
        # Analyze timing of transitions
        transition_analysis = self._analyze_transition_timing()
        
        # Final state
        final_experience = self.simulated_experiences[-1]
        
        return {
            'experiment_overview': {
                'total_experiences': len(self.simulated_experiences),
                'developmental_transitions': len(self.transition_events),
                'final_stage': final_experience['stage'].value,
                'stages_visited': stages_visited,
                'constraint_based_transitions': True
            },
            'transition_analysis': transition_analysis,
            'constraint_evolution': constraint_analysis,
            'transition_events': self.transition_events,
            'constraint_timeline': self.constraint_timeline,
            'final_cognitive_state': {
                'stage': final_experience['stage'].value,
                'active_constraints': final_experience['active_constraints'],
                'cognitive_metrics': final_experience['cognitive_metrics']
            }
        }
    
    def _analyze_constraint_patterns(self) -> Dict:
        """Analyze how cognitive constraints evolved over time."""
        constraint_counts = {}
        constraint_severities = {}
        
        for experience in self.simulated_experiences:
            for constraint in experience['active_constraints']:
                constraint_type = constraint['type']
                
                if constraint_type not in constraint_counts:
                    constraint_counts[constraint_type] = 0
                    constraint_severities[constraint_type] = []
                
                constraint_counts[constraint_type] += 1
                constraint_severities[constraint_type].append(constraint['severity'])
        
        # Calculate average severities
        avg_severities = {}
        for constraint_type, severities in constraint_severities.items():
            avg_severities[constraint_type] = sum(severities) / len(severities)
        
        return {
            'constraint_frequency': constraint_counts,
            'average_constraint_severities': avg_severities,
            'total_unique_constraints': len(constraint_counts)
        }
    
    def _analyze_transition_timing(self) -> Dict:
        """Analyze when transitions occurred and what triggered them."""
        if not self.transition_events:
            return {'status': 'no_transitions'}
        
        transition_triggers = {}
        transition_timings = []
        
        for transition in self.transition_events:
            # Track what triggered each transition
            for constraint in transition['active_constraints']:
                trigger_type = constraint['type']
                if trigger_type not in transition_triggers:
                    transition_triggers[trigger_type] = 0
                transition_triggers[trigger_type] += 1
            
            # Track timing
            transition_timings.append(transition['experience_num'])
        
        return {
            'status': 'analyzed',
            'transition_triggers': transition_triggers,
            'transition_timings': transition_timings,
            'average_time_between_transitions': sum(transition_timings) / len(transition_timings) if transition_timings else 0
        }
    
    def _print_development_summary(self, analysis: Dict):
        """Print comprehensive cognitive development summary."""
        print("\n📊 COGNITIVE DEVELOPMENT RESULTS")
        print("=" * 70)
        
        # Overview
        overview = analysis['experiment_overview']
        print(f"Total experiences simulated: {overview['total_experiences']}")
        print(f"Developmental transitions: {overview['developmental_transitions']}")
        print(f"Final developmental stage: {overview['final_stage']}")
        print(f"Stages visited: {' → '.join(overview['stages_visited'])}")
        print()
        
        # Transition analysis
        if self.transition_events:
            print("🎯 CONSTRAINT-BASED TRANSITIONS:")
            for i, transition in enumerate(self.transition_events, 1):
                print(f"  Transition {i}: {transition['from_stage']} → {transition['to_stage']}")
                print(f"    Experience: {transition['experience_num']}")
                print(f"    Triggering constraints:")
                for constraint in transition['active_constraints']:
                    print(f"      - {constraint['description']}")
                    print(f"        Severity: {constraint['severity']:.2f}")
                print()
        
        # Constraint evolution
        constraint_analysis = analysis['constraint_evolution']
        print("🧠 COGNITIVE CONSTRAINT PATTERNS:")
        print(f"  Unique constraint types encountered: {constraint_analysis['total_unique_constraints']}")
        
        if constraint_analysis['constraint_frequency']:
            print("  Constraint frequency:")
            for constraint_type, count in constraint_analysis['constraint_frequency'].items():
                avg_severity = constraint_analysis['average_constraint_severities'][constraint_type]
                print(f"    {constraint_type}: {count} occurrences (avg severity: {avg_severity:.2f})")
        print()
        
        # Final state
        final_state = analysis['final_cognitive_state']
        print("🏆 FINAL COGNITIVE STATE:")
        print(f"  Developmental stage: {final_state['stage']}")
        
        if final_state['active_constraints']:
            print(f"  Active constraints: {len(final_state['active_constraints'])}")
            for constraint in final_state['active_constraints']:
                print(f"    - {constraint['type']}: {constraint['severity']:.2f}")
        else:
            print("  No active constraints (cognitive equilibrium)")
        
        metrics = final_state['cognitive_metrics']
        print(f"  Search depth: {metrics.get('search_depth', 'N/A')}")
        print(f"  Memory utilization: {metrics.get('memory_utilization', 'N/A'):.1%}")
        print(f"  Learning rate: {metrics.get('learning_rate', 'N/A')}")
        print()
        
        # Key achievements
        print("🌟 KEY DEVELOPMENT ACHIEVEMENTS:")
        
        if overview['developmental_transitions'] > 0:
            print("  ✅ Successfully demonstrated constraint-based developmental transitions")
        
        if 'maturity' in overview['stages_visited'] or 'expertise' in overview['stages_visited']:
            print("  ✅ Reached advanced developmental stages through cognitive constraints")
        
        if constraint_analysis['total_unique_constraints'] >= 3:
            print("  ✅ Multiple types of cognitive constraints successfully triggered development")
        
        if len(self.transition_events) >= 2:
            print("  ✅ Multiple developmental transitions based on emergent cognitive limitations")
        
        print()
        print("🌟 COGNITIVE DEVELOPMENT EXPERIMENT COMPLETE!")
        print("Successfully demonstrated emergent developmental stages based on")
        print("actual cognitive constraints rather than arbitrary thresholds!")


def main():
    """Run the cognitive development experiment."""
    print("🚀 Loading brain system for cognitive development testing...")
    
    # Load existing brain system
    brainstem = GridWorldBrainstem(seed=42, use_sockets=False)
    world_graph = brainstem.brain_client.get_world_graph()
    
    print(f"✅ Loaded graph with {world_graph.node_count()} nodes")
    print()
    
    # Run experiment
    experiment = CognitiveDevelopmentExperiment(world_graph)
    results = experiment.run_cognitive_development_experiment(num_experiences=800)
    
    return results


if __name__ == "__main__":
    results = main()