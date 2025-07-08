#!/usr/bin/env python3
"""
Goal-Driven Robot Demo - Shows drives generating temporary objectives

This demonstrates the enhanced drive system where drives can create
specific, temporary goals when conditions warrant focused behavior.
"""

import pygame
import time
from core import WorldGraph
from simulation import GridWorldBrainstem
from visualization import IntegratedDisplay
from drives.goal_generation_drive import GoalGenerationDrive
from drives.survival_drive import SurvivalDrive
from drives.curiosity_drive import CuriosityDrive
from drives.exploration_drive import ExplorationDrive


class GoalDrivenAgent:
    """Agent that uses drive-generated goals to guide behavior."""
    
    def __init__(self):
        self.drives = {
            'survival': SurvivalDrive(base_weight=0.3),
            'curiosity': CuriosityDrive(base_weight=0.3),
            'exploration': ExplorationDrive(base_weight=0.3),
            'goal_generation': GoalGenerationDrive(base_weight=0.1)
        }
        self.recent_actions = []
        self.recent_positions = []
        self.step_count = 0
    
    def decide_action(self, state_dict) -> dict:
        """Use drives and goals to decide on action."""
        self.step_count += 1
        
        # Build drive context
        from drives.base_drive import DriveContext
        context = DriveContext(
            current_sensory=state_dict['sensors'],
            robot_health=state_dict['health'],
            robot_energy=state_dict['energy'],
            robot_position=tuple(state_dict['position']),
            robot_orientation=state_dict.get('orientation', 0),
            recent_experiences=[],  # Simplified for demo
            prediction_errors=[0.5],  # Simplified
            time_since_last_food=5,
            time_since_last_damage=10,
            threat_level='normal',  # Simplified
            step_count=self.step_count
        )
        
        # Add recent history to context for goal system
        context.recent_actions = self.recent_actions[-10:]
        context.recent_positions = self.recent_positions[-10:]
        
        # Update all drives
        for drive in self.drives.values():
            drive.update_drive_state(context)
        
        # Get goal info for display
        goal_drive = self.drives['goal_generation']
        active_goals = getattr(goal_drive, 'active_goals', [])
        
        # Generate candidate actions to evaluate
        candidates = self._generate_candidate_actions(state_dict)
        
        # Evaluate each candidate with all drives
        best_action = None
        best_score = -1
        best_reasoning = ""
        
        for action in candidates:
            total_score = 0
            reasoning_parts = []
            
            for drive_name, drive in self.drives.items():
                evaluation = drive.evaluate_action(action, context)
                weighted_score = evaluation.action_score * drive.current_weight
                total_score += weighted_score
                
                if evaluation.action_score > 0.6:  # Significant influence
                    reasoning_parts.append(f"{drive_name}: {evaluation.reasoning}")
            
            if total_score > best_score:
                best_score = total_score
                best_action = action
                best_reasoning = " | ".join(reasoning_parts)
        
        # Track history
        if best_action:
            self.recent_actions.append(best_action)
            self.recent_positions.append(tuple(state_dict['position']))
        
        # Print goals and decision info
        if active_goals:
            goal_info = ", ".join([f"{g.goal_type}({g.steps_remaining})" for g in active_goals])
            print(f"Step {self.step_count}: Active goals: {goal_info}")
            
        if best_reasoning and self.step_count % 5 == 0:
            print(f"  Decision reasoning: {best_reasoning}")
        
        return best_action or {'forward_motor': 0.2, 'turn_motor': 0.0, 'brake_motor': 0.0}
    
    def _generate_candidate_actions(self, state_dict) -> list:
        """Generate candidate actions to evaluate."""
        sensors = state_dict['sensors']
        distance_sensors = sensors[:4] if len(sensors) >= 4 else [1.0, 1.0, 1.0, 1.0]
        
        candidates = []
        
        # Forward movement options
        if distance_sensors[0] > 0.3:  # Clear ahead
            candidates.extend([
                {'forward_motor': 0.6, 'turn_motor': 0.0, 'brake_motor': 0.0},
                {'forward_motor': 0.4, 'turn_motor': 0.0, 'brake_motor': 0.0},
                {'forward_motor': 0.3, 'turn_motor': 0.2, 'brake_motor': 0.0},
                {'forward_motor': 0.3, 'turn_motor': -0.2, 'brake_motor': 0.0},
            ])
        
        # Turning options
        candidates.extend([
            {'forward_motor': 0.0, 'turn_motor': 0.7, 'brake_motor': 0.0},
            {'forward_motor': 0.0, 'turn_motor': -0.7, 'brake_motor': 0.0},
            {'forward_motor': 0.2, 'turn_motor': 0.5, 'brake_motor': 0.0},
            {'forward_motor': 0.2, 'turn_motor': -0.5, 'brake_motor': 0.0},
        ])
        
        # Conservative options
        candidates.extend([
            {'forward_motor': 0.0, 'turn_motor': 0.0, 'brake_motor': 0.5},
            {'forward_motor': 0.1, 'turn_motor': 0.0, 'brake_motor': 0.2},
        ])
        
        # Exploration options
        candidates.extend([
            {'forward_motor': 0.5, 'turn_motor': 0.3, 'brake_motor': 0.0},
            {'forward_motor': 0.5, 'turn_motor': -0.3, 'brake_motor': 0.0},
        ])
        
        return candidates


def main():
    """Launch goal-driven robot demo."""
    print("🎯 GOAL-DRIVEN ROBOT DEMO")
    print("=" * 50)
    print("Robot uses drive-generated goals for focused behavior:")
    print("• Survival drive creates 'find food' goals when energy low")
    print("• Curiosity drive creates 'investigate' goals for anomalies") 
    print("• Exploration drive creates 'map area' goals when stuck")
    print("• Goal generation creates 'break pattern' goals for loops")
    print()
    print("Watch console for active goals and decision reasoning!")
    print("=" * 50)
    
    try:
        # Create simulation
        brainstem = GridWorldBrainstem(world_width=15, world_height=15, seed=42)
        brain_graph = WorldGraph()
        
        # Create goal-driven agent
        agent = GoalDrivenAgent()
        
        # Create display
        display = IntegratedDisplay(brainstem, cell_size=25)
        display.set_brain_graph(brain_graph)
        display.set_learning_callback(lambda state: agent.decide_action(state))
        
        print(f"🖥️  Window: {display.window_width}x{display.window_height}")
        print("🚀 Launching goal-driven behavior...")
        print("   Watch for focused objectives in console output!")
        
        display.run(auto_step=True, step_delay=0.4)  # Slower for observation
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print final goal statistics
        if 'agent' in locals():
            goal_drive = agent.drives['goal_generation']
            stats = goal_drive.get_goal_statistics()
            print(f"\n📊 Final Goal Statistics:")
            print(f"   Active goals: {stats['active_goals']}")
            print(f"   Completed goals: {stats['completed_goals']}")
            print(f"   Recent goal history: {stats['recent_goal_history']}")
    
    print("✅ Goal-driven demo completed")


if __name__ == "__main__":
    main()