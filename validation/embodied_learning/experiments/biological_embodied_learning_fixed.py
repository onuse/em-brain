#!/usr/bin/env python3
"""
Fixed Biological Embodied Learning Experiment

This version includes:
1. Proper 25D sensory input (24 sensors + 1 reward)
2. Correct motor gradient interpretation
3. Enabled strategic planning and prediction systems
"""

import sys
import os
from pathlib import Path

# Import the original experiment
brain_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from biological_embodied_learning import *
from ..fix_motor_mapping import gradient_to_action, add_reward_signal


class FixedBiologicalEmbodiedLearning(BiologicalEmbodiedLearning):
    """Fixed version with proper motor mapping and reward signal."""
    
    def _run_learning_session(self, session_id: int, duration_minutes: int) -> SessionResults:
        """Run a single learning session with fixes."""
        print(f"\n{'='*60}")
        print(f"📚 Learning Session {session_id}")
        
        # Reset environment
        self.environment.reset()
        print("🔄 Environment reset to initial state")
        
        # Session parameters
        session_start = time.time()
        session_duration = duration_minutes * 60
        target_actions = int(duration_minutes * self.config.actions_per_minute)
        
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Target actions: {target_actions}")
        
        # Get initial brain state
        initial_brain_state = self._get_enhanced_brain_state()
        print(f"   🧬 Session {session_id} starting brain state:")
        print(f"      Self-modification: {initial_brain_state['self_modification']:.1f}%")
        print(f"      Working memory: {initial_brain_state['working_memory_patterns']} patterns")
        
        # Tracking variables
        actions_executed = 0
        prediction_errors = []
        light_distances = []
        action_counts = {'MOVE_FORWARD': 0, 'TURN_LEFT': 0, 'TURN_RIGHT': 0, 'STOP': 0}
        trajectory_points = []
        
        # Telemetry tracking
        confidence_samples = []
        heartbeat_interval = 300  # 5 minutes
        last_heartbeat = session_start
        
        # Store initial position for movement reward
        self.environment.robot_state.last_position = self.environment.robot_state.position.copy()
        
        while (time.time() - session_start) < session_duration and actions_executed < target_actions:
            
            try:
                # Get sensory input (24D)
                sensory_input = self.environment.get_sensory_input()
                
                # Add reward signal as 25th element
                brain_sensory_input = add_reward_signal(sensory_input, self.environment)
                
                # Send to brain and get prediction
                prediction = self.robot_client.get_action(brain_sensory_input, timeout=10.0)
                
                if prediction is None:
                    print("   ⚠️ No response from brain")
                    time.sleep(1)
                    continue
                
                # Convert motor gradients to discrete action
                action_type = gradient_to_action(prediction)
                action_vector = [0.0, 0.0, 0.0, 0.0]
                action_vector[action_type.value] = 1.0
                
                # Execute action in environment
                execution_result = self.environment.execute_action(action_vector)
                
                # Update last position for movement reward
                self.environment.robot_state.last_position = self.environment.robot_state.position.copy()
                
                # Record metrics
                metrics = execution_result['metrics']
                # Better prediction error: compare predicted movement to actual movement
                prediction_error = self._calculate_prediction_error(prediction, execution_result)
                
                prediction_errors.append(prediction_error)
                light_distances.append(metrics['min_light_distance'])
                self.light_distance_history.append(metrics['min_light_distance'])
                
                # Track action distribution
                action_name = action_type.name
                action_counts[action_name] += 1
                self.action_history.append(action_name)
                
                # Track collisions
                if action_name == 'MOVE_FORWARD' and not execution_result['success']:
                    self.collision_history.append(len(self.action_history) - 1)
                
                # Track trajectory
                robot_state = execution_result['robot_state']
                trajectory_points.append((robot_state['position'][0], robot_state['position'][1]))
                
                actions_executed += 1
                self.total_actions_executed += 1
                
                # Periodic telemetry heartbeat
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    self._print_heartbeat(
                        session_id, actions_executed, 
                        np.mean(light_distances[-50:]) if light_distances else 0,
                        confidence_samples
                    )
                    last_heartbeat = current_time
                
                # Track confidence
                if self.monitoring_client and actions_executed % 10 == 0:
                    brain_state = self.monitoring_client.get_brain_state()
                    if brain_state:
                        confidence = brain_state.get('prediction_confidence', 0.5)
                        confidence_samples.append(confidence)
                
            except Exception as e:
                print(f"   ⚠️ Error in action cycle: {e}")
                continue
        
        # Final heartbeat
        self._print_heartbeat(
            session_id, actions_executed,
            np.mean(light_distances[-50:]) if light_distances else 0,
            confidence_samples
        )
        
        # Calculate session results
        return self._calculate_session_results(
            session_id, session_start, actions_executed,
            prediction_errors, light_distances, action_counts,
            trajectory_points, confidence_samples
        )
    
    def _calculate_prediction_error(self, motor_prediction: List[float], 
                                  execution_result: Dict) -> float:
        """Calculate prediction error based on motor intent vs actual outcome."""
        # Extract motor intent
        forward_intent = motor_prediction[0] if len(motor_prediction) > 0 else 0.0
        turn_intent = motor_prediction[1] if len(motor_prediction) > 1 else 0.0
        
        # Compare with actual action executed
        action_executed = execution_result['action_executed']
        
        # Calculate error based on intent-action mismatch
        if action_executed == 0:  # MOVE_FORWARD
            error = abs(1.0 - forward_intent) + abs(turn_intent)
        elif action_executed == 1:  # TURN_LEFT
            error = abs(forward_intent) + abs(-1.0 - turn_intent)
        elif action_executed == 2:  # TURN_RIGHT
            error = abs(forward_intent) + abs(1.0 - turn_intent)
        else:  # STOP
            error = abs(forward_intent) + abs(turn_intent)
        
        return np.clip(error / 2.0, 0.0, 1.0)


def main():
    """Run the fixed experiment."""
    parser = argparse.ArgumentParser(
        description="Fixed Biological Embodied Learning Experiment"
    )
    parser.add_argument('--hours', type=float, default=4.0,
                       help='Experiment duration in hours')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, 
                       default='validation/embodied_learning/reports',
                       help='Directory for experiment results')
    
    args = parser.parse_args()
    
    # Configure experiment
    config = ExperimentConfig(
        duration_hours=args.hours,
        random_seed=args.seed
    )
    
    # Run experiment
    experiment = FixedBiologicalEmbodiedLearning(config, args.output_dir)
    experiment.run()


if __name__ == "__main__":
    main()