#!/usr/bin/env python3
"""
Profile Learning Bottlenecks

Analyze exactly where time is spent during experience learning to identify
the biggest bottlenecks for optimization.
"""

import sys
import os
import time
import numpy as np
from contextlib import contextmanager

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


@contextmanager
def timer(name):
    """Context manager to time code blocks."""
    start = time.time()
    yield
    elapsed = (time.time() - start) * 1000
    print(f"      {name}: {elapsed:.1f}ms")


def profile_learning_bottlenecks():
    """Profile exactly where time is spent during learning."""
    print("🔬 PROFILING LEARNING BOTTLENECKS")
    print("=" * 40)
    print("Analyzing where time is spent during experience learning")
    print()
    
    # Create brain for profiling
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Test with different experience counts to see scaling
    test_counts = [10, 50, 100]
    
    for exp_count in test_counts:
        print(f"📊 Profiling {exp_count} experiences:")
        
        # Clear brain state
        brain.experience_storage.clear()
        
        total_times = {
            'process_sensory_input': 0,
            'store_experience': 0,
            'similarity_updates': 0,
            'activation_updates': 0,
            'pattern_discovery': 0,
            'meta_learning': 0,
            'other': 0
        }
        
        overall_start = time.time()
        
        for i in range(exp_count):
            sensory = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01]
            
            # Profile: Sensory processing (prediction)
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            total_times['process_sensory_input'] += (time.time() - start_time) * 1000
            
            # Simulate outcome
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            
            # Profile: Experience storage and learning
            start_time = time.time()
            
            # This is where we need to profile internal components
            experience_id = profile_store_experience(brain, sensory, predicted_action, outcome, predicted_action, total_times)
            
            total_times['store_experience'] += (time.time() - start_time) * 1000
        
        overall_time = (time.time() - overall_start) * 1000
        
        # Calculate percentages
        print(f"   Total learning time: {overall_time:.1f}ms ({overall_time/exp_count:.1f}ms per experience)")
        print(f"   Time breakdown:")
        
        for component, total_time in total_times.items():
            if total_time > 0:
                percentage = (total_time / overall_time) * 100
                avg_per_exp = total_time / exp_count
                print(f"      {component}: {total_time:.1f}ms ({percentage:.1f}%, {avg_per_exp:.1f}ms/exp)")
        
        print()
    
    brain.finalize_session()
    
    print("🎯 BOTTLENECK ANALYSIS:")
    print("-" * 25)
    print("This profile shows where learning time is spent.")
    print("Focus optimization efforts on the highest percentages.")


def profile_store_experience(brain, sensory_input, action_taken, outcome, predicted_action, timers):
    """Profile the store_experience method to identify bottlenecks."""
    
    # Recreate store_experience with timing
    start_time = time.time()
    
    # Compute prediction error
    if predicted_action:
        prediction_error = brain._compute_prediction_error(predicted_action, outcome)
    else:
        prediction_error = 0.5
    
    # Compute intrinsic reward
    intrinsic_reward = brain.compute_intrinsic_reward(prediction_error)
    
    # Create experience
    from src.experience.models import Experience
    experience = Experience(
        sensory_input=sensory_input,
        action_taken=action_taken,
        outcome=outcome,
        prediction_error=prediction_error,
        timestamp=time.time(),
        prediction_utility=0.5,
        local_cluster_density=0.0
    )
    experience.intrinsic_reward = intrinsic_reward
    
    # Track learning outcomes
    brain.recent_learning_outcomes.append(intrinsic_reward)
    if len(brain.recent_learning_outcomes) > 50:
        brain.recent_learning_outcomes = brain.recent_learning_outcomes[-25:]
    
    # Store experience
    experience_id = brain.experience_storage.add_experience(experience)
    
    # Profile: Add to hierarchical index
    similarity_start = time.time()
    experience_vector = experience.get_context_vector()
    brain.similarity_engine.add_experience_to_index(experience_id, experience_vector)
    timers['similarity_updates'] += (time.time() - similarity_start) * 1000
    
    # Profile: Add to pattern analysis
    pattern_start = time.time()
    experience_data = {
        'experience_id': experience_id,
        'sensory_input': sensory_input,
        'action_taken': action_taken,
        'outcome': outcome,
        'prediction_error': prediction_error,
        'timestamp': experience.timestamp
    }
    brain.prediction_engine.add_experience_to_stream(experience_data)
    timers['pattern_discovery'] += (time.time() - pattern_start) * 1000
    
    # Profile: Activation updates
    activation_start = time.time()
    if brain.use_utility_based_activation:
        pass  # Utility-based activation emerges naturally
    else:
        base_activation = 0.8
        reward_modulated_activation = base_activation * (0.5 + intrinsic_reward * 0.5)
        brain.activation_dynamics.activate_experience(experience, strength=reward_modulated_activation)
    timers['activation_updates'] += (time.time() - activation_start) * 1000
    
    # Profile: Similarity connections and clustering
    similarity_update_start = time.time()
    brain._update_similarity_connections_and_clustering(experience)
    timers['similarity_updates'] += (time.time() - similarity_update_start) * 1000
    
    # Profile: Similarity learning outcomes
    if predicted_action and len(brain.experience_storage._experiences) > 0:
        similarity_learning_start = time.time()
        brain._record_similarity_learning_outcomes(sensory_input, predicted_action, outcome)
        timers['similarity_updates'] += (time.time() - similarity_learning_start) * 1000
    
    # Profile: Utility-based activation learning
    if brain.use_utility_based_activation and predicted_action:
        utility_start = time.time()
        prediction_success = brain._compute_prediction_success(predicted_action, outcome)
        working_memory = brain.activation_dynamics.get_working_memory_experiences()
        activated_experience_ids = [exp_id for exp_id, _ in working_memory]
        brain.activation_dynamics.record_prediction_outcome(activated_experience_ids, prediction_success)
        timers['activation_updates'] += (time.time() - utility_start) * 1000
    
    # Profile: Event-driven adaptation
    adaptation_start = time.time()
    brain.adaptive_trigger.record_prediction_outcome(prediction_error, brain.total_experiences)
    adaptation_triggers = brain.adaptive_trigger.check_adaptation_triggers(
        brain.total_experiences, 
        {"intrinsic_reward": intrinsic_reward, "prediction_error": prediction_error}
    )
    
    for trigger_type, trigger_reason, evidence in adaptation_triggers:
        brain._execute_triggered_adaptation(trigger_type, trigger_reason, evidence)
    timers['meta_learning'] += (time.time() - adaptation_start) * 1000
    
    brain.total_experiences += 1
    
    return experience_id


if __name__ == "__main__":
    profile_learning_bottlenecks()