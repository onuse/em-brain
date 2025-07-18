#!/usr/bin/env python3
"""
Manual Timing Analysis

Manually time each major component of learning to identify bottlenecks.
"""

import sys
import os
import time
import numpy as np

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def manual_timing_analysis():
    """Manually time learning components to find bottlenecks."""
    print("⏱️  MANUAL TIMING ANALYSIS")
    print("=" * 35)
    print("Testing 50 experiences with detailed timing...")
    print()
    
    brain = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        quiet_mode=True
    )
    
    # Track times for different components
    times = {
        'prediction': [],
        'similarity_learning': [],
        'pattern_analysis': [],
        'activation_learning': [],
        'total_store': []
    }
    
    for i in range(50):
        sensory = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.3 + i * 0.01, 0.4 + i * 0.01]
        
        # Time: Prediction
        start = time.time()
        predicted_action, brain_state = brain.process_sensory_input(sensory)
        times['prediction'].append((time.time() - start) * 1000)
        
        outcome = [a * 0.9 + 0.05 for a in predicted_action]
        
        # Time: Total storage (but we'll break it down)
        store_start = time.time()
        
        # Manually instrument store_experience to time components
        times_for_this_exp = time_store_experience_components(brain, sensory, predicted_action, outcome, predicted_action)
        
        times['similarity_learning'].append(times_for_this_exp['similarity'])
        times['pattern_analysis'].append(times_for_this_exp['pattern'])
        times['activation_learning'].append(times_for_this_exp['activation'])
        times['total_store'].append((time.time() - store_start) * 1000)
    
    brain.finalize_session()
    
    # Analyze timing results
    print("📊 TIMING BREAKDOWN (50 experiences):")
    print("-" * 40)
    
    for component, timing_list in times.items():
        avg_time = np.mean(timing_list)
        total_time = np.sum(timing_list)
        max_time = np.max(timing_list)
        std_time = np.std(timing_list)
        
        print(f"{component:18}: {avg_time:6.2f}ms avg, {total_time:7.1f}ms total, {max_time:6.2f}ms max (±{std_time:.2f})")
    
    # Calculate percentages of total learning time
    total_learning_time = np.sum(times['total_store']) + np.sum(times['prediction'])
    
    print()
    print("📈 PERCENTAGE BREAKDOWN:")
    print("-" * 25)
    
    prediction_pct = (np.sum(times['prediction']) / total_learning_time) * 100
    similarity_pct = (np.sum(times['similarity_learning']) / total_learning_time) * 100
    pattern_pct = (np.sum(times['pattern_analysis']) / total_learning_time) * 100
    activation_pct = (np.sum(times['activation_learning']) / total_learning_time) * 100
    storage_pct = (np.sum(times['total_store']) / total_learning_time) * 100
    
    print(f"Prediction:        {prediction_pct:5.1f}%")
    print(f"Similarity learning: {similarity_pct:5.1f}%")
    print(f"Pattern analysis:    {pattern_pct:5.1f}%")
    print(f"Activation learning: {activation_pct:5.1f}%")
    print(f"Storage overhead:    {storage_pct:5.1f}%")
    
    print()
    print("🎯 BOTTLENECK RANKING:")
    print("-" * 20)
    
    components = [
        ("Prediction", prediction_pct),
        ("Similarity learning", similarity_pct),
        ("Pattern analysis", pattern_pct),
        ("Activation learning", activation_pct),
        ("Storage overhead", storage_pct)
    ]
    
    components.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, pct) in enumerate(components, 1):
        if pct > 25:
            status = "🚨 CRITICAL"
        elif pct > 15:
            status = "⚠️  HIGH"
        elif pct > 5:
            status = "📊 MEDIUM"
        else:
            status = "✅ LOW"
        
        print(f"{i}. {name:18}: {pct:5.1f}% {status}")
    
    print()
    print("💡 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 32)
    
    top_bottleneck = components[0]
    if top_bottleneck[1] > 30:
        print(f"🎯 Focus on {top_bottleneck[0]} ({top_bottleneck[1]:.1f}% of time)")
        
        if "Similarity" in top_bottleneck[0]:
            print("   • Consider batch similarity updates")
            print("   • Implement lazy similarity learning")
            print("   • Use approximate similarity for non-critical updates")
        elif "Pattern" in top_bottleneck[0]:
            print("   • Reduce pattern discovery frequency")
            print("   • Use background pattern analysis")
            print("   • Cache pattern results more aggressively")
        elif "Prediction" in top_bottleneck[0]:
            print("   • This is core functionality - optimization complex")
            print("   • Consider prediction caching")
        elif "Activation" in top_bottleneck[0]:
            print("   • Batch activation updates")
            print("   • Reduce activation learning frequency")
    
    print()
    print("🚀 NEW KNOWLEDGE BUFFER FEASIBILITY:")
    print("-" * 38)
    
    core_prediction_time = np.mean(times['prediction'])
    learning_overhead_time = np.mean(times['total_store'])
    
    print(f"Core prediction: {core_prediction_time:.1f}ms (essential for real-time)")
    print(f"Learning overhead: {learning_overhead_time:.1f}ms (could be deferred)")
    print()
    
    if learning_overhead_time > core_prediction_time * 2:
        print("✅ STRONG CASE for new knowledge buffer:")
        print("   Learning takes much longer than prediction")
        print("   Could defer learning to background/sleep cycles")
        print(f"   Real-time performance: ~{core_prediction_time:.1f}ms")
        print(f"   Background learning: ~{learning_overhead_time:.1f}ms")
    else:
        print("📊 MODERATE CASE for new knowledge buffer:")
        print("   Learning overhead is manageable")
        print("   Consider other optimizations first")


def time_store_experience_components(brain, sensory_input, action_taken, outcome, predicted_action):
    """Time individual components of store_experience."""
    times = {'similarity': 0, 'pattern': 0, 'activation': 0}
    
    # Create experience quickly
    from src.experience.models import Experience
    prediction_error = brain._compute_prediction_error(predicted_action, outcome) if predicted_action else 0.5
    intrinsic_reward = brain.compute_intrinsic_reward(prediction_error)
    
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
    
    brain.recent_learning_outcomes.append(intrinsic_reward)
    if len(brain.recent_learning_outcomes) > 50:
        brain.recent_learning_outcomes = brain.recent_learning_outcomes[-25:]
    
    experience_id = brain.experience_storage.add_experience(experience)
    
    # Time: Similarity updates
    start = time.time()
    experience_vector = experience.get_context_vector()
    brain.similarity_engine.add_experience_to_index(experience_id, experience_vector)
    
    # Also time the similarity connection updates
    try:
        brain._update_similarity_connections_and_clustering(experience)
    except:
        pass  # Skip if it fails
    
    if predicted_action and len(brain.experience_storage._experiences) > 0:
        try:
            brain._record_similarity_learning_outcomes(sensory_input, predicted_action, outcome)
        except:
            pass
    
    times['similarity'] = (time.time() - start) * 1000
    
    # Time: Pattern analysis
    start = time.time()
    experience_data = {
        'experience_id': experience_id,
        'sensory_input': sensory_input,
        'action_taken': action_taken,
        'outcome': outcome,
        'prediction_error': prediction_error,
        'timestamp': experience.timestamp
    }
    brain.prediction_engine.add_experience_to_stream(experience_data)
    times['pattern'] = (time.time() - start) * 1000
    
    # Time: Activation learning
    start = time.time()
    if brain.use_utility_based_activation and predicted_action:
        try:
            prediction_success = brain._compute_prediction_success(predicted_action, outcome)
            working_memory = brain.activation_dynamics.get_working_memory_experiences()
            activated_experience_ids = [exp_id for exp_id, _ in working_memory]
            brain.activation_dynamics.record_prediction_outcome(activated_experience_ids, prediction_success)
        except:
            pass
    
    # Also time adaptation triggers
    try:
        brain.adaptive_trigger.record_prediction_outcome(prediction_error, brain.total_experiences)
        adaptation_triggers = brain.adaptive_trigger.check_adaptation_triggers(
            brain.total_experiences, 
            {"intrinsic_reward": intrinsic_reward, "prediction_error": prediction_error}
        )
        for trigger_type, trigger_reason, evidence in adaptation_triggers:
            brain._execute_triggered_adaptation(trigger_type, trigger_reason, evidence)
    except:
        pass
    
    times['activation'] = (time.time() - start) * 1000
    
    brain.total_experiences += 1
    
    return times


if __name__ == "__main__":
    manual_timing_analysis()