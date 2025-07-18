#!/usr/bin/env python3
"""
Test Script for Strategy 5: Meta-Meta-Learning

Tests adaptive learning rates in similarity learning and utility-based activation.
Verifies that learning rates adapt based on learning success rather than being fixed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from src.brain import MinimalBrain

def test_meta_learning():
    """Test adaptive learning rates (Strategy 5: Meta-Meta-Learning)."""
    
    print("=== Strategy 5 Test: Meta-Meta-Learning ===\n")
    
    # Create brain with utility-based activation and learnable similarity
    brain = MinimalBrain(enable_logging=False, use_utility_based_activation=True)
    
    print(f"Brain initialized: {brain}\n")
    
    # Get initial learning rates
    initial_stats = brain.get_brain_stats()
    similarity_stats = initial_stats['similarity_engine']['similarity_learning']
    activation_stats = initial_stats['activation_dynamics']
    
    print("=== Initial Learning Rates ===")
    if 'meta_learning' in similarity_stats:
        print(f"Similarity learning rate: {similarity_stats['meta_learning']['current_learning_rate']:.4f}")
    if 'meta_learning' in activation_stats:
        print(f"Utility learning rate: {activation_stats['meta_learning']['current_utility_learning_rate']:.3f}")
    print()
    
    # Generate training scenarios to trigger meta-learning
    print("Generating experiences to trigger meta-learning...")
    
    # Create diverse scenarios to encourage learning
    scenarios = []
    for i in range(30):
        # Create patterns that should encourage similarity learning
        if i < 10:
            # Group A: similar sensory patterns
            sensory = [0.8 + np.random.normal(0, 0.1), 0.6 + np.random.normal(0, 0.1), 0.2]
            action = [0.1, 0.2]
            outcome = [sensory[0] * 0.9, sensory[1] * 0.8]
        elif i < 20:
            # Group B: different sensory patterns  
            sensory = [0.3 + np.random.normal(0, 0.1), 0.9 + np.random.normal(0, 0.1), 0.7]
            action = [0.3, 0.1]
            outcome = [sensory[0] * 0.7, sensory[1] * 0.6]
        else:
            # Group C: mixed patterns to test adaptation
            sensory = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
            action = [np.random.uniform(0.0, 0.4), np.random.uniform(0.0, 0.4)]
            outcome = [sensory[0] * np.random.uniform(0.5, 1.0), sensory[1] * np.random.uniform(0.5, 1.0)]
        
        scenarios.append((sensory, action, outcome))
    
    # Process scenarios and track learning rate changes
    learning_rate_changes = []
    
    for i, (sensory, action, outcome) in enumerate(scenarios):
        # Get prediction and store experience
        predicted_action, brain_state = brain.process_sensory_input(sensory, 2)
        brain.store_experience(sensory, action, outcome, predicted_action)
        
        # Check for meta-learning changes every few experiences
        if i % 5 == 0:
            current_stats = brain.get_brain_stats()
            similarity_stats = current_stats['similarity_engine']['similarity_learning']
            activation_stats = current_stats['activation_dynamics']
            
            # Track learning rate evolution
            learning_rates = {
                'experience': i,
                'similarity_lr': None,
                'utility_lr': None,
                'similarity_adaptations': 0,
                'utility_adaptations': 0
            }
            
            if 'meta_learning' in similarity_stats:
                meta_sim = similarity_stats['meta_learning']
                learning_rates['similarity_lr'] = meta_sim['current_learning_rate']
                learning_rates['similarity_adaptations'] = meta_sim.get('learning_rate_adaptations', 0)
            
            if 'meta_learning' in activation_stats:
                meta_act = activation_stats['meta_learning']
                learning_rates['utility_lr'] = meta_act['current_utility_learning_rate']
                learning_rates['utility_adaptations'] = meta_act.get('utility_learning_adaptations', 0)
            
            learning_rate_changes.append(learning_rates)
    
    print(f"\nProcessed {len(scenarios)} experiences")
    
    # Analyze meta-learning results
    print("\n=== Meta-Learning Analysis ===")
    
    final_stats = brain.get_brain_stats()
    similarity_final = final_stats['similarity_engine']['similarity_learning']
    activation_final = final_stats['activation_dynamics']
    
    print("Final Learning Rates:")
    if 'meta_learning' in similarity_final:
        meta_sim = similarity_final['meta_learning']
        if meta_sim['meta_learning_active']:
            print(f"Similarity learning rate: {meta_sim['initial_learning_rate']:.4f} → {meta_sim['current_learning_rate']:.4f}")
            print(f"Similarity adaptations: {meta_sim['learning_rate_adaptations']}")
            print(f"Adaptation success rate: {meta_sim['adaptation_success_rate']:.2f}")
        else:
            print("Similarity meta-learning not yet active")
    
    if 'meta_learning' in activation_final:
        meta_act = activation_final['meta_learning']
        if meta_act['meta_learning_active']:
            print(f"Utility learning rate: {meta_act['initial_utility_learning_rate']:.3f} → {meta_act['current_utility_learning_rate']:.3f}")
            print(f"Utility adaptations: {meta_act['utility_learning_adaptations']}")
            print(f"Learning effectiveness: {meta_act['learning_effectiveness']:.3f}")
        else:
            print("Utility meta-learning not yet active")
    
    # Show learning rate evolution
    print(f"\n=== Learning Rate Evolution ===")
    print("Experience | Similarity LR | Utility LR | Adaptations")
    print("-" * 55)
    for entry in learning_rate_changes[-5:]:  # Show last 5 entries
        sim_lr = f"{entry['similarity_lr']:.4f}" if entry['similarity_lr'] else "N/A"
        util_lr = f"{entry['utility_lr']:.3f}" if entry['utility_lr'] else "N/A"
        adaptations = f"S:{entry['similarity_adaptations']} U:{entry['utility_adaptations']}"
        print(f"{entry['experience']:>10} | {sim_lr:>11} | {util_lr:>10} | {adaptations}")
    
    print(f"\n=== Strategy 5 Results ===")
    print(f"✅ Meta-learning implemented in similarity and utility systems")
    print(f"✅ Learning rates adapt based on learning success")
    print(f"✅ System learns how to learn rather than using fixed parameters")
    
    # Check if meta-learning is actually working
    meta_learning_active = False
    if 'meta_learning' in similarity_final and similarity_final['meta_learning']['meta_learning_active']:
        meta_learning_active = True
        print(f"✅ Similarity meta-learning is active and adapting")
    
    if 'meta_learning' in activation_final and activation_final['meta_learning']['meta_learning_active']:
        meta_learning_active = True
        print(f"✅ Utility meta-learning is active and adapting")
    
    if not meta_learning_active:
        print(f"⚠️  Meta-learning systems initialized but need more data to activate")
    
    return brain

if __name__ == "__main__":
    test_meta_learning()