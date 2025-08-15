#!/usr/bin/env python3
"""
Budget Selection Tracer - See what budgets are actually being selected
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
import time

def trace_budget_selection():
    """Trace what budgets are actually being selected."""
    print("🔍 TRACING BUDGET SELECTION")
    print("=" * 40)
    
    # Create brain
    brain = MinimalBrain(quiet_mode=True)
    
    # Track budget usage
    budget_usage = {'reflex': 0, 'habit': 0, 'deliberate': 0}
    
    # Make some predictions and track budgets
    for i in range(10):
        sensory_input = [1.0 + i*0.1, 2.0 + i*0.1, 3.0, 4.0]
        
        start_time = time.time()
        action, brain_state = brain.process_sensory_input(sensory_input, action_dimensions=2)
        elapsed = (time.time() - start_time) * 1000
        
        # Extract budget info from brain state
        temporal_hierarchy = brain_state.get('temporal_hierarchy', {})
        emergent_behaviors = temporal_hierarchy.get('emergent_behaviors', {})
        
        print(f"Prediction {i+1}: {elapsed:.1f}ms")
        print(f"  Brain state keys: {list(brain_state.keys())}")
        print(f"  Temporal hierarchy: {temporal_hierarchy}")
        print(f"  Emergent behaviors: {emergent_behaviors}")
        print()
        
        # Try to extract budget information
        # Look for budget-related info in brain state
        for key, value in brain_state.items():
            if 'budget' in key.lower() or 'reflex' in key.lower() or 'habit' in key.lower() or 'deliberate' in key.lower():
                print(f"  Budget info - {key}: {value}")
    
    print("📊 BUDGET USAGE SUMMARY")
    print("-" * 25)
    for budget, count in budget_usage.items():
        print(f"{budget}: {count} times")
    
    # Check if we can access the temporal hierarchy directly
    print("\n🔍 DIRECT TEMPORAL HIERARCHY ACCESS")
    print("-" * 40)
    try:
        vector_brain = brain.vector_brain
        temporal_hierarchy = vector_brain.emergent_hierarchy
        
        print(f"Temporal hierarchy available: {temporal_hierarchy is not None}")
        if temporal_hierarchy:
            print(f"Context pressure: {temporal_hierarchy.context_pressure}")
            print(f"Recent accuracies: {len(temporal_hierarchy.recent_accuracies)}")
            
            # Test budget selection directly
            from src.vector_stream.sparse_representations import SparsePatternEncoder
            encoder = SparsePatternEncoder(16, sparsity=0.02, quiet_mode=True)
            test_pattern = encoder.encode_top_k(
                [1.0, 2.0, 3.0, 4.0] + [0.0] * 12, 
                "test_pattern"
            )
            
            selected_budget = temporal_hierarchy._select_budget_adaptively(test_pattern, time.time())
            print(f"Selected budget for test pattern: {selected_budget}")
            
    except Exception as e:
        print(f"Error accessing temporal hierarchy: {e}")

if __name__ == "__main__":
    trace_budget_selection()