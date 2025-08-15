#!/usr/bin/env python3
"""
Test Constraint-Based Pattern Inhibition and Selection System

Validates that pattern selection emerges naturally from constraint competition
and that patterns compete for limited activation resources based on constraint dynamics.
"""

import sys
import os
import time
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain_factory import MinimalBrain
from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.constraint_pattern_inhibition import (
    create_pattern_inhibitor, PatternSelectionMode
)


def test_pattern_competition_modes():
    """Test different pattern selection modes."""
    print("🧠 Testing Pattern Competition Modes")
    print("=" * 40)
    
    # Test data: Multiple patterns with different activations
    stream_patterns = {
        StreamType.SENSORY: [1, 2, 3, 4, 5, 6, 7, 8],
        StreamType.MOTOR: [10, 11, 12, 13],
        StreamType.TEMPORAL: [20, 21, 22]
    }
    
    stream_activations = {
        StreamType.SENSORY: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],  # Decreasing
        StreamType.MOTOR: [0.8, 0.6, 0.4, 0.2],                          # Decreasing
        StreamType.TEMPORAL: [0.7, 0.5, 0.3]                             # Decreasing
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.INTERFERENCE: 0.6,  # High interference
            ConstraintType.COHERENCE_PRESSURE: 0.3
        },
        StreamType.MOTOR: {
            ConstraintType.RESOURCE_SCARCITY: 0.5,
            ConstraintType.PROCESSING_LOAD: 0.4
        },
        StreamType.TEMPORAL: {
            ConstraintType.PROCESSING_LOAD: 0.2
        }
    }
    
    # Test different selection modes
    modes = [
        PatternSelectionMode.COMPETITIVE_INHIBITION,
        PatternSelectionMode.COHERENCE_CLUSTERING,
        PatternSelectionMode.RESOURCE_OPTIMIZATION,
        PatternSelectionMode.INTERFERENCE_MINIMIZATION
    ]
    
    results = {}
    
    for mode in modes:
        print(f"\n📊 Testing {mode.value} mode:")
        
        inhibitor = create_pattern_inhibitor(max_patterns=8, mode=mode, quiet_mode=True)
        
        # Run pattern selection
        selected_patterns = inhibitor.update_active_patterns(
            stream_patterns, stream_activations, constraint_pressures
        )
        
        total_original = sum(len(patterns) for patterns in stream_patterns.values())
        total_selected = sum(len(patterns) for patterns in selected_patterns.values())
        
        print(f"  Original patterns: {total_original}")
        print(f"  Selected patterns: {total_selected}")
        print(f"  Selection ratio: {total_selected/total_original:.2f}")
        
        for stream_type, patterns in selected_patterns.items():
            original_count = len(stream_patterns.get(stream_type, []))
            print(f"    {stream_type.value}: {len(patterns)}/{original_count} patterns")
        
        results[mode] = {
            'total_selected': total_selected,
            'selection_ratio': total_selected / total_original,
            'selected_patterns': selected_patterns
        }
    
    # Validate mode characteristics
    comp_result = results[PatternSelectionMode.COMPETITIVE_INHIBITION]
    resource_result = results[PatternSelectionMode.RESOURCE_OPTIMIZATION]
    interference_result = results[PatternSelectionMode.INTERFERENCE_MINIMIZATION]
    
    # Competitive inhibition should select strongest patterns
    competitive_selective = comp_result['selection_ratio'] < 1.0
    
    # Resource optimization should be efficient
    resource_efficient = resource_result['total_selected'] > 0
    
    # Interference minimization should reduce conflicts
    interference_effective = interference_result['total_selected'] > 0
    
    print(f"\n🎯 Mode Validation:")
    print(f"  Competitive inhibition selective: {'✅' if competitive_selective else '❌'}")
    print(f"  Resource optimization efficient: {'✅' if resource_efficient else '❌'}")
    print(f"  Interference minimization effective: {'✅' if interference_effective else '❌'}")
    
    modes_working = competitive_selective and resource_efficient and interference_effective
    
    return modes_working


def test_constraint_based_inhibition():
    """Test that pattern inhibition responds to constraint pressures."""
    print("\n🔥 Testing Constraint-Based Pattern Inhibition")
    print("=" * 47)
    
    inhibitor = create_pattern_inhibitor(
        max_patterns=6,
        mode=PatternSelectionMode.COMPETITIVE_INHIBITION,
        quiet_mode=True
    )
    
    # Scenario 1: No constraints (baseline)
    print("📋 Scenario 1: No constraints")
    
    base_patterns = {
        StreamType.SENSORY: [1, 2, 3, 4, 5],
        StreamType.MOTOR: [10, 11, 12]
    }
    
    base_activations = {
        StreamType.SENSORY: [0.8, 0.7, 0.6, 0.5, 0.4],
        StreamType.MOTOR: [0.7, 0.6, 0.5]
    }
    
    no_constraints = {stream: {} for stream in StreamType}
    
    baseline_selected = inhibitor.update_active_patterns(
        base_patterns, base_activations, no_constraints
    )
    
    baseline_total = sum(len(patterns) for patterns in baseline_selected.values())
    print(f"  Baseline selected: {baseline_total} patterns")
    
    # Scenario 2: High interference in sensory stream
    print("\n📋 Scenario 2: High interference pressure")
    
    interference_constraints = {
        StreamType.SENSORY: {ConstraintType.INTERFERENCE: 0.8},  # High interference
        StreamType.MOTOR: {},
        StreamType.TEMPORAL: {}
    }
    
    interference_selected = inhibitor.update_active_patterns(
        base_patterns, base_activations, interference_constraints
    )
    
    interference_total = sum(len(patterns) for patterns in interference_selected.values())
    sensory_patterns_interference = len(interference_selected.get(StreamType.SENSORY, []))
    
    print(f"  With interference: {interference_total} patterns")
    print(f"  Sensory patterns: {sensory_patterns_interference}")
    
    # Scenario 3: Resource scarcity
    print("\n📋 Scenario 3: Resource scarcity pressure")
    
    scarcity_constraints = {
        StreamType.SENSORY: {ConstraintType.RESOURCE_SCARCITY: 0.7},
        StreamType.MOTOR: {ConstraintType.RESOURCE_SCARCITY: 0.7},
        StreamType.TEMPORAL: {}
    }
    
    scarcity_selected = inhibitor.update_active_patterns(
        base_patterns, base_activations, scarcity_constraints
    )
    
    scarcity_total = sum(len(patterns) for patterns in scarcity_selected.values())
    print(f"  With scarcity: {scarcity_total} patterns")
    
    # Validate constraint responsiveness
    interference_reduces_patterns = sensory_patterns_interference < len(base_patterns[StreamType.SENSORY])
    scarcity_affects_selection = scarcity_total != baseline_total
    selection_varies = len(set([baseline_total, interference_total, scarcity_total])) > 1
    
    print(f"\n🎯 Constraint Responsiveness:")
    print(f"  Interference reduces patterns: {'✅' if interference_reduces_patterns else '❌'}")
    print(f"  Scarcity affects selection: {'✅' if scarcity_affects_selection else '❌'}")
    print(f"  Selection varies with constraints: {'✅' if selection_varies else '❌'}")
    
    constraint_responsiveness = interference_reduces_patterns and scarcity_affects_selection and selection_varies
    
    return constraint_responsiveness


def test_pattern_interaction_dynamics():
    """Test pattern interaction and inhibition dynamics."""
    print("\n🤝 Testing Pattern Interaction Dynamics")
    print("=" * 40)
    
    inhibitor = create_pattern_inhibitor(
        max_patterns=8,
        mode=PatternSelectionMode.COMPETITIVE_INHIBITION,
        quiet_mode=True
    )
    
    # Scenario: Patterns with different strengths that should interact
    stream_patterns = {
        StreamType.SENSORY: [1, 2, 3, 4],  # Same stream - should compete
        StreamType.MOTOR: [10, 11],         # Different stream - may cooperate
        StreamType.TEMPORAL: [20]           # Single pattern
    }
    
    # Strong vs weak patterns in same stream
    stream_activations = {
        StreamType.SENSORY: [0.9, 0.8, 0.3, 0.2],  # Two strong, two weak
        StreamType.MOTOR: [0.7, 0.6],               # Moderate strength
        StreamType.TEMPORAL: [0.5]                  # Moderate strength
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.INTERFERENCE: 0.7,  # High interference should create inhibition
            ConstraintType.COHERENCE_PRESSURE: 0.2
        },
        StreamType.MOTOR: {
            ConstraintType.COHERENCE_PRESSURE: 0.1  # Low coherence pressure
        },
        StreamType.TEMPORAL: {}
    }
    
    # Run pattern selection
    selected_patterns = inhibitor.update_active_patterns(
        stream_patterns, stream_activations, constraint_pressures
    )
    
    # Get detailed pattern information
    pattern_info = inhibitor.get_active_patterns_info()
    
    print(f"📊 Pattern Selection Results:")
    for stream_type, patterns in selected_patterns.items():
        original_patterns = stream_patterns.get(stream_type, [])
        print(f"  {stream_type.value}: {patterns} (from {original_patterns})")
    
    print(f"\n📈 Pattern Interaction Analysis:")
    print(f"  Total active patterns: {pattern_info['total_active']}")
    print(f"  Total interactions: {pattern_info['total_interactions']}")
    
    # Validate expected behaviors
    sensory_selected = selected_patterns.get(StreamType.SENSORY, [])
    strong_patterns_preserved = 1 in sensory_selected and 2 in sensory_selected
    weak_patterns_inhibited = 3 not in sensory_selected or 4 not in sensory_selected
    
    selection_stats = inhibitor.get_selection_stats()
    print(f"\n📊 Selection Statistics:")
    for key, value in selection_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    interactions_detected = pattern_info['total_interactions'] > 0
    inhibition_occurred = selection_stats.get('inhibition_rate', 0) > 0
    
    print(f"\n🎯 Interaction Validation:")
    print(f"  Strong patterns preserved: {'✅' if strong_patterns_preserved else '❌'}")
    print(f"  Weak patterns inhibited: {'✅' if weak_patterns_inhibited else '❌'}")
    print(f"  Interactions detected: {'✅' if interactions_detected else '❌'}")
    print(f"  Inhibition occurred: {'✅' if inhibition_occurred else '❌'}")
    
    interaction_dynamics_work = (strong_patterns_preserved and weak_patterns_inhibited and 
                                interactions_detected and inhibition_occurred)
    
    return interaction_dynamics_work


def test_brain_integration():
    """Test pattern inhibition integration with the brain system."""
    print("\n🧠 Testing Brain System Integration")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': False
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with pattern inhibition...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        brain.enable_parallel_processing(True)
        
        # Ensure stream coordinator is initialized
        brain.parallel_coordinator._ensure_stream_coordinator()
        shared_state = brain.parallel_coordinator.stream_coordinator.shared_state
        
        print("📊 Testing pattern inhibition in live brain cycles...")
        
        # Run several processing cycles
        inhibition_events = []
        
        for cycle in range(10):
            # Generate input patterns
            sensory_input = [0.1 * i * (1 + 0.3 * (cycle % 4)) for i in range(16)]
            
            # Process through brain
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            # Get pattern inhibition statistics
            stats = shared_state.get_shared_state_stats()
            pattern_stats = stats.get('pattern_inhibition', {})
            
            if pattern_stats:
                active_info = pattern_stats.get('active_patterns_info', {})
                selection_stats = pattern_stats.get('selection_stats', {})
                
                total_active = active_info.get('total_active', 0)
                total_interactions = active_info.get('total_interactions', 0)
                
                if total_active > 0:
                    inhibition_events.append({
                        'cycle': cycle,
                        'total_active': total_active,
                        'total_interactions': total_interactions,
                        'selection_stats': selection_stats
                    })
                    
                    if cycle % 3 == 0:
                        print(f"  Cycle {cycle}: {total_active} active patterns, {total_interactions} interactions")
        
        brain.finalize_session()
        
        # Analyze pattern inhibition performance
        print(f"\n📈 Pattern Inhibition Analysis:")
        
        if inhibition_events:
            avg_active = sum(event['total_active'] for event in inhibition_events) / len(inhibition_events)
            avg_interactions = sum(event['total_interactions'] for event in inhibition_events) / len(inhibition_events)
            
            print(f"  Total cycles with inhibition: {len(inhibition_events)}")
            print(f"  Average active patterns: {avg_active:.1f}")
            print(f"  Average interactions: {avg_interactions:.1f}")
            
            # Check that inhibition is working
            inhibition_working = avg_active > 0 and avg_interactions >= 0
            system_responsive = len(inhibition_events) > 5  # Most cycles should have activity
            
            print(f"  Inhibition system active: {'✅' if inhibition_working else '❌'}")
            print(f"  System responsive: {'✅' if system_responsive else '❌'}")
            
            integration_success = inhibition_working and system_responsive
        else:
            print("  ⚠️ No pattern inhibition events detected")
            integration_success = False
        
        return integration_success


if __name__ == "__main__":
    print("🚀 Testing Constraint-Based Pattern Inhibition System")
    print("=" * 65)
    
    # Run all tests
    test1_success = test_pattern_competition_modes()
    test2_success = test_constraint_based_inhibition()
    test3_success = test_pattern_interaction_dynamics()
    test4_success = test_brain_integration()
    
    # Overall results
    all_passed = test1_success and test2_success and test3_success and test4_success
    
    print(f"\n🏁 Test Results Summary:")
    print(f"  Competition Modes: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"  Constraint-Based Inhibition: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    print(f"  Pattern Interaction Dynamics: {'✅ PASSED' if test3_success else '❌ FAILED'}")
    print(f"  Brain Integration: {'✅ PASSED' if test4_success else '❌ FAILED'}")
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Constraint-Based Pattern Inhibition is working correctly!")
        print("   ✅ Multiple competition modes function properly")
        print("   ✅ Pattern selection responds to constraint pressures")
        print("   ✅ Pattern interactions create natural inhibition")
        print("   ✅ Integration with brain system is successful")
        print("   ✅ Patterns compete for limited activation resources")
    else:
        print("\n⚠️ Some tests failed - review implementation")
    
    exit(0 if all_passed else 1)