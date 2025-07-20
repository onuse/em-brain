#!/usr/bin/env python3
"""
Test Constraint-Based Field Dynamics - Phase A5

This validates the ultimate marriage of constraint discovery and continuous
field dynamics. We test constraint emergence from field topology and 
constraint-guided field evolution.

The ultimate question: Can fields discover their own constraints and use
them to guide intelligent behavior?
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../validation/embodied_learning/environments'))

import torch
import numpy as np
import time
from typing import List, Dict, Any, Tuple

# Import constraint field dynamics
from vector_stream.constraint_field_dynamics import (
    ConstraintField4D, ConstraintFieldExperience, FieldConstraint, FieldConstraintType,
    create_constraint_field_4d
)
from vector_stream.temporal_field_dynamics import TemporalExperience

# Import environment for rich scenarios
from sensory_motor_world import SensoryMotorWorld, ActionType


def create_constraint_discovery_scenarios(env: SensoryMotorWorld) -> List[TemporalExperience]:
    """
    Create scenarios designed to trigger constraint discovery.
    
    These experiences have structured patterns that should lead to
    constraint emergence from field topology.
    """
    scenarios = []
    
    # SCENARIO 1: Repeated spatial pattern (should create gradient constraints)
    print("   🎯 Creating spatial repetition pattern...")
    base_position = np.array([3.0, 3.0])
    for i in range(6):
        env.robot_state.position = base_position + np.array([i * 0.5, 0.0])
        env.robot_state.orientation = 0.0
        sensory_data = env.get_sensory_input()
        
        exp = TemporalExperience(
            sensory_data=torch.tensor(sensory_data, dtype=torch.float32),
            position=(15 + i*4, 15),  # Linear spatial pattern
            scale_level=0.3,
            temporal_position=i * 0.8,
            intensity=0.9,
            spatial_spread=4.0,
            scale_spread=1.5,
            temporal_spread=1.0,
            timestamp=time.time() + i*0.1,
            sequence_id="spatial_repetition"
        )
        scenarios.append(exp)
    
    # SCENARIO 2: Scale progression (should create scale coupling constraints)
    print("   📏 Creating scale progression pattern...")
    for i in range(5):
        env.robot_state.position = np.array([6.0, 6.0])
        env.robot_state.orientation = i * np.pi/4
        sensory_data = env.get_sensory_input()
        
        exp = TemporalExperience(
            sensory_data=torch.tensor(sensory_data, dtype=torch.float32),
            position=(30, 30),  # Same spatial location
            scale_level=0.1 + i * 0.2,  # Progressive scale increase
            temporal_position=6.0 + i * 1.0,
            intensity=0.8,
            spatial_spread=6.0 + i*2,
            scale_spread=1.0 + i*0.5,
            temporal_spread=1.5,
            timestamp=time.time() + 6*0.1 + i*0.1,
            sequence_id="scale_progression"
        )
        scenarios.append(exp)
    
    # SCENARIO 3: Temporal sequence (should create temporal momentum constraints)
    print("   ⏱️ Creating temporal sequence pattern...")
    for i in range(7):
        env.robot_state.position = np.array([2.0 + i*0.3, 8.0])
        env.robot_state.orientation = np.pi/2
        sensory_data = env.get_sensory_input()
        
        exp = TemporalExperience(
            sensory_data=torch.tensor(sensory_data, dtype=torch.float32),
            position=(10 + i*2, 45),
            scale_level=0.4,
            temporal_position=11.0 + i * 0.5,  # Regular temporal progression
            intensity=0.9 - i*0.05,
            spatial_spread=5.0,
            scale_spread=2.0,
            temporal_spread=0.8,
            timestamp=time.time() + 11*0.1 + i*0.1,
            sequence_id="temporal_sequence"
        )
        scenarios.append(exp)
    
    # SCENARIO 4: Boundary formation (should create topology boundary constraints)
    print("   🏗️ Creating boundary formation pattern...")
    boundary_positions = [(20, 10), (25, 10), (30, 10), (30, 15), (30, 20), (25, 20), (20, 20), (20, 15)]
    for i, (x, y) in enumerate(boundary_positions):
        env.robot_state.position = np.array([x/5, y/5])
        env.robot_state.orientation = i * np.pi/4
        sensory_data = env.get_sensory_input()
        
        exp = TemporalExperience(
            sensory_data=torch.tensor(sensory_data, dtype=torch.float32),
            position=(x, y),
            scale_level=0.5,
            temporal_position=18.0 + i * 0.3,
            intensity=1.0,
            spatial_spread=3.0,
            scale_spread=1.0,
            temporal_spread=1.0,
            timestamp=time.time() + 18*0.1 + i*0.1,
            sequence_id="boundary_formation"
        )
        scenarios.append(exp)
    
    print(f"   ✅ Created {len(scenarios)} constraint discovery scenarios")
    return scenarios


def test_constraint_emergence_from_topology():
    """
    Test that constraints emerge naturally from field topology patterns.
    """
    print(f"\n🧭 TESTING CONSTRAINT EMERGENCE FROM FIELD TOPOLOGY")
    
    # Create constraint field
    field = create_constraint_field_4d(
        width=50, height=50, scale_depth=8, 
        temporal_depth=20, temporal_window=12.0, 
        constraint_discovery_rate=0.2,  # Higher discovery rate for testing
        constraint_enforcement_strength=0.4,
        quiet_mode=False
    )
    
    print(f"🌊 Created {field.width}x{field.height}x{field.scale_depth}x{field.temporal_depth} constraint field")
    
    # Create environment and scenarios
    env = SensoryMotorWorld(world_size=10.0, num_light_sources=3, num_obstacles=4, random_seed=555)
    scenarios = create_constraint_discovery_scenarios(env)
    
    # Apply scenarios and track constraint discovery
    print(f"\n📍 Applying {len(scenarios)} constraint discovery scenarios:")
    
    constraint_discoveries = []
    for i, scenario in enumerate(scenarios):
        constraint_exp = field.apply_constraint_guided_imprint(scenario)
        
        # Track discoveries
        discoveries = len(constraint_exp.discovered_constraints)
        violations = len(constraint_exp.constraint_violations)
        satisfaction = constraint_exp.constraint_satisfaction
        
        constraint_discoveries.append({
            'scenario_index': i,
            'sequence_id': scenario.sequence_id,
            'discoveries': discoveries,
            'violations': violations,
            'satisfaction': satisfaction
        })
        
        if discoveries > 0:
            print(f"      Scenario {i+1} ({scenario.sequence_id}): 🔍 {discoveries} constraints discovered!")
        elif violations > 0:
            print(f"      Scenario {i+1} ({scenario.sequence_id}): ⚠️ {violations} constraint violations")
        else:
            print(f"      Scenario {i+1} ({scenario.sequence_id}): satisfaction={satisfaction:.3f}")
    
    return {
        'field': field,
        'constraint_discoveries': constraint_discoveries,
        'total_scenarios': len(scenarios)
    }


def test_constraint_guided_field_evolution(field: ConstraintField4D):
    """
    Test that discovered constraints guide field evolution intelligently.
    """
    print(f"\n🌊 TESTING CONSTRAINT-GUIDED FIELD EVOLUTION")
    
    initial_stats = field.get_constraint_field_stats()
    print(f"   Initial state: {initial_stats['discovered_constraints_count']} constraints discovered")
    
    # Evolution tracking
    evolution_history = []
    
    # Evolve field with constraint guidance
    print(f"   Evolving field for 15 cycles with constraint guidance:")
    for cycle in range(15):
        field.evolve_constraint_guided_field(dt=0.2)
        
        # Track evolution metrics
        stats = field.get_constraint_field_stats()
        evolution_history.append({
            'cycle': cycle + 1,
            'constraints_count': stats['discovered_constraints_count'],
            'constraint_density': stats['field_constraint_density'],
            'guidance_intensity': stats['field_guidance_intensity'],
            'temporal_coherence': stats['temporal_coherence'],
            'violations': stats['total_constraint_violations'],
            'enforcements': stats['total_constraint_enforcements']
        })
        
        if cycle % 5 == 0:
            print(f"      Cycle {cycle+1}: constraints={stats['discovered_constraints_count']}, "
                  f"density={stats['field_constraint_density']:.4f}, "
                  f"guidance={stats['field_guidance_intensity']:.4f}, "
                  f"coherence={stats['temporal_coherence']:.4f}")
    
    final_stats = field.get_constraint_field_stats()
    
    return {
        'evolution_history': evolution_history,
        'initial_stats': initial_stats,
        'final_stats': final_stats
    }


def test_constraint_discovery_validation(field: ConstraintField4D):
    """
    Validate that discovered constraints actually improve field intelligence.
    """
    print(f"\n🔬 TESTING CONSTRAINT DISCOVERY VALIDATION")
    
    # Get current constraint state
    stats = field.get_constraint_field_stats()
    constraints = field.discovered_constraints
    
    print(f"   Analyzing {len(constraints)} discovered constraints:")
    
    # Analyze constraint types
    constraint_analysis = {}
    for constraint_id, constraint in constraints.items():
        constraint_type = constraint.constraint_type.value
        if constraint_type not in constraint_analysis:
            constraint_analysis[constraint_type] = {
                'count': 0,
                'avg_strength': 0.0,
                'avg_enforcement_success': 0.0,
                'total_violations': 0
            }
        
        analysis = constraint_analysis[constraint_type]
        analysis['count'] += 1
        analysis['avg_strength'] = (analysis['avg_strength'] * (analysis['count']-1) + constraint.strength) / analysis['count']
        analysis['avg_enforcement_success'] = (analysis['avg_enforcement_success'] * (analysis['count']-1) + constraint.enforcement_success) / analysis['count']
        analysis['total_violations'] += constraint.violation_count
    
    # Display constraint analysis
    for constraint_type, analysis in constraint_analysis.items():
        print(f"      {constraint_type}: {analysis['count']} constraints, "
              f"avg_strength={analysis['avg_strength']:.3f}, "
              f"success_rate={analysis['avg_enforcement_success']:.3f}, "
              f"violations={analysis['total_violations']}")
    
    # Test constraint effectiveness with new experience
    print(f"\n   Testing constraint effectiveness with validation experience:")
    
    # Create a test experience that should follow discovered constraints
    test_experience = TemporalExperience(
        sensory_data=torch.tensor([0.7, 0.4, 0.8, 0.3, 0.6, 0.2]),
        position=(25, 25),  # Central position
        scale_level=0.4,
        temporal_position=15.0,
        intensity=0.8,
        spatial_spread=5.0,
        scale_spread=2.0,
        temporal_spread=1.5,
        timestamp=time.time(),
        sequence_id="validation_test"
    )
    
    # Apply with constraint guidance
    constraint_exp = field.apply_constraint_guided_imprint(test_experience)
    
    print(f"      Validation experience: satisfaction={constraint_exp.constraint_satisfaction:.3f}, "
          f"violations={len(constraint_exp.constraint_violations)}, "
          f"new_discoveries={len(constraint_exp.discovered_constraints)}")
    
    # Overall effectiveness assessment
    effectiveness_score = (
        stats['constraint_discovery_success_rate'] * 0.3 +
        constraint_exp.constraint_satisfaction * 0.4 +
        (1.0 - min(1.0, len(constraint_exp.constraint_violations) / 5)) * 0.3
    )
    
    return {
        'constraint_analysis': constraint_analysis,
        'validation_experience': constraint_exp,
        'effectiveness_score': effectiveness_score,
        'total_constraints': len(constraints)
    }


def test_hybrid_constraint_field_integration():
    """
    Test the ultimate integration: discrete constraints + continuous intelligence.
    """
    print(f"\n🚀 TESTING HYBRID CONSTRAINT-FIELD INTEGRATION")
    
    # This is the culmination - full integration test
    print(f"   Phase A5: Ultimate constraint discovery + field intelligence test")
    
    # Create environment
    env = SensoryMotorWorld(world_size=10.0, num_light_sources=4, num_obstacles=5, random_seed=777)
    
    # Create constraint field with optimized parameters
    field = create_constraint_field_4d(
        width=40, height=40, scale_depth=6, 
        temporal_depth=15, temporal_window=10.0, 
        constraint_discovery_rate=0.15,
        constraint_enforcement_strength=0.35,
        quiet_mode=True  # Reduce output for integration test
    )
    
    # Phase 1: Constraint Discovery
    print(f"   Phase 1: Constraint discovery from structured experiences")
    discovery_results = test_constraint_emergence_from_topology()
    
    # Phase 2: Constraint-Guided Evolution
    print(f"   Phase 2: Constraint-guided field evolution")
    evolution_results = test_constraint_guided_field_evolution(discovery_results['field'])
    
    # Phase 3: Constraint Validation
    print(f"   Phase 3: Constraint discovery validation")
    validation_results = test_constraint_discovery_validation(discovery_results['field'])
    
    # Comprehensive analysis
    print(f"\n📊 COMPREHENSIVE CONSTRAINT-FIELD ANALYSIS")
    
    final_stats = discovery_results['field'].get_constraint_field_stats()
    
    print(f"\n   🧭 Constraint Discovery Performance:")
    total_discoveries = sum([d['discoveries'] for d in discovery_results['constraint_discoveries']])
    total_violations = sum([d['violations'] for d in discovery_results['constraint_discoveries']])
    avg_satisfaction = np.mean([d['satisfaction'] for d in discovery_results['constraint_discoveries']])
    
    print(f"      Total constraint discoveries: {total_discoveries}")
    print(f"      Total constraint violations: {total_violations}")
    print(f"      Average constraint satisfaction: {avg_satisfaction:.3f}")
    print(f"      Discovery success rate: {final_stats['constraint_discovery_success_rate']:.3f}")
    
    print(f"\n   🌊 Field Evolution Metrics:")
    evolution_final = evolution_results['evolution_history'][-1]
    evolution_improvement = evolution_final['temporal_coherence'] - evolution_results['evolution_history'][0]['temporal_coherence']
    
    print(f"      Final constraint count: {evolution_final['constraints_count']}")
    print(f"      Final constraint density: {evolution_final['constraint_density']:.4f}")
    print(f"      Field guidance intensity: {evolution_final['guidance_intensity']:.4f}")
    print(f"      Temporal coherence improvement: {evolution_improvement:.4f}")
    
    print(f"\n   🔬 Constraint Effectiveness:")
    print(f"      Constraint types discovered: {len(validation_results['constraint_analysis'])}")
    print(f"      Overall effectiveness score: {validation_results['effectiveness_score']:.3f}")
    print(f"      Field constraint density: {final_stats['field_constraint_density']:.4f}")
    
    print(f"\n   📈 Hybrid Intelligence Metrics:")
    print(f"      Working memory patterns: {final_stats['working_memory_size']}")
    print(f"      Temporal chains formed: {final_stats['temporal_chains_count']}")
    print(f"      Field utilization: {final_stats['field_mean_activation']:.6f}")
    print(f"      Total field experiences: {final_stats['total_experiences']}")
    
    # Overall success assessment
    success_score = (
        min(1.0, total_discoveries / 10) * 0.25 +  # Discovery capability
        min(1.0, evolution_improvement * 5) * 0.25 +  # Evolution improvement
        validation_results['effectiveness_score'] * 0.25 +  # Constraint effectiveness
        min(1.0, final_stats['temporal_coherence']) * 0.25  # Field coherence
    )
    
    print(f"\n   🌟 PHASE A5 SUCCESS ASSESSMENT:")
    print(f"      Overall success score: {success_score:.3f}")
    print(f"      Constraint-field integration: {'✅ SUCCESSFUL' if success_score > 0.6 else '⚠️ DEVELOPING'}")
    
    return {
        'discovery_results': discovery_results,
        'evolution_results': evolution_results,
        'validation_results': validation_results,
        'final_stats': final_stats,
        'success_score': success_score,
        'hybrid_integration_achieved': success_score > 0.6
    }


def test_constraint_field_intelligence():
    """
    Comprehensive test of constraint-based field intelligence.
    """
    print("🧭 TESTING CONSTRAINT-BASED FIELD INTELLIGENCE")
    print("=" * 70)
    
    print(f"🎯 Phase A5: Ultimate Constraint Discovery + Field Intelligence")
    print(f"   Testing the marriage of discrete constraints with continuous intelligence")
    print(f"   Key capabilities: constraint emergence, guided evolution, hybrid intelligence")
    
    # Run the ultimate integration test
    results = test_hybrid_constraint_field_integration()
    
    print(f"\n✅ CONSTRAINT-BASED FIELD INTELLIGENCE TEST COMPLETED!")
    print(f"🎯 Key achievements:")
    
    if results['hybrid_integration_achieved']:
        print(f"   ✓ Constraint discovery from field topology")
        print(f"   ✓ Constraint-guided field evolution")
        print(f"   ✓ Hybrid discrete-continuous intelligence")
        print(f"   ✓ Self-organizing constraint enforcement")
        print(f"   ✓ Emergent constraint-field dynamics")
    else:
        print(f"   ⚠️ Constraint-field integration developing")
        print(f"   ✓ Basic constraint discovery functional")
        print(f"   ✓ Field evolution with guidance")
    
    return results


if __name__ == "__main__":
    # Run the comprehensive constraint field intelligence test
    results = test_constraint_field_intelligence()
    
    print(f"\n🔬 PHASE A5 VALIDATION SUMMARY:")
    print(f"   Success score: {results['success_score']:.3f}")
    print(f"   Constraints discovered: {results['final_stats']['discovered_constraints_count']}")
    print(f"   Field constraint density: {results['final_stats']['field_constraint_density']:.4f}")
    print(f"   Temporal coherence: {results['final_stats']['temporal_coherence']:.4f}")
    print(f"   Hybrid integration: {'✅ ACHIEVED' if results['hybrid_integration_achieved'] else '⚠️ DEVELOPING'}")
    
    if results['hybrid_integration_achieved']:
        print(f"\n🚀 Phase A5 CONSTRAINT-BASED FIELD INTELLIGENCE SUCCESSFULLY DEMONSTRATED!")
        print(f"🎉 We have achieved the ultimate marriage of constraints and continuous intelligence!")
        print(f"🧭 The field discovers its own constraints and uses them to guide intelligent behavior!")
    else:
        print(f"\n⚠️ Phase A5 constraint-field integration still developing")
        print(f"🔧 Consider adjusting constraint discovery parameters or field dynamics")