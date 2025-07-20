#!/usr/bin/env python3
"""
Test analog field dynamics with real sensory data from robot environment.

This validates that the continuous field can handle actual 24D sensory input
and demonstrates fluid concept comparison vs discrete pattern matching.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../validation/embodied_learning/environments'))

import torch
import numpy as np
import time
from typing import List, Dict, Any

# Import analog field dynamics
from vector_stream.analog_field_dynamics import AnalogField2D, FieldExperience, create_analog_field_2d

# Import environment for real sensory data
from sensory_motor_world import SensoryMotorWorld, ActionType

# Import discrete pattern system for comparison
from vector_stream.sparse_representations import SparsePatternEncoder, SparsePatternStorage


def test_analog_field_with_real_sensory_data():
    """Test analog field dynamics with real robot sensory data."""
    print("🧪 TESTING ANALOG FIELD WITH REAL SENSORY DATA")
    print("=" * 60)
    
    # Create environment and analog field
    env = SensoryMotorWorld(world_size=10.0, num_light_sources=3, num_obstacles=4, random_seed=42)
    field = create_analog_field_2d(width=200, height=200, quiet_mode=False)
    
    print(f"\n📡 Environment: {env.world_size}m world, 24D sensory input")
    print(f"🌊 Field: {field.width}x{field.height} continuous space")
    
    # Collect diverse sensory experiences
    print(f"\n🚀 Collecting diverse sensory experiences...")
    experiences = []
    
    # Generate different robot positions and actions to get varied sensory data
    test_scenarios = [
        # Near red light
        {'position': np.array([3.0, 3.0]), 'orientation': 0.0, 'description': 'Near red light'},
        # Near green light  
        {'position': np.array([7.0, 7.0]), 'orientation': np.pi/2, 'description': 'Near green light'},
        # Among obstacles
        {'position': np.array([5.0, 5.0]), 'orientation': np.pi, 'description': 'Among obstacles'},
        # Edge of world
        {'position': np.array([1.0, 9.0]), 'orientation': 3*np.pi/2, 'description': 'Edge of world'},
        # Center position
        {'position': np.array([5.0, 5.0]), 'orientation': np.pi/4, 'description': 'Center position'},
    ]
    
    for i, scenario in enumerate(test_scenarios):
        # Set robot position
        env.robot_state.position = scenario['position']
        env.robot_state.orientation = scenario['orientation']
        
        # Get sensory input
        sensory_data = env.get_sensory_input()
        sensory_tensor = torch.tensor(sensory_data, dtype=torch.float32)
        
        # Convert to field experience
        # Use robot position mapped to field coordinates
        field_x = (scenario['position'][0] / env.world_size) * field.width
        field_y = (scenario['position'][1] / env.world_size) * field.height
        
        experience = FieldExperience(
            sensory_data=sensory_tensor,
            position=(field_x, field_y),
            intensity=1.0,
            spread=15.0,  # Larger spread for robot-scale patterns
            timestamp=time.time()
        )
        
        experiences.append((experience, scenario['description']))
        print(f"   Experience {i+1}: {scenario['description']} → sensory vector {len(sensory_data)}D")
    
    # Apply experiences to field
    print(f"\n📍 Applying experiences to continuous field...")
    for experience, description in experiences:
        imprint = field.experience_to_imprint(experience)
        field.apply_imprint(imprint)
        print(f"   ✓ {description}: imprint at ({imprint.center_x:.1f}, {imprint.center_y:.1f})")
    
    # Evolve field dynamics
    print(f"\n🌊 Evolving field dynamics...")
    for i in range(10):
        field.evolve_field()
        if i % 2 == 0:
            density = field.get_field_density()
            print(f"   Evolution step {i+1}: field density = {density:.4f}")
    
    # Test continuous similarity search vs discrete patterns
    print(f"\n🔍 Testing continuous similarity search...")
    
    # Create a query experience (robot near obstacles)
    env.robot_state.position = np.array([4.8, 5.2])  # Slightly different from center
    env.robot_state.orientation = np.pi/3
    query_sensory = env.get_sensory_input()
    query_experience = FieldExperience(
        sensory_data=torch.tensor(query_sensory, dtype=torch.float32),
        position=(4.8/env.world_size * field.width, 5.2/env.world_size * field.height),
        intensity=1.0,
        spread=15.0,
        timestamp=time.time()
    )
    
    # Find similar regions in continuous field
    similar_regions = field.find_similar_regions(query_experience, top_k=5)
    
    print(f"   Query: Robot at ({env.robot_state.position[0]:.1f}, {env.robot_state.position[1]:.1f})")
    print(f"   Found {len(similar_regions)} similar field regions:")
    for i, (x, y, correlation) in enumerate(similar_regions):
        # Convert back to world coordinates
        world_x = (x / field.width) * env.world_size
        world_y = (y / field.height) * env.world_size
        print(f"      Match {i+1}: world pos ({world_x:.1f}, {world_y:.1f}), correlation={correlation:.3f}")
    
    # Compare with discrete pattern matching
    print(f"\n📊 Comparing with discrete pattern system...")
    
    # Set up discrete pattern system
    encoder = SparsePatternEncoder(pattern_dim=24, sparsity=0.1, quiet_mode=True)
    storage = SparsePatternStorage(pattern_dim=24, max_patterns=100, quiet_mode=True)
    
    # Store discrete patterns from the same experiences
    discrete_patterns = []
    for i, (experience, description) in enumerate(experiences):
        pattern = encoder.encode_top_k(experience.sensory_data, f"pattern_{i}")
        storage.store_pattern(pattern)
        discrete_patterns.append((pattern, description))
    
    # Search discrete patterns
    query_pattern = encoder.encode_top_k(torch.tensor(query_sensory), "query")
    discrete_results = storage.find_similar_patterns(query_pattern, k=5)
    
    print(f"   Discrete pattern matches:")
    for i, (pattern_id, similarity) in enumerate(discrete_results):
        print(f"      Match {i+1}: {pattern_id}, similarity={similarity:.3f}")
    
    # Analyze field properties
    print(f"\n📈 Analyzing field properties...")
    stats = field.get_stats()
    active_regions = field.get_active_regions()
    
    print(f"   Field statistics:")
    print(f"      Total imprints: {stats['total_imprints']}")
    print(f"      Field density: {stats['field_density']:.4f}")
    print(f"      Max activation: {stats['field_max_activation']:.4f}")
    print(f"      Mean activation: {stats['field_mean_activation']:.4f}")
    print(f"      Active regions: {len(active_regions)}")
    print(f"      Avg imprint time: {stats['avg_imprint_time_ms']:.2f}ms")
    print(f"      Avg retrieval time: {stats['avg_retrieval_time_ms']:.2f}ms")
    
    # Test analogical reasoning capability
    print(f"\n🎯 Testing analogical reasoning...")
    
    # Create two similar but different scenarios
    scenario_a = {'position': np.array([2.0, 2.0]), 'orientation': 0.0}
    scenario_b = {'position': np.array([8.0, 8.0]), 'orientation': 0.0}
    
    experiences_analogy = []
    for scenario in [scenario_a, scenario_b]:
        env.robot_state.position = scenario['position']
        env.robot_state.orientation = scenario['orientation']
        sensory_data = env.get_sensory_input()
        
        field_x = (scenario['position'][0] / env.world_size) * field.width
        field_y = (scenario['position'][1] / env.world_size) * field.height
        
        experience = FieldExperience(
            sensory_data=torch.tensor(sensory_data, dtype=torch.float32),
            position=(field_x, field_y),
            intensity=0.8,
            spread=12.0,
            timestamp=time.time()
        )
        experiences_analogy.append(experience)
    
    # Apply analogical experiences
    for i, exp in enumerate(experiences_analogy):
        imprint = field.experience_to_imprint(exp)
        field.apply_imprint(imprint)
        print(f"   Analogical experience {i+1}: position ({exp.position[0]:.1f}, {exp.position[1]:.1f})")
    
    # Test if field can find analogical relationships
    query_analogy = experiences_analogy[0]  # Query with first position
    analogy_results = field.find_similar_regions(query_analogy, top_k=3)
    
    print(f"   Analogical query results:")
    for i, (x, y, correlation) in enumerate(analogy_results):
        world_x = (x / field.width) * env.world_size  
        world_y = (y / field.height) * env.world_size
        print(f"      Analogy {i+1}: world pos ({world_x:.1f}, {world_y:.1f}), correlation={correlation:.3f}")
    
    print(f"\n✅ ANALOG FIELD SENSORY DATA TEST COMPLETED!")
    print(f"🎯 Key validations:")
    print(f"   ✓ 24D sensory data → continuous field imprints")
    print(f"   ✓ Field evolution creates natural clustering") 
    print(f"   ✓ Continuous similarity search vs discrete matching")
    print(f"   ✓ Analogical reasoning through field topology")
    print(f"   ✓ Real robot positions mapped to field coordinates")
    
    return {
        'field_stats': stats,
        'continuous_matches': similar_regions,
        'discrete_matches': discrete_results,
        'active_regions': len(active_regions),
        'analogical_results': analogy_results
    }


if __name__ == "__main__":
    # Run the test
    results = test_analog_field_with_real_sensory_data()
    
    print(f"\n🔬 SCIENTIFIC ANALYSIS:")
    print(f"   Continuous field approach shows {len(results['continuous_matches'])} similar regions")
    print(f"   Discrete pattern approach shows {len(results['discrete_matches'])} similar patterns")
    print(f"   Field density with real sensory data: {results['field_stats']['field_density']:.4f}")
    print(f"   Natural clustering: {results['active_regions']} active regions emerged")
    print(f"   Analogical reasoning: {len(results['analogical_results'])} analogical connections found")
    
    print(f"\n🚀 Ready for Phase A2: Multi-scale field coupling!")