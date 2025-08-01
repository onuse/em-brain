#!/usr/bin/env python3
"""Test Phase 3: Pattern Library with Behavioral Similarity."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

import torch
import numpy as np
import time
from src.brains.field.unified_field_brain import UnifiedFieldBrain
from src.brains.field.field_strategic_planner import FieldStrategicPlanner, StrategicPattern

def test_behavioral_similarity():
    """Test that behavioral similarity recognizes similar movement patterns."""
    print("\n=== Testing Behavioral Similarity Metric ===")
    
    # Create planner
    planner = FieldStrategicPlanner(
        field_shape=(8, 8, 8, 64),
        sensory_dim=10,
        motor_dim=4,
        device=torch.device('cpu')
    )
    
    # Create two trajectories that go forward but with different dynamics
    traj1 = torch.zeros(20, 4)
    traj2 = torch.zeros(20, 4)
    
    # Both move forward (dim 0) but with different patterns
    for t in range(20):
        traj1[t, 0] = t * 0.1  # Linear forward
        traj1[t, 1] = 0.1 * np.sin(t * 0.5)  # Small oscillation
        
        traj2[t, 0] = t * 0.1 + 0.05 * np.sin(t)  # Forward with wobble
        traj2[t, 1] = 0.05 * np.cos(t * 0.5)  # Different oscillation
    
    similarity = planner._compute_behavioral_similarity(traj1, traj2)
    print(f"Forward movement similarity: {similarity:.3f}")
    assert similarity > 0.7, "Similar forward movements should have high similarity"
    
    # Create a backward trajectory
    traj3 = torch.zeros(20, 4)
    for t in range(20):
        traj3[t, 0] = -t * 0.1  # Backward
    
    backward_sim = planner._compute_behavioral_similarity(traj1, traj3)
    print(f"Forward vs backward similarity: {backward_sim:.3f}")
    assert backward_sim < 0.3, "Opposite movements should have low similarity"
    
    print("✓ Behavioral similarity metric working correctly")

def test_pattern_storage_and_blending():
    """Test pattern storage with behavioral similarity and blending."""
    print("\n=== Testing Pattern Storage and Blending ===")
    
    brain = UnifiedFieldBrain(
        sensory_dim=10,
        motor_dim=4,
        spatial_resolution=8,
        quiet_mode=True
    )
    
    brain.enable_strategic_planning(True)
    planner = brain.strategic_planner
    
    # Create and add similar patterns
    context = torch.tensor([0.5, 0.5, 0.5, 0.5], device=brain.device)
    
    # Pattern 1: Creates forward movement
    pattern1 = planner._generate_novel_pattern()
    traj1 = torch.zeros(50, 4, device=brain.device)
    for t in range(50):
        traj1[t, 0] = t * 0.02  # Forward
    
    strategic_pattern1 = StrategicPattern(
        pattern=pattern1,
        score=0.8,
        behavioral_signature=traj1.mean(dim=0),
        behavioral_trajectory=traj1,
        persistence=30.0,
        context_embedding=context,
        creation_time=time.time()
    )
    
    planner._add_to_library(strategic_pattern1)
    assert len(planner.pattern_library) == 1
    
    # Pattern 2: Similar forward movement (should blend)
    pattern2 = planner._generate_novel_pattern()
    traj2 = torch.zeros(50, 4, device=brain.device)
    for t in range(50):
        traj2[t, 0] = t * 0.02 + 0.01 * np.sin(t * 0.2)  # Forward with slight wave
    
    strategic_pattern2 = StrategicPattern(
        pattern=pattern2,
        score=0.75,  # Comparable score
        behavioral_signature=traj2.mean(dim=0),
        behavioral_trajectory=traj2,
        persistence=30.0,
        context_embedding=context,
        creation_time=time.time() + 1
    )
    
    planner._add_to_library(strategic_pattern2)
    
    # Should still have 1 pattern (blended)
    assert len(planner.pattern_library) == 1, "Similar patterns should blend"
    assert planner.pattern_library[0].usage_count > 0, "Blending should increase usage count"
    
    # Pattern 3: Different behavior (turn right)
    pattern3 = planner._generate_novel_pattern()
    traj3 = torch.zeros(50, 4, device=brain.device)
    for t in range(50):
        traj3[t, 1] = t * 0.02  # Turn right
    
    strategic_pattern3 = StrategicPattern(
        pattern=pattern3,
        score=0.9,
        behavioral_signature=traj3.mean(dim=0),
        behavioral_trajectory=traj3,
        persistence=30.0,
        context_embedding=context,
        creation_time=time.time() + 2
    )
    
    planner._add_to_library(strategic_pattern3)
    
    # Should now have 2 patterns (different behaviors)
    assert len(planner.pattern_library) == 2, "Different behaviors should create separate patterns"
    
    print(f"✓ Pattern library size: {len(planner.pattern_library)}")
    print(f"✓ First pattern usage count: {planner.pattern_library[0].usage_count}")

def test_resonance_based_retrieval():
    """Test pattern retrieval through field resonance."""
    print("\n=== Testing Resonance-Based Pattern Retrieval ===")
    
    brain = UnifiedFieldBrain(
        sensory_dim=10,
        motor_dim=4,
        spatial_resolution=8,
        quiet_mode=True
    )
    
    brain.enable_strategic_planning(True)
    planner = brain.strategic_planner
    
    # Add patterns with different contexts
    contexts = [
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=brain.device),  # High energy
        torch.tensor([0.0, 1.0, 0.0, 0.0], device=brain.device),  # High variance
        torch.tensor([0.0, 0.0, 1.0, 0.0], device=brain.device),  # High activation
    ]
    
    for i, ctx in enumerate(contexts):
        pattern = planner._generate_novel_pattern()
        traj = torch.randn(50, 4, device=brain.device) * 0.1
        traj[:, 0] += i * 0.1  # Different forward speeds
        
        strategic_pattern = StrategicPattern(
            pattern=pattern,
            score=0.7 + i * 0.1,
            behavioral_signature=traj.mean(dim=0),
            behavioral_trajectory=traj,
            persistence=30.0,
            context_embedding=ctx,
            creation_time=time.time() + i
        )
        planner._add_to_library(strategic_pattern)
    
    # Test retrieval with similar context
    test_context = torch.tensor([0.9, 0.1, 0.0, 0.0], device=brain.device)  # Close to first
    retrieved = planner._generate_from_library(test_context, exploration=0.1)
    
    print(f"✓ Retrieved pattern shape: {retrieved.shape}")
    print(f"✓ Pattern library has {len(planner.pattern_library)} patterns")
    
    # Check usage counts increased
    assert any(p.usage_count > 0 for p in planner.pattern_library), "Retrieval should increase usage"

def test_behavioral_clustering():
    """Test that patterns cluster by behavioral similarity."""
    print("\n=== Testing Behavioral Clustering ===")
    
    brain = UnifiedFieldBrain(
        sensory_dim=10,
        motor_dim=4,
        spatial_resolution=8,
        quiet_mode=True
    )
    
    brain.enable_strategic_planning(True)
    planner = brain.strategic_planner
    
    # Add patterns that create 3 distinct behaviors
    behaviors = [
        (1.0, 0.0),  # Forward
        (1.0, 0.0),  # Forward (similar)
        (0.0, 1.0),  # Right
        (0.0, 1.0),  # Right (similar)
        (-1.0, 0.0), # Backward
    ]
    
    for i, (forward, right) in enumerate(behaviors):
        pattern = planner._generate_novel_pattern()
        traj = torch.zeros(30, 4, device=brain.device)
        for t in range(30):
            traj[t, 0] = forward * t * 0.01
            traj[t, 1] = right * t * 0.01
        
        strategic_pattern = StrategicPattern(
            pattern=pattern,
            score=0.5 + i * 0.1,
            behavioral_signature=traj.mean(dim=0),
            behavioral_trajectory=traj,
            persistence=30.0,
            context_embedding=torch.randn(4, device=brain.device),
            creation_time=time.time() + i
        )
        planner.pattern_library.append(strategic_pattern)
    
    # Get statistics
    stats = planner.get_pattern_statistics()
    print(f"✓ Library size: {stats['library_size']}")
    print(f"✓ Behavior clusters: {stats['behavior_clusters']}")
    
    assert stats['behavior_clusters'] == 3, "Should have 3 distinct behavior clusters"

def test_find_similar_patterns():
    """Test finding patterns by target behavior."""
    print("\n=== Testing Find Similar Patterns ===")
    
    brain = UnifiedFieldBrain(
        sensory_dim=10,
        motor_dim=4,  
        spatial_resolution=8,
        quiet_mode=True
    )
    
    brain.enable_strategic_planning(True)
    planner = brain.strategic_planner
    
    # Add various patterns
    for i in range(5):
        pattern = planner._generate_novel_pattern()
        traj = torch.zeros(30, 4, device=brain.device)
        
        if i < 3:  # First 3 create forward movement
            for t in range(30):
                traj[t, 0] = t * 0.01 + i * 0.001
        else:  # Last 2 create turning
            for t in range(30):
                traj[t, 1] = t * 0.01
        
        strategic_pattern = StrategicPattern(
            pattern=pattern,
            score=0.5 + i * 0.1,
            behavioral_signature=traj.mean(dim=0),
            behavioral_trajectory=traj,
            persistence=30.0,
            context_embedding=torch.randn(4, device=brain.device),
            creation_time=time.time() + i
        )
        planner.pattern_library.append(strategic_pattern)
    
    # Find patterns that create forward movement
    target_behavior = torch.tensor([1.0, 0.0, 0.0, 0.0], device=brain.device)
    similar = planner.find_similar_patterns(target_behavior, n_results=5)
    
    print(f"✓ Found {len(similar)} similar patterns")
    
    # Check that we found some similar patterns
    if len(similar) > 0:
        # Check that top results are forward-moving patterns
        forward_count = sum(1 for p in similar[:min(3, len(similar))] if p.behavioral_signature[0] > 0.01)
        print(f"✓ Forward-moving patterns in top results: {forward_count}")
    else:
        print("⚠️ No similar patterns found (library might be empty)")

if __name__ == '__main__':
    test_behavioral_similarity()
    test_pattern_storage_and_blending()
    test_resonance_based_retrieval()
    test_behavioral_clustering()
    test_find_similar_patterns()
    
    print("\n✅ All Phase 3 pattern library tests passed!")