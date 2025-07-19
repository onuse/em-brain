#!/usr/bin/env python3
"""
Test Emergent Attention Allocation System

Validates that attention emerges naturally from resource competition between streams
based on their activation levels, constraint pressures, and processing needs.
"""

import sys
import os
import time
import tempfile

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.brain import MinimalBrain
from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.emergent_attention_allocation import (
    create_attention_allocator, AttentionCompetitionMode, AttentionBid
)


def test_attention_competition_modes():
    """Test different attention competition modes."""
    print("🧠 Testing Attention Competition Modes")
    print("=" * 45)
    
    # Create mock stream states for testing
    class MockStreamState:
        def __init__(self, activation_strength=0.5, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    # Test scenario: High activation sensory vs moderate motor vs low temporal
    stream_states = {
        StreamType.SENSORY: MockStreamState(0.9, "active", [1, 2, 3, 4, 5]),
        StreamType.MOTOR: MockStreamState(0.6, "active", [1, 2]),
        StreamType.TEMPORAL: MockStreamState(0.3, "completed", [1])
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.URGENCY_SIGNAL: 0.8,  # High urgency
            ConstraintType.PROCESSING_LOAD: 0.2   # Low load
        },
        StreamType.MOTOR: {
            ConstraintType.PROCESSING_LOAD: 0.6,  # Moderate load
            ConstraintType.RESOURCE_SCARCITY: 0.3
        },
        StreamType.TEMPORAL: {
            ConstraintType.PROCESSING_LOAD: 0.1   # Very low load
        }
    }
    
    # Test different competition modes
    modes = [
        AttentionCompetitionMode.BIOLOGICAL_INHIBITION,
        AttentionCompetitionMode.PROPORTIONAL,
        AttentionCompetitionMode.WINNER_TAKE_ALL,
        AttentionCompetitionMode.THRESHOLD_GATING
    ]
    
    results = {}
    
    for mode in modes:
        print(f"\n📊 Testing {mode.value} mode:")
        
        allocator = create_attention_allocator(budget=1.0, mode=mode, quiet_mode=True)
        
        # Compute bids and allocate attention
        bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
        allocations = allocator.allocate_attention(bids)
        
        # Analyze results
        total_allocated = sum(alloc.allocated_attention for alloc in allocations)
        winner = max(allocations, key=lambda a: a.allocated_attention) if allocations else None
        
        print(f"  Winner: {winner.stream_type.value if winner else 'None'}")
        print(f"  Winner allocation: {winner.allocated_attention:.3f}" if winner else "  No winner")
        print(f"  Total allocated: {total_allocated:.3f}")
        print(f"  Allocations:")
        for alloc in sorted(allocations, key=lambda a: a.allocated_attention, reverse=True):
            print(f"    {alloc.stream_type.value}: {alloc.allocated_attention:.3f} "
                  f"(rank: {alloc.bid_rank}, pressure: {alloc.competition_pressure:.3f})")
        
        results[mode] = {
            'winner': winner.stream_type if winner else None,
            'winner_allocation': winner.allocated_attention if winner else 0.0,
            'total_allocated': total_allocated,
            'allocations': {alloc.stream_type: alloc.allocated_attention for alloc in allocations}
        }
    
    # Validate expected behaviors
    print(f"\n🎯 Mode Validation:")
    
    # Winner-take-all should give most to winner
    wta_result = results[AttentionCompetitionMode.WINNER_TAKE_ALL]
    wta_correct = (wta_result['winner'] == StreamType.SENSORY and 
                   wta_result['winner_allocation'] > 0.5)
    print(f"  Winner-take-all behavior: {'✅' if wta_correct else '❌'}")
    
    # Proportional should distribute more evenly
    prop_result = results[AttentionCompetitionMode.PROPORTIONAL]
    prop_allocations = list(prop_result['allocations'].values())
    prop_spread = max(prop_allocations) - min(prop_allocations) if prop_allocations else 0
    prop_correct = prop_spread < 0.8  # More even distribution
    print(f"  Proportional distribution: {'✅' if prop_correct else '❌'}")
    
    # Biological inhibition should be between winner-take-all and proportional
    bio_result = results[AttentionCompetitionMode.BIOLOGICAL_INHIBITION]
    bio_correct = (bio_result['winner'] == StreamType.SENSORY and
                   0.3 < bio_result['winner_allocation'] < 0.9)
    print(f"  Biological inhibition balance: {'✅' if bio_correct else '❌'}")
    
    modes_working = wta_correct and prop_correct and bio_correct
    
    return modes_working


def test_constraint_based_attention_dynamics():
    """Test that attention allocation responds to constraint pressures."""
    print("\n🔥 Testing Constraint-Based Attention Dynamics")
    print("=" * 50)
    
    allocator = create_attention_allocator(
        budget=1.0, 
        mode=AttentionCompetitionMode.BIOLOGICAL_INHIBITION, 
        quiet_mode=True
    )
    
    class MockStreamState:
        def __init__(self, activation_strength=0.5, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    # Scenario 1: Sensory urgency should win attention
    print("📋 Scenario 1: High sensory urgency")
    
    stream_states = {
        StreamType.SENSORY: MockStreamState(0.6, "active", [1, 2, 3]),
        StreamType.MOTOR: MockStreamState(0.6, "active", [1, 2, 3]),
        StreamType.TEMPORAL: MockStreamState(0.6, "active", [1, 2, 3])
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {ConstraintType.URGENCY_SIGNAL: 0.9},  # High urgency
        StreamType.MOTOR: {},
        StreamType.TEMPORAL: {}
    }
    
    bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
    allocations = allocator.allocate_attention(bids)
    
    sensory_allocation = next((a.allocated_attention for a in allocations 
                              if a.stream_type == StreamType.SENSORY), 0.0)
    motor_allocation = next((a.allocated_attention for a in allocations 
                            if a.stream_type == StreamType.MOTOR), 0.0)
    
    urgency_wins = sensory_allocation > motor_allocation
    print(f"  Sensory allocation: {sensory_allocation:.3f}")
    print(f"  Motor allocation: {motor_allocation:.3f}")
    print(f"  Urgency wins attention: {'✅' if urgency_wins else '❌'}")
    
    # Scenario 2: Processing load should reduce attention allocation
    print("\n📋 Scenario 2: High processing load penalty")
    
    constraint_pressures = {
        StreamType.SENSORY: {ConstraintType.PROCESSING_LOAD: 0.9},  # High load (overwhelmed)
        StreamType.MOTOR: {},
        StreamType.TEMPORAL: {}
    }
    
    bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
    allocations = allocator.allocate_attention(bids)
    
    sensory_allocation_loaded = next((a.allocated_attention for a in allocations 
                                     if a.stream_type == StreamType.SENSORY), 0.0)
    motor_allocation_normal = next((a.allocated_attention for a in allocations 
                                   if a.stream_type == StreamType.MOTOR), 0.0)
    
    load_penalty_works = sensory_allocation_loaded < motor_allocation_normal
    print(f"  Sensory (high load): {sensory_allocation_loaded:.3f}")
    print(f"  Motor (normal): {motor_allocation_normal:.3f}")
    print(f"  Load penalty works: {'✅' if load_penalty_works else '❌'}")
    
    # Scenario 3: Resource scarcity should boost attention
    print("\n📋 Scenario 3: Resource scarcity boost")
    
    constraint_pressures = {
        StreamType.SENSORY: {ConstraintType.RESOURCE_SCARCITY: 0.8},  # High scarcity
        StreamType.MOTOR: {},
        StreamType.TEMPORAL: {}
    }
    
    bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
    allocations = allocator.allocate_attention(bids)
    
    sensory_allocation_scarce = next((a.allocated_attention for a in allocations 
                                     if a.stream_type == StreamType.SENSORY), 0.0)
    motor_allocation_normal2 = next((a.allocated_attention for a in allocations 
                                    if a.stream_type == StreamType.MOTOR), 0.0)
    
    scarcity_boost_works = sensory_allocation_scarce > motor_allocation_normal2
    print(f"  Sensory (scarce): {sensory_allocation_scarce:.3f}")
    print(f"  Motor (normal): {motor_allocation_normal2:.3f}")
    print(f"  Scarcity boost works: {'✅' if scarcity_boost_works else '❌'}")
    
    constraint_dynamics_work = urgency_wins and load_penalty_works and scarcity_boost_works
    
    return constraint_dynamics_work


def test_brain_integration():
    """Test emergent attention allocation integration with the brain system."""
    print("\n🧠 Testing Brain System Integration")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Configuration for testing
        config = {
            'memory': {
                'persistent_memory_path': temp_dir,
                'enable_persistence': False
            },
            'brain': {
                'type': 'sparse_goldilocks',
                'sensory_dim': 16,
                'motor_dim': 4,
                'target_cycle_time_ms': 25.0,
                'enable_biological_timing': True,
                'enable_parallel_processing': True
            },
            'logging': {
                'log_brain_cycles': False
            }
        }
        
        print("🔧 Creating brain with emergent attention allocation...")
        brain = MinimalBrain(config=config, enable_logging=False, quiet_mode=False)
        brain.enable_parallel_processing(True)
        
        # Ensure stream coordinator is initialized
        brain.parallel_coordinator._ensure_stream_coordinator()
        shared_state = brain.parallel_coordinator.stream_coordinator.shared_state
        
        print("📊 Testing attention allocation in live brain cycles...")
        
        # Run several processing cycles with varied input
        attention_winners = []
        attention_allocations = []
        
        for cycle in range(20):
            # Generate input with varying complexity to trigger different constraints
            if cycle < 5:
                # Low complexity - should favor efficient processing
                sensory_input = [0.1 * i for i in range(16)]
            elif cycle < 10:
                # High complexity - should trigger constraint responses
                sensory_input = [0.5 * i * (1 + 0.8 * (cycle % 3)) for i in range(16)]
            elif cycle < 15:
                # Moderate complexity with patterns
                sensory_input = [0.2 * i * (1 + 0.3 * ((cycle + i) % 4)) for i in range(16)]
            else:
                # Very high complexity - should trigger resource competition
                sensory_input = [0.8 * i * (1 + 0.9 * (cycle % 2)) for i in range(16)]
            
            # Process through brain
            motor_output, brain_state = brain.process_sensory_input(sensory_input)
            
            # Get attention allocation statistics
            stats = shared_state.get_shared_state_stats()
            attention_stats = stats.get('attention_allocation', {})
            current_allocations = attention_stats.get('current_allocations', {})
            
            if current_allocations:
                # Find winner (stream with most attention)
                winner = max(current_allocations.items(), key=lambda x: x[1])
                attention_winners.append(winner[0])
                attention_allocations.append(current_allocations.copy())
                
                if cycle % 5 == 0 and current_allocations:
                    print(f"  Cycle {cycle}: Winner = {winner[0].value}, "
                          f"Allocation = {winner[1]:.3f}")
        
        brain.finalize_session()
        
        # Analyze attention allocation patterns
        print(f"\n📈 Attention Allocation Analysis:")
        
        if attention_winners:
            winner_counts = {}
            for winner in attention_winners:
                winner_counts[winner] = winner_counts.get(winner, 0) + 1
            
            print(f"  Total cycles with allocation: {len(attention_winners)}")
            print(f"  Winner distribution:")
            for stream, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(attention_winners)) * 100
                print(f"    {stream.value}: {count} times ({percentage:.1f}%)")
            
            # Check that allocation varies (system is responsive)
            attention_variance = len(set(attention_winners))
            responsive_system = attention_variance > 1
            print(f"  Attention variance: {attention_variance} different winners")
            print(f"  System responsiveness: {'✅' if responsive_system else '❌'}")
            
            # Check that allocations are reasonable (not all zeros)
            total_allocations = sum(sum(alloc.values()) for alloc in attention_allocations if alloc)
            reasonable_allocations = total_allocations > 0.1
            print(f"  Total allocations: {total_allocations:.3f}")
            print(f"  Reasonable allocation levels: {'✅' if reasonable_allocations else '❌'}")
            
            integration_success = responsive_system and reasonable_allocations
        else:
            print("  ⚠️ No attention allocations detected")
            integration_success = False
        
        return integration_success


if __name__ == "__main__":
    print("🚀 Testing Emergent Attention Allocation System")
    print("=" * 60)
    
    # Run all tests
    test1_success = test_attention_competition_modes()
    test2_success = test_constraint_based_attention_dynamics()
    test3_success = test_brain_integration()
    
    # Overall results
    all_passed = test1_success and test2_success and test3_success
    
    print(f"\n🏁 Test Results Summary:")
    print(f"  Competition Modes: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"  Constraint Dynamics: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    print(f"  Brain Integration: {'✅ PASSED' if test3_success else '❌ FAILED'}")
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Emergent Attention Allocation is working correctly!")
        print("   ✅ Multiple competition modes function properly")
        print("   ✅ Attention responds to constraint pressures")
        print("   ✅ Integration with brain system is successful")
        print("   ✅ Attention emerges naturally from resource competition")
    else:
        print("\n⚠️ Some tests failed - review implementation")
    
    exit(0 if all_passed else 1)