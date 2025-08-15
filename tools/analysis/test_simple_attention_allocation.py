#!/usr/bin/env python3
"""
Simple Test for Emergent Attention Allocation

Tests the core attention allocation functionality with direct instantiation.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from server.src.vector_stream.stream_types import StreamType, ConstraintType
from server.src.vector_stream.emergent_attention_allocation import (
    create_attention_allocator, AttentionCompetitionMode
)


def test_basic_attention_allocation():
    """Test basic attention allocation functionality."""
    print("🧠 Testing Basic Attention Allocation")
    print("=" * 40)
    
    # Create attention allocator
    allocator = create_attention_allocator(
        budget=1.0, 
        mode=AttentionCompetitionMode.BIOLOGICAL_INHIBITION, 
        quiet_mode=False
    )
    
    # Mock stream states
    class MockStreamState:
        def __init__(self, activation_strength=0.5, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    # Create test scenario: High sensory activation with urgency
    stream_states = {
        StreamType.SENSORY: MockStreamState(0.9, "active", [1, 2, 3, 4, 5]),  # High activation
        StreamType.MOTOR: MockStreamState(0.5, "active", [1, 2]),              # Moderate activation
        StreamType.TEMPORAL: MockStreamState(0.3, "completed", [1])             # Low activation
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.URGENCY_SIGNAL: 0.8,  # High urgency in sensory
            ConstraintType.PROCESSING_LOAD: 0.2
        },
        StreamType.MOTOR: {
            ConstraintType.PROCESSING_LOAD: 0.6,
            ConstraintType.RESOURCE_SCARCITY: 0.3
        },
        StreamType.TEMPORAL: {
            ConstraintType.PROCESSING_LOAD: 0.1
        }
    }
    
    print(f"\n📊 Input Scenario:")
    for stream_type, state in stream_states.items():
        constraints = constraint_pressures.get(stream_type, {})
        print(f"  {stream_type.value}:")
        print(f"    Activation: {state.activation_strength:.2f}")
        print(f"    Phase: {state.processing_phase}")
        print(f"    Patterns: {len(state.active_patterns)}")
        if constraints:
            for constraint_type, intensity in constraints.items():
                print(f"    {constraint_type.value}: {intensity:.2f}")
    
    # Compute attention bids
    bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
    
    print(f"\n💰 Attention Bids:")
    for bid in sorted(bids, key=lambda b: b.bid_strength, reverse=True):
        print(f"  {bid.stream_type.value}:")
        print(f"    Bid strength: {bid.bid_strength:.3f}")
        print(f"    Base activation: {bid.base_activation:.3f}")
        print(f"    Urgency pressure: {bid.urgency_pressure:.3f}")
        print(f"    Resource need: {bid.resource_need:.3f}")
    
    # Allocate attention
    allocations = allocator.allocate_attention(bids)
    
    print(f"\n🎯 Attention Allocations:")
    total_allocated = 0.0
    for allocation in sorted(allocations, key=lambda a: a.allocated_attention, reverse=True):
        total_allocated += allocation.allocated_attention
        print(f"  {allocation.stream_type.value}:")
        print(f"    Allocated: {allocation.allocated_attention:.3f}")
        print(f"    Allocation ratio: {allocation.allocation_ratio:.3f}")
        print(f"    Bid rank: {allocation.bid_rank}")
        print(f"    Competition pressure: {allocation.competition_pressure:.3f}")
    
    print(f"\n📈 Summary:")
    print(f"  Total attention allocated: {total_allocated:.3f}")
    print(f"  Winner: {allocations[0].stream_type.value if allocations else 'None'}")
    print(f"  Winner allocation: {allocations[0].allocated_attention:.3f}" if allocations else "N/A")
    
    # Validate expected behavior
    sensory_wins = (allocations[0].stream_type == StreamType.SENSORY if allocations else False)
    reasonable_allocation = (allocations[0].allocated_attention > 0.3 if allocations else False)
    total_reasonable = (0.5 < total_allocated <= 1.0)
    
    print(f"\n🎯 Validation:")
    print(f"  Sensory wins (high activation + urgency): {'✅' if sensory_wins else '❌'}")
    print(f"  Winner gets reasonable allocation (>30%): {'✅' if reasonable_allocation else '❌'}")
    print(f"  Total allocation reasonable (50-100%): {'✅' if total_reasonable else '❌'}")
    
    basic_allocation_works = sensory_wins and reasonable_allocation and total_reasonable
    
    return basic_allocation_works


def test_competition_modes():
    """Test different competition modes produce expected behaviors."""
    print("\n🏆 Testing Competition Modes")
    print("=" * 30)
    
    # Mock stream states
    class MockStreamState:
        def __init__(self, activation_strength=0.5, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    stream_states = {
        StreamType.SENSORY: MockStreamState(0.8, "active", [1, 2, 3, 4]),
        StreamType.MOTOR: MockStreamState(0.6, "active", [1, 2]),
        StreamType.TEMPORAL: MockStreamState(0.4, "active", [1])
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {ConstraintType.URGENCY_SIGNAL: 0.5},
        StreamType.MOTOR: {ConstraintType.PROCESSING_LOAD: 0.3},
        StreamType.TEMPORAL: {}
    }
    
    modes = [
        AttentionCompetitionMode.BIOLOGICAL_INHIBITION,
        AttentionCompetitionMode.PROPORTIONAL,
        AttentionCompetitionMode.WINNER_TAKE_ALL
    ]
    
    results = {}
    
    for mode in modes:
        print(f"\n📋 Mode: {mode.value}")
        
        allocator = create_attention_allocator(budget=1.0, mode=mode, quiet_mode=True)
        bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
        allocations = allocator.allocate_attention(bids)
        
        winner = allocations[0] if allocations else None
        winner_allocation = winner.allocated_attention if winner else 0.0
        total_allocated = sum(alloc.allocated_attention for alloc in allocations)
        
        print(f"  Winner: {winner.stream_type.value if winner else 'None'}")
        print(f"  Winner allocation: {winner_allocation:.3f}")
        print(f"  Total allocated: {total_allocated:.3f}")
        
        results[mode] = {
            'winner_allocation': winner_allocation,
            'total_allocated': total_allocated
        }
    
    # Validate mode characteristics
    bio_result = results[AttentionCompetitionMode.BIOLOGICAL_INHIBITION]
    prop_result = results[AttentionCompetitionMode.PROPORTIONAL]
    wta_result = results[AttentionCompetitionMode.WINNER_TAKE_ALL]
    
    # Winner-take-all should give winner the most
    wta_gives_most = wta_result['winner_allocation'] >= bio_result['winner_allocation']
    
    # Proportional should allocate most evenly
    prop_most_total = prop_result['total_allocated'] >= bio_result['total_allocated']
    
    # Biological should be balanced
    bio_balanced = 0.3 <= bio_result['winner_allocation'] <= 0.9
    
    print(f"\n🎯 Mode Validation:")
    print(f"  Winner-take-all gives most to winner: {'✅' if wta_gives_most else '❌'}")
    print(f"  Proportional allocates most total: {'✅' if prop_most_total else '❌'}")
    print(f"  Biological is balanced: {'✅' if bio_balanced else '❌'}")
    
    modes_work = wta_gives_most and prop_most_total and bio_balanced
    
    return modes_work


def test_constraint_responsiveness():
    """Test that attention responds to different constraint pressures."""
    print("\n🔥 Testing Constraint Responsiveness")
    print("=" * 35)
    
    # Base scenario
    class MockStreamState:
        def __init__(self, activation_strength=0.6, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    base_stream_states = {
        StreamType.SENSORY: MockStreamState(),
        StreamType.MOTOR: MockStreamState(),
        StreamType.TEMPORAL: MockStreamState()
    }
    
    allocator = create_attention_allocator(
        budget=1.0, 
        mode=AttentionCompetitionMode.BIOLOGICAL_INHIBITION, 
        quiet_mode=True
    )
    
    # Test 1: No constraints (baseline)
    print(f"\n📋 Test 1: No constraints")
    no_constraints = {stream: {} for stream in StreamType}
    bids = allocator.compute_attention_bids(base_stream_states, no_constraints)
    allocations = allocator.allocate_attention(bids)
    
    baseline_winner = allocations[0].stream_type if allocations else None
    baseline_allocation = allocations[0].allocated_attention if allocations else 0.0
    print(f"  Winner: {baseline_winner.value if baseline_winner else 'None'}")
    print(f"  Allocation: {baseline_allocation:.3f}")
    
    # Test 2: High urgency in motor stream
    print(f"\n📋 Test 2: High urgency in motor")
    urgency_constraints = {
        StreamType.SENSORY: {},
        StreamType.MOTOR: {ConstraintType.URGENCY_SIGNAL: 0.9},
        StreamType.TEMPORAL: {}
    }
    bids = allocator.compute_attention_bids(base_stream_states, urgency_constraints)
    allocations = allocator.allocate_attention(bids)
    
    urgency_winner = allocations[0].stream_type if allocations else None
    urgency_allocation = allocations[0].allocated_attention if allocations else 0.0
    print(f"  Winner: {urgency_winner.value if urgency_winner else 'None'}")
    print(f"  Allocation: {urgency_allocation:.3f}")
    
    # Test 3: High processing load in temporal stream
    print(f"\n📋 Test 3: High processing load in temporal")
    load_constraints = {
        StreamType.SENSORY: {},
        StreamType.MOTOR: {},
        StreamType.TEMPORAL: {ConstraintType.PROCESSING_LOAD: 0.9}
    }
    bids = allocator.compute_attention_bids(base_stream_states, load_constraints)
    allocations = allocator.allocate_attention(bids)
    
    load_winner = allocations[0].stream_type if allocations else None
    load_allocation = allocations[0].allocated_attention if allocations else 0.0
    print(f"  Winner: {load_winner.value if load_winner else 'None'}")
    print(f"  Allocation: {load_allocation:.3f}")
    
    # Validate responsiveness
    urgency_responsive = (urgency_winner == StreamType.MOTOR)
    load_responsive = (load_winner != StreamType.TEMPORAL)  # High load should reduce attention
    allocation_changes = len(set([baseline_allocation, urgency_allocation, load_allocation])) > 1
    
    print(f"\n🎯 Responsiveness Validation:")
    print(f"  Urgency boosts attention: {'✅' if urgency_responsive else '❌'}")
    print(f"  Load reduces attention: {'✅' if load_responsive else '❌'}")
    print(f"  Allocations change with constraints: {'✅' if allocation_changes else '❌'}")
    
    responsiveness_works = urgency_responsive and load_responsive and allocation_changes
    
    return responsiveness_works


if __name__ == "__main__":
    print("🚀 Simple Emergent Attention Allocation Test")
    print("=" * 55)
    
    # Run tests
    test1_success = test_basic_attention_allocation()
    test2_success = test_competition_modes()
    test3_success = test_constraint_responsiveness()
    
    # Results
    all_passed = test1_success and test2_success and test3_success
    
    print(f"\n🏁 Test Results:")
    print(f"  Basic Allocation: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"  Competition Modes: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    print(f"  Constraint Responsiveness: {'✅ PASSED' if test3_success else '❌ FAILED'}")
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Emergent Attention Allocation works correctly!")
        print("   ✅ Attention emerges from competitive resource dynamics")
        print("   ✅ Multiple competition modes work as expected")
        print("   ✅ System responds appropriately to constraint pressures")
        print("   ✅ Resource allocation follows biological principles")
    
    exit(0 if all_passed else 1)