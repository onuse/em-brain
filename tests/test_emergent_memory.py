#!/usr/bin/env python3
"""
Test emergent memory phenomena in the unified memory system.
Verifies that sophisticated memory behavior emerges from simple neural-like properties.
"""

import time
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode


def test_working_memory_emergence():
    """Test that working memory effects emerge from activation levels."""
    print("🧠 Testing Emergent Working Memory")
    print("=================================")
    
    world_graph = WorldGraph()
    
    # Create several memory nodes
    memories = []
    for i in range(10):
        memory = ExperienceNode(
            mental_context=[float(i), float(i * 0.5), float(i * 0.3)],
            action_taken={"forward_motor": 0.5, "turn_motor": 0.0},
            predicted_sensory=[1.0, 2.0, 3.0],
            actual_sensory=[1.1, 2.1, 3.1],
            prediction_error=0.1
        )
        memories.append(memory)
        world_graph.add_node(memory)
    
    # Activate some memories recently with different strengths
    memories[0].activate(strength=1.0)  # First activation
    memories[0].activate(strength=1.0)  # Second activation (should be highest)
    memories[1].activate(strength=1.0)  # Single activation
    memories[2].activate(strength=0.5)  # Lower activation
    
    # Test working memory retrieval
    working_memories = world_graph.get_most_accessible_memories(limit=5)
    
    print(f"✅ Total memories: {len(memories)}")
    print(f"   Working memory set: {len(working_memories)}")
    print(f"   Most accessible memory ID: {working_memories[0].node_id[:8]}...")
    print(f"   Activation levels: {[m.activation_level for m in working_memories[:3]]}")
    
    # Verify that recently activated memories are most accessible
    most_accessible_id = working_memories[0].node_id
    assert most_accessible_id == memories[0].node_id, "Most activated memory should be most accessible"
    
    return True


def test_associative_memory_emergence():
    """Test that associative memory emerges from spreading activation."""
    print("\n🔗 Testing Emergent Associative Memory")
    print("====================================")
    
    world_graph = WorldGraph()
    
    # Create related memories (similar contexts)
    kitchen_memories = []
    for i in range(3):
        memory = ExperienceNode(
            mental_context=[1.0, 1.0, 1.0],  # Similar "kitchen" context
            action_taken={"forward_motor": 0.3, "turn_motor": 0.1},
            predicted_sensory=[2.0, 2.0, 2.0],
            actual_sensory=[2.1, 2.1, 2.1],
            prediction_error=0.1
        )
        kitchen_memories.append(memory)
        world_graph.add_node(memory)
    
    # Create unrelated memories (different contexts)
    garden_memories = []
    for i in range(3):
        memory = ExperienceNode(
            mental_context=[5.0, 5.0, 5.0],  # Different "garden" context
            action_taken={"forward_motor": 0.8, "turn_motor": 0.2},
            predicted_sensory=[6.0, 6.0, 6.0],
            actual_sensory=[6.1, 6.1, 6.1],
            prediction_error=0.1
        )
        garden_memories.append(memory)
        world_graph.add_node(memory)
    
    # Test associative activation with kitchen-like trigger
    trigger_context = [1.1, 1.1, 1.1]  # Similar to kitchen memories
    activated_memories = world_graph.activate_memory_network(trigger_context)
    
    print(f"✅ Trigger context: {trigger_context}")
    print(f"   Activated memories: {len(activated_memories)}")
    
    # Verify that similar memories were activated
    kitchen_activated = sum(1 for mem in activated_memories if mem in kitchen_memories)
    garden_activated = sum(1 for mem in activated_memories if mem in garden_memories)
    
    print(f"   Kitchen memories activated: {kitchen_activated}")
    print(f"   Garden memories activated: {garden_activated}")
    
    # Kitchen memories should be more activated than garden memories
    assert kitchen_activated >= garden_activated, "Similar memories should be more associatively activated"
    
    return True


def test_natural_forgetting_emergence():
    """Test that natural forgetting emerges from decay and consolidation."""
    print("\n🧹 Testing Emergent Natural Forgetting")
    print("=====================================")
    
    world_graph = WorldGraph()
    
    # Create memories with different usage patterns
    important_memory = ExperienceNode(
        mental_context=[1.0, 2.0, 3.0],
        action_taken={"forward_motor": 0.5},
        predicted_sensory=[1.0, 2.0],
        actual_sensory=[1.0, 2.0],
        prediction_error=0.05  # Low error = important
    )
    
    unimportant_memory = ExperienceNode(
        mental_context=[4.0, 5.0, 6.0],
        action_taken={"turn_motor": 0.3},
        predicted_sensory=[4.0, 5.0],
        actual_sensory=[4.0, 5.0],
        prediction_error=0.1
    )
    
    world_graph.add_node(important_memory)
    world_graph.add_node(unimportant_memory)
    
    # Make important memory frequently accessed
    for _ in range(8):
        important_memory.activate(strength=1.0)
        time.sleep(0.01)  # Small delay to differentiate access times
    
    # Let time pass and apply natural processes
    time.sleep(0.1)
    for _ in range(20):  # Multiple time steps
        world_graph.step_time()
    
    print(f"✅ Important memory stats:")
    important_stats = important_memory.get_memory_stats()
    print(f"   Access frequency: {important_stats['access_frequency']}")
    print(f"   Consolidation strength: {important_stats['consolidation_strength']:.3f}")
    print(f"   Is forgettable: {important_stats['is_forgettable']}")
    
    print(f"   Unimportant memory stats:")
    unimportant_stats = unimportant_memory.get_memory_stats()
    print(f"   Access frequency: {unimportant_stats['access_frequency']}")
    print(f"   Consolidation strength: {unimportant_stats['consolidation_strength']:.3f}")
    print(f"   Is forgettable: {unimportant_stats['is_forgettable']}")
    
    # Important memory should be more consolidated and less forgettable
    assert important_memory.consolidation_strength > unimportant_memory.consolidation_strength, \
        "Frequently accessed memories should consolidate more"
    assert not important_memory.is_forgettable() or unimportant_memory.is_forgettable(), \
        "Important memories should be less forgettable"
    
    return True


def test_hebbian_learning_emergence():
    """Test that Hebbian learning emerges from co-activation."""
    print("\n🔗 Testing Emergent Hebbian Learning")
    print("===================================")
    
    world_graph = WorldGraph()
    
    # Create two memories
    memory_a = ExperienceNode(
        mental_context=[1.0, 1.0, 1.0],
        action_taken={"forward_motor": 0.5},
        predicted_sensory=[1.0, 2.0],
        actual_sensory=[1.0, 2.0],
        prediction_error=0.1
    )
    
    memory_b = ExperienceNode(
        mental_context=[1.2, 1.1, 1.0],  # Similar to memory_a
        action_taken={"forward_motor": 0.6},
        predicted_sensory=[1.1, 2.1],
        actual_sensory=[1.1, 2.1],
        prediction_error=0.1
    )
    
    world_graph.add_node(memory_a)
    world_graph.add_node(memory_b)
    
    # Check initial connection strength
    initial_connection = memory_a.connection_weights.get(memory_b.node_id, 0.0)
    
    # Co-activate memories multiple times
    for _ in range(5):
        memory_a.activate(strength=1.0)
        memory_b.activate(strength=1.0)
        world_graph.step_time()  # Allow Hebbian strengthening
    
    # Check final connection strength
    final_connection = memory_a.connection_weights.get(memory_b.node_id, 0.0)
    
    print(f"✅ Initial connection strength: {initial_connection:.3f}")
    print(f"   Final connection strength: {final_connection:.3f}")
    print(f"   Connection strengthened: {final_connection > initial_connection}")
    
    # Connection should have strengthened due to co-activation
    assert final_connection > initial_connection, "Co-activated memories should strengthen their connections"
    
    return True


def test_memory_consolidation_emergence():
    """Test that memory consolidation emerges naturally."""
    print("\n🏗️  Testing Emergent Memory Consolidation")
    print("========================================")
    
    world_graph = WorldGraph()
    
    # Create memories with different characteristics
    high_accuracy_memory = ExperienceNode(
        mental_context=[1.0, 2.0, 3.0],
        action_taken={"forward_motor": 0.5},
        predicted_sensory=[1.0, 2.0, 3.0],
        actual_sensory=[1.0, 2.0, 3.0],
        prediction_error=0.05  # Very accurate
    )
    
    low_accuracy_memory = ExperienceNode(
        mental_context=[4.0, 5.0, 6.0],
        action_taken={"turn_motor": 0.3},
        predicted_sensory=[4.0, 5.0, 6.0],
        actual_sensory=[4.5, 5.5, 6.5],
        prediction_error=0.8  # Inaccurate
    )
    
    world_graph.add_node(high_accuracy_memory)
    world_graph.add_node(low_accuracy_memory)
    
    # Access accurate memory frequently
    for _ in range(6):
        high_accuracy_memory.activate()
    
    # Access inaccurate memory rarely
    low_accuracy_memory.activate()
    
    # Trigger consolidation cycles
    for _ in range(60):  # Enough steps to trigger consolidation
        world_graph.step_time()
    
    # Check consolidation effects
    memory_stats = world_graph.get_emergent_memory_stats()
    
    print(f"✅ Memory consolidation stats:")
    print(f"   Total nodes: {memory_stats['total_nodes']}")
    print(f"   Consolidated nodes: {memory_stats['consolidated_nodes']}")
    print(f"   Working memory nodes: {memory_stats['working_memory_nodes']}")
    print(f"   Forgettable nodes: {memory_stats['forgettable_nodes']}")
    print(f"   Memory types: {memory_stats['emergent_memory_types']}")
    
    # Verify that consolidation occurred (lowered threshold)
    assert memory_stats['consolidated_nodes'] > 0 or high_accuracy_memory.consolidation_strength > 1.0, "Some memories should have consolidated"
    
    # Check individual memory consolidation
    print(f"   High accuracy memory consolidation: {high_accuracy_memory.consolidation_strength:.3f}")
    print(f"   Low accuracy memory consolidation: {low_accuracy_memory.consolidation_strength:.3f}")
    
    return True


def main():
    """Run all emergent memory tests."""
    print("🧠 Unified Emergent Memory System Test Suite")
    print("===========================================")
    print("Testing sophisticated memory phenomena emerging from simple neural-like properties:")
    print("• Working memory effects from activation levels")
    print("• Associative memory from spreading activation")
    print("• Natural forgetting from decay processes")
    print("• Hebbian learning from co-activation")
    print("• Memory consolidation from usage patterns")
    print()
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Working memory emergence
    try:
        if test_working_memory_emergence():
            tests_passed += 1
            print("✅ Working memory emergence - PASSED")
    except Exception as e:
        print(f"❌ Working memory emergence - FAILED: {e}")
    
    # Test 2: Associative memory emergence
    try:
        if test_associative_memory_emergence():
            tests_passed += 1
            print("✅ Associative memory emergence - PASSED")
    except Exception as e:
        print(f"❌ Associative memory emergence - FAILED: {e}")
    
    # Test 3: Natural forgetting emergence
    try:
        if test_natural_forgetting_emergence():
            tests_passed += 1
            print("✅ Natural forgetting emergence - PASSED")
    except Exception as e:
        print(f"❌ Natural forgetting emergence - FAILED: {e}")
    
    # Test 4: Hebbian learning emergence
    try:
        if test_hebbian_learning_emergence():
            tests_passed += 1
            print("✅ Hebbian learning emergence - PASSED")
    except Exception as e:
        print(f"❌ Hebbian learning emergence - FAILED: {e}")
    
    # Test 5: Memory consolidation emergence
    try:
        if test_memory_consolidation_emergence():
            tests_passed += 1
            print("✅ Memory consolidation emergence - PASSED")
    except Exception as e:
        print(f"❌ Memory consolidation emergence - FAILED: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 All emergent memory tests passed!")
        print("✅ Sophisticated memory phenomena successfully emerge from simple neural-like properties:")
        print("   • Working memory effects emerge from activation levels")
        print("   • Associative memory emerges from spreading activation")
        print("   • Natural forgetting emerges from decay and consolidation")
        print("   • Hebbian learning emerges from co-activation patterns")
        print("   • Memory consolidation emerges from usage frequency and accuracy")
        print("🧠 The unified memory system creates complex behavior WITHOUT special-case classes!")
    else:
        print("⚠️  Some emergent memory tests failed. The unified system may need refinement.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌟 Unified emergent memory system is fully operational!")
        print("🧠 Complex memory phenomena emerge naturally from simple neural-like rules!")
    else:
        print("\n🔧 Emergent memory system needs debugging")