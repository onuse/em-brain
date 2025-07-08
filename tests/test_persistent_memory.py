#!/usr/bin/env python3
"""
Test the persistent memory system.
Verifies that the brain can save and restore its state across sessions,
enabling lifelong learning and experience accumulation.
"""

import os
import time
import shutil
import tempfile
from datetime import datetime
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from core.experience_node import ExperienceNode
from predictor.multi_drive_predictor import MultiDrivePredictor


def test_basic_persistence():
    """Test basic save and load functionality."""
    print("💾 Testing Basic Persistence")
    print("===========================")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="robot_memory_test_")
    print(f"Using test directory: {temp_dir}")
    
    try:
        # Create brain with persistence
        predictor = MultiDrivePredictor(base_time_budget=0.05)
        brain1 = BrainInterface(predictor, memory_path=temp_dir, enable_persistence=True)
        
        # Start memory session
        session_id = brain1.start_memory_session("Basic persistence test")
        print(f"Started session: {session_id}")
        
        # Create some experiences
        for i in range(5):
            sensory_packet = SensoryPacket(
                sensor_values=[float(i), float(i*2), float(i*3)],
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=i + 1
            )
            mental_context = [1.0, 2.0, 3.0]
            brain1.process_sensory_input(sensory_packet, mental_context)
        
        # Check brain state before saving
        stats1 = brain1.get_brain_statistics()
        experiences_before = stats1['graph_stats']['total_nodes']
        adaptations_before = stats1['adaptive_tuning_stats']['total_adaptations']
        
        print(f"Before save: {experiences_before} experiences, {adaptations_before} adaptations")
        
        # Save current state
        save_result = brain1.save_current_state()
        print(f"Save result: {save_result}")
        
        # End session
        session_summary = brain1.end_memory_session()
        print(f"Session ended: {session_summary}")
        
        # Create new brain and load previous state
        predictor2 = MultiDrivePredictor(base_time_budget=0.05)
        brain2 = BrainInterface(predictor2, memory_path=temp_dir, enable_persistence=True)
        
        # Check loaded state
        stats2 = brain2.get_brain_statistics()
        experiences_after = stats2['graph_stats']['total_nodes']
        adaptations_after = stats2['adaptive_tuning_stats']['total_adaptations']
        
        print(f"After load: {experiences_after} experiences, {adaptations_after} adaptations")
        
        # Verify persistence worked
        persistence_success = (experiences_after == experiences_before and
                              experiences_after > 0)
        
        print(f"✅ Persistence successful: {persistence_success}")
        return persistence_success
    
    finally:
        # Clean up test directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cross_session_learning():
    """Test learning accumulation across multiple sessions."""
    print("\n🧠 Testing Cross-Session Learning")
    print("===============================")
    
    temp_dir = tempfile.mkdtemp(prefix="robot_memory_learning_")
    print(f"Using test directory: {temp_dir}")
    
    try:
        total_experiences = 0
        total_adaptations = 0
        
        # Session 1: Initial learning
        print("\n📚 Session 1: Initial Learning")
        predictor1 = MultiDrivePredictor(base_time_budget=0.05)
        brain1 = BrainInterface(predictor1, memory_path=temp_dir, enable_persistence=True)
        brain1.start_memory_session("Session 1: Initial learning")
        
        # Create experiences with pattern
        for i in range(3):
            sensory_packet = SensoryPacket(
                sensor_values=[1.0, 2.0, 3.0, float(i)],
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=i + 1
            )
            mental_context = [1.0, 1.0, 1.0, float(i)]
            brain1.process_sensory_input(sensory_packet, mental_context)
        
        stats1 = brain1.get_brain_statistics()
        session1_experiences = stats1['graph_stats']['total_nodes']
        session1_adaptations = stats1['adaptive_tuning_stats']['total_adaptations']
        total_experiences += session1_experiences
        total_adaptations += session1_adaptations
        
        print(f"Session 1: {session1_experiences} experiences, {session1_adaptations} adaptations")
        brain1.end_memory_session()
        
        # Session 2: Continued learning
        print("\n📚 Session 2: Continued Learning")
        predictor2 = MultiDrivePredictor(base_time_budget=0.05)
        brain2 = BrainInterface(predictor2, memory_path=temp_dir, enable_persistence=True)
        brain2.start_memory_session("Session 2: Continued learning")
        
        # Brain should have loaded previous experiences
        initial_stats2 = brain2.get_brain_statistics()
        loaded_experiences = initial_stats2['graph_stats']['total_nodes']
        loaded_adaptations = initial_stats2['adaptive_tuning_stats']['total_adaptations']
        
        print(f"Loaded from session 1: {loaded_experiences} experiences, {loaded_adaptations} adaptations")
        
        # Add more experiences
        for i in range(3, 6):
            sensory_packet = SensoryPacket(
                sensor_values=[1.0, 2.0, 3.0, float(i)],
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=i + 1
            )
            mental_context = [1.0, 1.0, 1.0, float(i)]
            brain2.process_sensory_input(sensory_packet, mental_context)
        
        final_stats2 = brain2.get_brain_statistics()
        final_experiences = final_stats2['graph_stats']['total_nodes']
        final_adaptations = final_stats2['adaptive_tuning_stats']['total_adaptations']
        
        print(f"Session 2 final: {final_experiences} experiences, {final_adaptations} adaptations")
        brain2.end_memory_session()
        
        # Verify accumulation
        experiences_accumulated = final_experiences >= loaded_experiences
        learning_continued = final_experiences > session1_experiences
        
        print(f"✅ Experiences accumulated: {experiences_accumulated}")
        print(f"✅ Learning continued across sessions: {learning_continued}")
        
        return experiences_accumulated and learning_continued
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_memory_archiving():
    """Test memory archiving and categorization."""
    print("\n🗂️  Testing Memory Archiving")
    print("==========================")
    
    temp_dir = tempfile.mkdtemp(prefix="robot_memory_archive_")
    
    try:
        predictor = MultiDrivePredictor(base_time_budget=0.05)
        brain = BrainInterface(predictor, memory_path=temp_dir, enable_persistence=True)
        brain.start_memory_session("Memory archiving test")
        
        # Create diverse experiences
        experiences_created = 0
        
        # High importance experiences (low prediction error, high frequency)
        for i in range(3):
            experience = ExperienceNode(
                mental_context=[1.0, 1.0, 1.0],  # Consistent spatial pattern
                action_taken={"forward_motor": 0.5},
                predicted_sensory=[2.0, 2.0, 2.0],
                actual_sensory=[2.05, 2.02, 1.98],  # Low prediction error
                prediction_error=0.05
            )
            experience.access_frequency = 8  # High frequency
            brain.world_graph.add_node(experience)
            experiences_created += 1
        
        # Spatial experiences
        for i in range(2):
            experience = ExperienceNode(
                mental_context=[float(i), float(i+1), 0.0, 1.0, 2.0, 3.0],  # Spatial-like context
                action_taken={"turn_motor": 0.3},
                predicted_sensory=[1.0, 1.0],
                actual_sensory=[1.2, 0.8],
                prediction_error=0.2
            )
            brain.world_graph.add_node(experience)
            experiences_created += 1
        
        # Motor skill experiences
        for i in range(2):
            experience = ExperienceNode(
                mental_context=[2.0, 2.0, 2.0],
                action_taken={"forward_motor": 0.8, "turn_motor": 0.1},  # Motor action
                predicted_sensory=[3.0, 3.0],
                actual_sensory=[3.1, 2.9],  # Good prediction
                prediction_error=0.1
            )
            brain.world_graph.add_node(experience)
            experiences_created += 1
        
        print(f"Created {experiences_created} diverse experiences")
        
        # Save and trigger archiving
        brain.save_current_state()
        
        # Test archive retrieval
        high_importance = brain.get_archived_experiences('high_importance')
        spatial_memories = brain.get_archived_experiences('spatial')
        skill_memories = brain.get_archived_experiences('skills')
        recent_memories = brain.get_archived_experiences('recent')
        
        print(f"Archived experiences:")
        print(f"  High importance: {len(high_importance)}")
        print(f"  Spatial memories: {len(spatial_memories)}")
        print(f"  Skill memories: {len(skill_memories)}")
        print(f"  Recent memories: {len(recent_memories)}")
        
        # Test memory statistics
        memory_stats = brain.get_memory_statistics()
        print(f"Memory statistics: {memory_stats.get('archive_summary', {})}")
        
        # Verify archiving worked
        archiving_success = (len(high_importance) > 0 or 
                           len(spatial_memories) > 0 or 
                           len(skill_memories) > 0) and len(recent_memories) > 0
        
        print(f"✅ Archiving successful: {archiving_success}")
        
        brain.end_memory_session()
        return archiving_success
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_2d_world_persistence():
    """Test persistence in the context of 2D world simulation."""
    print("\n🌍 Testing 2D World Persistence")
    print("=============================")
    
    temp_dir = tempfile.mkdtemp(prefix="robot_memory_2d_")
    
    try:
        # Session 1: Explore and learn
        print("\n🤖 Session 1: Initial Exploration")
        predictor1 = MultiDrivePredictor(base_time_budget=0.05)
        brain1 = BrainInterface(predictor1, memory_path=temp_dir, enable_persistence=True)
        brain1.start_memory_session("2D World Session 1: Initial exploration")
        
        # Simulate 2D world exploration with spatial sensory data
        spatial_experiences = []
        for x in range(3):
            for y in range(3):
                # Simulate spatial sensory input (position, walls, etc.)
                sensory_packet = SensoryPacket(
                    sensor_values=[
                        float(x), float(y),  # Position
                        1.0 if x == 0 else 0.0,  # Left wall
                        1.0 if x == 2 else 0.0,  # Right wall
                        1.0 if y == 0 else 0.0,  # Bottom wall
                        1.0 if y == 2 else 0.0,  # Top wall
                        0.8, 0.9  # Other sensors
                    ],
                    actuator_positions=[0.0, 0.0, 0.0],
                    timestamp=datetime.now(),
                    sequence_id=len(spatial_experiences) + 1
                )
                
                mental_context = [float(x), float(y), 0.5, 0.8, 0.7]
                brain1.process_sensory_input(sensory_packet, mental_context)
                spatial_experiences.append((x, y))
        
        session1_stats = brain1.get_brain_statistics()
        session1_experiences = session1_stats['graph_stats']['total_nodes']
        
        print(f"Session 1: Explored {len(spatial_experiences)} locations, {session1_experiences} experiences")
        
        # Save spatial memory
        brain1.save_current_state()
        brain1.end_memory_session()
        
        # Session 2: Return to same world
        print("\n🤖 Session 2: Return to Known World")
        predictor2 = MultiDrivePredictor(base_time_budget=0.05)
        brain2 = BrainInterface(predictor2, memory_path=temp_dir, enable_persistence=True)
        brain2.start_memory_session("2D World Session 2: Return to known world")
        
        # Check if spatial memories were loaded
        loaded_stats = brain2.get_brain_statistics()
        loaded_experiences = loaded_stats['graph_stats']['total_nodes']
        
        print(f"Session 2: Loaded {loaded_experiences} experiences from previous exploration")
        
        # Test searching for similar spatial experiences
        similar_experiences = brain2.search_similar_experiences([1.0, 1.0, 0.5, 0.8, 0.7])
        spatial_memories = brain2.get_archived_experiences('spatial')
        
        print(f"Found {len(similar_experiences)} similar experiences")
        print(f"Found {len(spatial_memories)} spatial memories in archive")
        
        # Add some new exploration
        for i in range(2):
            sensory_packet = SensoryPacket(
                sensor_values=[3.0, 3.0, 0.0, 1.0, 0.0, 1.0, 0.6, 0.7],  # New area
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=100 + i
            )
            mental_context = [3.0, 3.0, 0.6, 0.7, 0.8]
            brain2.process_sensory_input(sensory_packet, mental_context)
        
        final_stats = brain2.get_brain_statistics()
        final_experiences = final_stats['graph_stats']['total_nodes']
        
        print(f"Session 2 final: {final_experiences} total experiences")
        
        # Verify world persistence
        world_persistence_success = (loaded_experiences >= session1_experiences and
                                   final_experiences > loaded_experiences and
                                   len(spatial_memories) > 0)
        
        print(f"✅ 2D world persistence successful: {world_persistence_success}")
        
        brain2.end_memory_session()
        return world_persistence_success
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_storage_efficiency():
    """Test storage efficiency and compression."""
    print("\n💿 Testing Storage Efficiency")
    print("============================")
    
    temp_dir = tempfile.mkdtemp(prefix="robot_memory_storage_")
    
    try:
        predictor = MultiDrivePredictor(base_time_budget=0.05)
        brain = BrainInterface(predictor, memory_path=temp_dir, enable_persistence=True)
        brain.start_memory_session("Storage efficiency test")
        
        # Create many experiences to test storage
        num_experiences = 50
        for i in range(num_experiences):
            sensory_packet = SensoryPacket(
                sensor_values=[float(j) for j in range(10)],  # Large sensory vector
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=i + 1
            )
            mental_context = [float(j) for j in range(8)]
            brain.process_sensory_input(sensory_packet, mental_context)
        
        # Save and check storage
        brain.save_current_state()
        
        # Get storage statistics
        memory_stats = brain.get_memory_statistics()
        storage_usage = memory_stats.get('storage_usage', {})
        
        total_bytes = storage_usage.get('total_bytes', 0)
        graphs_bytes = storage_usage.get('graphs_bytes', 0)
        
        print(f"Storage usage:")
        print(f"  Total: {total_bytes} bytes")
        print(f"  Graphs: {graphs_bytes} bytes")
        print(f"  Experiences: {num_experiences}")
        print(f"  Bytes per experience: {graphs_bytes / num_experiences:.1f}")
        
        # Check that compression is working (storage should be reasonable)
        reasonable_storage = total_bytes < 1000000  # Less than 1MB for 50 experiences
        storage_exists = total_bytes > 0
        
        print(f"✅ Storage exists: {storage_exists}")
        print(f"✅ Storage reasonable: {reasonable_storage}")
        
        brain.end_memory_session()
        return storage_exists and reasonable_storage
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all persistent memory tests."""
    print("💾 Persistent Memory System Test Suite")
    print("=====================================")
    print("Testing lifelong learning and experience accumulation:")
    print("• Basic save and load functionality")
    print("• Cross-session learning accumulation")
    print("• Memory archiving and categorization")
    print("• 2D world spatial memory persistence")
    print("• Storage efficiency and compression")
    print()
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Basic persistence
    try:
        if test_basic_persistence():
            tests_passed += 1
            print("✅ Basic persistence - PASSED")
    except Exception as e:
        print(f"❌ Basic persistence - FAILED: {e}")
    
    # Test 2: Cross-session learning
    try:
        if test_cross_session_learning():
            tests_passed += 1
            print("✅ Cross-session learning - PASSED")
    except Exception as e:
        print(f"❌ Cross-session learning - FAILED: {e}")
    
    # Test 3: Memory archiving
    try:
        if test_memory_archiving():
            tests_passed += 1
            print("✅ Memory archiving - PASSED")
    except Exception as e:
        print(f"❌ Memory archiving - FAILED: {e}")
    
    # Test 4: 2D world persistence
    try:
        if test_2d_world_persistence():
            tests_passed += 1
            print("✅ 2D world persistence - PASSED")
    except Exception as e:
        print(f"❌ 2D world persistence - FAILED: {e}")
    
    # Test 5: Storage efficiency
    try:
        if test_storage_efficiency():
            tests_passed += 1
            print("✅ Storage efficiency - PASSED")
    except Exception as e:
        print(f"❌ Storage efficiency - FAILED: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("\n🎉 All persistent memory tests passed!")
        print("✅ The brain now has:")
        print("   • Lifelong learning capability")
        print("   • Experience accumulation across sessions")
        print("   • Intelligent memory archiving and categorization")
        print("   • Spatial memory for navigation")
        print("   • Efficient storage with compression")
        print("🧠 The robot can now build expertise and wisdom over time!")
        print("🌍 2D world robots will remember locations, paths, and strategies!")
    else:
        print("⚠️  Some persistent memory tests failed. The system may need refinement.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌟 Persistent memory system is fully operational!")
        print("🧠 The robot brain can now accumulate lifelong learning!")
        print("💾 Ready for 2TB of accumulated wisdom!")
    else:
        print("\n🔧 Persistent memory system needs debugging")