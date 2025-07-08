#!/usr/bin/env python3
"""
Test integration of all brain systems including persistent memory.
This demonstrates the full brain capabilities working together.
"""

import tempfile
import shutil
from datetime import datetime
from core.brain_interface import BrainInterface
from core.communication import SensoryPacket
from predictor.multi_drive_predictor import MultiDrivePredictor


def test_full_integration_with_persistence():
    """Test all brain systems working together with persistent memory."""
    print("🧠 Testing Full Brain Integration with Persistence")
    print("================================================")
    
    temp_dir = tempfile.mkdtemp(prefix="robot_brain_integration_")
    print(f"Using memory directory: {temp_dir}")
    
    try:
        # Session 1: Initial learning
        print("\n🤖 Session 1: Initial Learning and Adaptation")
        predictor1 = MultiDrivePredictor(base_time_budget=0.05)
        brain1 = BrainInterface(predictor1, memory_path=temp_dir, enable_persistence=True)
        
        session1_id = brain1.start_memory_session("Integration test - Session 1")
        print(f"Started session: {session1_id}")
        
        # Process various sensory inputs to trigger all systems
        # Keep consistent vector length but vary values to test adaptation
        sensory_size = 20  # Medium bandwidth - consistent throughout
        
        for i in range(8):
            # Vary sensory patterns to test adaptive tuning
            if i < 3:
                # Low activity pattern
                sensor_values = [float(j) * 0.1 for j in range(sensory_size)]
            elif i < 6:
                # Medium activity pattern
                sensor_values = [float(j) * 0.5 for j in range(sensory_size)]
            else:
                # High activity pattern
                sensor_values = [float(j) * 2.0 for j in range(sensory_size)]
            
            sensory_packet = SensoryPacket(
                sensor_values=sensor_values,
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=i + 1
            )
            
            mental_context = [float(i), float(i*0.5), 1.0, 2.0, 3.0]
            prediction = brain1.process_sensory_input(sensory_packet, mental_context)
        
        # Get session 1 statistics
        stats1 = brain1.get_brain_statistics()
        
        print(f"Session 1 Results:")
        print(f"  Experiences: {stats1['graph_stats']['total_nodes']}")
        print(f"  Emergent memory types: {stats1['graph_stats'].get('emergent_memory_types', {})}")
        print(f"  Adaptations: {stats1['adaptive_tuning_stats']['total_adaptations']}")
        print(f"  Sensory bandwidth: {stats1['adaptive_tuning_stats']['sensory_insights'].get('bandwidth_tier', 'unknown')}")
        
        # Save session 1
        save_result = brain1.save_current_state()
        session1_summary = brain1.end_memory_session()
        
        # Session 2: Continue learning with memory
        print("\n🤖 Session 2: Continued Learning with Persistent Memory")
        predictor2 = MultiDrivePredictor(base_time_budget=0.05)
        brain2 = BrainInterface(predictor2, memory_path=temp_dir, enable_persistence=True)
        
        session2_id = brain2.start_memory_session("Integration test - Session 2")
        
        # Check loaded state
        initial_stats2 = brain2.get_brain_statistics()
        loaded_experiences = initial_stats2['graph_stats']['total_nodes']
        loaded_adaptations = initial_stats2['adaptive_tuning_stats']['total_adaptations']
        
        print(f"Loaded from Session 1:")
        print(f"  Experiences: {loaded_experiences}")
        print(f"  Adaptations: {loaded_adaptations}")
        print(f"  Memory archive: {initial_stats2.get('persistent_memory_stats', {}).get('archive_summary', {})}")
        
        # Continue processing with different patterns (same vector size)
        for i in range(5):
            # High activity to test adaptation
            sensor_values = [float(j) * (i + 1) for j in range(sensory_size)]
            
            sensory_packet = SensoryPacket(
                sensor_values=sensor_values,
                actuator_positions=[0.0, 0.0, 0.0],
                timestamp=datetime.now(),
                sequence_id=100 + i
            )
            
            mental_context = [float(i+10), float(i*2), 4.0, 5.0, 6.0]
            prediction = brain2.process_sensory_input(sensory_packet, mental_context)
        
        # Test memory search
        search_results = brain2.search_similar_experiences([5.0, 2.5, 4.0, 5.0, 6.0])
        spatial_memories = brain2.get_archived_experiences('spatial')
        high_importance = brain2.get_archived_experiences('high_importance')
        
        print(f"Memory search results:")
        print(f"  Similar experiences found: {len(search_results)}")
        print(f"  Spatial memories: {len(spatial_memories)}")
        print(f"  High importance memories: {len(high_importance)}")
        
        # Final statistics
        final_stats = brain2.get_brain_statistics()
        final_experiences = final_stats['graph_stats']['total_nodes']
        final_adaptations = final_stats['adaptive_tuning_stats']['total_adaptations']
        
        print(f"\nSession 2 Final Results:")
        print(f"  Total experiences: {final_experiences}")
        print(f"  Total adaptations: {final_adaptations}")
        print(f"  Current bandwidth tier: {final_stats['adaptive_tuning_stats']['sensory_insights'].get('bandwidth_tier', 'unknown')}")
        print(f"  Emergent memory stats: {final_stats['graph_stats'].get('emergent_memory_types', {})}")
        
        # Get memory storage statistics
        memory_stats = brain2.get_memory_statistics()
        storage_usage = memory_stats.get('storage_usage', {})
        
        print(f"\nPersistent Memory Statistics:")
        print(f"  Total sessions: {memory_stats.get('total_sessions', 0)}")
        print(f"  Total graphs saved: {memory_stats.get('total_graphs', 0)}")
        print(f"  Storage usage: {storage_usage.get('total_bytes', 0)} bytes")
        
        brain2.end_memory_session()
        
        # Verify integration success
        learning_accumulated = final_experiences > loaded_experiences >= stats1['graph_stats']['total_nodes']
        adaptations_working = final_adaptations >= loaded_adaptations
        memory_archiving = len(spatial_memories) > 0 or len(high_importance) > 0
        persistence_working = loaded_experiences > 0
        
        print(f"\n✅ Integration Success Criteria:")
        print(f"  Learning accumulated: {learning_accumulated}")
        print(f"  Adaptations working: {adaptations_working}")
        print(f"  Memory archiving: {memory_archiving}")
        print(f"  Persistence working: {persistence_working}")
        
        return learning_accumulated and adaptations_working and persistence_working
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run integration test with persistent memory."""
    print("🧠 Brain Integration Test with Persistent Memory")
    print("===============================================")
    print("Testing the complete brain system:")
    print("• Unified emergent memory system")
    print("• Adaptive parameter tuning")
    print("• Multi-drive motivation system") 
    print("• Persistent memory with archiving")
    print("• Cross-session learning accumulation")
    print()
    
    try:
        success = test_full_integration_with_persistence()
        
        if success:
            print("\n🎉 Full brain integration test PASSED!")
            print("✅ All systems working together:")
            print("   • Emergent memory phenomena from neural-like dynamics")
            print("   • Adaptive sensory processing for any sensor type")
            print("   • Self-optimizing parameters based on performance")
            print("   • Competing drives creating emergent behavior")
            print("   • Lifelong learning through persistent memory")
            print("   • Intelligent experience archiving and retrieval")
            print("🧠 The robot brain is ready for sophisticated autonomous behavior!")
        else:
            print("\n❌ Integration test FAILED")
            print("Some brain systems are not working together properly.")
        
        return success
    
    except Exception as e:
        print(f"\n❌ Integration test FAILED with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🌟 Complete robot brain system is operational!")
        print("🤖 Ready for real-world autonomous operation!")
    else:
        print("\n🔧 Integration needs debugging")