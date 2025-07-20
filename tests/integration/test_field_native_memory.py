#!/usr/bin/env python3
"""
Test Field-Native Memory System: Phase B3

Comprehensive testing of the field-native memory system including formation,
retrieval, consolidation, and integration with the field brain.

This validates that field topology can serve as persistent memory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

import torch
import numpy as np
import time
import math
import tempfile
from typing import List, Dict, Any, Tuple

# Import field-native components
try:
    from field_native_brain import create_unified_field_brain
    from field_native_memory import FieldNativeMemorySystem, FieldMemoryType, create_field_native_memory_system
    from field_native_robot_interface import create_field_native_robot_interface
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure field-native modules are in server/src/")
    sys.exit(1)


def generate_test_field_state(pattern_type: str, variation: float = 0.0, field_dims: int = 37) -> torch.Tensor:
    """Generate test field states with different patterns."""
    
    if pattern_type == "oscillatory":
        # Create oscillatory pattern (strong in oscillatory dimensions 5-10)
        field = torch.zeros(field_dims)
        field[5:11] = torch.tensor([0.8, 0.6, 0.9, 0.7, 0.5, 0.8]) + torch.randn(6) * variation
        field[11:19] = torch.randn(8) * 0.2  # Small flow component
        return field
    
    elif pattern_type == "flow":
        # Create flow pattern (strong in flow dimensions 11-18)
        field = torch.zeros(field_dims)
        field[11:19] = torch.tensor([0.9, 0.7, 0.8, 0.6, 0.9, 0.5, 0.7, 0.8]) + torch.randn(8) * variation
        field[5:11] = torch.randn(6) * 0.2  # Small oscillatory component
        return field
    
    elif pattern_type == "topology":
        # Create topology pattern (strong in topology dimensions 19-24)
        field = torch.zeros(field_dims)
        field[19:25] = torch.tensor([0.8, 0.9, 0.7, 0.6, 0.8, 0.5]) + torch.randn(6) * variation
        field[25:29] = torch.randn(4) * 0.2  # Small energy component
        return field
    
    elif pattern_type == "energy":
        # Create energy pattern (strong in energy dimensions 25-28)
        field = torch.zeros(field_dims)
        field[25:29] = torch.tensor([0.9, 0.8, 0.7, 0.6]) + torch.randn(4) * variation
        field[19:25] = torch.randn(6) * 0.2  # Small topology component
        return field
    
    elif pattern_type == "random":
        # Random pattern
        return torch.randn(field_dims) + torch.ones(field_dims) * variation
    
    else:
        # Default neutral pattern
        return torch.ones(field_dims) * 0.5 + torch.randn(field_dims) * variation


def test_memory_formation():
    """Test basic memory formation from field states."""
    print("🧠 TESTING MEMORY FORMATION")
    
    # Create field brain and memory system
    field_brain = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
    memory_system = create_field_native_memory_system(field_brain, memory_capacity=100)
    
    print(f"   Created memory system with {memory_system.memory_capacity} capacity")
    
    # Test forming different types of memories
    test_cases = [
        ("oscillatory", FieldMemoryType.EXPERIENCE, 1.0),
        ("flow", FieldMemoryType.SKILL, 1.2),
        ("topology", FieldMemoryType.CONCEPT, 0.8),
        ("energy", FieldMemoryType.REFLEX, 1.5),
        ("random", FieldMemoryType.WORKING, 0.6)
    ]
    
    formed_memories = []
    
    for pattern_type, memory_type, importance in test_cases:
        field_state = generate_test_field_state(pattern_type, field_dims=field_brain.total_dimensions)
        memory_id = memory_system.form_memory(field_state, memory_type, importance)
        
        if memory_id >= 0:
            formed_memories.append((memory_id, pattern_type, memory_type))
            print(f"      Formed {memory_type.value} memory #{memory_id} from {pattern_type} pattern")
        else:
            print(f"      ⚠️ Failed to form memory from {pattern_type} pattern")
    
    # Test duplicate memory rejection
    duplicate_field = generate_test_field_state("oscillatory", field_dims=field_brain.total_dimensions)  # Same as first
    duplicate_id = memory_system.form_memory(duplicate_field, FieldMemoryType.EXPERIENCE, 1.0)
    
    print(f"   Memory formation results:")
    print(f"      Unique memories formed: {len(formed_memories)}")
    print(f"      Duplicate rejected: {'✅ YES' if duplicate_id == -1 else '⚠️ NO'}")
    print(f"      Total memories: {len(memory_system.memory_traces)}")
    
    return {
        'memory_system': memory_system,
        'formed_memories': formed_memories,
        'formation_success': len(formed_memories) >= 4,
        'duplicate_rejection': duplicate_id == -1
    }


def test_memory_retrieval():
    """Test memory retrieval through field resonance."""
    print("\n🔍 TESTING MEMORY RETRIEVAL")
    
    # Create memory system and form some memories
    field_brain = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
    memory_system = create_field_native_memory_system(field_brain, memory_capacity=100)
    
    # Form memories with known patterns
    original_patterns = {}
    memory_ids = []
    
    for i, pattern_type in enumerate(["oscillatory", "flow", "topology", "energy"]):
        field_state = generate_test_field_state(pattern_type, field_dims=field_brain.total_dimensions)
        memory_id = memory_system.form_memory(field_state, FieldMemoryType.EXPERIENCE, 1.0)
        original_patterns[pattern_type] = field_state
        memory_ids.append((memory_id, pattern_type))
    
    print(f"   Formed {len(memory_ids)} test memories")
    
    # Test retrieval with similar patterns
    retrieval_results = []
    
    for pattern_type, original_field in original_patterns.items():
        # Create similar but not identical query
        query_field = generate_test_field_state(pattern_type, variation=0.1, field_dims=field_brain.total_dimensions)
        
        retrieved = memory_system.retrieve_memories(query_field, max_memories=3, similarity_threshold=0.2)
        
        print(f"   Query {pattern_type}:")
        print(f"      Retrieved {len(retrieved)} memories")
        
        if retrieved:
            best_memory_id, best_similarity = retrieved[0]
            print(f"      Best match: memory #{best_memory_id} (similarity: {best_similarity:.3f})")
            
            # Check if the best match is correct type
            correct_type = False
            for mid, mtype in memory_ids:
                if mid == best_memory_id and mtype == pattern_type:
                    correct_type = True
                    break
            
            retrieval_results.append({
                'pattern_type': pattern_type,
                'retrieved_count': len(retrieved),
                'best_similarity': best_similarity,
                'correct_type': correct_type
            })
        else:
            print(f"      ⚠️ No memories retrieved")
            retrieval_results.append({
                'pattern_type': pattern_type,
                'retrieved_count': 0,
                'best_similarity': 0.0,
                'correct_type': False
            })
    
    # Test retrieval with completely different pattern
    random_query = generate_test_field_state("random", variation=1.0)
    random_retrieved = memory_system.retrieve_memories(random_query, max_memories=3, similarity_threshold=0.2)
    
    print(f"   Random query retrieved: {len(random_retrieved)} memories")
    
    # Analysis
    successful_retrievals = sum(1 for r in retrieval_results if r['retrieved_count'] > 0)
    correct_type_retrievals = sum(1 for r in retrieval_results if r['correct_type'])
    avg_similarity = np.mean([r['best_similarity'] for r in retrieval_results if r['retrieved_count'] > 0])
    
    print(f"   Retrieval analysis:")
    print(f"      Successful retrievals: {successful_retrievals}/{len(retrieval_results)}")
    print(f"      Correct type matches: {correct_type_retrievals}/{len(retrieval_results)}")
    print(f"      Average similarity: {avg_similarity:.3f}")
    
    return {
        'memory_system': memory_system,
        'retrieval_results': retrieval_results,
        'successful_retrievals': successful_retrievals,
        'correct_type_retrievals': correct_type_retrievals,
        'avg_similarity': avg_similarity,
        'retrieval_success': successful_retrievals >= 3
    }


def test_memory_consolidation():
    """Test memory consolidation and forgetting processes."""
    print("\n🌙 TESTING MEMORY CONSOLIDATION")
    
    # Create memory system with fast forgetting for testing
    field_brain = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
    memory_system = create_field_native_memory_system(
        field_brain, 
        memory_capacity=50,
        forgetting_rate=0.1  # Fast forgetting for testing
    )
    
    # Form many memories with different importance levels
    print(f"   Forming memories with varying importance:")
    
    important_memories = []
    unimportant_memories = []
    
    for i in range(20):
        pattern_type = ["oscillatory", "flow", "topology", "energy"][i % 4]
        field_state = generate_test_field_state(pattern_type, variation=i * 0.1)
        
        # Alternate between important and unimportant memories
        importance = 1.5 if i % 2 == 0 else 0.3
        memory_type = FieldMemoryType.SKILL if importance > 1.0 else FieldMemoryType.WORKING
        
        memory_id = memory_system.form_memory(field_state, memory_type, importance)
        
        if importance > 1.0:
            important_memories.append(memory_id)
        else:
            unimportant_memories.append(memory_id)
    
    initial_count = len(memory_system.memory_traces)
    print(f"      Total memories formed: {initial_count}")
    print(f"      Important memories: {len(important_memories)}")
    print(f"      Unimportant memories: {len(unimportant_memories)}")
    
    # Simulate time passage and access some memories
    print(f"   Simulating memory access and aging:")
    
    # Access important memories multiple times
    for memory_id in important_memories[:5]:
        if memory_id in memory_system.memory_traces:
            query_field = memory_system.memory_traces[memory_id].field_coordinates + torch.randn(field_brain.total_dimensions) * 0.1
            memory_system.retrieve_memories(query_field, max_memories=3)
    
    # Perform consolidation
    print(f"   Performing memory consolidation:")
    memory_system.consolidate_memories(force_consolidation=True)
    
    after_consolidation_count = len(memory_system.memory_traces)
    
    # Check which memories survived
    important_survived = sum(1 for mid in important_memories if mid in memory_system.memory_traces)
    unimportant_survived = sum(1 for mid in unimportant_memories if mid in memory_system.memory_traces)
    
    print(f"      Memories after consolidation: {after_consolidation_count}")
    print(f"      Important memories survived: {important_survived}/{len(important_memories)}")
    print(f"      Unimportant memories survived: {unimportant_survived}/{len(unimportant_memories)}")
    
    # Test sleep mode consolidation
    print(f"   Testing sleep mode consolidation:")
    memory_system.enter_sleep_mode()
    memory_system.exit_sleep_mode()
    
    final_count = len(memory_system.memory_traces)
    print(f"      Memories after sleep: {final_count}")
    
    consolidation_success = (important_survived > unimportant_survived and 
                           memory_system.sleep_cycles_completed > 0)
    
    return {
        'memory_system': memory_system,
        'initial_count': initial_count,
        'after_consolidation_count': after_consolidation_count,
        'final_count': final_count,
        'important_survived': important_survived,
        'unimportant_survived': unimportant_survived,
        'consolidation_success': consolidation_success
    }


def test_memory_persistence():
    """Test memory saving and loading."""
    print("\n💾 TESTING MEMORY PERSISTENCE")
    
    # Create memory system and form memories
    field_brain = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
    memory_system = create_field_native_memory_system(field_brain, memory_capacity=100)
    
    # Form some memorable patterns
    test_patterns = []
    for i, pattern_type in enumerate(["oscillatory", "flow", "topology"]):
        field_state = generate_test_field_state(pattern_type)
        memory_id = memory_system.form_memory(field_state, FieldMemoryType.CONCEPT, 1.0)
        test_patterns.append((memory_id, pattern_type, field_state))
    
    original_count = len(memory_system.memory_traces)
    original_stats = memory_system.get_memory_statistics()
    
    print(f"   Original state: {original_count} memories")
    
    # Save memory state
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mem.gz') as tmp_file:
        memory_file = tmp_file.name
    
    try:
        memory_system.save_memory_state(memory_file)
        print(f"   Saved memory state to temporary file")
        
        # Create new memory system and load
        field_brain2 = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
        memory_system2 = create_field_native_memory_system(field_brain2, memory_capacity=100)
        
        load_success = memory_system2.load_memory_state(memory_file)
        loaded_count = len(memory_system2.memory_traces)
        loaded_stats = memory_system2.get_memory_statistics()
        
        print(f"   Load success: {'✅ YES' if load_success else '❌ NO'}")
        print(f"   Loaded state: {loaded_count} memories")
        
        # Test that loaded memories can be retrieved
        retrieval_test_passed = True
        for memory_id, pattern_type, original_field in test_patterns:
            if memory_id in memory_system2.memory_traces:
                # Try to retrieve this memory
                query_field = original_field + torch.randn(field_brain.total_dimensions) * 0.1
                retrieved = memory_system2.retrieve_memories(query_field, max_memories=1)
                
                if not retrieved or retrieved[0][0] != memory_id:
                    retrieval_test_passed = False
                    break
            else:
                retrieval_test_passed = False
                break
        
        print(f"   Memory retrieval after load: {'✅ PASSED' if retrieval_test_passed else '⚠️ FAILED'}")
        
        persistence_success = (load_success and 
                             loaded_count == original_count and 
                             retrieval_test_passed)
        
        return {
            'save_success': True,
            'load_success': load_success,
            'original_count': original_count,
            'loaded_count': loaded_count,
            'retrieval_test_passed': retrieval_test_passed,
            'persistence_success': persistence_success
        }
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(memory_file)
        except:
            pass


def test_memory_influence():
    """Test how memory influences current field dynamics."""
    print("\n🔗 TESTING MEMORY INFLUENCE")
    
    # Create memory system and form strong memories
    field_brain = create_unified_field_brain(spatial_resolution=6, quiet_mode=True)
    memory_system = create_field_native_memory_system(field_brain, memory_capacity=100)
    
    # Form strong memory patterns
    strong_patterns = {}
    for pattern_type in ["oscillatory", "flow", "topology"]:
        field_state = generate_test_field_state(pattern_type)
        memory_id = memory_system.form_memory(field_state, FieldMemoryType.CONCEPT, 2.0)  # High importance
        strong_patterns[pattern_type] = field_state
        
        # Access the memory multiple times to strengthen it
        for _ in range(5):
            query = field_state + torch.randn(field_brain.total_dimensions) * 0.05
            memory_system.retrieve_memories(query, max_memories=1)
    
    print(f"   Formed {len(strong_patterns)} strong memory patterns")
    
    # Test memory influence on similar field states
    influence_results = []
    
    for pattern_type, original_field in strong_patterns.items():
        # Create partially similar field state
        test_field = generate_test_field_state(pattern_type, variation=0.3)
        
        # Update memory influence
        memory_system.update_memory_influence(test_field)
        
        # Get memory-influenced field
        influenced_field = memory_system.get_memory_influenced_field(test_field)
        
        # Calculate influence strength
        influence_vector = influenced_field - test_field
        influence_strength = torch.norm(influence_vector).item()
        
        # Calculate direction of influence (toward original memory?)
        if torch.norm(original_field) > 0 and torch.norm(test_field) > 0:
            direction_to_memory = original_field - test_field
            direction_to_memory = direction_to_memory / torch.norm(direction_to_memory)
            
            if torch.norm(influence_vector) > 0:
                influence_direction = influence_vector / torch.norm(influence_vector)
                alignment = torch.dot(direction_to_memory, influence_direction).item()
            else:
                alignment = 0.0
        else:
            alignment = 0.0
        
        influence_results.append({
            'pattern_type': pattern_type,
            'influence_strength': influence_strength,
            'direction_alignment': alignment
        })
        
        print(f"   {pattern_type} influence:")
        print(f"      Strength: {influence_strength:.4f}")
        print(f"      Direction alignment: {alignment:.3f}")
    
    # Test influence on completely different field
    random_field = generate_test_field_state("random", variation=1.0)
    memory_system.update_memory_influence(random_field)
    random_influenced = memory_system.get_memory_influenced_field(random_field)
    random_influence = torch.norm(random_influenced - random_field).item()
    
    print(f"   Random field influence: {random_influence:.4f}")
    
    # Analysis
    avg_influence = np.mean([r['influence_strength'] for r in influence_results])
    avg_alignment = np.mean([r['direction_alignment'] for r in influence_results])
    
    influence_success = (avg_influence > 0.01 and  # Some influence exists
                        avg_alignment > 0.1 and   # Generally toward memories
                        random_influence < avg_influence)  # Less influence on random
    
    print(f"   Memory influence analysis:")
    print(f"      Average influence strength: {avg_influence:.4f}")
    print(f"      Average direction alignment: {avg_alignment:.3f}")
    print(f"      Influence working: {'✅ YES' if influence_success else '⚠️ WEAK'}")
    
    return {
        'memory_system': memory_system,
        'influence_results': influence_results,
        'avg_influence': avg_influence,
        'avg_alignment': avg_alignment,
        'random_influence': random_influence,
        'influence_success': influence_success
    }


def run_comprehensive_memory_test():
    """Run comprehensive test of field-native memory system."""
    print("🧠 COMPREHENSIVE FIELD-NATIVE MEMORY SYSTEM TEST")
    print("=" * 70)
    
    print(f"🎯 Phase B3: Field-Native Memory and Persistence")
    print(f"   Testing memory formation, retrieval, consolidation, and influence")
    
    # Test 1: Memory Formation
    formation_results = test_memory_formation()
    
    # Test 2: Memory Retrieval
    retrieval_results = test_memory_retrieval()
    
    # Test 3: Memory Consolidation
    consolidation_results = test_memory_consolidation()
    
    # Test 4: Memory Persistence
    persistence_results = test_memory_persistence()
    
    # Test 5: Memory Influence
    influence_results = test_memory_influence()
    
    # Comprehensive Analysis
    print(f"\n📊 COMPREHENSIVE MEMORY SYSTEM ANALYSIS")
    
    print(f"\n   🧠 Memory Formation:")
    print(f"      Formation success: {'✅ YES' if formation_results['formation_success'] else '⚠️ NO'}")
    print(f"      Duplicate rejection: {'✅ YES' if formation_results['duplicate_rejection'] else '⚠️ NO'}")
    print(f"      Memories formed: {len(formation_results['formed_memories'])}")
    
    print(f"\n   🔍 Memory Retrieval:")
    print(f"      Retrieval success: {'✅ YES' if retrieval_results['retrieval_success'] else '⚠️ NO'}")
    print(f"      Successful retrievals: {retrieval_results['successful_retrievals']}/4")
    print(f"      Correct type matches: {retrieval_results['correct_type_retrievals']}/4")
    print(f"      Average similarity: {retrieval_results['avg_similarity']:.3f}")
    
    print(f"\n   🌙 Memory Consolidation:")
    print(f"      Consolidation working: {'✅ YES' if consolidation_results['consolidation_success'] else '⚠️ NO'}")
    print(f"      Important survived: {consolidation_results['important_survived']}")
    print(f"      Unimportant survived: {consolidation_results['unimportant_survived']}")
    print(f"      Memory reduction: {consolidation_results['initial_count']} → {consolidation_results['final_count']}")
    
    print(f"\n   💾 Memory Persistence:")
    print(f"      Save/load working: {'✅ YES' if persistence_results['persistence_success'] else '⚠️ NO'}")
    print(f"      Load success: {'✅ YES' if persistence_results['load_success'] else '❌ NO'}")
    print(f"      Memory count preserved: {persistence_results['original_count']} → {persistence_results['loaded_count']}")
    
    print(f"\n   🔗 Memory Influence:")
    print(f"      Influence working: {'✅ YES' if influence_results['influence_success'] else '⚠️ WEAK'}")
    print(f"      Average influence: {influence_results['avg_influence']:.4f}")
    print(f"      Direction alignment: {influence_results['avg_alignment']:.3f}")
    
    # Overall Assessment
    success_metrics = [
        formation_results['formation_success'],
        retrieval_results['retrieval_success'],
        consolidation_results['consolidation_success'],
        persistence_results['persistence_success'],
        influence_results['influence_success']
    ]
    
    success_count = sum(success_metrics)
    success_rate = success_count / len(success_metrics)
    
    print(f"\n   🌟 OVERALL MEMORY SYSTEM ASSESSMENT:")
    print(f"      Success metrics: {success_count}/{len(success_metrics)}")
    print(f"      Overall success rate: {success_rate:.3f}")
    print(f"      Field-native memory: {'✅ FULLY FUNCTIONAL' if success_rate >= 0.8 else '⚠️ DEVELOPING'}")
    
    if success_rate >= 0.8:
        print(f"\n🚀 PHASE B3: FIELD-NATIVE MEMORY SUCCESSFULLY IMPLEMENTED!")
        print(f"🎯 Key achievements:")
        print(f"   ✓ Memory formation from field topology")
        print(f"   ✓ Memory retrieval through field resonance")
        print(f"   ✓ Biological consolidation and forgetting")
        print(f"   ✓ Persistent memory storage with compression")
        print(f"   ✓ Memory influence on current field dynamics")
        print(f"   ✓ Field-native memory system fully operational!")
    else:
        print(f"\n⚠️ Phase B3 field-native memory system still developing")
        print(f"🔧 Areas needing improvement:")
        if not formation_results['formation_success']:
            print(f"   • Memory formation from field states")
        if not retrieval_results['retrieval_success']:
            print(f"   • Memory retrieval accuracy")
        if not consolidation_results['consolidation_success']:
            print(f"   • Memory consolidation effectiveness")
        if not persistence_results['persistence_success']:
            print(f"   • Memory persistence and loading")
        if not influence_results['influence_success']:
            print(f"   • Memory influence on field dynamics")
    
    return {
        'formation_results': formation_results,
        'retrieval_results': retrieval_results,
        'consolidation_results': consolidation_results,
        'persistence_results': persistence_results,
        'influence_results': influence_results,
        'success_rate': success_rate,
        'phase_b3_achieved': success_rate >= 0.8
    }


if __name__ == "__main__":
    # Run the comprehensive field-native memory test
    results = run_comprehensive_memory_test()
    
    print(f"\n🔬 PHASE B3 VALIDATION SUMMARY:")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Memory formation: {'✅' if results['formation_results']['formation_success'] else '⚠️'}")
    print(f"   Memory retrieval: {'✅' if results['retrieval_results']['retrieval_success'] else '⚠️'}")
    print(f"   Memory consolidation: {'✅' if results['consolidation_results']['consolidation_success'] else '⚠️'}")
    print(f"   Memory persistence: {'✅' if results['persistence_results']['persistence_success'] else '⚠️'}")
    print(f"   Memory influence: {'✅' if results['influence_results']['influence_success'] else '⚠️'}")
    print(f"   Field-native memory: {'✅ ACHIEVED' if results['phase_b3_achieved'] else '⚠️ DEVELOPING'}")
    
    if results['phase_b3_achieved']:
        print(f"\n🌊 Phase B3 FIELD-NATIVE MEMORY SUCCESSFULLY DEMONSTRATED!")
        print(f"🎉 Ready for Phase B4: Complete Field-Native Intelligence System!")
        print(f"🧠 Continuous field intelligence now has persistent memory!")