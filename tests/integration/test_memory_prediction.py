#!/usr/bin/env python3
"""
Test framework for memory and prediction capabilities of the unified field brain.

Memory and prediction are fundamentally linked:
- Memory: Can the brain recall patterns it has seen before?
- Prediction: Can the brain anticipate what comes next based on memory?
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import numpy as np
import torch
from server.src.brains.field.core_brain import UnifiedFieldBrain

class MemoryPredictionTester:
    """Test framework for evaluating field-based memory and prediction."""
    
    def __init__(self, spatial_resolution=8, quiet_mode=True):
        self.brain = UnifiedFieldBrain(spatial_resolution=spatial_resolution, quiet_mode=quiet_mode)
        self.test_results = {}
    
    def test_pattern_recall(self):
        """Test 1: Can the brain recall a pattern after seeing it multiple times?"""
        print("\n=== TEST 1: Pattern Recall ===")
        
        # Create two distinct patterns
        pattern_a = [0.9, 0.1] * 12  # Alternating high-low
        pattern_b = [0.1, 0.9] * 12  # Opposite pattern
        
        # Train on pattern A
        print("Training on pattern A (5 presentations)...")
        for i in range(5):
            action, _ = self.brain.process_robot_cycle(pattern_a)
        
        initial_regions = len(self.brain.topology_regions)
        print(f"Topology regions after training: {initial_regions}")
        
        # Test recall with partial pattern
        partial_a = pattern_a[:12] + [0.5] * 12  # First half of pattern A
        print("\nPresenting partial pattern A...")
        action_recall, _ = self.brain.process_robot_cycle(partial_a)
        
        # Check if action suggests completion of pattern A
        # Strong positive values in first half of action would suggest recall
        recall_strength = np.mean(action_recall[:12]) - np.mean(action_recall[12:])
        print(f"Recall strength: {recall_strength:.3f}")
        
        success = recall_strength > 0.1 and initial_regions > 0
        print(f"Pattern recall: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results['pattern_recall'] = success
        return success
    
    def test_sequence_prediction(self):
        """Test 2: Can the brain predict the next item in a sequence?"""
        print("\n=== TEST 2: Sequence Prediction ===")
        
        # Create a simple repeating sequence
        sequence = [
            [0.9, 0.1, 0.1, 0.1] + [0.5] * 20,  # State 1
            [0.1, 0.9, 0.1, 0.1] + [0.5] * 20,  # State 2  
            [0.1, 0.1, 0.9, 0.1] + [0.5] * 20,  # State 3
            [0.1, 0.1, 0.1, 0.9] + [0.5] * 20,  # State 4
        ]
        
        # Train on sequence (3 full cycles)
        print("Training on sequence (3 cycles)...")
        for cycle in range(3):
            for state in sequence:
                self.brain.process_robot_cycle(state)
        
        # Test prediction: present states 1,2,3 and see if brain predicts state 4
        print("\nTesting prediction...")
        predictions = []
        
        for i, state in enumerate(sequence[:3]):
            action, _ = self.brain.process_robot_cycle(state)
            # Look at which "channel" has highest activation
            pred_channel = np.argmax(action[:4])
            predictions.append(pred_channel)
            print(f"  After state {i+1}, predicted channel: {pred_channel}")
        
        # Success if it predicts the next state in sequence
        expected = [1, 2, 3]  # After state 1 predict 2, after 2 predict 3, etc
        success = predictions == expected
        print(f"Sequence prediction: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results['sequence_prediction'] = success
        return success
    
    def test_temporal_association(self):
        """Test 3: Can the brain learn temporal associations?"""
        print("\n=== TEST 3: Temporal Association ===")
        
        # Create cause-effect pairs
        cause = [0.9] * 12 + [0.1] * 12   # Strong signal in first half
        effect = [0.1] * 12 + [0.9] * 12  # Strong signal in second half
        
        # Train association (cause always followed by effect)
        print("Training cause->effect association (10 pairs)...")
        for i in range(10):
            self.brain.process_robot_cycle(cause)
            self.brain.process_robot_cycle(effect)
        
        # Test: present cause, does brain anticipate effect?
        print("\nPresenting cause...")
        action_before, _ = self.brain.process_robot_cycle(cause)
        
        # Check if action shows anticipation of effect (higher values in second half)
        anticipation = np.mean(action_before[12:]) - np.mean(action_before[:12])
        print(f"Anticipation strength: {anticipation:.3f}")
        
        success = anticipation > 0.05
        print(f"Temporal association: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results['temporal_association'] = success
        return success
    
    def test_memory_persistence(self):
        """Test 4: Do memories persist over time without reinforcement?"""
        print("\n=== TEST 4: Memory Persistence ===")
        
        # Create a unique pattern
        unique_pattern = [0.8, 0.2, 0.7, 0.3, 0.9, 0.1] * 4
        
        # Present it several times
        print("Creating memory (5 presentations)...")
        for _ in range(5):
            self.brain.process_robot_cycle(unique_pattern)
        
        initial_regions = len(self.brain.topology_regions)
        initial_max = self.brain.unified_field.max().item()
        print(f"Initial: {initial_regions} regions, max field value: {initial_max:.3f}")
        
        # Run neutral cycles (no strong patterns)
        print("\nRunning 20 neutral cycles...")
        neutral = [0.5] * 24
        for _ in range(20):
            self.brain.process_robot_cycle(neutral)
        
        final_regions = len(self.brain.topology_regions)
        final_max = self.brain.unified_field.max().item()
        print(f"Final: {final_regions} regions, max field value: {final_max:.3f}")
        
        # Test recall of original pattern
        print("\nTesting recall of original pattern...")
        action_recall, _ = self.brain.process_robot_cycle(unique_pattern)
        
        # Check if topology regions survived
        persistence_ratio = final_regions / max(initial_regions, 1)
        print(f"Persistence ratio: {persistence_ratio:.2f}")
        
        success = persistence_ratio > 0.5 and final_max > 0.02  # Some activity remains
        print(f"Memory persistence: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results['memory_persistence'] = success
        return success
    
    def test_interference_resistance(self):
        """Test 5: Can memories resist interference from new patterns?"""
        print("\n=== TEST 5: Interference Resistance ===")
        
        # Create and learn pattern A
        pattern_a = [0.9, 0.1, 0.1] * 8
        print("Learning pattern A (5 presentations)...")
        for _ in range(5):
            self.brain.process_robot_cycle(pattern_a)
        
        # Record field state
        regions_after_a = len(self.brain.topology_regions)
        
        # Learn interfering pattern B
        pattern_b = [0.1, 0.9, 0.1] * 8
        print(f"\nLearning interfering pattern B (5 presentations)...")
        for _ in range(5):
            self.brain.process_robot_cycle(pattern_b)
        
        regions_after_b = len(self.brain.topology_regions)
        
        # Test if pattern A can still be recalled
        print("\nTesting recall of pattern A...")
        action_a, _ = self.brain.process_robot_cycle(pattern_a)
        
        # Success if we have regions for both patterns
        success = regions_after_b >= regions_after_a
        print(f"Regions: A={regions_after_a}, After B={regions_after_b}")
        print(f"Interference resistance: {'✅ PASS' if success else '❌ FAIL'}")
        
        self.test_results['interference_resistance'] = success
        return success
    
    def run_all_tests(self):
        """Run all memory/prediction tests."""
        print("="*60)
        print("MEMORY/PREDICTION TEST SUITE")
        print("="*60)
        
        tests = [
            self.test_pattern_recall,
            self.test_sequence_prediction,
            self.test_temporal_association,
            self.test_memory_persistence,
            self.test_interference_resistance
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"Test failed with error: {e}")
                self.test_results[test.__name__] = False
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        total = len(self.test_results)
        passed = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            print(f"{test_name}: {'✅ PASS' if result else '❌ FAIL'}")
        
        print(f"\nTotal: {passed}/{total} passed ({100*passed/total:.0f}%)")
        
        return passed == total


if __name__ == "__main__":
    tester = MemoryPredictionTester(spatial_resolution=8, quiet_mode=True)
    success = tester.run_all_tests()
    
    if not success:
        print("\n⚠️  Some tests failed. This is expected as the field-based memory")
        print("    system is still being refined. Key areas for improvement:")
        print("    - Topology region persistence across cycles")
        print("    - Pattern completion mechanisms") 
        print("    - Temporal association through field dynamics")