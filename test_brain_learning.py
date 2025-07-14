#!/usr/bin/env python3
"""
Scientific Brain Learning Test

This test provides controlled input patterns to the brain server
and analyzes if it's actually learning and improving predictions.

We'll test:
1. Pattern recognition - repetitive sensory-action pairs
2. Prediction accuracy improvement over time
3. Memory formation and retrieval
4. Learning rate adaptation
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Tuple

# Add server directory to path
server_dir = os.path.join(os.path.dirname(__file__), 'server')
sys.path.insert(0, server_dir)

from src.communication import MinimalBrainClient

class BrainLearningTester:
    """
    Scientific tester for brain learning capabilities.
    """
    
    def __init__(self, host='localhost', port=9999):
        self.client = MinimalBrainClient(host, port)
        self.test_results = {
            'pattern_recognition': {},
            'prediction_accuracy': [],
            'learning_progression': [],
            'memory_formation': {}
        }
        
    def connect(self) -> bool:
        """Connect to brain server."""
        print("🔗 Connecting to brain server for learning test...")
        return self.client.connect()
    
    def disconnect(self):
        """Disconnect from brain server."""
        self.client.disconnect()
    
    def test_simple_pattern_learning(self, repetitions=20) -> Dict:
        """
        Test if brain learns simple input-output patterns.
        
        Pattern: [1.0, 0.0, 0.0, 0.0] → should predict [1.0, 0.0, 0.0, 0.0]
        """
        print(f"\n🧪 Testing Simple Pattern Learning ({repetitions} repetitions)")
        print("   Teaching: [1,0,0,0] → [1,0,0,0]")
        
        pattern_input = [1.0, 0.0, 0.0, 0.0]
        expected_output = [1.0, 0.0, 0.0, 0.0]
        
        predictions = []
        errors = []
        
        for i in range(repetitions):
            # Get prediction
            prediction = self.client.get_action(pattern_input, timeout=2.0)
            predictions.append(prediction.copy())
            
            # Calculate error
            error = np.mean(np.abs(np.array(prediction) - np.array(expected_output)))
            errors.append(error)
            
            if i % 5 == 0 or i < 3 or i >= repetitions - 3:
                print(f"   Rep {i+1:2d}: Predicted {[f'{x:.2f}' for x in prediction]} | Error: {error:.3f}")
            
            time.sleep(0.1)  # Small delay
        
        # Analyze learning
        initial_error = np.mean(errors[:3])
        final_error = np.mean(errors[-3:])
        improvement = initial_error - final_error
        
        result = {
            'pattern': 'simple_1000',
            'repetitions': repetitions,
            'initial_error': initial_error,
            'final_error': final_error,
            'improvement': improvement,
            'learning_detected': improvement > 0.1,
            'all_errors': errors,
            'all_predictions': predictions
        }
        
        print(f"   📊 Results:")
        print(f"      Initial error: {initial_error:.3f}")
        print(f"      Final error: {final_error:.3f}")
        print(f"      Improvement: {improvement:.3f}")
        print(f"      Learning detected: {'✅' if result['learning_detected'] else '❌'}")
        
        return result
    
    def test_two_pattern_discrimination(self, repetitions=30) -> Dict:
        """
        Test if brain can learn to discriminate between two patterns.
        
        Pattern A: [1.0, 0.0, 0.0, 0.0] → [1.0, 0.0, 0.0, 0.0] 
        Pattern B: [0.0, 1.0, 0.0, 0.0] → [0.0, 1.0, 0.0, 0.0]
        """
        print(f"\n🧪 Testing Two-Pattern Discrimination ({repetitions} repetitions)")
        print("   Teaching: [1,0,0,0] → [1,0,0,0] and [0,1,0,0] → [0,1,0,0]")
        
        patterns = [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        ]
        
        pattern_errors = {'A': [], 'B': []}
        
        for i in range(repetitions):
            # Alternate between patterns
            pattern_idx = i % 2
            pattern_name = 'A' if pattern_idx == 0 else 'B'
            input_pattern, expected_output = patterns[pattern_idx]
            
            # Get prediction
            prediction = self.client.get_action(input_pattern, timeout=2.0)
            
            # Calculate error
            error = np.mean(np.abs(np.array(prediction) - np.array(expected_output)))
            pattern_errors[pattern_name].append(error)
            
            if i % 6 == 0 or i < 4 or i >= repetitions - 4:
                print(f"   Rep {i+1:2d} ({pattern_name}): {[f'{x:.2f}' for x in prediction]} | Error: {error:.3f}")
            
            time.sleep(0.1)
        
        # Analyze learning for each pattern
        results = {}
        for pattern_name, errors in pattern_errors.items():
            if len(errors) >= 6:
                initial_error = np.mean(errors[:3])
                final_error = np.mean(errors[-3:])
                improvement = initial_error - final_error
                
                results[f'pattern_{pattern_name}'] = {
                    'initial_error': initial_error,
                    'final_error': final_error,
                    'improvement': improvement,
                    'learning_detected': improvement > 0.1
                }
        
        overall_learning = all(r['learning_detected'] for r in results.values())
        
        print(f"   📊 Results:")
        for pattern_name, result in results.items():
            print(f"      {pattern_name}: {result['improvement']:.3f} improvement "
                  f"({'✅' if result['learning_detected'] else '❌'})")
        print(f"      Overall discrimination: {'✅' if overall_learning else '❌'}")
        
        return {
            'test_type': 'two_pattern_discrimination',
            'repetitions': repetitions,
            'pattern_results': results,
            'overall_learning': overall_learning,
            'pattern_errors': pattern_errors
        }
    
    def test_memory_persistence(self) -> Dict:
        """
        Test if brain remembers learned patterns after a delay.
        """
        print(f"\n🧪 Testing Memory Persistence")
        
        # Teach a pattern
        print("   Phase 1: Teaching pattern [0.5, 0.5, 0.0, 0.0] → [0.5, 0.5, 0.0, 0.0]")
        pattern_input = [0.5, 0.5, 0.0, 0.0]
        expected_output = [0.5, 0.5, 0.0, 0.0]
        
        # Training phase
        for i in range(15):
            prediction = self.client.get_action(pattern_input, timeout=2.0)
            time.sleep(0.05)
        
        # Get post-training prediction
        post_training_pred = self.client.get_action(pattern_input, timeout=2.0)
        post_training_error = np.mean(np.abs(np.array(post_training_pred) - np.array(expected_output)))
        
        print(f"   Post-training prediction: {[f'{x:.2f}' for x in post_training_pred]} | Error: {post_training_error:.3f}")
        
        # Wait and test with different inputs to see if memory persists
        print("   Phase 2: Testing with different inputs (5 seconds delay)")
        time.sleep(5)
        
        # Send some different inputs
        for i in range(10):
            different_input = [np.random.random(), np.random.random(), np.random.random(), np.random.random()]
            self.client.get_action(different_input, timeout=2.0)
            time.sleep(0.05)
        
        # Test original pattern recall
        print("   Phase 3: Testing original pattern recall")
        recall_prediction = self.client.get_action(pattern_input, timeout=2.0)
        recall_error = np.mean(np.abs(np.array(recall_prediction) - np.array(expected_output)))
        
        print(f"   Recall prediction: {[f'{x:.2f}' for x in recall_prediction]} | Error: {recall_error:.3f}")
        
        # Check if memory persisted
        memory_retained = recall_error <= post_training_error * 1.2  # Allow 20% degradation
        
        result = {
            'test_type': 'memory_persistence',
            'post_training_error': post_training_error,
            'recall_error': recall_error,
            'memory_retained': memory_retained,
            'error_increase': recall_error - post_training_error
        }
        
        print(f"   📊 Memory persistence: {'✅' if memory_retained else '❌'}")
        print(f"      Error increase: {result['error_increase']:.3f}")
        
        return result
    
    def run_comprehensive_test(self):
        """Run all learning tests."""
        print("🧠 COMPREHENSIVE BRAIN LEARNING TEST")
        print("=" * 60)
        print("Testing if the brain server is actually learning from experience")
        
        if not self.connect():
            print("❌ Could not connect to brain server")
            print("Please ensure brain server is running: python3 server/brain_server.py")
            return
        
        try:
            # Test 1: Simple pattern learning
            result1 = self.test_simple_pattern_learning(repetitions=20)
            self.test_results['pattern_recognition']['simple'] = result1
            
            # Test 2: Two pattern discrimination  
            result2 = self.test_two_pattern_discrimination(repetitions=30)
            self.test_results['pattern_recognition']['discrimination'] = result2
            
            # Test 3: Memory persistence
            result3 = self.test_memory_persistence()
            self.test_results['memory_formation'] = result3
            
            # Overall assessment
            self.print_final_assessment()
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.disconnect()
    
    def print_final_assessment(self):
        """Print final assessment of brain learning capability."""
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 40)
        
        # Check each capability
        simple_learning = self.test_results['pattern_recognition']['simple']['learning_detected']
        discrimination = self.test_results['pattern_recognition']['discrimination']['overall_learning']
        memory = self.test_results['memory_formation']['memory_retained']
        
        print(f"✅ Simple Pattern Learning:  {'PASS' if simple_learning else 'FAIL'}")
        print(f"✅ Pattern Discrimination:   {'PASS' if discrimination else 'FAIL'}")
        print(f"✅ Memory Persistence:       {'PASS' if memory else 'FAIL'}")
        
        total_passed = sum([simple_learning, discrimination, memory])
        
        print(f"\n📊 Overall Score: {total_passed}/3 tests passed")
        
        if total_passed == 3:
            print("🎉 EXCELLENT: Brain is learning effectively!")
        elif total_passed == 2:
            print("⚠️  MODERATE: Brain shows some learning, needs investigation")
        elif total_passed == 1:
            print("🔍 WEAK: Brain shows minimal learning, likely configuration issue")
        else:
            print("❌ CRITICAL: Brain shows no learning, major issue detected")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if not simple_learning:
            print("   - Check brain learning rate and similarity function")
            print("   - Verify experience storage is working")
        if not discrimination:
            print("   - Check pattern analysis and working memory")
            print("   - Verify activation dynamics are functioning")
        if not memory:
            print("   - Check experience persistence and retrieval")
            print("   - Verify similarity search is finding relevant experiences")
        
        if total_passed == 3:
            print("   - Brain is working well! Consider more complex tasks")


def main():
    """Run brain learning test."""
    tester = BrainLearningTester()
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()