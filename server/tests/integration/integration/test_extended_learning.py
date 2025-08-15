#!/usr/bin/env python3
"""
Extended Brain Learning Test

Test if longer training sequences produce stronger learning.
We'll run the same patterns but with much more repetitions to see
if the brain converges to better predictions.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Add server directory to path
# From tests/ we need to go up one level to brain/, then into server/
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

from src.communication import MinimalBrainClient

class ExtendedLearningTester:
    """Test brain learning with extended training sequences."""
    
    def __init__(self, host='localhost', port=9999):
        self.client = MinimalBrainClient(host, port)
        
    def connect(self) -> bool:
        """Connect to brain server."""
        print("🔗 Connecting to brain server for extended learning test...")
        return self.client.connect()
    
    def disconnect(self):
        """Disconnect from brain server."""
        self.client.disconnect()
    
    def test_extended_pattern_learning(self, repetitions=100) -> Dict:
        """
        Test extended pattern learning with many repetitions.
        
        Pattern: [1.0, 0.0, 0.0, 0.0] → should predict [1.0, 0.0, 0.0, 0.0]
        """
        print(f"\n🧪 Extended Pattern Learning Test ({repetitions} repetitions)")
        print("   Teaching: [1,0,0,0] → [1,0,0,0]")
        print("   Tracking learning curve...")
        
        pattern_input = [1.0, 0.0, 0.0, 0.0]
        expected_output = [1.0, 0.0, 0.0, 0.0]
        
        errors = []
        predictions = []
        
        # Track progress at key intervals
        report_intervals = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100]
        if repetitions < 100:
            report_intervals = [i for i in report_intervals if i <= repetitions]
        
        for i in range(repetitions):
            # Get prediction
            prediction = self.client.get_action(pattern_input, timeout=3.0)
            if prediction is None:
                print(f"   ⚠️ No response at rep {i+1}")
                prediction = [0.0, 0.0, 0.0, 0.0]
            
            predictions.append(prediction.copy())
            
            # Calculate error
            error = np.mean(np.abs(np.array(prediction) - np.array(expected_output)))
            errors.append(error)
            
            # Report progress at intervals
            if (i + 1) in report_intervals:
                print(f"   Rep {i+1:3d}: Predicted {[f'{x:.2f}' for x in prediction]} | Error: {error:.3f}")
            
            # Small delay to allow brain processing
            time.sleep(0.05)
        
        # Analyze learning curve
        initial_errors = errors[:5] if len(errors) >= 5 else errors[:len(errors)//2]
        final_errors = errors[-5:] if len(errors) >= 5 else errors[len(errors)//2:]
        
        initial_avg = np.mean(initial_errors)
        final_avg = np.mean(final_errors)
        total_improvement = initial_avg - final_avg
        
        # Check for convergence
        final_10_errors = errors[-10:] if len(errors) >= 10 else errors
        convergence_variance = np.var(final_10_errors)
        converged = convergence_variance < 0.1 and final_avg < 0.3
        
        # Find best prediction
        best_error_idx = np.argmin(errors)
        best_prediction = predictions[best_error_idx]
        best_error = errors[best_error_idx]
        
        result = {
            'repetitions': repetitions,
            'initial_avg_error': initial_avg,
            'final_avg_error': final_avg,
            'total_improvement': total_improvement,
            'best_error': best_error,
            'best_prediction': best_prediction,
            'best_at_rep': best_error_idx + 1,
            'converged': converged,
            'convergence_variance': convergence_variance,
            'strong_learning': total_improvement > 0.3,
            'all_errors': errors,
            'all_predictions': predictions
        }
        
        return result
    
    def test_extended_discrimination(self, repetitions=200) -> Dict:
        """
        Test extended pattern discrimination with many repetitions.
        
        Pattern A: [1.0, 0.0, 0.0, 0.0] → [1.0, 0.0, 0.0, 0.0] 
        Pattern B: [0.0, 1.0, 0.0, 0.0] → [0.0, 1.0, 0.0, 0.0]
        """
        print(f"\n🧪 Extended Pattern Discrimination ({repetitions} repetitions)")
        print("   Teaching: [1,0,0,0] → [1,0,0,0] and [0,1,0,0] → [0,1,0,0]")
        
        patterns = [
            ([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])
        ]
        
        pattern_errors = {'A': [], 'B': []}
        pattern_predictions = {'A': [], 'B': []}
        
        # Report intervals for longer test
        report_intervals = [1, 2, 5, 10, 20, 50, 100, 150, 200]
        if repetitions < 200:
            report_intervals = [i for i in report_intervals if i <= repetitions]
        
        for i in range(repetitions):
            # Alternate between patterns (but could also randomize)
            pattern_idx = i % 2
            pattern_name = 'A' if pattern_idx == 0 else 'B'
            input_pattern, expected_output = patterns[pattern_idx]
            
            # Get prediction
            prediction = self.client.get_action(input_pattern, timeout=3.0)
            if prediction is None:
                print(f"   ⚠️ No response at rep {i+1}")
                prediction = [0.0, 0.0, 0.0, 0.0]
            
            pattern_predictions[pattern_name].append(prediction.copy())
            
            # Calculate error
            error = np.mean(np.abs(np.array(prediction) - np.array(expected_output)))
            pattern_errors[pattern_name].append(error)
            
            # Report progress
            if (i + 1) in report_intervals:
                print(f"   Rep {i+1:3d} ({pattern_name}): {[f'{x:.2f}' for x in prediction]} | Error: {error:.3f}")
            
            time.sleep(0.05)
        
        # Analyze learning for each pattern
        results = {}
        for pattern_name, errors in pattern_errors.items():
            if len(errors) >= 10:
                initial_errors = errors[:5]
                final_errors = errors[-5:]
                
                initial_avg = np.mean(initial_errors)
                final_avg = np.mean(final_errors)
                improvement = initial_avg - final_avg
                
                # Check convergence
                final_10_errors = errors[-10:]
                convergence_variance = np.var(final_10_errors)
                converged = convergence_variance < 0.1 and final_avg < 0.3
                
                # Best performance
                best_error_idx = np.argmin(errors)
                best_error = errors[best_error_idx]
                
                results[f'pattern_{pattern_name}'] = {
                    'initial_avg_error': initial_avg,
                    'final_avg_error': final_avg,
                    'improvement': improvement,
                    'best_error': best_error,
                    'converged': converged,
                    'strong_learning': improvement > 0.3,
                    'all_errors': errors
                }
        
        # Overall assessment
        both_learning = all(r['strong_learning'] for r in results.values())
        both_converged = all(r['converged'] for r in results.values())
        
        return {
            'test_type': 'extended_discrimination',
            'repetitions': repetitions,
            'pattern_results': results,
            'both_patterns_learning': both_learning,
            'both_patterns_converged': both_converged,
            'pattern_errors': pattern_errors,
            'pattern_predictions': pattern_predictions
        }
    
    def run_extended_tests(self):
        """Run extended learning tests."""
        print("🚀 EXTENDED BRAIN LEARNING TEST")
        print("=" * 60)
        print("Testing brain learning with extended training sequences")
        print("This will take several minutes...")
        
        if not self.connect():
            print("❌ Could not connect to brain server")
            return
        
        try:
            # Extended simple pattern test
            print(f"\n⏱️  Extended Simple Pattern Test (estimated 5 minutes)")
            result1 = self.test_extended_pattern_learning(repetitions=100)
            
            # Extended discrimination test
            print(f"\n⏱️  Extended Discrimination Test (estimated 10 minutes)")
            result2 = self.test_extended_discrimination(repetitions=200)
            
            # Final assessment
            self.print_extended_assessment(result1, result2)
            
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()
    
    def print_extended_assessment(self, simple_result, discrimination_result):
        """Print extended assessment."""
        print(f"\n🎯 EXTENDED LEARNING ASSESSMENT")
        print("=" * 50)
        
        # Simple pattern results
        print(f"📈 Simple Pattern Learning (100 reps):")
        print(f"   Initial error: {simple_result['initial_avg_error']:.3f}")
        print(f"   Final error: {simple_result['final_avg_error']:.3f}")
        print(f"   Total improvement: {simple_result['total_improvement']:.3f}")
        print(f"   Best error achieved: {simple_result['best_error']:.3f} (rep {simple_result['best_at_rep']})")
        print(f"   Converged: {'✅' if simple_result['converged'] else '❌'}")
        print(f"   Strong learning: {'✅' if simple_result['strong_learning'] else '❌'}")
        
        # Discrimination results
        print(f"\n📈 Pattern Discrimination (200 reps):")
        for pattern_name, result in discrimination_result['pattern_results'].items():
            print(f"   {pattern_name}:")
            print(f"     Improvement: {result['improvement']:.3f}")
            print(f"     Best error: {result['best_error']:.3f}")
            print(f"     Converged: {'✅' if result['converged'] else '❌'}")
            print(f"     Strong learning: {'✅' if result['strong_learning'] else '❌'}")
        
        print(f"   Both patterns learning: {'✅' if discrimination_result['both_patterns_learning'] else '❌'}")
        print(f"   Both patterns converged: {'✅' if discrimination_result['both_patterns_converged'] else '❌'}")
        
        # Overall conclusions
        print(f"\n🏆 CONCLUSIONS:")
        simple_good = simple_result['strong_learning'] and simple_result['converged']
        discrimination_good = discrimination_result['both_patterns_learning']
        
        if simple_good and discrimination_good:
            print("   🎉 EXCELLENT: Extended training shows strong learning!")
            print("   The brain learns effectively with sufficient repetitions.")
        elif simple_good:
            print("   ✅ GOOD: Simple patterns learned well with extended training.")
            print("   Discrimination may need even more repetitions or different approach.")
        else:
            print("   ⚠️  LIMITED: Brain shows some learning but may need parameter tuning.")
        
        # Learning curve insights
        if len(simple_result['all_errors']) >= 20:
            early_20 = np.mean(simple_result['all_errors'][:20])
            late_20 = np.mean(simple_result['all_errors'][-20:])
            print(f"\n📊 Learning Curve Insights:")
            print(f"   First 20 reps average: {early_20:.3f}")
            print(f"   Last 20 reps average: {late_20:.3f}")
            print(f"   Learning stability: {(early_20 - late_20):.3f}")


def main():
    """Run extended learning test."""
    print("⏱️  Warning: This test takes 10-15 minutes to complete!")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Test cancelled.")
        return
    
    tester = ExtendedLearningTester()
    tester.run_extended_tests()

if __name__ == "__main__":
    main()