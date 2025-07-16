#!/usr/bin/env python3
"""
Experience Flow Tracer

This tool traces the complete flow of experiences through the brain system
to identify where the learning breakdown occurs.
"""

import sys
import os
import json
import time
import numpy as np
from typing import List, Dict, Any

# Add server directory to path
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

from src.brain import MinimalBrain
from src.experience.models import Experience

class ExperienceFlowTracer:
    """Traces complete experience flow through the brain system."""
    
    def __init__(self):
        self.brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        self.trace_log = []
    
    def trace_experience_lifecycle(self, sensory_input: List[float], 
                                 expected_action: List[float]) -> Dict[str, Any]:
        """Trace a single experience through the complete lifecycle."""
        
        self.log_trace("=== Starting Experience Lifecycle Trace ===")
        self.log_trace(f"Input: {sensory_input}")
        self.log_trace(f"Expected: {expected_action}")
        
        # Step 1: Get prediction from brain
        self.log_trace("STEP 1: Getting prediction from brain")
        predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
        self.log_trace(f"Predicted: {predicted_action}")
        self.log_trace(f"Prediction method: {brain_state.get('prediction_method', 'unknown')}")
        self.log_trace(f"Confidence: {brain_state.get('prediction_confidence', 0.0)}")
        self.log_trace(f"Similar experiences used: {brain_state.get('num_similar_experiences', 0)}")
        
        # Step 2: Store experience
        self.log_trace("STEP 2: Storing experience")
        initial_storage_size = len(self.brain.experience_storage._experiences)
        self.log_trace(f"Storage size before: {initial_storage_size}")
        
        # Simulate perfect outcome (what should have happened)
        perfect_outcome = expected_action.copy()
        
        exp_id = self.brain.store_experience(
            sensory_input=sensory_input,
            action_taken=expected_action,  # What should have been done
            outcome=perfect_outcome,       # Perfect outcome
            predicted_action=predicted_action
        )
        
        final_storage_size = len(self.brain.experience_storage._experiences)
        self.log_trace(f"Storage size after: {final_storage_size}")
        self.log_trace(f"Experience ID: {exp_id}")
        self.log_trace(f"Storage increased: {final_storage_size > initial_storage_size}")
        
        # Step 3: Verify experience was stored
        self.log_trace("STEP 3: Verifying experience storage")
        stored_exp = self.brain.experience_storage.get_experience(exp_id)
        if stored_exp:
            self.log_trace(f"Experience retrieved: YES")
            self.log_trace(f"Stored sensory: {stored_exp.sensory_input}")
            self.log_trace(f"Stored action: {stored_exp.action_taken}")
            self.log_trace(f"Stored outcome: {stored_exp.outcome}")
            self.log_trace(f"Prediction error: {stored_exp.prediction_error}")
            self.log_trace(f"Activation level: {stored_exp.activation_level}")
        else:
            self.log_trace(f"Experience retrieved: NO - ERROR!")
        
        # Step 4: Test if experience would be found in similarity search
        self.log_trace("STEP 4: Testing similarity search")
        
        # Get all experience vectors
        experience_vectors = []
        experience_ids = []
        for exp_id_inner, exp in self.brain.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id_inner)
        
        if experience_vectors:
            # Search for similar experiences to our input
            similar_experiences = self.brain.similarity_engine.find_similar_experiences(
                sensory_input, experience_vectors, experience_ids,
                max_results=10, min_similarity=0.0  # No minimum to see all
            )
            
            self.log_trace(f"Similar experiences found: {len(similar_experiences)}")
            for i, (sim_exp_id, similarity) in enumerate(similar_experiences[:5]):
                sim_exp = self.brain.experience_storage._experiences[sim_exp_id]
                self.log_trace(f"  {i+1}. ID: {sim_exp_id[:8]}... Similarity: {similarity:.3f}")
                self.log_trace(f"      Sensory: {sim_exp.sensory_input}")
                self.log_trace(f"      Action: {sim_exp.action_taken}")
                
                # Check if this is our stored experience
                if sim_exp_id == exp_id:
                    self.log_trace(f"      *** THIS IS OUR STORED EXPERIENCE ***")
        
        # Step 5: Test prediction on same input again
        self.log_trace("STEP 5: Testing prediction on same input again")
        predicted_action_2, brain_state_2 = self.brain.process_sensory_input(sensory_input)
        self.log_trace(f"Second prediction: {predicted_action_2}")
        self.log_trace(f"Second method: {brain_state_2.get('prediction_method', 'unknown')}")
        self.log_trace(f"Second confidence: {brain_state_2.get('prediction_confidence', 0.0)}")
        self.log_trace(f"Second similar experiences: {brain_state_2.get('num_similar_experiences', 0)}")
        
        # Calculate if prediction improved
        error_1 = np.mean(np.abs(np.array(predicted_action) - np.array(expected_action)))
        error_2 = np.mean(np.abs(np.array(predicted_action_2) - np.array(expected_action)))
        improvement = error_1 - error_2
        
        self.log_trace(f"Prediction error 1: {error_1:.3f}")
        self.log_trace(f"Prediction error 2: {error_2:.3f}")
        self.log_trace(f"Improvement: {improvement:.3f}")
        self.log_trace(f"Learning detected: {improvement > 0.05}")
        
        # Return comprehensive trace
        return {
            'sensory_input': sensory_input,
            'expected_action': expected_action,
            'predicted_action_1': predicted_action,
            'predicted_action_2': predicted_action_2,
            'brain_state_1': brain_state,
            'brain_state_2': brain_state_2,
            'experience_id': exp_id,
            'experience_stored': stored_exp is not None,
            'storage_size_change': final_storage_size - initial_storage_size,
            'similar_experiences_found': len(similar_experiences) if 'similar_experiences' in locals() else 0,
            'prediction_error_1': error_1,
            'prediction_error_2': error_2,
            'improvement': improvement,
            'learning_detected': improvement > 0.05,
            'trace_log': self.trace_log.copy()
        }
    
    def log_trace(self, message: str):
        """Add message to trace log."""
        print(message)
        self.trace_log.append(message)
    
    def test_learning_pattern(self, pattern_input: List[float], 
                            pattern_output: List[float], 
                            repetitions: int = 5) -> Dict[str, Any]:
        """Test learning of a specific pattern over multiple repetitions."""
        
        print(f"\n🔍 TESTING LEARNING PATTERN")
        print(f"Pattern: {pattern_input} → {pattern_output}")
        print(f"Repetitions: {repetitions}")
        
        self.brain.reset_brain()
        
        results = []
        
        for i in range(repetitions):
            print(f"\n--- Repetition {i+1} ---")
            self.trace_log = []  # Clear trace log for each repetition
            
            result = self.trace_experience_lifecycle(pattern_input, pattern_output)
            result['repetition'] = i + 1
            results.append(result)
            
            # Brief pause
            time.sleep(0.1)
        
        # Analyze learning progression
        errors = [r['prediction_error_1'] for r in results]
        early_error = np.mean(errors[:2]) if len(errors) >= 2 else errors[0]
        late_error = np.mean(errors[-2:]) if len(errors) >= 2 else errors[-1]
        total_improvement = early_error - late_error
        
        print(f"\n📊 LEARNING ANALYSIS")
        print(f"Early error: {early_error:.3f}")
        print(f"Late error: {late_error:.3f}")
        print(f"Total improvement: {total_improvement:.3f}")
        print(f"Learning detected: {'✅' if total_improvement > 0.1 else '❌'}")
        
        return {
            'pattern_input': pattern_input,
            'pattern_output': pattern_output,
            'repetitions': repetitions,
            'results': results,
            'early_error': early_error,
            'late_error': late_error,
            'total_improvement': total_improvement,
            'learning_detected': total_improvement > 0.1,
            'final_storage_size': len(self.brain.experience_storage._experiences)
        }

def main():
    """Run experience flow tracing."""
    tracer = ExperienceFlowTracer()
    
    # Test a simple pattern
    pattern_input = [1.0, 0.0, 0.0, 0.0]
    pattern_output = [1.0, 0.0, 0.0, 0.0]
    
    # Test learning over 5 repetitions
    results = tracer.test_learning_pattern(pattern_input, pattern_output, repetitions=5)
    
    # Save results
    timestamp = int(time.time())
    results_file = f"logs/experience_flow_trace_{timestamp}.json"
    
    os.makedirs('logs', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed trace saved to: {results_file}")
    
    # Test another pattern
    print("\n" + "="*60)
    pattern_input_2 = [0.0, 1.0, 0.0, 0.0]
    pattern_output_2 = [0.0, 1.0, 0.0, 0.0]
    
    results_2 = tracer.test_learning_pattern(pattern_input_2, pattern_output_2, repetitions=5)
    
    # Save second results
    results_file_2 = f"logs/experience_flow_trace_2_{timestamp}.json"
    with open(results_file_2, 'w') as f:
        json.dump(results_2, f, indent=2, default=str)
    
    print(f"\n📄 Second trace saved to: {results_file_2}")

if __name__ == "__main__":
    main()