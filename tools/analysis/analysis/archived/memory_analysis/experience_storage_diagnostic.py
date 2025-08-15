#!/usr/bin/env python3
"""
Experience Storage Diagnostic Tool

This tool investigates the brain's experience storage system to identify
why it's not learning from experience properly.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

# Add server directory to path
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

from src.brain import MinimalBrain
from src.experience.models import Experience
from src.experience.storage import ExperienceStorage

class ExperienceStorageDiagnostic:
    """Diagnostic tool for experience storage system."""
    
    def __init__(self):
        self.brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        self.test_results = {}
    
    def test_experience_creation(self) -> Dict[str, Any]:
        """Test if experiences can be created properly."""
        print("🔬 Testing Experience Creation...")
        
        # Create a test experience
        sensory_input = [1.0, 0.0, 0.5, 0.0]
        action_taken = [0.8, 0.2, 0.0, 0.0]
        outcome = [1.0, 0.1, 0.4, 0.1]
        
        experience = Experience(
            sensory_input=sensory_input,
            action_taken=action_taken,
            outcome=outcome,
            prediction_error=0.3,
            timestamp=time.time()
        )
        
        result = {
            "experience_created": experience is not None,
            "experience_id": experience.experience_id if experience else None,
            "sensory_dimensions": len(experience.sensory_input) if experience else 0,
            "action_dimensions": len(experience.action_taken) if experience else 0,
            "outcome_dimensions": len(experience.outcome) if experience else 0,
            "has_timestamp": hasattr(experience, 'timestamp') if experience else False,
            "has_prediction_error": hasattr(experience, 'prediction_error') if experience else False
        }
        
        print(f"   Experience created: {'✅' if result['experience_created'] else '❌'}")
        print(f"   Experience ID: {result['experience_id']}")
        print(f"   Dimensions: S={result['sensory_dimensions']}, A={result['action_dimensions']}, O={result['outcome_dimensions']}")
        
        return result
    
    def test_experience_storage(self) -> Dict[str, Any]:
        """Test if experiences are stored properly."""
        print("🔬 Testing Experience Storage...")
        
        storage = ExperienceStorage()
        initial_size = storage.size()
        
        # Create and store test experiences
        test_experiences = []
        for i in range(5):
            experience = Experience(
                sensory_input=[float(i), 0.0, 0.5, 0.0],
                action_taken=[0.8, float(i) * 0.1, 0.0, 0.0],
                outcome=[1.0, 0.1, 0.4, float(i) * 0.1],
                prediction_error=0.3 + i * 0.1,
                timestamp=time.time() + i * 0.1
            )
            test_experiences.append(experience)
            exp_id = storage.add_experience(experience)
            print(f"   Stored experience {i+1}: {exp_id}")
        
        final_size = storage.size()
        
        # Test retrieval
        retrieved_count = 0
        for exp in test_experiences:
            retrieved = storage.get_experience(exp.experience_id)
            if retrieved:
                retrieved_count += 1
        
        result = {
            "initial_size": initial_size,
            "final_size": final_size,
            "experiences_added": len(test_experiences),
            "storage_size_increased": final_size > initial_size,
            "retrieved_count": retrieved_count,
            "all_retrieved": retrieved_count == len(test_experiences),
            "storage_statistics": storage.get_statistics()
        }
        
        print(f"   Storage size: {initial_size} → {final_size}")
        print(f"   Retrieved: {retrieved_count}/{len(test_experiences)}")
        print(f"   All stored and retrieved: {'✅' if result['all_retrieved'] else '❌'}")
        
        return result
    
    def test_brain_experience_flow(self) -> Dict[str, Any]:
        """Test the complete experience flow through the brain."""
        print("🔬 Testing Brain Experience Flow...")
        
        # Reset brain to clean state
        self.brain.reset_brain()
        initial_count = self.brain.total_experiences
        
        # Send some sensory inputs and store experiences
        test_cycles = []
        for i in range(3):
            sensory_input = [1.0, float(i) * 0.5, 0.0, 0.0]
            
            # Get prediction from brain
            predicted_action, brain_state = self.brain.process_sensory_input(sensory_input)
            
            # Simulate outcome
            actual_outcome = [predicted_action[0] + 0.1, predicted_action[1], 
                            predicted_action[2], predicted_action[3] + 0.05]
            
            # Store experience
            exp_id = self.brain.store_experience(
                sensory_input=sensory_input,
                action_taken=predicted_action,
                outcome=actual_outcome,
                predicted_action=predicted_action
            )
            
            test_cycles.append({
                "cycle": i + 1,
                "sensory_input": sensory_input,
                "predicted_action": predicted_action,
                "actual_outcome": actual_outcome,
                "experience_id": exp_id,
                "brain_state": brain_state
            })
            
            print(f"   Cycle {i+1}: Experience {exp_id} stored")
            time.sleep(0.1)
        
        final_count = self.brain.total_experiences
        
        # Check if experiences are accessible
        storage_size = len(self.brain.experience_storage._experiences)
        recent_experiences = self.brain.experience_storage.get_recent_experiences(5)
        
        result = {
            "initial_experience_count": initial_count,
            "final_experience_count": final_count,
            "experiences_processed": len(test_cycles),
            "count_increased": final_count > initial_count,
            "storage_size": storage_size,
            "recent_experiences_count": len(recent_experiences),
            "test_cycles": test_cycles,
            "brain_stats": self.brain.get_brain_stats()
        }
        
        print(f"   Experience count: {initial_count} → {final_count}")
        print(f"   Storage size: {storage_size}")
        print(f"   Recent experiences: {len(recent_experiences)}")
        print(f"   Experiences flowing through brain: {'✅' if result['count_increased'] else '❌'}")
        
        return result
    
    def test_similarity_search(self) -> Dict[str, Any]:
        """Test if similarity search is working."""
        print("🔬 Testing Similarity Search...")
        
        # First populate with some experiences
        for i in range(10):
            sensory_input = [float(i % 3), float(i % 2), 0.0, 0.0]
            predicted_action, _ = self.brain.process_sensory_input(sensory_input)
            self.brain.store_experience(
                sensory_input=sensory_input,
                action_taken=predicted_action,
                outcome=predicted_action,  # Perfect prediction for testing
                predicted_action=predicted_action
            )
        
        # Test similarity search
        test_input = [1.0, 0.0, 0.0, 0.0]  # Should be similar to some stored experiences
        
        # Get all experiences for similarity testing
        experience_vectors = []
        experience_ids = []
        for exp_id, exp in self.brain.experience_storage._experiences.items():
            experience_vectors.append(exp.get_context_vector())
            experience_ids.append(exp_id)
        
        if experience_vectors:
            similar_experiences = self.brain.similarity_engine.find_similar_experiences(
                test_input, experience_vectors, experience_ids,
                max_results=5, min_similarity=0.0
            )
        else:
            similar_experiences = []
        
        result = {
            "total_experiences": len(self.brain.experience_storage._experiences),
            "experience_vectors_available": len(experience_vectors),
            "similar_experiences_found": len(similar_experiences),
            "similarity_search_working": len(similar_experiences) > 0,
            "similarity_scores": [sim for _, sim in similar_experiences],
            "avg_similarity": sum(sim for _, sim in similar_experiences) / len(similar_experiences) if similar_experiences else 0.0
        }
        
        print(f"   Total experiences: {result['total_experiences']}")
        print(f"   Similar experiences found: {result['similar_experiences_found']}")
        print(f"   Similarity search working: {'✅' if result['similarity_search_working'] else '❌'}")
        
        return result
    
    def test_learning_progression(self) -> Dict[str, Any]:
        """Test if the brain shows any learning progression."""
        print("🔬 Testing Learning Progression...")
        
        # Reset brain
        self.brain.reset_brain()
        
        # Teach a simple pattern repeatedly
        pattern_input = [1.0, 0.0, 0.0, 0.0]
        expected_action = [1.0, 0.0, 0.0, 0.0]
        
        predictions = []
        errors = []
        
        for i in range(10):
            # Get prediction
            predicted_action, _ = self.brain.process_sensory_input(pattern_input)
            predictions.append(predicted_action.copy())
            
            # Calculate error
            error = sum(abs(p - e) for p, e in zip(predicted_action, expected_action)) / len(expected_action)
            errors.append(error)
            
            # Store experience with the expected action as outcome
            self.brain.store_experience(
                sensory_input=pattern_input,
                action_taken=expected_action,  # What should have been done
                outcome=expected_action,       # Perfect outcome
                predicted_action=predicted_action
            )
            
            print(f"   Cycle {i+1}: Error = {error:.3f}, Prediction = {[f'{x:.2f}' for x in predicted_action]}")
            time.sleep(0.05)
        
        # Analyze learning
        early_error = sum(errors[:3]) / 3
        late_error = sum(errors[-3:]) / 3
        improvement = early_error - late_error
        
        result = {
            "total_cycles": len(predictions),
            "early_error": early_error,
            "late_error": late_error,
            "improvement": improvement,
            "learning_detected": improvement > 0.1,
            "all_errors": errors,
            "all_predictions": predictions,
            "final_experience_count": self.brain.total_experiences
        }
        
        print(f"   Early error: {early_error:.3f}")
        print(f"   Late error: {late_error:.3f}")
        print(f"   Improvement: {improvement:.3f}")
        print(f"   Learning detected: {'✅' if result['learning_detected'] else '❌'}")
        
        return result
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete diagnostic suite."""
        print("🧠 EXPERIENCE STORAGE DIAGNOSTIC")
        print("=" * 50)
        
        results = {}
        
        # Test 1: Experience creation
        results['experience_creation'] = self.test_experience_creation()
        print()
        
        # Test 2: Experience storage
        results['experience_storage'] = self.test_experience_storage()
        print()
        
        # Test 3: Brain experience flow
        results['brain_experience_flow'] = self.test_brain_experience_flow()
        print()
        
        # Test 4: Similarity search
        results['similarity_search'] = self.test_similarity_search()
        print()
        
        # Test 5: Learning progression
        results['learning_progression'] = self.test_learning_progression()
        print()
        
        # Generate summary
        self.generate_diagnostic_summary(results)
        
        return results
    
    def generate_diagnostic_summary(self, results: Dict[str, Any]):
        """Generate diagnostic summary."""
        print("📊 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        
        # Check each component
        experience_creation_ok = results['experience_creation']['experience_created']
        storage_ok = results['experience_storage']['all_retrieved']
        brain_flow_ok = results['brain_experience_flow']['count_increased']
        similarity_ok = results['similarity_search']['similarity_search_working']
        learning_ok = results['learning_progression']['learning_detected']
        
        print(f"✅ Experience Creation:  {'PASS' if experience_creation_ok else 'FAIL'}")
        print(f"✅ Experience Storage:   {'PASS' if storage_ok else 'FAIL'}")
        print(f"✅ Brain Experience Flow: {'PASS' if brain_flow_ok else 'FAIL'}")
        print(f"✅ Similarity Search:    {'PASS' if similarity_ok else 'FAIL'}")
        print(f"✅ Learning Progression: {'PASS' if learning_ok else 'FAIL'}")
        
        total_passed = sum([experience_creation_ok, storage_ok, brain_flow_ok, similarity_ok, learning_ok])
        
        print(f"\n📊 Overall Score: {total_passed}/5 tests passed")
        
        if total_passed >= 4:
            print("🎉 GOOD: Most components working, minor issues only")
        elif total_passed >= 2:
            print("⚠️  MODERATE: Some components broken, needs investigation")
        else:
            print("❌ CRITICAL: Major system failures detected")
        
        # Specific recommendations
        print(f"\n💡 Issue Analysis:")
        if not experience_creation_ok:
            print("   - Experience objects cannot be created properly")
        if not storage_ok:
            print("   - Experience storage/retrieval is broken")
        if not brain_flow_ok:
            print("   - Brain is not processing experiences through the full pipeline")
        if not similarity_ok:
            print("   - Similarity search system is not finding relevant experiences")
        if not learning_ok:
            print("   - Brain shows no learning improvement over time")
        
        if total_passed >= 4:
            print("   - System is mostly functional, check configuration parameters")

def main():
    """Run experience storage diagnostic."""
    diagnostic = ExperienceStorageDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    # Save results to file
    timestamp = int(time.time())
    results_file = f"logs/experience_diagnostic_{timestamp}.json"
    
    os.makedirs('logs', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()