#!/usr/bin/env python3
"""
Debug Prediction Learning Test

Analyze exactly why prediction learning is showing 0% to achieve our 30% goal.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory

def debug_prediction_learning():
    """Debug the prediction learning mechanism step by step"""
    
    print("🔍 GOAL: Diagnose why prediction learning shows 0%")
    print("🎯 TARGET: Achieve 30% prediction learning capability")
    print("="*60)
    
    # Create brain with field implementation
    brain = BrainFactory(config={'brain_implementation': 'field'}, quiet_mode=True)
    
    # Simple repeating pattern
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D
    
    print(f"🧪 Test Pattern: {pattern[:8]}... (16D)")
    print(f"📊 Running prediction learning analysis...")
    
    prediction_confidences = []
    prediction_efficiencies = []
    field_evolution_cycles = []
    field_energies = []
    
    for i in range(20):  # Smaller test for detailed analysis
        action, brain_state = brain.process_sensory_input(pattern)
        
        # Extract all relevant metrics
        prediction_confidence = brain_state.get('prediction_confidence', 0.0)
        prediction_efficiency = brain_state.get('prediction_efficiency', 0.0)
        field_evolution = brain_state.get('field_evolution_cycles', 0)
        field_energy = brain_state.get('field_energy', 0.0)
        learning_detected = brain_state.get('learning_detected', False)
        
        prediction_confidences.append(prediction_confidence)
        prediction_efficiencies.append(prediction_efficiency)
        field_evolution_cycles.append(field_evolution)
        field_energies.append(field_energy)
        
        if i < 5 or i % 5 == 0:  # Report first 5 and every 5th cycle
            print(f"  Cycle {i+1:2d}: conf={prediction_confidence:.3f}, eff={prediction_efficiency:.3f}, "
                  f"evol={field_evolution}, energy={field_energy:.1f}, learning={learning_detected}")
    
    print(f"\n📈 ANALYSIS RESULTS:")
    
    # Analyze prediction confidence
    if all(conf == 0.0 for conf in prediction_confidences):
        print(f"❌ PROBLEM: prediction_confidence is always 0.0")
    else:
        early_conf = np.mean(prediction_confidences[:5])
        late_conf = np.mean(prediction_confidences[-5:])
        print(f"📊 Prediction confidence: {early_conf:.3f} → {late_conf:.3f}")
    
    # Analyze prediction efficiency
    if all(eff == 0.0 for eff in prediction_efficiencies):
        print(f"❌ PROBLEM: prediction_efficiency is always 0.0")
    else:
        early_eff = np.mean(prediction_efficiencies[:5])
        late_eff = np.mean(prediction_efficiencies[-5:])
        print(f"📊 Prediction efficiency: {early_eff:.3f} → {late_eff:.3f}")
    
    # Analyze field evolution
    max_evolution = max(field_evolution_cycles)
    if max_evolution == 0:
        print(f"❌ PROBLEM: field_evolution_cycles never increases (stuck at 0)")
    else:
        print(f"📊 Field evolution: 0 → {max_evolution} cycles")
    
    # Analyze field energy
    if all(energy == 0.0 for energy in field_energies):
        print(f"❌ PROBLEM: field_energy is always 0.0")
    else:
        early_energy = np.mean(field_energies[:5])
        late_energy = np.mean(field_energies[-5:])
        print(f"📊 Field energy: {early_energy:.1f} → {late_energy:.1f}")
    
    # Check what brain state keys are actually available
    print(f"\n🔑 Available brain_state keys: {list(brain_state.keys())}")
    
    # Calculate learning score using our current algorithm
    prediction_errors = [1.0 - conf for conf in prediction_confidences]
    early_error = np.mean(prediction_errors[:5])
    late_error = np.mean(prediction_errors[-5:])
    
    if early_error == 0:
        learning_score = 0.0
    else:
        improvement = max(0.0, (early_error - late_error) / early_error)
        learning_score = min(1.0, improvement)
    
    print(f"\n🎯 CURRENT LEARNING SCORE: {learning_score:.3f} (target: 0.30)")
    print(f"   Early error: {early_error:.3f}")
    print(f"   Late error: {late_error:.3f}")
    print(f"   Improvement: {((early_error - late_error) / early_error if early_error > 0 else 0):.3f}")
    
    # Identify the primary issue
    print(f"\n🔧 DIAGNOSTIC SUMMARY:")
    if all(conf == 0.0 for conf in prediction_confidences):
        print(f"🚨 PRIMARY ISSUE: prediction_confidence mechanism not working")
        print(f"   The brain is not generating prediction confidence values")
        print(f"   This could be due to:")
        print(f"   - Prediction system not initialized")
        print(f"   - Prediction confidence not being calculated")
        print(f"   - Brain state not including prediction metrics")
    
    if max_evolution == 0:
        print(f"🚨 SECONDARY ISSUE: field_evolution_cycles not increasing")
        print(f"   The field brain is not evolving/learning")
        print(f"   This suggests the field dynamics are not engaging")
    
    # Cleanup
    try:
        brain.finalize_session()
    except:
        pass
    
    return {
        'learning_score': learning_score,
        'prediction_confidences': prediction_confidences,
        'prediction_efficiencies': prediction_efficiencies,
        'field_evolution_cycles': field_evolution_cycles,
        'field_energies': field_energies,
        'brain_state_keys': list(brain_state.keys())
    }

def suggest_fixes(debug_results):
    """Suggest specific fixes based on diagnostic results"""
    
    print(f"\n🛠️  SUGGESTED FIXES TO REACH 30% PREDICTION LEARNING:")
    print("="*60)
    
    if all(conf == 0.0 for conf in debug_results['prediction_confidences']):
        print(f"1. 🔧 FIX PREDICTION CONFIDENCE CALCULATION")
        print(f"   - Check if prediction system is properly initialized")
        print(f"   - Verify prediction_confidence is calculated in brain state")
        print(f"   - Ensure prediction validation is enabled")
        
    if max(debug_results['field_evolution_cycles']) == 0:
        print(f"2. 🔧 ENABLE FIELD EVOLUTION")
        print(f"   - Check field dynamics are running")
        print(f"   - Verify field imprinting is working")
        print(f"   - Ensure field evolution thresholds are reachable")
        
    if all(eff == 0.0 for eff in debug_results['prediction_efficiencies']):
        print(f"3. 🔧 IMPLEMENT PREDICTION EFFICIENCY")
        print(f"   - Add prediction efficiency calculation to brain state")
        print(f"   - Connect prediction accuracy to efficiency metrics")
        print(f"   - Track prediction improvement over time")
    
    print(f"\n🎯 DEVELOPMENT STRATEGY:")
    print(f"   Start with fixing prediction_confidence calculation")
    print(f"   Then enable field evolution mechanics")
    print(f"   Finally add prediction efficiency tracking")
    print(f"   Test after each fix to measure progress toward 30% goal")

if __name__ == "__main__":
    debug_results = debug_prediction_learning()
    suggest_fixes(debug_results)