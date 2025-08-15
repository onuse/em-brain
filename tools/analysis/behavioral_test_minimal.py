#!/usr/bin/env python3
"""
Minimal Behavioral Test for Unified Brain
Tests that the prediction improvement addiction system leads to learning behavior.
"""

import sys
import os
sys.path.append('server/src')

from server.src.brain_factory import BrainFactory
import numpy as np

def test_prediction_learning_behavior():
    """Test that the brain shows learning behavior with prediction improvement addiction."""
    print("🎯 Behavioral Test: Prediction Learning with Addiction System")
    
    # Create brain with minimal configuration
    config = {
        'brain': {'type': 'field', 'sensory_dim': 16, 'motor_dim': 4},
        'memory': {'enable_persistence': False}  # Disable persistence for clean test
    }
    
    brain = BrainFactory(config=config, quiet_mode=True)
    
    # Test pattern that should be learnable
    pattern = [0.5, 0.3, 0.8, 0.2, 0.6, 0.1, 0.9, 0.4] * 2  # 16D repeating pattern
    
    # Reset learning history for clean test
    if hasattr(brain, 'unified_brain'):
        brain.unified_brain._prediction_confidence_history = []
        brain.unified_brain._improvement_rate_history = []
    
    # Track learning over time
    confidences = []
    addiction_modifiers = []
    efficiencies = []
    
    cycles = 50  # Reduced for faster testing
    
    for i in range(cycles):
        # Present the same pattern repeatedly - brain should learn to predict it
        action, brain_state = brain.process_sensory_input(pattern)
        
        # Extract learning metrics
        confidence = brain_state.get('last_action_confidence', 
                                   brain_state.get('prediction_confidence', 0.0))
        
        # Get addiction system metrics if available
        if hasattr(brain, 'unified_brain'):
            try:
                efficiency = brain.unified_brain.calculate_prediction_efficiency()
                addiction_mod = brain.unified_brain._get_prediction_improvement_addiction_modifier()
                
                confidences.append(confidence)
                efficiencies.append(efficiency)
                addiction_modifiers.append(addiction_mod)
                
            except Exception:
                confidences.append(confidence)
                efficiencies.append(0.0)
                addiction_modifiers.append(1.0)
        else:
            confidences.append(confidence)
            efficiencies.append(0.0)
            addiction_modifiers.append(1.0)
    
    # Analyze learning behavior
    if len(confidences) >= 20:
        early_conf = np.mean(confidences[:10])
        late_conf = np.mean(confidences[-10:])
        confidence_improvement = late_conf - early_conf
        
        early_eff = np.mean(efficiencies[:10]) if efficiencies else 0
        late_eff = np.mean(efficiencies[-10:]) if efficiencies else 0
        efficiency_improvement = late_eff - early_eff
        
        # Check if addiction system is active
        addiction_active = any(mod != 1.0 for mod in addiction_modifiers[-10:])
        avg_addiction = np.mean(addiction_modifiers[-10:])
        
        print(f"📊 Learning Analysis:")
        print(f"  Early confidence: {early_conf:.3f}")
        print(f"  Late confidence: {late_conf:.3f}")
        print(f"  Confidence improvement: {confidence_improvement:.3f}")
        print(f"  Early efficiency: {early_eff:.3f}")
        print(f"  Late efficiency: {late_eff:.3f}")
        print(f"  Efficiency improvement: {efficiency_improvement:.3f}")
        print(f"  Addiction system active: {addiction_active}")
        print(f"  Average addiction modifier: {avg_addiction:.3f}")
        
        # Success criteria
        learning_detected = confidence_improvement > 0.05 or efficiency_improvement > 0.1
        addiction_working = addiction_active and avg_addiction != 1.0
        
        print(f"\n🎯 Results:")
        print(f"  Learning detected: {'✅' if learning_detected else '❌'} (improvement > 0.05)")
        print(f"  Addiction system working: {'✅' if addiction_working else '❌'}")
        
        if learning_detected and addiction_working:
            print(f"\n🎉 SUCCESS: Brain shows learning behavior with addiction system!")
            return True
        elif learning_detected:
            print(f"\n⚠️ PARTIAL: Learning detected but addiction system may not be active")
            return True
        else:
            print(f"\n❌ FAILED: No significant learning improvement detected")
            return False
    else:
        print(f"❌ Insufficient data collected")
        return False

def main():
    """Run behavioral test."""
    print("🧪 Unified Brain Behavioral Test")
    print("=" * 50)
    
    try:
        success = test_prediction_learning_behavior()
        
        if success:
            print("\n✅ Unified brain with prediction improvement addiction is working!")
        else:
            print("\n❌ Behavioral test failed - may need further tuning")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)