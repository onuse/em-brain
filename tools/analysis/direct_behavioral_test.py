#!/usr/bin/env python3
"""
Direct Behavioral Test - Bypasses BrainFactory to test core brain performance
"""

import sys
import time
sys.path.append('server/src')

from server.src.brains.field.core_brain import UnifiedFieldBrain
import numpy as np

def test_learning_behavior():
    """Test learning behavior with direct brain access."""
    print("🧠 Direct Behavioral Test (Bypassing BrainFactory)")
    
    # Create brain directly with small field for speed
    brain = UnifiedFieldBrain(spatial_resolution=8, quiet_mode=True)
    print(f"✅ Brain created: {brain.spatial_resolution}×{brain.spatial_resolution} field")
    
    # Test repeating pattern
    pattern = [0.2, 0.4, 0.6, 0.8]
    print(f"Testing pattern: {pattern}")
    
    # Track learning over multiple cycles
    confidences = []
    efficiencies = []
    addiction_mods = []
    
    print("\n📊 Running 30 cycles...")
    start_time = time.time()
    
    for i in range(30):
        actions, state = brain.process_robot_cycle(pattern)
        
        # Extract metrics
        confidence = state.get('last_action_confidence', 0.0)
        efficiency = state.get('prediction_efficiency', 0.0) 
        addiction = state.get('learning_addiction_modifier', 1.0)
        
        confidences.append(confidence)
        efficiencies.append(efficiency)
        addiction_mods.append(addiction)
        
        if i < 3 or i % 10 == 9:  # Show progress
            print(f"  Cycle {i+1:2d}: conf={confidence:.3f}, eff={efficiency:.3f}, addiction={addiction:.2f}")
    
    elapsed = time.time() - start_time
    cps = 30 / elapsed
    
    # Analyze learning progression
    early_conf = np.mean(confidences[:10])
    late_conf = np.mean(confidences[-10:])
    conf_improvement = late_conf - early_conf
    
    early_eff = np.mean(efficiencies[:10])
    late_eff = np.mean(efficiencies[-10:])
    eff_improvement = late_eff - early_eff
    
    avg_addiction = np.mean(addiction_mods[-10:])
    addiction_active = any(abs(mod - 1.0) > 0.01 for mod in addiction_mods[-10:])
    
    print(f"\n📈 Learning Analysis:")
    print(f"  Performance: {cps:.1f} cycles/sec ({elapsed:.2f}s total)")
    print(f"  Confidence: {early_conf:.3f} → {late_conf:.3f} (Δ{conf_improvement:+.3f})")
    print(f"  Efficiency: {early_eff:.3f} → {late_eff:.3f} (Δ{eff_improvement:+.3f})")
    print(f"  Addiction modifier: {avg_addiction:.3f} (active: {addiction_active})")
    
    # Success criteria
    learning_detected = conf_improvement > 0.02 or eff_improvement > 0.1
    good_performance = cps > 5.0
    addiction_working = addiction_active
    
    print(f"\n🎯 Results:")
    print(f"  Learning detected: {'✅' if learning_detected else '❌'}")
    print(f"  Performance acceptable: {'✅' if good_performance else '❌'}")
    print(f"  Addiction system active: {'✅' if addiction_working else '❌'}")
    
    if learning_detected and good_performance and addiction_working:
        print(f"\n🎉 SUCCESS: Brain shows learning behavior with good CPU performance!")
        return True
    elif learning_detected:
        print(f"\n⚠️ PARTIAL: Learning detected but performance/addiction issues")
        return True
    else:
        print(f"\n❌ FAILED: No significant learning detected")
        return False

if __name__ == "__main__":
    test_learning_behavior()