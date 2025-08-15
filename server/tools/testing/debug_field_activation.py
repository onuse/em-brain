#!/usr/bin/env python3
"""
Debug Field Activation

Investigate why field_energy = 0.0 and field_evolution_cycles = 0
This is blocking our 30% prediction learning goal.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory

def debug_field_activation():
    """Debug why the field isn't activating or evolving"""
    
    print("🔍 DEBUGGING FIELD ACTIVATION")
    print("🎯 GOAL: Find why field_energy = 0.0 and evolution_cycles = 0")
    print("="*60)
    
    # Create brain
    brain = BrainFactory(quiet_mode=False)  # Verbose mode for debugging
    
    # Simple pattern
    pattern = [0.8, 0.2, 0.6, 0.4] * 4  # Strong signal
    
    print(f"🧪 Test Pattern: {pattern}")
    print(f"📊 Pattern energy: {sum(abs(x) for x in pattern):.1f}")
    
    print(f"\n🔍 Processing single cycle with verbose output...")
    
    # Single cycle with detailed analysis
    action, brain_state = brain.process_sensory_input(pattern)
    
    print(f"\n📊 FIRST CYCLE RESULTS:")
    print(f"   Action output: {[f'{x:.3f}' for x in action]}")
    print(f"   Brain architecture: {brain_state.get('architecture', 'unknown')}")
    
    # Check what we actually get in brain_state
    field_keys = [k for k in brain_state.keys() if 'field' in k.lower()]
    energy_keys = [k for k in brain_state.keys() if 'energy' in k.lower()]
    evolution_keys = [k for k in brain_state.keys() if 'evolution' in k.lower()]
    
    print(f"\n🔑 Field-related keys: {field_keys}")
    print(f"🔑 Energy-related keys: {energy_keys}")
    print(f"🔑 Evolution-related keys: {evolution_keys}")
    
    # Check if we're using the right brain type
    print(f"\n🧠 BRAIN TYPE ANALYSIS:")
    if 'sparse_goldilocks' in brain_state.get('architecture', ''):
        print("   ❌ Using sparse_goldilocks brain (not field brain!)")
        print("   This explains why field_energy = 0.0")
        print("   Need to force field brain usage")
    elif 'field' in brain_state.get('architecture', ''):
        print("   ✅ Using field brain architecture")
        # Dive deeper into field activation
        for key in field_keys:
            value = brain_state.get(key, 'NOT_FOUND')
            print(f"   {key}: {value}")
    else:
        print(f"   ❓ Unknown architecture: {brain_state.get('architecture', 'none')}")
    
    # Try more cycles to see if field activates
    print(f"\n🔄 Processing 5 more cycles...")
    for i in range(5):
        action, brain_state = brain.process_sensory_input(pattern)
        field_energy = brain_state.get('field_energy', 0.0)
        evolution_cycles = brain_state.get('field_evolution_cycles', 0)
        prediction_conf = brain_state.get('prediction_confidence', 0.0)
        
        print(f"   Cycle {i+2}: energy={field_energy:.1f}, evolution={evolution_cycles}, conf={prediction_conf:.3f}")
    
    # Check field implementation directly if possible
    print(f"\n🔧 BRAIN FACTORY ANALYSIS:")
    print(f"   Brain implementation: {brain.brain_implementation}")
    
    if hasattr(brain, 'field_brain') or hasattr(brain, 'brain_adapter'):
        print("   ✅ Has field brain component")
        if hasattr(brain, 'brain_adapter'):
            adapter = brain.brain_adapter
            if hasattr(adapter, 'field_brain'):
                field_brain = adapter.field_brain
                print(f"   Field brain type: {type(field_brain).__name__}")
                
                # Check field implementation
                if hasattr(field_brain, 'field_impl'):
                    field_impl = field_brain.field_impl
                    field_stats = field_impl.get_field_statistics()
                    print(f"   Field stats: {field_stats}")
                    
                    # Check if field tensors are actually being updated
                    if hasattr(field_impl, 'unified_field'):
                        field_tensor = field_impl.unified_field
                        tensor_stats = {
                            'shape': list(field_tensor.shape),
                            'sum': field_tensor.sum().item(),
                            'mean': field_tensor.mean().item(),
                            'max': field_tensor.max().item(),
                            'nonzero_count': torch.count_nonzero(field_tensor).item()
                        }
                        print(f"   Field tensor: {tensor_stats}")
                    elif hasattr(field_impl, 'field_tensors'):
                        total_sum = sum(tensor.sum().item() for tensor in field_impl.field_tensors.values())
                        total_nonzero = sum(torch.count_nonzero(tensor).item() for tensor in field_impl.field_tensors.values())
                        print(f"   Multi-tensor field: sum={total_sum:.3f}, nonzero={total_nonzero}")
    else:
        print("   ❌ No field brain component found")
    
    # Cleanup
    try:
        brain.finalize_session()
    except:
        pass
    
    return brain_state

def suggest_field_activation_fixes(brain_state):
    """Suggest fixes to activate the field brain"""
    
    print(f"\n🛠️  SUGGESTED FIXES FOR FIELD ACTIVATION:")
    print("="*60)
    
    architecture = brain_state.get('architecture', '')
    
    if 'sparse_goldilocks' in architecture:
        print(f"1. 🔧 FORCE FIELD BRAIN USAGE")
        print(f"   - Current: Using {architecture}")
        print(f"   - Solution: Configure brain factory to use field brain")
        print(f"   - Code: BrainFactory(config={{'brain_implementation': 'field'}})")
        
    elif 'field' in architecture:
        print(f"1. 🔧 ENABLE FIELD IMPRINTING")
        print(f"   - Field brain is active but not imprinting patterns")
        print(f"   - Check if sensory input reaches field implementation")
        print(f"   - Verify field imprinting thresholds are appropriate")
        
        print(f"2. 🔧 CHECK FIELD EVOLUTION TRIGGERS")
        print(f"   - Field evolution cycles stuck at 0")
        print(f"   - Check evolution triggers and thresholds")
        print(f"   - Verify field dynamics are running")
    
    print(f"\n🎯 IMMEDIATE ACTION:")
    print(f"   Test with explicit field brain configuration")
    print(f"   Monitor field tensor updates during processing")
    print(f"   Check field energy accumulation with strong input patterns")

if __name__ == "__main__":
    brain_state = debug_field_activation()
    suggest_field_activation_fixes(brain_state)