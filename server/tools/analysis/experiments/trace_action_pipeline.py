#!/usr/bin/env python3
"""
Trace Standard Action Pipeline

Complete investigation of the action generation pipeline to find where 
amplified gradients are getting lost between manual output and standard processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add server path for imports
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)
sys.path.insert(0, os.path.join(server_path, '..'))

from src.brain_factory import BrainFactory
from src.brains.field.tcp_adapter import create_field_brain_tcp_adapter
import torch

def trace_action_pipeline():
    """
    Trace the complete action generation pipeline step by step.
    """
    print("🔍 Standard Action Pipeline Investigation")
    print("=" * 60)
    
    # === PHASE 1: Create both pathways ===
    print("\n📋 Phase 1: Setting up both pathways")
    
    # Pathway A: BrainFactory (standard pipeline - currently broken)
    config = {
        'brain': {
            'type': 'field',
            'sensory_dim': 8,
            'motor_dim': 4,
            'enable_enhanced_dynamics': True,
            'enable_attention_guidance': True,
            'enable_hierarchical_processing': True
        }
    }
    
    print("   Creating BrainFactory (standard pipeline)...")
    brain_factory = BrainFactory(config=config, quiet_mode=True)
    
    # Pathway B: Direct TCP Adapter (manual test - working)
    print("   Creating TCP Adapter (manual test)...")
    tcp_adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=8,
        motor_dimensions=4,
        spatial_resolution=20,
        quiet_mode=True
    )
    
    # === PHASE 2: Compare internal structure ===
    print("\n🏗️ Phase 2: Internal structure comparison")
    
    # BrainFactory internals
    print("   BrainFactory structure:")
    print(f"      Brain type: {brain_factory.brain_type}")
    print(f"      Vector brain type: {type(brain_factory.vector_brain)}")
    
    if hasattr(brain_factory.vector_brain, 'field_brain'):
        field_brain_a = brain_factory.vector_brain.field_brain
        print(f"      Field brain type: {type(field_brain_a)}")
        print(f"      Field impl type: {type(field_brain_a.field_impl)}")
    else:
        print("      ❌ No field_brain attribute found!")
        field_brain_a = None
    
    # TCP Adapter internals
    print("   TCP Adapter structure:")
    field_brain_b = tcp_adapter.field_brain
    print(f"      Field brain type: {type(field_brain_b)}")
    print(f"      Field impl type: {type(field_brain_b.field_impl)}")
    
    # === PHASE 3: Test sensory processing ===
    print("\n📊 Phase 3: Sensory processing comparison")
    
    sensory_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"   Test input: {sensory_input}")
    
    # Process through BrainFactory
    print("\n   🅰️ BrainFactory pathway:")
    try:
        action_a, brain_state_a = brain_factory.process_sensory_input(sensory_input)
        print(f"      Action: {[f'{x:.6f}' for x in action_a]}")
        print(f"      Magnitude: {sum(abs(x) for x in action_a):.6f}")
        print(f"      Brain state keys: {list(brain_state_a.keys())}")
        print(f"      Field evolution cycles: {brain_state_a.get('field_evolution_cycles', 'NOT_FOUND')}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        action_a, brain_state_a = None, None
    
    # Process through TCP Adapter  
    print("\n   🅱️ TCP Adapter pathway:")
    try:
        action_b, brain_state_b = tcp_adapter.process_sensory_input(sensory_input)
        print(f"      Action: {[f'{x:.6f}' for x in action_b]}")
        print(f"      Magnitude: {sum(abs(x) for x in action_b):.6f}")
        print(f"      Brain state keys: {list(brain_state_b.keys())}")
        print(f"      Field evolution cycles: {brain_state_b.get('field_evolution_cycles', 'NOT_FOUND')}")
    except Exception as e:
        print(f"      ❌ Error: {e}")
        action_b, brain_state_b = None, None
    
    # === PHASE 4: Deep dive into field implementations ===
    print("\n🔬 Phase 4: Field implementation deep dive")
    
    if field_brain_a and field_brain_b:
        # Compare field implementations
        field_impl_a = field_brain_a.field_impl
        field_impl_b = field_brain_b.field_impl
        
        print("   Field implementation comparison:")
        print(f"      A (BrainFactory): {type(field_impl_a).__name__}")
        print(f"      B (TCP Adapter):  {type(field_impl_b).__name__}")
        
        # Check gradient flows
        print("   Gradient flows comparison:")
        grad_flows_a = getattr(field_impl_a, 'gradient_flows', {})
        grad_flows_b = getattr(field_impl_b, 'gradient_flows', {})
        
        print(f"      A gradient count: {len(grad_flows_a)}")
        print(f"      B gradient count: {len(grad_flows_b)}")
        
        if grad_flows_a:
            print("      A gradient keys:", list(grad_flows_a.keys())[:3], "...")
        if grad_flows_b:
            print("      B gradient keys:", list(grad_flows_b.keys())[:3], "...")
        
        # Test manual output generation on both
        print("   Manual output generation test:")
        try:
            dummy_coords = torch.zeros(4, device=field_impl_a.field_device)
            manual_a = field_impl_a.generate_field_output(dummy_coords)
            print(f"      A manual output: {[f'{x:.6f}' for x in manual_a.tolist()]}")
            print(f"      A manual magnitude: {manual_a.abs().sum():.6f}")
        except Exception as e:
            print(f"      A manual error: {e}")
        
        try:
            dummy_coords = torch.zeros(4, device=field_impl_b.field_device)
            manual_b = field_impl_b.generate_field_output(dummy_coords)
            print(f"      B manual output: {[f'{x:.6f}' for x in manual_b.tolist()]}")
            print(f"      B manual magnitude: {manual_b.abs().sum():.6f}")
        except Exception as e:
            print(f"      B manual error: {e}")
    
    # === PHASE 5: Trace output mapping ===
    print("\n🎯 Phase 5: Output mapping investigation")
    
    if field_brain_a:
        print("   BrainFactory output mapping:")
        # Check if there's any output mapping/scaling
        if hasattr(field_brain_a, 'output_mapping'):
            print(f"      Output mapping found: {field_brain_a.output_mapping}")
        else:
            print("      No output mapping attribute")
        
        # Check stream capabilities
        if hasattr(field_brain_a, 'stream_capabilities'):
            caps = field_brain_a.stream_capabilities
            print(f"      Stream capabilities: {caps}")
        
        # Check if there's any action processing in the field brain
        print("      Methods containing 'output' or 'action':")
        for attr_name in dir(field_brain_a):
            if 'output' in attr_name.lower() or 'action' in attr_name.lower():
                print(f"         - {attr_name}")
    
    # === PHASE 6: Identify the disconnect ===
    print("\n❌ Phase 6: Disconnect Analysis")
    
    if action_a is not None and action_b is not None:
        mag_a = sum(abs(x) for x in action_a)
        mag_b = sum(abs(x) for x in action_b)
        
        print(f"   Action magnitude comparison:")
        print(f"      BrainFactory: {mag_a:.6f}")
        print(f"      TCP Adapter:  {mag_b:.6f}")
        print(f"      Ratio: {mag_b/max(mag_a, 1e-10):.1f}x")
        
        if mag_a < 0.01 and mag_b > 0.01:
            print("   🎯 DISCONNECT IDENTIFIED: BrainFactory action generation is broken")
            print("      TCP Adapter generates meaningful actions, BrainFactory doesn't")
            return "brainfactory_broken"
        elif mag_a > 0.01 and mag_b < 0.01:
            print("   🎯 DISCONNECT IDENTIFIED: TCP Adapter action generation is broken")
            return "tcp_adapter_broken"
        elif mag_a < 0.01 and mag_b < 0.01:
            print("   🎯 BOTH PATHWAYS BROKEN: Neither generates meaningful actions")
            return "both_broken"
        else:
            print("   ✅ BOTH PATHWAYS WORKING: Actions generated successfully")
            return "both_working"
    else:
        print("   ❌ CRITICAL ERROR: Could not process sensory input through one or both pathways")
        return "processing_error"

if __name__ == "__main__":
    result = trace_action_pipeline()
    print(f"\n🏁 Investigation Result: {result}")