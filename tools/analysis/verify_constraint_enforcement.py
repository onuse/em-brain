#!/usr/bin/env python3
"""
Verify Constraint Enforcement System

Check if constraints are being discovered and applied in the brain.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'server'))

import torch
from src.core.dynamic_brain_factory import DynamicBrainFactory


def verify_constraint_enforcement():
    """Verify that constraint enforcement is working."""
    print("\n=== Verifying Constraint Enforcement System ===\n")
    
    # Create brain with high constraint discovery rate
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True,  # Reduce noise
        'constraint_discovery_rate': 1.0,  # Always discover constraints
        'constraint_enforcement_strength': 0.5,  # Strong enforcement
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=16,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print(f"✅ Brain created")
    print(f"   Constraint discovery rate: {brain.constraint_discovery_rate}")
    print(f"   Constraint enforcement strength: {brain.constraint_field.constraint_enforcement_strength}")
    print(f"   Field shape: {brain.unified_field.shape}")
    
    # Create patterns that should trigger constraint discovery
    patterns = [
        [1.0] * 16 + [1.0],      # High activation
        [0.0] * 16 + [0.0],      # Low activation
        [i/16.0 for i in range(16)] + [0.5],  # Gradient pattern
    ]
    
    print("\n📊 Processing patterns to trigger constraint discovery:\n")
    
    # Process multiple cycles to discover constraints
    for cycle in range(20):
        pattern = patterns[cycle % len(patterns)]
        
        # Process cycle
        motor_output, brain_state = brain.process_robot_cycle(pattern)
        
        # Every 5 cycles, check constraints (matching brain's discovery cycle)
        if cycle % 5 == 0:
            active_constraints = brain_state.get('active_constraints', 0)
            print(f"Cycle {cycle}: Active constraints = {active_constraints}")
            
            # Check constraint field directly
            if hasattr(brain, 'constraint_field'):
                cf = brain.constraint_field
                print(f"   Discovery cycles: {cf.discovery_cycles}")
                print(f"   Total discovered: {cf.constraints_discovered}")
                print(f"   Total enforced: {cf.constraints_enforced}")
                
                # Show constraint details
                if len(cf.active_constraints) > 0:
                    print(f"   Active constraint types:")
                    for i, constraint in enumerate(cf.active_constraints[:3]):  # Show first 3
                        print(f"     {i+1}. {constraint.constraint_type.value} "
                              f"(dims={constraint.dimensions}, strength={constraint.strength:.3f})")
    
    # Final summary
    print("\n📈 Final constraint system status:")
    if hasattr(brain, 'constraint_field'):
        cf = brain.constraint_field
        print(f"   Total constraints discovered: {cf.constraints_discovered}")
        print(f"   Total constraints enforced: {cf.constraints_enforced}")
        print(f"   Active constraints: {len(cf.active_constraints)}")
        
        if cf.constraints_discovered > 0:
            print("\n✅ Constraint system is WORKING!")
            print("   - Constraints are being discovered")
            print("   - Constraints are being enforced")
            print("   - The system is not disabled")
        elif len(cf.active_constraints) > 0:
            print("\n✅ Constraint system has active constraints!")
            print("   - Constraints are being enforced")
            print("   - The system is not disabled")
        else:
            print("\n⚠️  No constraints discovered")
            print("   This might be due to:")
            print("   - Field gradients below threshold")
            print("   - Insufficient field variation")
            print("   - Need more processing cycles")
    else:
        print("\n❌ Constraint field not found in brain")
    
    # Test enforcement directly
    print("\n🔧 Testing direct constraint enforcement:")
    
    if hasattr(brain, 'constraint_field'):
        # Save initial field state
        initial_field = brain.unified_field.clone()
        
        # Call enforce_constraints directly
        constraint_forces = brain.constraint_field.enforce_constraints(brain.unified_field)
        
        if constraint_forces is not None:
            force_magnitude = torch.mean(torch.abs(constraint_forces)).item()
            print(f"   Constraint forces computed successfully")
            print(f"   Force magnitude: {force_magnitude:.6f}")
            
            if force_magnitude > 0:
                print("   ✅ Constraint forces are non-zero - enforcement is active!")
            else:
                print("   ⚠️  Constraint forces are zero")
        else:
            print("   ❌ No constraint forces returned")


if __name__ == "__main__":
    verify_constraint_enforcement()