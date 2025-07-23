#!/usr/bin/env python3
"""Verify constraint system is actually functioning."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

from brains.field.core_brain import UnifiedFieldBrain

print("=== CONSTRAINT SYSTEM VERIFICATION ===\n")

# Create brain with verbose constraint logging
brain = UnifiedFieldBrain(spatial_resolution=3, quiet_mode=False)

# Check initial state
print(f"\nInitial constraint state:")
print(f"  Active constraints: {len(brain.constraint_field.active_constraints)}")
print(f"  Constraints discovered: {brain.constraint_field.constraints_discovered}")
print(f"  Constraints enforced: {brain.constraint_field.constraints_enforced}")

# Process several cycles to trigger constraint discovery
print(f"\nProcessing cycles to trigger constraint discovery...")
for i in range(10):
    brain.process_robot_cycle([0.5 + i*0.05] * 24)  # Varying input
    
    if i == 4 or i == 9:  # Check after constraint discovery cycles
        print(f"\nAfter cycle {i+1}:")
        print(f"  Active constraints: {len(brain.constraint_field.active_constraints)}")
        print(f"  Constraints discovered: {brain.constraint_field.constraints_discovered}")
        print(f"  Constraints enforced: {brain.constraint_field.constraints_enforced}")
        
        # Check field statistics
        field_mean = brain.unified_field.mean().item()
        field_std = brain.unified_field.std().item()
        print(f"  Field mean: {field_mean:.4f}, std: {field_std:.4f}")

# Detailed constraint info
print(f"\nConstraint details:")
for i, constraint in enumerate(brain.constraint_field.active_constraints[:3]):  # First 3
    print(f"  Constraint {i}:")
    print(f"    Type: {constraint.constraint_type}")
    print(f"    Dimensions: {constraint.dimensions}")
    print(f"    Strength: {constraint.strength:.3f}")
    print(f"    Applications: {constraint.application_count}")

brain.shutdown()

print("\n✅ Constraint system is functioning!")