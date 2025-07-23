#!/usr/bin/env python3
"""Test the constraint system to understand the dimension issues."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
from brains.field.dynamics.constraint_field_nd import ConstraintFieldND, FieldConstraintND, FieldConstraintType

print("=== CONSTRAINT SYSTEM TEST ===\n")

# Create a test field matching brain dimensions
field_shape = [3, 3, 3, 10, 15, 4, 4, 3, 2, 3, 2]  # 11D field
print(f"Field shape: {field_shape} ({len(field_shape)}D)")

# Create constraint field
constraint_field = ConstraintFieldND(
    field_shape=field_shape,
    dimension_names=[f"dim_{i}" for i in range(len(field_shape))],
    constraint_discovery_rate=0.1,
    constraint_enforcement_strength=0.3
)

# Create test field
field = torch.rand(field_shape) * 0.1

print("\n1. Testing constraint creation:")
# Create a simple constraint
test_constraint = FieldConstraintND(
    constraint_type=FieldConstraintType.GRADIENT_FLOW,
    dimensions=[0, 1, 2],  # Spatial dimensions
    region_centers=torch.tensor([[1.5, 1.5, 1.5]]),
    region_radii=torch.tensor([[1.0, 1.0, 1.0]]),
    gradient_direction=torch.tensor([0.1, 0.1, 0.1]),
    strength=0.5
)

print(f"  Created constraint on dimensions: {test_constraint.dimensions}")
print(f"  Region center: {test_constraint.region_centers}")

print("\n2. Testing constraint application:")
# Test with a coordinate point
test_coord = torch.tensor([1, 1, 1, 5, 7, 2, 2, 1, 1, 1, 1], dtype=torch.float32)
print(f"  Test coordinate: {test_coord}")

# Check if constraint applies
applies = test_constraint.applies_to_point(test_coord)
print(f"  Constraint applies: {applies}")

# Compute force
force = test_constraint.compute_constraint_force(test_coord)
print(f"  Force shape: {force.shape}")
print(f"  Force magnitude: {torch.norm(force).item():.4f}")

print("\n3. Testing constraint enforcement:")
# Add constraint to field
constraint_field.constraints.append(test_constraint)
constraint_field.active_constraints = [test_constraint]

# Try to enforce constraints
try:
    constraint_forces = constraint_field.enforce_constraints(field)
    print(f"  Constraint forces shape: {constraint_forces.shape}")
    print(f"  Non-zero forces: {torch.count_nonzero(constraint_forces).item()}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n4. Analyzing the issue:")
print("  The constraint system expects coordinate vectors with all dimensions")
print("  But it's being given indices that need to be expanded to full coordinates")
print("  Solution: Modify constraint system to work with field indices directly")