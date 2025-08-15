#!/usr/bin/env python3
"""
Test the 37D constraint system integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
from brains.field.core_brain import create_unified_field_brain
from brains.field.dynamics.constraint_field_nd import ConstraintFieldND, FieldConstraintND, FieldConstraintType


def test_constraint_discovery():
    """Test that constraints can be discovered in 37D space."""
    print("Testing 37D Constraint Discovery")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        constraint_discovery_rate=0.5,  # High rate for testing
        quiet_mode=False
    )
    
    print(f"\nField shape: {brain.unified_field.shape}")
    print(f"Total dimensions: {len(brain.unified_field.shape)}")
    print(f"Constraint system dimensions: {brain.constraint_field.n_dimensions}")
    
    # Process several cycles to trigger constraint discovery
    print("\nProcessing cycles to discover constraints...")
    
    for i in range(20):
        # Create varied input to generate interesting field dynamics
        input_data = [
            0.5 + 0.3 * np.sin(i * 0.2),
            0.5 + 0.3 * np.cos(i * 0.3),
            0.5 + 0.2 * np.sin(i * 0.1)
        ] + [0.5 + 0.1 * np.random.randn() for _ in range(21)]
        
        action, state = brain.process_robot_cycle(input_data)
        
        if i % 5 == 0:
            stats = brain.constraint_field.get_constraint_stats()
            print(f"\nCycle {i}:")
            print(f"  Active constraints: {stats['active_constraints']}")
            print(f"  Discovered total: {stats['constraints_discovered']}")
            print(f"  Enforced: {stats['constraints_enforced']}")
            if stats['constraint_types']:
                print(f"  Types: {stats['constraint_types']}")
    
    # Final analysis
    final_stats = brain.constraint_field.get_constraint_stats()
    print(f"\n\nFinal Constraint Statistics:")
    print(f"  Total discovered: {final_stats['constraints_discovered']}")
    print(f"  Currently active: {final_stats['active_constraints']}")
    print(f"  Total enforced: {final_stats['constraints_enforced']}")
    
    if final_stats['constraints_discovered'] > 0:
        print("✅ SUCCESS: Constraints discovered in 37D space!")
    else:
        print("⚠️  WARNING: No constraints discovered")
    
    return brain


def test_constraint_enforcement():
    """Test that constraints affect field evolution."""
    print("\n\nTesting Constraint Enforcement")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=8,
        constraint_discovery_rate=0.8,
        quiet_mode=True
    )
    
    # Manually create a test constraint
    test_constraint = FieldConstraintND(
        constraint_type=FieldConstraintType.GRADIENT_FLOW,
        dimensions=[0, 1, 2],  # Spatial dimensions
        region_centers=torch.tensor([[4.0, 4.0, 4.0]]),
        region_radii=torch.tensor([[2.0, 2.0, 2.0]]),
        strength=1.0,
        gradient_direction=torch.tensor([1.0, 0.0, -1.0]),
        discovery_timestamp=0.0
    )
    
    brain.constraint_field.active_constraints.append(test_constraint)
    
    # Measure field before and after constraint enforcement
    field_before = brain.unified_field.clone()
    
    # Apply constraint
    constraint_forces = brain.constraint_field.enforce_constraints(brain.unified_field)
    brain.unified_field += constraint_forces * 0.1
    
    # Check if field changed
    field_diff = torch.abs(brain.unified_field - field_before)
    total_change = torch.sum(field_diff).item()
    
    print(f"Field change from constraint: {total_change:.6f}")
    print(f"Max change: {torch.max(field_diff).item():.6f}")
    print(f"Affected elements: {torch.sum(field_diff > 0).item()}")
    
    if total_change > 0:
        print("✅ SUCCESS: Constraints successfully modify field!")
    else:
        print("❌ FAIL: Constraints had no effect")


def test_dimension_specific_constraints():
    """Test constraints that target specific dimension families."""
    print("\n\nTesting Dimension-Specific Constraints")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        quiet_mode=True
    )
    
    # Create constraints for different dimension families
    constraint_tests = [
        ("Spatial", [0, 1, 2]),
        ("Temporal", [4]),
        ("Oscillatory", [5, 6, 7]),
        ("Energy", [8, 9, 10])
    ]
    
    for name, dims in constraint_tests:
        # Create constraint
        constraint = FieldConstraintND(
            constraint_type=FieldConstraintType.ACTIVATION_THRESHOLD,
            dimensions=dims,
            region_centers=torch.zeros(1, len(dims)),
            region_radii=torch.ones(1, len(dims)) * float('inf'),
            strength=0.5,
            threshold_value=0.5,
            discovery_timestamp=0.0
        )
        
        brain.constraint_field.active_constraints = [constraint]
        
        # Test enforcement
        forces = brain.constraint_field.enforce_constraints(brain.unified_field)
        affected_dims = torch.any(forces != 0, dim=tuple(i for i in range(len(forces.shape)) if i not in dims))
        
        print(f"\n{name} constraint (dims {dims}):")
        print(f"  Forces generated: {torch.any(forces != 0).item()}")
        print(f"  Affects correct dims: {torch.any(affected_dims).item()}")


def test_constraint_performance():
    """Test performance with many constraints."""
    print("\n\nTesting Constraint System Performance")
    print("=" * 60)
    
    import time
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Add many constraints
    for i in range(50):
        constraint = FieldConstraintND(
            constraint_type=FieldConstraintType.GRADIENT_FLOW,
            dimensions=list(range(i % 10, min(i % 10 + 3, 37))),
            region_centers=torch.randn(1, 3) * 5 + 7.5,
            region_radii=torch.ones(1, 3) * 2.0,
            strength=0.3,
            gradient_direction=torch.randn(3),
            discovery_timestamp=0.0
        )
        brain.constraint_field.active_constraints.append(constraint)
    
    print(f"Testing with {len(brain.constraint_field.active_constraints)} constraints")
    
    # Time constraint enforcement
    start_time = time.perf_counter()
    
    for _ in range(10):
        forces = brain.constraint_field.enforce_constraints(brain.unified_field)
    
    elapsed = (time.perf_counter() - start_time) * 1000
    
    print(f"Time for 10 enforcements: {elapsed:.2f}ms")
    print(f"Average per enforcement: {elapsed/10:.2f}ms")
    
    if elapsed < 100:  # Less than 100ms for 10 enforcements
        print("✅ SUCCESS: Constraint system is performant!")
    else:
        print("⚠️  WARNING: Constraint system may be too slow")


def analyze_constraint_coverage():
    """Analyze which dimensions get constraints."""
    print("\n\nAnalyzing Constraint Dimension Coverage")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=10,
        constraint_discovery_rate=0.8,
        quiet_mode=True
    )
    
    # Run many cycles
    for i in range(50):
        input_data = [0.5 + 0.2 * np.random.randn() for _ in range(24)]
        brain.process_robot_cycle(input_data)
    
    # Analyze dimension coverage
    dimension_coverage = [0] * 37
    
    for constraint in brain.constraint_field.active_constraints:
        for dim in constraint.dimensions:
            if dim < 37:
                dimension_coverage[dim] += 1
    
    print("\nDimension coverage by constraints:")
    families = [
        ("Spatial (0-2)", range(0, 3)),
        ("Scale (3)", range(3, 4)),
        ("Time (4)", range(4, 5)),
        ("Oscillatory (5-7)", range(5, 8)),
        ("Flow (8-10)", range(8, 11)),
        ("Topology (11-12)", range(11, 13)),
        ("Energy (13-14)", range(13, 15)),
        ("Coupling (15-16)", range(15, 17)),
        ("Emergence (17-18)", range(17, 19))
    ]
    
    for family_name, dims in families:
        coverage = sum(dimension_coverage[d] for d in dims if d < 37)
        print(f"  {family_name}: {coverage} constraints")
    
    total_covered = sum(1 for c in dimension_coverage if c > 0)
    print(f"\nTotal dimensions with constraints: {total_covered}/37 ({total_covered/37*100:.1f}%)")
    
    if total_covered >= 10:
        print("✅ SUCCESS: Good dimension coverage!")
    else:
        print("⚠️  WARNING: Limited dimension coverage")


if __name__ == "__main__":
    # Run all tests
    brain = test_constraint_discovery()
    test_constraint_enforcement()
    test_dimension_specific_constraints()
    test_constraint_performance()
    analyze_constraint_coverage()
    
    print("\n\n✅ 37D Constraint System Testing Complete!")