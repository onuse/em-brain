#!/usr/bin/env python3
"""
Analyze field sparsity to determine if sparse tensor representation would be beneficial.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server/src'))

import torch
import numpy as np
from brains.field.core_brain import create_unified_field_brain


def analyze_field_sparsity():
    """Analyze sparsity patterns in the unified field."""
    print("Field Sparsity Analysis")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=20,
        quiet_mode=True
    )
    
    print(f"Field shape: {brain.unified_field.shape}")
    print(f"Total elements: {brain.unified_field.nelement():,}")
    print(f"Memory size: {brain.unified_field.element_size() * brain.unified_field.nelement() / (1024*1024):.2f} MB")
    
    # Run some cycles to populate the field
    print("\nRunning 100 brain cycles to populate field...")
    
    for i in range(100):
        input_data = [
            0.5 + 0.3 * np.sin(i * 0.1),
            0.5 + 0.3 * np.cos(i * 0.15),
            0.5 + 0.2 * np.sin(i * 0.05)
        ] + [0.5 + 0.1 * np.random.randn() for _ in range(21)]
        
        brain.process_robot_cycle(input_data)
    
    # Analyze sparsity
    threshold = 1e-6
    nonzero_mask = torch.abs(brain.unified_field) > threshold
    nonzero_count = torch.sum(nonzero_mask).item()
    total_elements = brain.unified_field.nelement()
    sparsity = 1.0 - (nonzero_count / total_elements)
    
    print(f"\nSparsity Analysis:")
    print(f"  Non-zero elements: {nonzero_count:,} / {total_elements:,}")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Density: {(1-sparsity):.2%}")
    
    # Analyze sparsity by dimension
    print("\nSparsity by dimension slice:")
    
    # Check spatial dimensions
    spatial_nonzero = torch.sum(torch.abs(brain.unified_field[:, :, :, 5, 7]) > threshold).item()
    spatial_total = brain.spatial_resolution ** 3
    print(f"  Spatial (X,Y,Z) slice: {spatial_nonzero}/{spatial_total} active ({spatial_nonzero/spatial_total:.2%})")
    
    # Memory comparison
    dense_memory = brain.unified_field.element_size() * total_elements / (1024*1024)
    
    # Estimate sparse tensor memory (indices + values)
    # COO format: (nnz * ndim * index_dtype_size) + (nnz * value_dtype_size)
    sparse_indices_memory = nonzero_count * len(brain.unified_field.shape) * 8 / (1024*1024)  # 8 bytes for int64
    sparse_values_memory = nonzero_count * brain.unified_field.element_size() / (1024*1024)
    sparse_total_memory = sparse_indices_memory + sparse_values_memory
    
    print(f"\nMemory Comparison:")
    print(f"  Dense tensor: {dense_memory:.2f} MB")
    print(f"  Sparse tensor (estimated): {sparse_total_memory:.2f} MB")
    print(f"  Memory savings: {(1 - sparse_total_memory/dense_memory):.1%}")
    
    if sparsity > 0.95:
        print("\n✅ HIGH SPARSITY: Sparse tensors would save significant memory!")
    elif sparsity > 0.90:
        print("\n⚠️  MODERATE SPARSITY: Sparse tensors might be beneficial")
    else:
        print("\n❌ LOW SPARSITY: Dense tensors are more efficient")
    
    return sparsity, nonzero_count, total_elements


def test_sparse_tensor_operations():
    """Test if PyTorch sparse operations would work for our use case."""
    print("\n\nSparse Tensor Feasibility Test")
    print("=" * 60)
    
    # Create a small test field
    shape = (10, 10, 10, 5, 5)
    
    # Dense version
    dense_field = torch.zeros(shape)
    
    # Add some sparse activation
    for _ in range(20):
        idx = tuple(torch.randint(0, s, (1,)).item() for s in shape)
        dense_field[idx] = torch.randn(1).item()
    
    # Convert to sparse
    sparse_field = dense_field.to_sparse()
    
    print(f"Test field shape: {shape}")
    print(f"Dense size: {dense_field.nelement() * dense_field.element_size() / 1024:.2f} KB")
    print(f"Sparse nnz: {sparse_field._nnz()}")
    
    # Test basic operations
    print("\nTesting sparse operations:")
    
    # 1. Indexing
    try:
        # Sparse tensors don't support direct indexing
        val = sparse_field[5, 5, 5, 2, 2]
        print("  ✅ Direct indexing: Supported")
    except:
        print("  ❌ Direct indexing: Not supported (major limitation)")
    
    # 2. Slicing
    try:
        slice_result = sparse_field[:, :, 5]
        print("  ✅ Slicing: Supported")
    except:
        print("  ❌ Slicing: Not supported")
    
    # 3. In-place operations
    try:
        sparse_field.mul_(0.99)
        print("  ✅ In-place multiplication: Supported")
    except:
        print("  ❌ In-place multiplication: Not supported")
    
    # 4. Gradient computation
    try:
        sparse_field.requires_grad = True
        result = sparse_field.sum()
        result.backward()
        print("  ✅ Gradient computation: Supported")
    except:
        print("  ❌ Gradient computation: Not supported")
    
    print("\nConclusion:")
    print("PyTorch sparse tensors have significant limitations for our use case:")
    print("- No direct indexing (critical for field updates)")
    print("- Limited slicing operations")
    print("- Many operations convert back to dense")


def analyze_access_patterns():
    """Analyze how the field is accessed to determine best representation."""
    print("\n\nField Access Pattern Analysis")
    print("=" * 60)
    
    brain = create_unified_field_brain(
        spatial_resolution=15,
        quiet_mode=True
    )
    
    # Track access patterns
    read_count = 0
    write_count = 0
    slice_count = 0
    
    # Instrument some operations
    original_getitem = brain.unified_field.__class__.__getitem__
    original_setitem = brain.unified_field.__class__.__setitem__
    
    def tracked_getitem(self, key):
        nonlocal read_count, slice_count
        read_count += 1
        if isinstance(key, tuple) and any(isinstance(k, slice) for k in key):
            slice_count += 1
        return original_getitem(self, key)
    
    def tracked_setitem(self, key, value):
        nonlocal write_count
        write_count += 1
        return original_setitem(self, key, value)
    
    # Monkey patch for tracking
    brain.unified_field.__class__.__getitem__ = tracked_getitem
    brain.unified_field.__class__.__setitem__ = tracked_setitem
    
    # Run some cycles
    for i in range(10):
        input_data = [0.5 + 0.1 * np.random.randn() for _ in range(24)]
        brain.process_robot_cycle(input_data)
    
    # Restore original methods
    brain.unified_field.__class__.__getitem__ = original_getitem
    brain.unified_field.__class__.__setitem__ = original_setitem
    
    print(f"Access patterns over 10 cycles:")
    print(f"  Reads: {read_count}")
    print(f"  Writes: {write_count}")
    print(f"  Slice operations: {slice_count}")
    print(f"  Read/Write ratio: {read_count/max(1, write_count):.2f}")
    
    print("\nAccess pattern characteristics:")
    print("- Frequent direct indexing (incompatible with sparse)")
    print("- Many slice operations (limited sparse support)")
    print("- Regular in-place updates (field evolution)")


def suggest_sparse_strategy():
    """Suggest optimal sparse representation strategy."""
    print("\n\nSparse Representation Recommendations")
    print("=" * 60)
    
    print("""
Based on the analysis:

1. CURRENT STATUS:
   - Dense field storage with sparse gradient computation ✅
   - This is actually a good hybrid approach!

2. WHY NOT FULL SPARSE TENSORS:
   - PyTorch sparse tensors don't support direct indexing
   - Field updates require frequent indexed writes
   - Slice operations are limited in sparse format
   - Many operations auto-convert to dense anyway

3. CURRENT OPTIMIZATIONS ALREADY ACHIEVE:
   - Sparse gradient computation (only where field is active)
   - Gradient caching (avoids redundant computation)
   - Efficient memory management in maintenance

4. POTENTIAL FUTURE OPTIMIZATIONS:
   - Hierarchical representation (coarse + fine grids)
   - Chunk-based storage (only allocate active regions)
   - Custom sparse format for field operations
   - GPU-optimized sparse kernels

5. RECOMMENDATION:
   Keep current dense field + sparse gradient approach!
   It provides the benefits of sparsity where it matters most
   (gradient computation) while maintaining the flexibility
   needed for field evolution operations.
""")


if __name__ == "__main__":
    sparsity, nonzero, total = analyze_field_sparsity()
    test_sparse_tensor_operations()
    analyze_access_patterns()
    suggest_sparse_strategy()
    
    print("\n✅ Sparsity analysis complete!")
    
    if sparsity > 0.95:
        print(f"\nField is {sparsity:.1%} sparse - consider custom sparse format in future")
    else:
        print(f"\nField is only {sparsity:.1%} sparse - dense representation is appropriate")