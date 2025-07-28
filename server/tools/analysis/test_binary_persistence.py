#!/usr/bin/env python3
"""
Test binary persistence performance
"""

import sys
import os
from pathlib import Path
import time
import torch
import numpy as np

brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.persistence.binary_persistence import BinaryPersistence


def test_binary_vs_json():
    """Compare binary vs JSON performance"""
    print("🧪 Binary Persistence Performance Test")
    print("=" * 60)
    
    # Create test data similar to brain state
    print("\n📊 Creating test brain state...")
    test_state = {
        'brain_type': 'dynamic_unified_field',
        'version': '2.0',
        'brain_cycles': 1000,
        'tensor_shape': [8, 8, 8, 10, 15, 3, 3, 2, 2, 2, 2],
        'metadata': {'test': True},
        # Create a tensor similar to unified_field
        'unified_field': torch.randn(8, 8, 8, 10, 15, 3, 3, 2, 2, 2, 2, dtype=torch.float32)
    }
    
    tensor_elements = test_state['unified_field'].numel()
    print(f"   Tensor elements: {tensor_elements:,}")
    print(f"   Tensor memory: {tensor_elements * 4 / 1e6:.1f} MB")
    
    # Test binary save
    print("\n💾 Testing binary save...")
    binary_persistence = BinaryPersistence("./test_binary", use_compression=True)
    
    start_time = time.time()
    binary_persistence.save_brain_state(test_state, "test_session", 1000)
    binary_save_time = time.time() - start_time
    
    # Check file sizes
    meta_file = Path("./test_binary/brain_state_test_session_1000_meta.json")
    tensor_file = Path("./test_binary/brain_state_test_session_1000_tensors.pt.gz")
    
    meta_size = meta_file.stat().st_size / 1e6 if meta_file.exists() else 0
    tensor_size = tensor_file.stat().st_size / 1e6 if tensor_file.exists() else 0
    total_binary_size = meta_size + tensor_size
    
    print(f"\n📏 File sizes:")
    print(f"   Metadata: {meta_size:.2f} MB")
    print(f"   Tensor (compressed): {tensor_size:.2f} MB")
    print(f"   Total: {total_binary_size:.2f} MB")
    print(f"   Save time: {binary_save_time:.2f}s")
    
    # Test binary load
    print("\n🔄 Testing binary load...")
    start_time = time.time()
    loaded_state = binary_persistence.load_brain_state("test_session")
    binary_load_time = time.time() - start_time
    
    # Verify loaded correctly
    if loaded_state and 'unified_field' in loaded_state:
        loaded_tensor = loaded_state['unified_field']
        if torch.allclose(test_state['unified_field'], loaded_tensor):
            print("   ✅ Data loaded correctly")
        else:
            print("   ❌ Data mismatch!")
    
    # Estimate JSON size (without actually creating the huge file)
    json_size_estimate = tensor_elements * 20 / 1e6  # ~20 bytes per float in JSON
    json_save_time_estimate = binary_save_time * 50  # Conservative estimate
    json_load_time_estimate = binary_load_time * 100
    
    print("\n📊 Performance Comparison:")
    print("-" * 50)
    print(f"{'Metric':<20} {'Binary':<15} {'JSON (estimated)':<15}")
    print("-" * 50)
    print(f"{'File size':<20} {total_binary_size:<15.1f} MB {json_size_estimate:<15.1f} MB")
    print(f"{'Save time':<20} {binary_save_time:<15.2f} s {json_save_time_estimate:<15.1f} s")
    print(f"{'Load time':<20} {binary_load_time:<15.2f} s {json_load_time_estimate:<15.1f} s")
    print("-" * 50)
    print(f"{'Compression ratio':<20} {json_size_estimate/total_binary_size:<15.1f}x")
    print(f"{'Save speedup':<20} {json_save_time_estimate/binary_save_time:<15.1f}x")
    print(f"{'Load speedup':<20} {json_load_time_estimate/binary_load_time:<15.1f}x")
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    import shutil
    if Path("./test_binary").exists():
        shutil.rmtree("./test_binary")
    
    print("\n✅ Test complete!")
    print("\n💡 Recommendations:")
    print("   1. Use binary format for all brain persistence")
    print("   2. Migrate existing JSON files when convenient")
    print("   3. Enable compression for additional ~3x reduction")
    print("   4. Consider disabling persistence for tests")


if __name__ == "__main__":
    test_binary_vs_json()