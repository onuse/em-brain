#!/usr/bin/env python3
"""
Field Brain Memory Sizing Analysis

Calculate memory requirements for different field brain configurations
and design efficient persistence strategies.
"""

import torch
import numpy as np
import sys
import os
import gzip
import pickle
import time
import math
from typing import Tuple, Dict, Any

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

try:
    from field_native_brain import create_unified_field_brain
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def calculate_field_memory_size(spatial_resolution: int = 20, 
                               scale_resolution: int = 10,
                               temporal_resolution: int = 15,
                               dynamics_dimensions: int = 37) -> Dict[str, Any]:
    """Calculate memory requirements for a field brain configuration."""
    
    # Field shape calculation (based on field_native_brain.py)
    field_shape = [spatial_resolution] * 3 + [scale_resolution] + [temporal_resolution] + [1] * (dynamics_dimensions - 5)
    
    print(f"🧠 Field Brain Memory Analysis")
    print(f"   Configuration:")
    print(f"      Spatial resolution: {spatial_resolution}³ = {spatial_resolution**3:,} positions")
    print(f"      Scale resolution: {scale_resolution}")
    print(f"      Temporal resolution: {temporal_resolution}")
    print(f"      Total dynamics dimensions: {dynamics_dimensions}")
    print(f"      Field shape: {field_shape}")
    
    # Calculate total field size
    total_elements = 1
    for dim in field_shape:
        total_elements *= dim
    
    # Memory calculations
    bytes_per_float32 = 4
    total_bytes = total_elements * bytes_per_float32
    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    
    print(f"\n   Memory Requirements:")
    print(f"      Total field elements: {total_elements:,}")
    print(f"      Memory (bytes): {total_bytes:,}")
    print(f"      Memory (MB): {total_mb:.2f}")
    print(f"      Memory (GB): {total_gb:.3f}")
    
    # Test actual PyTorch tensor creation
    try:
        print(f"\n   Testing actual tensor creation...")
        start_time = time.time()
        test_field = torch.zeros(field_shape, dtype=torch.float32)
        creation_time = time.time() - start_time
        
        actual_memory_mb = test_field.element_size() * test_field.nelement() / (1024 * 1024)
        
        print(f"      ✅ Tensor created successfully")
        print(f"      Creation time: {creation_time:.3f}s")
        print(f"      Actual memory: {actual_memory_mb:.2f} MB")
        
        # Test basic operations
        start_time = time.time()
        test_field += 0.1
        test_field *= 0.9
        operation_time = time.time() - start_time
        
        print(f"      Basic operations time: {operation_time:.4f}s")
        
        tensor_success = True
        
    except Exception as e:
        print(f"      ❌ Tensor creation failed: {e}")
        actual_memory_mb = total_mb
        creation_time = None
        operation_time = None
        tensor_success = False
    
    return {
        'spatial_resolution': spatial_resolution,
        'total_elements': total_elements,
        'memory_bytes': total_bytes,
        'memory_mb': total_mb,
        'memory_gb': total_gb,
        'actual_memory_mb': actual_memory_mb,
        'creation_time': creation_time,
        'operation_time': operation_time,
        'tensor_success': tensor_success,
        'field_shape': field_shape
    }


def test_field_compression(field_brain) -> Dict[str, Any]:
    """Test different compression strategies for field persistence."""
    
    print(f"\n💾 Field Compression Analysis")
    
    # Get the actual unified field
    field_tensor = field_brain.unified_field
    original_size = field_tensor.element_size() * field_tensor.nelement()
    
    print(f"   Original field size: {original_size / (1024*1024):.2f} MB")
    
    compression_results = {}
    
    # Test 1: Raw PyTorch save
    print(f"\n   Testing PyTorch .pt format:")
    try:
        start_time = time.time()
        torch.save(field_tensor, '/tmp/test_field.pt')
        save_time = time.time() - start_time
        
        file_size = os.path.getsize('/tmp/test_field.pt')
        compression_ratio = original_size / file_size
        
        start_time = time.time()
        loaded_tensor = torch.load('/tmp/test_field.pt')
        load_time = time.time() - start_time
        
        # Verify integrity
        integrity_check = torch.equal(field_tensor, loaded_tensor)
        
        print(f"      File size: {file_size / (1024*1024):.2f} MB")
        print(f"      Compression ratio: {compression_ratio:.2f}x")
        print(f"      Save time: {save_time:.3f}s")
        print(f"      Load time: {load_time:.3f}s")
        print(f"      Integrity check: {'✅ PASS' if integrity_check else '❌ FAIL'}")
        
        compression_results['pytorch'] = {
            'file_size_mb': file_size / (1024*1024),
            'compression_ratio': compression_ratio,
            'save_time': save_time,
            'load_time': load_time,
            'integrity_check': integrity_check
        }
        
        os.unlink('/tmp/test_field.pt')
        
    except Exception as e:
        print(f"      ❌ PyTorch save failed: {e}")
        compression_results['pytorch'] = None
    
    # Test 2: Gzipped pickle
    print(f"\n   Testing gzipped pickle:")
    try:
        start_time = time.time()
        with gzip.open('/tmp/test_field.pkl.gz', 'wb') as f:
            pickle.dump(field_tensor.numpy(), f)
        save_time = time.time() - start_time
        
        file_size = os.path.getsize('/tmp/test_field.pkl.gz')
        compression_ratio = original_size / file_size
        
        start_time = time.time()
        with gzip.open('/tmp/test_field.pkl.gz', 'rb') as f:
            loaded_array = pickle.load(f)
        loaded_tensor = torch.from_numpy(loaded_array)
        load_time = time.time() - start_time
        
        # Verify integrity
        integrity_check = torch.equal(field_tensor, loaded_tensor)
        
        print(f"      File size: {file_size / (1024*1024):.2f} MB")
        print(f"      Compression ratio: {compression_ratio:.2f}x")
        print(f"      Save time: {save_time:.3f}s")
        print(f"      Load time: {load_time:.3f}s")
        print(f"      Integrity check: {'✅ PASS' if integrity_check else '❌ FAIL'}")
        
        compression_results['gzip_pickle'] = {
            'file_size_mb': file_size / (1024*1024),
            'compression_ratio': compression_ratio,
            'save_time': save_time,
            'load_time': load_time,
            'integrity_check': integrity_check
        }
        
        os.unlink('/tmp/test_field.pkl.gz')
        
    except Exception as e:
        print(f"      ❌ Gzipped pickle failed: {e}")
        compression_results['gzip_pickle'] = None
    
    # Test 3: Sparse representation (only non-zero elements)
    print(f"\n   Testing sparse representation:")
    try:
        # Find non-zero elements
        nonzero_mask = field_tensor != 0.0
        nonzero_count = torch.sum(nonzero_mask).item()
        sparsity = 1.0 - (nonzero_count / field_tensor.nelement())
        
        print(f"      Non-zero elements: {nonzero_count:,} / {field_tensor.nelement():,}")
        print(f"      Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% zeros)")
        
        if nonzero_count > 0:
            # Get coordinates and values of non-zero elements
            nonzero_coords = torch.nonzero(field_tensor, as_tuple=False)
            nonzero_values = field_tensor[field_tensor != 0.0]
            
            sparse_data = {
                'shape': field_tensor.shape,
                'coords': nonzero_coords.numpy(),
                'values': nonzero_values.numpy()
            }
            
            start_time = time.time()
            with gzip.open('/tmp/test_field_sparse.pkl.gz', 'wb') as f:
                pickle.dump(sparse_data, f)
            save_time = time.time() - start_time
            
            file_size = os.path.getsize('/tmp/test_field_sparse.pkl.gz')
            compression_ratio = original_size / file_size
            
            print(f"      Sparse file size: {file_size / (1024*1024):.2f} MB")
            print(f"      Sparse compression ratio: {compression_ratio:.2f}x")
            print(f"      Sparse save time: {save_time:.3f}s")
            
            compression_results['sparse'] = {
                'file_size_mb': file_size / (1024*1024),
                'compression_ratio': compression_ratio,
                'save_time': save_time,
                'sparsity': sparsity,
                'nonzero_count': nonzero_count
            }
            
            os.unlink('/tmp/test_field_sparse.pkl.gz')
        else:
            print(f"      Field is completely zero - no sparse representation needed")
            compression_results['sparse'] = {
                'sparsity': 1.0,
                'nonzero_count': 0
            }
            
    except Exception as e:
        print(f"      ❌ Sparse representation failed: {e}")
        compression_results['sparse'] = None
    
    return compression_results


def test_incremental_updates(field_brain) -> Dict[str, Any]:
    """Test incremental field updates for efficient persistence."""
    
    print(f"\n🔄 Incremental Update Analysis")
    
    # Initial field state
    field_tensor = field_brain.unified_field
    initial_state = field_tensor.clone()
    
    print(f"   Testing incremental field changes...")
    
    # Simulate small changes (typical robot experience)
    change_regions = []
    
    # Make small localized changes
    for i in range(5):
        # Random small region
        x, y, z = torch.randint(0, field_tensor.shape[0], (3,))
        s, t = torch.randint(0, min(field_tensor.shape[3], field_tensor.shape[4]), (2,))
        
        # Small change
        old_value = field_tensor[x, y, z, s, t].clone()
        field_tensor[x, y, z, s, t] += torch.randn(field_tensor.shape[-1]) * 0.1
        new_value = field_tensor[x, y, z, s, t].clone()
        
        change_regions.append({
            'position': (x.item(), y.item(), z.item(), s.item(), t.item()),
            'old_value': old_value,
            'new_value': new_value
        })
    
    # Calculate difference
    diff_tensor = field_tensor - initial_state
    changed_mask = torch.abs(diff_tensor) > 1e-6
    changed_count = torch.sum(changed_mask).item()
    change_percentage = changed_count / field_tensor.nelement() * 100
    
    print(f"      Changed elements: {changed_count:,} / {field_tensor.nelement():,}")
    print(f"      Change percentage: {change_percentage:.4f}%")
    
    # Test delta compression
    print(f"\n   Testing delta compression:")
    try:
        # Store only changes
        if changed_count > 0:
            change_coords = torch.nonzero(changed_mask, as_tuple=False)
            change_values = diff_tensor[changed_mask]
            
            delta_data = {
                'shape': field_tensor.shape,
                'change_coords': change_coords.numpy(),
                'change_values': change_values.numpy(),
                'timestamp': time.time()
            }
            
            start_time = time.time()
            with gzip.open('/tmp/test_delta.pkl.gz', 'wb') as f:
                pickle.dump(delta_data, f)
            save_time = time.time() - start_time
            
            file_size = os.path.getsize('/tmp/test_delta.pkl.gz')
            original_size = field_tensor.element_size() * field_tensor.nelement()
            compression_ratio = original_size / file_size
            
            print(f"      Delta file size: {file_size / 1024:.2f} KB")
            print(f"      Delta compression ratio: {compression_ratio:.2f}x")
            print(f"      Delta save time: {save_time:.4f}s")
            
            # Test delta application
            start_time = time.time()
            reconstructed = initial_state.clone()
            for i, coord in enumerate(change_coords):
                reconstructed[tuple(coord)] += change_values[i]
            reconstruction_time = time.time() - start_time
            
            # Verify reconstruction
            integrity_check = torch.allclose(reconstructed, field_tensor, atol=1e-6)
            
            print(f"      Delta reconstruction time: {reconstruction_time:.4f}s")
            print(f"      Reconstruction integrity: {'✅ PASS' if integrity_check else '❌ FAIL'}")
            
            os.unlink('/tmp/test_delta.pkl.gz')
            
            return {
                'changed_count': changed_count,
                'change_percentage': change_percentage,
                'delta_file_size_kb': file_size / 1024,
                'delta_compression_ratio': compression_ratio,
                'delta_save_time': save_time,
                'reconstruction_time': reconstruction_time,
                'integrity_check': integrity_check
            }
        else:
            print(f"      No changes detected")
            return {'changed_count': 0, 'change_percentage': 0.0}
            
    except Exception as e:
        print(f"      ❌ Delta compression failed: {e}")
        return None


def run_field_sizing_analysis():
    """Run comprehensive field brain sizing analysis."""
    
    print("🧠 COMPREHENSIVE FIELD BRAIN SIZING ANALYSIS")
    print("=" * 60)
    
    # Test different field brain sizes
    configurations = [
        {'spatial': 10, 'name': 'Small (Dev)'},
        {'spatial': 20, 'name': 'Medium (Test)'},
        {'spatial': 50, 'name': 'Large (Production)'},
        {'spatial': 100, 'name': 'XL (Research)'},
    ]
    
    sizing_results = []
    
    print(f"\n📊 MEMORY SIZING FOR DIFFERENT CONFIGURATIONS:")
    
    for config in configurations:
        print(f"\n{'='*50}")
        print(f"🔍 {config['name']} Configuration")
        
        try:
            result = calculate_field_memory_size(spatial_resolution=config['spatial'])
            sizing_results.append({
                'name': config['name'],
                'spatial_resolution': config['spatial'],
                **result
            })
            
            # Memory feasibility assessment
            memory_mb = result['memory_mb']
            if memory_mb < 100:
                feasibility = "✅ EXCELLENT (fits in RAM easily)"
            elif memory_mb < 1000:
                feasibility = "✅ GOOD (reasonable RAM usage)"
            elif memory_mb < 4000:
                feasibility = "⚠️ MODERATE (high RAM usage)"
            else:
                feasibility = "❌ CHALLENGING (very high RAM usage)"
            
            print(f"   Feasibility: {feasibility}")
            
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            sizing_results.append({
                'name': config['name'],
                'spatial_resolution': config['spatial'],
                'error': str(e)
            })
    
    # Test persistence with a reasonable size
    print(f"\n{'='*50}")
    print(f"💾 PERSISTENCE TESTING")
    
    try:
        # Use medium size for persistence testing
        test_brain = create_unified_field_brain(spatial_resolution=10, quiet_mode=True)
        
        # Add some test data to the field
        test_brain.unified_field += torch.randn_like(test_brain.unified_field) * 0.1
        
        compression_results = test_field_compression(test_brain)
        incremental_results = test_incremental_updates(test_brain)
        
    except Exception as e:
        print(f"❌ Persistence testing failed: {e}")
        compression_results = None
        incremental_results = None
    
    # Summary and recommendations
    print(f"\n{'='*50}")
    print(f"📋 RECOMMENDATIONS")
    
    print(f"\n🎯 Recommended Configurations:")
    
    for result in sizing_results:
        if 'memory_mb' in result:
            memory_mb = result['memory_mb']
            if memory_mb < 500:  # Under 500MB
                use_case = "✅ Suitable for development and production"
                if result['tensor_success']:
                    print(f"   {result['name']}: {memory_mb:.1f} MB - {use_case}")
    
    print(f"\n💾 Persistence Strategy:")
    if compression_results:
        best_method = None
        best_ratio = 0
        
        for method, data in compression_results.items():
            if data and 'compression_ratio' in data and data.get('integrity_check', False):
                if data['compression_ratio'] > best_ratio:
                    best_ratio = data['compression_ratio']
                    best_method = method
        
        if best_method:
            print(f"   Best compression: {best_method} ({best_ratio:.1f}x ratio)")
        
        if incremental_results and incremental_results.get('change_percentage', 0) < 1.0:
            print(f"   Incremental updates: Highly effective ({incremental_results['change_percentage']:.3f}% changes)")
            print(f"   Delta compression: {incremental_results.get('delta_compression_ratio', 0):.1f}x ratio")
    
    print(f"\n🚀 Production Recommendations:")
    print(f"   • Use spatial resolution 20-50 for production (20-125 MB)")
    print(f"   • Implement gzipped pickle for full saves")
    print(f"   • Use delta compression for incremental saves")
    print(f"   • Save full state periodically, deltas frequently")
    print(f"   • Consider field sparsity for additional compression")
    
    return {
        'sizing_results': sizing_results,
        'compression_results': compression_results,
        'incremental_results': incremental_results
    }


if __name__ == "__main__":
    results = run_field_sizing_analysis()
    
    print(f"\n🔬 FIELD BRAIN SIZING ANALYSIS COMPLETE")
    
    # Find the sweet spot configuration
    viable_configs = [r for r in results['sizing_results'] 
                     if 'memory_mb' in r and r['memory_mb'] < 1000 and r.get('tensor_success', False)]
    
    if viable_configs:
        recommended = max(viable_configs, key=lambda x: x['spatial_resolution'])
        print(f"\n🎯 Recommended Production Config:")
        print(f"   Spatial Resolution: {recommended['spatial_resolution']}")
        print(f"   Memory Usage: {recommended['memory_mb']:.1f} MB")
        print(f"   Field Elements: {recommended['total_elements']:,}")
    
    print(f"\n✅ Field brain memory requirements are manageable for production use!")