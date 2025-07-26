#!/usr/bin/env python3
"""
Calculate memory requirements for the brain at different resolutions
"""

def calculate_memory(spatial_res, config="minimal"):
    """Calculate memory usage for different configurations"""
    
    if config == "minimal":
        # Minimal viable dimensions for embedded
        field_shape = [spatial_res, spatial_res, spatial_res, 
                      8, 10, 2, 2, 2, 1, 2, 1]
    else:
        # Current M1 Mac configuration
        field_shape = [spatial_res, spatial_res, spatial_res,
                      10, 15, 4, 4, 3, 2, 3, 2]
    
    # Calculate tensor size
    total_elements = 1
    for dim in field_shape:
        total_elements *= dim
    
    # Memory calculations
    tensor_bytes = total_elements * 4  # float32
    tensor_mb = tensor_bytes / (1024 * 1024)
    
    # Additional memory overhead
    # - PyTorch framework: ~100-200MB
    # - Python runtime: ~50-100MB  
    # - Brain data structures: ~50MB
    # - OS overhead: ~100MB
    overhead_mb = 300  # Conservative estimate
    
    # Topology regions (each ~1KB)
    topology_mb = 0.001 * 100  # Assume 100 regions
    
    # Experience buffer
    experience_mb = 0.0002 * 1000  # 1000 experiences at 200 bytes each
    
    total_mb = tensor_mb + overhead_mb + topology_mb + experience_mb
    
    return {
        'tensor_elements': total_elements,
        'tensor_mb': tensor_mb,
        'overhead_mb': overhead_mb,
        'total_mb': total_mb,
        'shape': field_shape
    }


def main():
    print("🧮 Brain Memory Requirements")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        ("M1 Mac (4³)", 4, "full"),
        ("Pi Zero 2 (3³)", 3, "full"),
        ("Pi Zero 2 (3³ minimal)", 3, "minimal"),
        ("Pi Zero 2 (2³ minimal)", 2, "minimal"),
    ]
    
    print(f"{'Configuration':<25} | {'Tensor MB':<10} | {'Total MB':<10} | {'Feasible?'}")
    print("-" * 70)
    
    for name, res, config in configs:
        result = calculate_memory(res, config)
        feasible = "✅ Yes" if result['total_mb'] < 400 else "❌ No"
        
        print(f"{name:<25} | {result['tensor_mb']:<10.1f} | {result['total_mb']:<10.1f} | {feasible}")
        
        if "minimal" in name:
            print(f"  Shape: {result['shape']}")
    
    print("\nPi Zero 2 WH has 512MB total RAM")
    print("After OS, need ~400MB free for brain")


if __name__ == "__main__":
    main()