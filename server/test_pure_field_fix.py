#!/usr/bin/env python3
"""
Test that PureFieldBrain works with actual robot dimensions
"""

import sys
import torch
sys.path.append('.')

from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS

def test_hardware_constrained():
    """Test with the actual hardware-constrained configuration"""
    print("Testing PureFieldBrain with hardware-constrained config...")
    
    # Use the actual dimensions from the error
    input_dim = 24  # Robot sends 24 sensory channels
    output_dim = 4   # Robot expects 4 motor outputs
    
    # Create brain with hardware_constrained config (6³×64)
    brain = PureFieldBrain(
        input_dim=input_dim,
        output_dim=output_dim,
        scale_config=SCALE_CONFIGS['hardware_constrained'],
        aggressive=True
    )
    
    print(f"✅ Brain created: {brain.scale_config.name}")
    print(f"   Input: {input_dim} channels")
    print(f"   Output: {output_dim} channels")
    print(f"   Field: {brain.scale_config.levels[0][0]}³×{brain.scale_config.levels[0][1]}")
    
    # Test forward pass
    sensory_input = torch.randn(input_dim)
    
    try:
        motor_output = brain(sensory_input)
        print(f"✅ Forward pass successful!")
        print(f"   Motor output shape: {motor_output.shape}")
        print(f"   Motor values: {motor_output.detach().numpy()}")
        
        # Test multiple cycles
        for i in range(10):
            motor = brain(torch.randn(input_dim), reward=0.1 if i % 3 == 0 else 0.0)
        
        print(f"✅ 10 cycles completed successfully")
        print(f"   Final metrics: {brain._practical_metrics}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_variable_inputs():
    """Test that brain handles variable input dimensions"""
    print("\nTesting variable input dimensions...")
    
    brain = PureFieldBrain(
        input_dim=10,  # Initialize with 10
        output_dim=4,
        scale_config=SCALE_CONFIGS['hardware_constrained']
    )
    
    # Test with different input sizes
    test_sizes = [10, 16, 24, 32]
    
    for size in test_sizes:
        try:
            sensory = torch.randn(size)
            motor = brain(sensory)
            print(f"✅ Input size {size}: OK (output: {motor.shape})")
        except Exception as e:
            print(f"❌ Input size {size}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("PureFieldBrain Dimension Fix Test")
    print("="*60)
    
    success = True
    
    # Test 1: Hardware constrained with actual robot dimensions
    if not test_hardware_constrained():
        success = False
    
    # Test 2: Variable input handling
    if not test_variable_inputs():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed! PureFieldBrain is ready for deployment.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("="*60)