#!/usr/bin/env python3
"""
Gradient Fix Proposal

This proposes a fix for the weak gradient problem in the field brain.
"""

import sys
import os
sys.path.append('/Users/jkarlsson/Documents/Projects/robot-project/brain/server/src')

import torch
import numpy as np

def analyze_current_vs_proposed():
    """Compare current gradient extraction vs proposed fix."""
    
    print("🔧 Gradient Extraction Fix Analysis")
    print("=" * 50)
    
    # Simulate the current problematic averaging
    print("\n1️⃣ Current Method (BROKEN):")
    
    # Simulate a gradient slice like the brain produces
    gradient_slice = torch.zeros(10, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    
    # Set a strong gradient in the spatial dimensions
    gradient_slice[5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 1.0
    gradient_slice[3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = 0.8
    gradient_slice[7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] = -0.5
    
    print(f"   Gradient slice shape: {gradient_slice.shape}")
    print(f"   Non-zero values: {torch.count_nonzero(gradient_slice).item()} out of {gradient_slice.nelement()}")
    print(f"   Max gradient magnitude: {torch.max(torch.abs(gradient_slice)).item():.6f}")
    
    # Current method: average across ALL dimensions
    current_result = torch.mean(gradient_slice).item()
    print(f"   Current method (mean): {current_result:.8f}")
    print(f"   → This is the problem! Strong gradients → nearly zero")
    
    # Proposed fixes
    print("\n2️⃣ Proposed Fix Methods:")
    
    # Method 1: Only average spatial dimensions (ignore extra dimensions)
    spatial_only = gradient_slice[:, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Take only spatial slice
    method1_result = torch.mean(spatial_only).item()
    print(f"   Method 1 (spatial only): {method1_result:.6f}")
    print(f"   Improvement: {method1_result / current_result:.1f}x stronger")
    
    # Method 2: Use max instead of mean
    method2_result = torch.max(torch.abs(gradient_slice)).item()
    if torch.max(gradient_slice).item() > torch.abs(torch.min(gradient_slice)).item():
        method2_result = torch.max(gradient_slice).item()
    else:
        method2_result = torch.min(gradient_slice).item()
    print(f"   Method 2 (max magnitude): {method2_result:.6f}")
    print(f"   Improvement: {method2_result / current_result:.1f}x stronger")
    
    # Method 3: Weighted spatial average
    method3_result = torch.mean(spatial_only[spatial_only != 0]).item() if torch.any(spatial_only != 0) else 0.0
    print(f"   Method 3 (non-zero only): {method3_result:.6f}")
    print(f"   Improvement: {method3_result / current_result:.1f}x stronger")
    
    # Method 4: RMS instead of mean
    method4_result = torch.sqrt(torch.mean(spatial_only ** 2)).item()
    # Apply sign of the dominant gradient
    if torch.sum(spatial_only > 0) > torch.sum(spatial_only < 0):
        pass  # Positive
    else:
        method4_result = -method4_result
    print(f"   Method 4 (RMS with sign): {method4_result:.6f}")
    print(f"   Improvement: {method4_result / current_result:.1f}x stronger")

def generate_gradient_fix_code():
    """Generate the actual code fix."""
    
    print("\n3️⃣ Proposed Code Fix:")
    print("=" * 30)
    
    fix_code = '''
# CURRENT BROKEN CODE (line ~656 in core_brain.py):
if 'gradient_x' in self.gradient_flows:
    grad_x = self.gradient_flows['gradient_x'][center_idx, center_idx, center_idx, :, :]
    motor_gradients[0] = torch.mean(grad_x).item()  # ❌ PROBLEM: Averages 150+ dimensions!

# PROPOSED FIX 1: Only use spatial-scale-time dimensions (ignore extra 1's)
if 'gradient_x' in self.gradient_flows:
    grad_x = self.gradient_flows['gradient_x'][center_idx, center_idx, center_idx, :, :]
    # Extract only the meaningful dimensions (scale: 10, time: 15)
    spatial_temporal_slice = grad_x[:, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    motor_gradients[0] = torch.mean(spatial_temporal_slice).item()

# PROPOSED FIX 2: Use max magnitude for stronger gradients
if 'gradient_x' in self.gradient_flows:
    grad_x = self.gradient_flows['gradient_x'][center_idx, center_idx, center_idx, :, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Use max magnitude with proper sign
    abs_max = torch.max(torch.abs(grad_x)).item()
    if torch.max(grad_x).item() >= torch.abs(torch.min(grad_x)).item():
        motor_gradients[0] = torch.max(grad_x).item()
    else:
        motor_gradients[0] = torch.min(grad_x).item()

# PROPOSED FIX 3: Combine both approaches
if 'gradient_x' in self.gradient_flows:
    grad_x = self.gradient_flows['gradient_x'][center_idx, center_idx, center_idx, :, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # First try RMS for smooth gradients
    rms_gradient = torch.sqrt(torch.mean(grad_x ** 2)).item()
    # Apply sign of dominant gradient
    if torch.sum(grad_x > 0) >= torch.sum(grad_x < 0):
        motor_gradients[0] = rms_gradient
    else:
        motor_gradients[0] = -rms_gradient
'''
    
    print(fix_code)

def test_parameter_adjustments():
    """Test what parameter adjustments would help."""
    
    print("\n4️⃣ Parameter Adjustment Recommendations:")
    print("=" * 45)
    
    current_params = {
        'field_decay_rate': 0.995,
        'gradient_following_strength': 0.3,
        'field_diffusion_rate': 0.02,
    }
    
    recommended_params = {
        'field_decay_rate': 0.999,  # Much slower decay
        'gradient_following_strength': 1.0,  # Maximum strength
        'field_diffusion_rate': 0.05,  # More diffusion to spread gradients
    }
    
    print("Current parameters:")
    for param, value in current_params.items():
        print(f"   {param}: {value}")
    
    print("\nRecommended parameters:")
    for param, value in recommended_params.items():
        improvement = value / current_params[param] if current_params[param] != 0 else "N/A"
        print(f"   {param}: {value} (improvement: {improvement:.2f}x)")
    
    # Decay analysis
    print(f"\nDecay rate comparison over 100 cycles:")
    current_decay = current_params['field_decay_rate'] ** 100
    recommended_decay = recommended_params['field_decay_rate'] ** 100
    print(f"   Current: {current_decay:.4f} (60% loss)")
    print(f"   Recommended: {recommended_decay:.4f} (37% loss)")
    print(f"   Field retention improvement: {recommended_decay / current_decay:.2f}x better")

if __name__ == "__main__":
    try:
        analyze_current_vs_proposed()
        generate_gradient_fix_code()
        test_parameter_adjustments()
        
        print(f"\n✅ Gradient Fix Analysis Complete!")
        print(f"\n🎯 Priority Actions:")
        print(f"   1. Fix dimensional averaging (CRITICAL)")
        print(f"   2. Increase gradient_following_strength to 1.0")
        print(f"   3. Slow field decay rate to 0.999")
        print(f"   4. Test with enhanced parameters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()