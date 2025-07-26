#!/usr/bin/env python3
"""
Deep dive into prediction learning mechanism
Understanding exactly what happens at each step
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add brain server to path
brain_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_server_path))

from src.brain_factory import BrainFactory
from src.adaptive_configuration import load_adaptive_configuration

def deep_dive_prediction_mechanism():
    """Deep dive into every aspect of prediction learning"""
    print("🔬 DEEP DIVE: PREDICTION LEARNING MECHANISM")
    print("=" * 60)
    
    # Create brain with minimal quiet mode for debugging
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    ub = brain.brain
    
    print("\n📋 System Overview:")
    print(f"   Field shape: {ub.unified_field.shape}")
    print(f"   Field elements: {ub.unified_field.numel():,}")
    print(f"   Spatial resolution: {ub.spatial_resolution}³")
    print(f"   Prediction region: 3x3x3 around center")
    
    # Test pattern
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    
    print("\n🔍 CYCLE-BY-CYCLE ANALYSIS:")
    print("-" * 60)
    
    # Cycle 0: Initial state
    print("\n🔄 Initial State:")
    print(f"   Field max: {ub.unified_field.max().item():.6f}")
    print(f"   Field mean: {ub.unified_field.mean().item():.6f}")
    print(f"   Field non-zero elements: {(ub.unified_field != 0).sum().item()}")
    print(f"   Has _predicted_field: {hasattr(ub, '_predicted_field')}")
    
    # Cycle 1: First prediction
    print("\n🔄 Cycle 1 - First Input:")
    print("   Step 1: No prediction comparison (no _predicted_field yet)")
    
    # Manually trace through the first cycle
    # Save field state before processing
    field_before = ub.unified_field.clone()
    
    _, state1 = brain.process_sensory_input(pattern)
    
    # Check field after sensory input but before evolution
    print(f"   Step 2: After sensory input applied")
    print(f"      Field max: {ub.unified_field.max().item():.6f}")
    print(f"      Field changed by sensory: {torch.mean(torch.abs(ub.unified_field - field_before)).item():.8f}")
    
    print(f"   Step 3: Field evolution occurred")
    print(f"      Now has _predicted_field: {hasattr(ub, '_predicted_field')}")
    
    if hasattr(ub, '_predicted_field'):
        print(f"      Predicted field max: {ub._predicted_field.max().item():.6f}")
        pred_diff = torch.mean(torch.abs(ub._predicted_field - ub.unified_field)).item()
        print(f"      Difference between predicted and current: {pred_diff:.8f}")
    
    print(f"   Prediction confidence: {state1['prediction_confidence']:.6f}")
    
    # Cycle 2: First real prediction
    print("\n🔄 Cycle 2 - First Real Prediction:")
    
    # Let's manually check what the prediction will be
    center = ub.spatial_resolution // 2
    region_slice = slice(center-1, center+2)
    
    # Get the predicted region BEFORE processing
    predicted_region = ub._predicted_field[region_slice, region_slice, region_slice].clone()
    print(f"   Predicted region stats:")
    print(f"      Shape: {predicted_region.shape}")
    print(f"      Max: {predicted_region.max().item():.6f}")
    print(f"      Mean: {predicted_region.mean().item():.6f}")
    print(f"      Non-zero: {(predicted_region != 0).sum().item()}")
    
    # Get actual region BEFORE sensory input
    actual_region_before = ub.unified_field[region_slice, region_slice, region_slice].clone()
    print(f"   Actual region BEFORE sensory:")
    print(f"      Max: {actual_region_before.max().item():.6f}")
    print(f"      Mean: {actual_region_before.mean().item():.6f}")
    
    # Calculate what the error SHOULD be
    manual_error = torch.mean(torch.abs(predicted_region - actual_region_before)).item()
    print(f"   Manual prediction error: {manual_error:.8f}")
    print(f"   This should give confidence: {1.0 / (1.0 + manual_error * 100.0):.6f}")
    
    # Now process and see what actually happens
    _, state2 = brain.process_sensory_input(pattern)
    print(f"   Actual confidence reported: {state2['prediction_confidence']:.6f}")
    
    # Investigate further cycles
    print("\n🔄 Cycles 3-10 - Pattern Analysis:")
    for i in range(3, 11):
        # Check prediction region evolution
        pred_region = ub._predicted_field[region_slice, region_slice, region_slice]
        actual_region = ub.unified_field[region_slice, region_slice, region_slice]
        
        _, state = brain.process_sensory_input(pattern)
        
        print(f"   Cycle {i}:")
        print(f"      Predicted region max: {pred_region.max().item():.3f}")
        print(f"      Actual region max: {actual_region.max().item():.3f}")
        print(f"      Confidence: {state['prediction_confidence']:.6f}")
    
    # Deep analysis of the prediction region
    print("\n🔬 PREDICTION REGION ANALYSIS:")
    print("-" * 60)
    
    # What's in the 3x3x3 region?
    final_pred_region = ub._predicted_field[region_slice, region_slice, region_slice]
    final_actual_region = ub.unified_field[region_slice, region_slice, region_slice]
    
    print(f"   Region shape: {final_pred_region.shape}")
    print(f"   Total elements in region: {final_pred_region.numel():,}")
    
    # Check spatial slices
    print("\n   Spatial structure (center z-slice):")
    center_z = 1  # Middle of 3x3x3
    spatial_slice = final_pred_region[:, :, center_z, 0, 0, 0, 0, 0, 0, 0, 0]
    print(f"   {spatial_slice}")
    
    # Why is prediction error so small?
    print("\n🤔 HYPOTHESIS TESTING:")
    print("-" * 60)
    
    print("\n1. Is the field evolution too subtle?")
    field_before_evolution = ub.unified_field.clone()
    ub._evolve_unified_field()
    evolution_change = torch.mean(torch.abs(ub.unified_field - field_before_evolution)).item()
    print(f"   Field change from evolution: {evolution_change:.8f}")
    
    print("\n2. Is the prediction region too small?")
    print(f"   Current region: 3x3x3 = 27 spatial points")
    print(f"   But full tensor at each point: {final_pred_region[0,0,0].numel()} dimensions")
    print(f"   Total prediction elements: {27 * final_pred_region[0,0,0].numel()}")
    
    print("\n3. Are the patterns too repetitive?")
    print(f"   Input pattern: {pattern[:8]}...")
    print(f"   Pattern uniqueness: {len(set(pattern))} unique values out of {len(pattern)}")
    
    print("\n4. Is the sensitivity still wrong?")
    test_errors = [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    print("   Error -> Confidence mapping:")
    for err in test_errors:
        conf = 1.0 / (1.0 + err * 100.0)
        print(f"      {err:.5f} -> {conf:.6f}")

def analyze_field_evolution_impact():
    """Analyze how field evolution affects predictions"""
    print("\n\n🔬 FIELD EVOLUTION IMPACT ANALYSIS")
    print("=" * 60)
    
    config = load_adaptive_configuration("settings.json")
    brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
    ub = brain.brain
    
    pattern = [0.5, 0.8, 0.3, 0.6] * 4
    
    # Build up some field state
    for _ in range(5):
        brain.process_sensory_input(pattern)
    
    print("\n📊 Evolution Component Analysis:")
    
    # Test each evolution component
    print("\n1. Field Decay:")
    field_before = ub.unified_field.clone()
    ub.unified_field *= ub.field_decay_rate
    decay_change = torch.mean(torch.abs(ub.unified_field - field_before)).item()
    print(f"   Decay rate: {ub.field_decay_rate}")
    print(f"   Change from decay: {decay_change:.8f}")
    
    print("\n2. Spatial Diffusion:")
    field_before = ub.unified_field.clone()
    ub._apply_spatial_diffusion()
    diffusion_change = torch.mean(torch.abs(ub.unified_field - field_before)).item()
    print(f"   Diffusion rate: {ub.field_diffusion_rate}")
    print(f"   Change from diffusion: {diffusion_change:.8f}")
    
    print("\n3. Constraint Evolution:")
    field_before = ub.unified_field.clone()
    ub._apply_constraint_guided_evolution()
    constraint_change = torch.mean(torch.abs(ub.unified_field - field_before)).item()
    print(f"   Change from constraints: {constraint_change:.8f}")
    
    print("\n4. Energy Normalization:")
    current_energy = torch.sum(torch.abs(ub.unified_field)).item()
    print(f"   Current energy: {current_energy:.0f}")
    print(f"   Normalization threshold: {100000 * 1.5:.0f}")
    print(f"   Normalization active: {current_energy > 150000}")

def test_different_patterns():
    """Test prediction with different input patterns"""
    print("\n\n🔬 TESTING DIFFERENT PATTERNS")
    print("=" * 60)
    
    config = load_adaptive_configuration("settings.json")
    
    patterns = [
        ([0.5, 0.8, 0.3, 0.6] * 4, "Original repetitive"),
        ([np.random.random() for _ in range(16)], "Random"),
        ([0.1] * 16, "All low"),
        ([0.9] * 16, "All high"),
        ([i/15 for i in range(16)], "Linear gradient"),
        ([0.0 if i % 2 == 0 else 1.0 for i in range(16)], "Binary alternating"),
    ]
    
    for pattern, desc in patterns:
        print(f"\n📊 Pattern: {desc}")
        print(f"   Values: {pattern[:4]}...")
        
        # Fresh brain for each pattern
        brain = BrainFactory(config=config, enable_logging=False, quiet_mode=True)
        
        confidences = []
        for i in range(10):
            _, state = brain.process_sensory_input(pattern)
            conf = state['prediction_confidence']
            confidences.append(conf)
        
        print(f"   Confidences: {[f'{c:.3f}' for c in confidences[:5]]}...")
        print(f"   Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")

def recommendation_summary():
    """Summarize findings and recommendations"""
    print("\n\n📋 FINDINGS AND RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n🔍 Key Findings:")
    print("1. Prediction comparison happens at the right time (before sensory input)")
    print("2. Field evolution is too subtle to create meaningful prediction errors")
    print("3. The 3x3x3 prediction region might be too small")
    print("4. Sensitivity of 100 might still be too high")
    
    print("\n💡 Recommendations:")
    print("1. Increase field evolution impact (stronger dynamics)")
    print("2. Use larger prediction region or full-field comparison")
    print("3. Add noise or variation to make prediction non-trivial")
    print("4. Consider prediction of change rather than absolute state")

if __name__ == "__main__":
    deep_dive_prediction_mechanism()
    analyze_field_evolution_impact()
    test_different_patterns()
    recommendation_summary()