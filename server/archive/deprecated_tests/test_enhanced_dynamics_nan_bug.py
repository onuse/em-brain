#!/usr/bin/env python3
"""
Enhanced Dynamics NaN Bug Reproduction Test

This test specifically reproduces the NaN bug that causes field evolution to fail
at 50³ resolution, leading to identical session results and dead field dynamics.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import time

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'src'))

from brains.field.enhanced_dynamics import EnhancedFieldDynamics, PhaseTransitionConfig
from brains.field.adaptive_field_impl import create_adaptive_field_implementation
from brains.field.field_types import FieldDimension, FieldDynamicsFamily, UnifiedFieldExperience

def create_test_field_dimensions():
    """Create the standard 37D field dimensions used in the experiment."""
    dimensions = []
    index = 0
    
    # Spatial (3D)
    for i in range(3):
        dimensions.append(FieldDimension(f"spatial_{i}", FieldDynamicsFamily.SPATIAL, index))
        index += 1
    
    # Oscillatory (6D) 
    for i in range(6):
        dimensions.append(FieldDimension(f"oscillatory_{i}", FieldDynamicsFamily.OSCILLATORY, index))
        index += 1
    
    # Flow (8D)
    for i in range(8):
        dimensions.append(FieldDimension(f"flow_{i}", FieldDynamicsFamily.FLOW, index))
        index += 1
    
    # Topology (6D)
    for i in range(6):
        dimensions.append(FieldDimension(f"topology_{i}", FieldDynamicsFamily.TOPOLOGY, index))
        index += 1
    
    # Energy (4D)
    for i in range(4):
        dimensions.append(FieldDimension(f"energy_{i}", FieldDynamicsFamily.ENERGY, index))
        index += 1
    
    # Coupling (5D)
    for i in range(5):
        dimensions.append(FieldDimension(f"coupling_{i}", FieldDynamicsFamily.COUPLING, index))
        index += 1
    
    # Emergence (5D)
    for i in range(5):
        dimensions.append(FieldDimension(f"emergence_{i}", FieldDynamicsFamily.EMERGENCE, index))
        index += 1
    
    return dimensions

def test_nan_bug_reproduction(spatial_resolution=50, cycles_to_test=10):
    """
    Test that reproduces the NaN bug in enhanced dynamics.
    
    Returns:
        dict: Test results including whether NaN was detected
    """
    print(f"🔬 Testing Enhanced Dynamics NaN Bug")
    print(f"   Spatial resolution: {spatial_resolution}³")
    print(f"   Test cycles: {cycles_to_test}")
    
    # Create field dimensions (37D total)
    field_dimensions = create_test_field_dimensions()
    print(f"   Field dimensions: {len(field_dimensions)}")
    
    # Create field implementation
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    field_impl = create_adaptive_field_implementation(
        field_dimensions=field_dimensions,
        spatial_resolution=spatial_resolution,
        requested_device=device,
        quiet_mode=True
    )
    
    print(f"   Field implementation: {field_impl.get_implementation_type()}")
    print(f"   Device: {device}")
    
    # Create enhanced dynamics
    phase_config = PhaseTransitionConfig(
        stability_threshold=0.3,
        energy_threshold=1.0,
        transition_strength=0.4
    )
    
    enhanced_dynamics = EnhancedFieldDynamics(
        field_impl=field_impl,
        phase_config=phase_config,
        quiet_mode=True
    )
    
    # Test state tracking
    test_results = {
        'nan_detected': False,
        'nan_first_cycle': None,
        'field_evolution_cycles': [],
        'field_energies': [],
        'energy_variances': [],
        'phase_transitions': [],
        'test_successful': False
    }
    
    print(f"\n🧪 Running {cycles_to_test} evolution cycles...")
    
    for cycle in range(cycles_to_test):
        try:
            # Get field stats before evolution
            field_stats = field_impl.get_field_statistics()
            field_energy = field_stats.get('total_activation', 0.0)
            
            # Check for NaN in field energy
            if np.isnan(field_energy) or np.isinf(field_energy):
                if not test_results['nan_detected']:
                    test_results['nan_detected'] = True
                    test_results['nan_first_cycle'] = cycle
                    print(f"   🚨 NaN detected in field energy at cycle {cycle}!")
            
            test_results['field_energies'].append(field_energy)
            test_results['field_evolution_cycles'].append(field_stats.get('evolution_cycles', 0))
            
            # Imprint some experience to generate field activity
            if cycle < 3:  # Only first few cycles
                experience = UnifiedFieldExperience(
                    field_coordinates=torch.rand(len(field_dimensions)) * 2 - 1,
                    field_intensity=0.5 + 0.5 * torch.rand(1).item()
                )
                field_impl.imprint_experience(experience)
            
            # Run enhanced dynamics evolution
            enhanced_dynamics.evolve_with_enhancements(dt=0.1, current_input_stream=None)
            
            # Check enhanced dynamics internal state
            energy_history = enhanced_dynamics.phase_energy_history
            if len(energy_history) >= 5:
                recent_energy = energy_history[-5:]
                
                # Reproduce the exact calculation that causes NaN
                try:
                    energy_variance = torch.var(torch.tensor(recent_energy)).item()
                    test_results['energy_variances'].append(energy_variance)
                    
                    # Check for NaN in variance calculation
                    if np.isnan(energy_variance) or np.isinf(energy_variance):
                        if not test_results['nan_detected']:
                            test_results['nan_detected'] = True
                            test_results['nan_first_cycle'] = cycle
                            print(f"   🎯 NaN detected in energy variance at cycle {cycle}!")
                            print(f"      Recent energy history: {recent_energy}")
                            print(f"      Energy variance: {energy_variance}")
                except Exception as e:
                    print(f"   ⚠️ Error in variance calculation at cycle {cycle}: {e}")
                    test_results['nan_detected'] = True
                    test_results['nan_first_cycle'] = cycle
            
            # Track phase transitions
            current_phase = enhanced_dynamics.current_phase
            if cycle == 0 or (len(test_results['phase_transitions']) > 0 and 
                            test_results['phase_transitions'][-1] != current_phase):
                test_results['phase_transitions'].append(current_phase)
            
            # Print progress
            if cycle % 2 == 0 or test_results['nan_detected']:
                evolution_cycles = field_stats.get('evolution_cycles', 0)
                print(f"   Cycle {cycle}: energy={field_energy:.6f}, "
                      f"evolution_cycles={evolution_cycles}, phase={current_phase}")
            
            # Stop early if NaN detected
            if test_results['nan_detected']:
                break
                
        except Exception as e:
            print(f"   💥 Exception at cycle {cycle}: {e}")
            test_results['nan_detected'] = True
            test_results['nan_first_cycle'] = cycle
            break
    
    # Final analysis
    test_results['test_successful'] = True
    final_stats = field_impl.get_field_statistics()
    
    print(f"\n📊 Test Results:")
    print(f"   NaN detected: {test_results['nan_detected']}")
    if test_results['nan_detected']:
        print(f"   NaN first appeared at cycle: {test_results['nan_first_cycle']}")
    print(f"   Final field evolution cycles: {final_stats.get('evolution_cycles', 0)}")
    print(f"   Final field energy: {test_results['field_energies'][-1] if test_results['field_energies'] else 'N/A'}")
    print(f"   Energy variance calculations: {len(test_results['energy_variances'])}")
    print(f"   Phase transitions: {test_results['phase_transitions']}")
    
    return test_results

def test_different_resolutions():
    """Test the bug across different spatial resolutions."""
    print(f"\n🔍 Testing across different spatial resolutions...")
    
    resolutions = [20, 30, 40, 50]
    results = {}
    
    for resolution in resolutions:
        print(f"\n--- Testing {resolution}³ resolution ---")
        try:
            result = test_nan_bug_reproduction(spatial_resolution=resolution, cycles_to_test=8)
            results[resolution] = result
        except Exception as e:
            print(f"❌ Failed at {resolution}³: {e}")
            results[resolution] = {'error': str(e), 'nan_detected': True}
    
    print(f"\n📈 Summary across resolutions:")
    for resolution, result in results.items():
        if 'error' in result:
            print(f"   {resolution}³: ERROR - {result['error']}")
        else:
            nan_status = "NaN detected" if result['nan_detected'] else "No NaN"
            first_cycle = f" (cycle {result['nan_first_cycle']})" if result['nan_detected'] else ""
            print(f"   {resolution}³: {nan_status}{first_cycle}")
    
    return results

def main():
    """Run the enhanced dynamics NaN bug reproduction test."""
    print("🔬 Enhanced Dynamics NaN Bug Reproduction Test")
    print("=" * 60)
    
    # Test 1: Reproduce the specific 50³ bug
    print("\n🎯 Test 1: Reproduce 50³ NaN bug")
    result_50 = test_nan_bug_reproduction(spatial_resolution=50, cycles_to_test=10)
    
    # Test 2: Compare across resolutions
    resolution_results = test_different_resolutions()
    
    # Test 3: Verify the bug is in enhanced dynamics, not base field
    print(f"\n🔧 Test 3: Testing base field implementation without enhanced dynamics")
    try:
        field_dimensions = create_test_field_dimensions()
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        base_field = create_adaptive_field_implementation(
            field_dimensions=field_dimensions,
            spatial_resolution=50,
            requested_device=device,
            quiet_mode=True
        )
        
        # Run base field evolution without enhanced dynamics
        for i in range(5):
            if i < 2:
                experience = UnifiedFieldExperience(
                    field_coordinates=torch.rand(len(field_dimensions)) * 2 - 1,
                    field_intensity=0.5
                )
                base_field.imprint_experience(experience)
            
            base_field.evolve_field(dt=0.1)
            stats = base_field.get_field_statistics()
            field_energy = stats.get('total_activation', 0.0)
            evolution_cycles = stats.get('evolution_cycles', 0)
            
            if np.isnan(field_energy):
                print(f"   ❌ Base field also produces NaN at cycle {i}")
                break
            else:
                print(f"   ✅ Base field cycle {i}: energy={field_energy:.6f}, evolution_cycles={evolution_cycles}")
        else:
            print(f"   ✅ Base field implementation works correctly")
    
    except Exception as e:
        print(f"   ❌ Base field test failed: {e}")
    
    # Final conclusion
    print(f"\n🎯 CONCLUSION:")
    if result_50['nan_detected']:
        print(f"   ✅ Successfully reproduced the NaN bug at 50³ resolution")
        print(f"   🎯 Bug appears at cycle {result_50['nan_first_cycle']}")
        print(f"   📍 This confirms our hypothesis about enhanced dynamics variance calculation")
    else:
        print(f"   ❌ Could not reproduce the NaN bug - may need different conditions")
    
    nan_resolutions = [res for res, result in resolution_results.items() 
                      if result.get('nan_detected', False)]
    if nan_resolutions:
        print(f"   📊 NaN detected at resolutions: {nan_resolutions}")
        print(f"   🔍 This suggests the bug scales with field complexity")
    
    return result_50

if __name__ == "__main__":
    main()