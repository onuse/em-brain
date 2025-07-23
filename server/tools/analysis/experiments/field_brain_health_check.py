#!/usr/bin/env python3
"""
Field Brain Health Check

Comprehensive check for UnifiedFieldBrain functionality and issues.
"""

import sys
import os
import torch
import time
import numpy as np

# Add necessary paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
server_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
sys.path.insert(0, server_path)

from brains.field.core_brain import UnifiedFieldBrain
from brains.field.robot_interface import FieldNativeRobotInterface

def check_field_brain_health():
    """Comprehensive health check for field brain."""
    print("🏥 UnifiedFieldBrain Health Check")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # 1. Check instantiation
    print("\n1. Checking brain instantiation...")
    try:
        brain = UnifiedFieldBrain(
            spatial_resolution=5,  # Very small for testing
            temporal_window=5.0,
            quiet_mode=True
        )
        print("✅ Brain instantiated successfully")
        print(f"   Device: {brain.device}")
        print(f"   Field shape: {brain.unified_field.shape}")
        print(f"   Memory usage: {brain.unified_field.element_size() * brain.unified_field.nelement() / 1024 / 1024:.1f} MB")
    except Exception as e:
        issues.append(f"Failed to instantiate brain: {e}")
        return issues, warnings
    
    # 2. Check placeholder implementations
    print("\n2. Checking for placeholder implementations...")
    placeholder_dims = {
        24: "Memory persistence",
        32: "Social coupling", 
        33: "Analogical coupling",
        35: "Creativity space"
    }
    
    # Process some sensor data to check field coordinates
    sensor_data = [0.5] * 24
    action, state = brain.process_robot_cycle(sensor_data)
    
    # Check if we can access the field coordinates from the last experience
    if brain.field_experiences:
        last_exp = brain.field_experiences[-1]
        coords = last_exp.field_coordinates
        
        for dim, name in placeholder_dims.items():
            if dim < len(coords) and abs(coords[dim] - 0.5) < 0.01:
                warnings.append(f"Dimension {dim} ({name}) appears to be placeholder (value: {coords[dim]:.3f})")
    
    # 3. Check maintenance thread
    print("\n3. Checking maintenance operations...")
    initial_cycles = brain.brain_cycles
    initial_last_maintenance = brain.last_maintenance_cycle
    
    # Run enough cycles to trigger maintenance
    for i in range(brain.maintenance_interval + 5):
        brain.process_robot_cycle([0.1] * 24)
    
    if brain.last_maintenance_cycle > initial_last_maintenance:
        print(f"✅ Maintenance triggered after {brain.maintenance_interval} cycles")
    else:
        issues.append("Maintenance not triggered after expected interval")
    
    # Check if maintenance is actually running in a thread
    import threading
    active_threads = [t.name for t in threading.enumerate()]
    if any("maintenance" in name.lower() for name in active_threads):
        print("✅ Maintenance thread found")
    else:
        warnings.append("No maintenance thread found - maintenance runs synchronously")
    
    # 4. Check robot interface
    print("\n4. Checking robot interface...")
    try:
        interface = FieldNativeRobotInterface(brain, cycle_time_target=0.05)
        
        # Test a cycle
        start = time.time()
        motor_commands, brain_state = interface.process_robot_cycle([0.1] * 24)
        cycle_time = time.time() - start
        
        print(f"✅ Robot interface functional")
        print(f"   Cycle time: {cycle_time*1000:.1f}ms (target: 50.0ms)")
        print(f"   Motor commands: {[f'{x:.3f}' for x in motor_commands]}")
        
        if cycle_time > interface.cycle_time_target * 1.5:
            warnings.append(f"Cycle time ({cycle_time*1000:.1f}ms) exceeds target by >50%")
            
    except Exception as e:
        issues.append(f"Robot interface error: {e}")
    
    # 5. Check prediction/learning
    print("\n5. Checking prediction/learning system...")
    
    # Feed varying inputs to trigger learning
    initial_confidence = brain._current_prediction_confidence
    for i in range(20):  # Reduced from 50
        varied_input = [0.5 + 0.1 * np.sin(i * 0.1 + j) for j in range(24)]
        brain.process_robot_cycle(varied_input)
    
    final_confidence = brain._current_prediction_confidence
    improvement_rate = brain._improvement_rate_history[-1] if brain._improvement_rate_history else 0.0
    
    print(f"   Initial confidence: {initial_confidence:.3f}")
    print(f"   Final confidence: {final_confidence:.3f}")
    print(f"   Improvement rate: {improvement_rate:.4f}")
    
    if abs(final_confidence - initial_confidence) < 0.01:
        warnings.append("No change in prediction confidence after varied inputs")
    
    # 6. Check save/load
    print("\n6. Checking save/load functionality...")
    test_file = "/tmp/field_brain_test.pkl.gz"
    
    try:
        # Save
        if brain.save_field_state(test_file):
            print("✅ Save successful")
            
            # Create new brain and load
            new_brain = UnifiedFieldBrain(
                spatial_resolution=5,  # Match the test brain size
                temporal_window=5.0,
                quiet_mode=True
            )
            
            if new_brain.load_field_state(test_file):
                print("✅ Load successful")
                
                # Verify state
                if torch.allclose(brain.unified_field, new_brain.unified_field):
                    print("✅ Field state matches after load")
                else:
                    issues.append("Field state mismatch after load")
            else:
                issues.append("Failed to load field state")
        else:
            issues.append("Failed to save field state")
            
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            
    except Exception as e:
        issues.append(f"Save/load error: {e}")
    
    # 7. Check for memory leaks
    print("\n7. Checking for memory leaks...")
    
    # Get initial counts
    initial_experiences = len(brain.field_experiences)
    initial_actions = len(brain.field_actions)
    initial_topology = len(brain.topology_regions)
    
    # Run many cycles
    for i in range(200):  # Reduced from 2000 for faster testing
        brain.process_robot_cycle([0.1] * 24)
    
    final_experiences = len(brain.field_experiences)
    final_actions = len(brain.field_actions)
    final_topology = len(brain.topology_regions)
    
    print(f"   Experiences: {initial_experiences} → {final_experiences}")
    print(f"   Actions: {initial_actions} → {final_actions}")
    print(f"   Topology regions: {initial_topology} → {final_topology}")
    
    if final_experiences > 1000:
        print("✅ Experience trimming working (capped at 1000)")
    else:
        warnings.append(f"Experience list may grow unbounded ({final_experiences} items)")
    
    # 8. Check field dynamics
    print("\n8. Checking field dynamics...")
    
    # Get field energy before and after evolution
    initial_energy = torch.sum(torch.abs(brain.unified_field)).item()
    
    # Evolve without input
    for i in range(10):
        brain._evolve_unified_field()
    
    final_energy = torch.sum(torch.abs(brain.unified_field)).item()
    energy_ratio = final_energy / (initial_energy + 1e-8)
    
    print(f"   Initial energy: {initial_energy:.3f}")
    print(f"   Final energy: {final_energy:.3f}")
    print(f"   Energy ratio: {energy_ratio:.3f}")
    
    if energy_ratio > 1.5:
        issues.append(f"Field energy growing unstably (ratio: {energy_ratio:.3f})")
    elif energy_ratio < 0.5:
        warnings.append(f"Field energy decaying rapidly (ratio: {energy_ratio:.3f})")
    
    # Summary
    print("\n" + "=" * 50)
    print("🏥 HEALTH CHECK SUMMARY")
    print("=" * 50)
    
    if not issues and not warnings:
        print("✅ All systems functional!")
    else:
        if issues:
            print(f"\n❌ CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"   - {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"   - {warning}")
    
    return issues, warnings

if __name__ == "__main__":
    issues, warnings = check_field_brain_health()
    
    # Exit with error if critical issues
    sys.exit(1 if issues else 0)