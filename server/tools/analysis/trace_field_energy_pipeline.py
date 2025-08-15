#!/usr/bin/env python3
"""
Field Energy Pipeline Tracer

Comprehensive test to trace exactly where field energy values spike
beyond reasonable bounds. Adds logging at every critical point in the pipeline.
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

from brains.field.generic_brain import GenericFieldBrain
from brains.field.enhanced_dynamics import EnhancedFieldDynamics, PhaseTransitionConfig
from brains.field.field_types import FieldDimension, FieldDynamicsFamily, UnifiedFieldExperience

class EnergyTracer:
    """Traces energy values through the field pipeline."""
    
    def __init__(self):
        self.energy_log = []
        self.intensity_log = []
        self.step_count = 0
    
    def log_step(self, location: str, value: float, description: str = ""):
        """Log energy/intensity at a specific pipeline step."""
        self.step_count += 1
        entry = {
            'step': self.step_count,
            'location': location,
            'value': value,
            'description': description,
            'timestamp': time.time()
        }
        
        if 'energy' in location.lower():
            self.energy_log.append(entry)
        else:
            self.intensity_log.append(entry)
        
        # Print immediately for real-time tracking
        if value > 10.0:  # Flag suspicious values
            print(f"🚨 Step {self.step_count}: {location} = {value:.6f} - {description}")
        elif value > 1.0:
            print(f"⚠️  Step {self.step_count}: {location} = {value:.6f} - {description}")
        else:
            print(f"✅ Step {self.step_count}: {location} = {value:.6f} - {description}")
    
    def print_summary(self):
        """Print comprehensive summary of energy progression."""
        print("\n" + "="*80)
        print("🔍 FIELD ENERGY PIPELINE TRACE SUMMARY")
        print("="*80)
        
        print(f"\n📊 Total Steps Traced: {self.step_count}")
        
        # Find maximum values
        max_intensity = max((entry['value'] for entry in self.intensity_log), default=0)
        max_energy = max((entry['value'] for entry in self.energy_log), default=0)
        
        print(f"📈 Maximum Intensity: {max_intensity:.6f}")
        print(f"📈 Maximum Energy: {max_energy:.6f}")
        
        # Find first spike
        first_spike = None
        for entry in self.intensity_log + self.energy_log:
            if entry['value'] > 10.0:
                first_spike = entry
                break
        
        if first_spike:
            print(f"\n🚨 FIRST SPIKE DETECTED:")
            print(f"   Location: {first_spike['location']}")
            print(f"   Value: {first_spike['value']:.6f}")
            print(f"   Description: {first_spike['description']}")
        
        # Print progression
        print(f"\n📋 Energy Progression:")
        for entry in self.energy_log:
            status = "🚨" if entry['value'] > 10.0 else "⚠️" if entry['value'] > 1.0 else "✅"
            print(f"   {status} {entry['location']}: {entry['value']:.6f}")
        
        print(f"\n📋 Intensity Progression:")
        for entry in self.intensity_log:
            status = "🚨" if entry['value'] > 10.0 else "⚠️" if entry['value'] > 1.0 else "✅"
            print(f"   {status} {entry['location']}: {entry['value']:.6f}")

# Global tracer instance
tracer = EnergyTracer()

def create_test_field_dimensions():
    """Create the standard 37D field dimensions."""
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

def test_input_stream_processing():
    """Test the input stream to field experience conversion."""
    print("🧪 Testing Input Stream Processing...")
    
    # Create a simple input stream
    test_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 16D input
    
    tracer.log_step("input_stream_max", max(test_input), "Maximum value in input stream")
    tracer.log_step("input_stream_norm", np.linalg.norm(test_input), "L2 norm of input stream")
    
    # Create field brain for testing
    field_brain = GenericFieldBrain(
        spatial_resolution=20,  # Smaller for faster testing
        quiet_mode=True
    )
    
    # Negotiate stream capabilities
    from brains.field.field_types import StreamCapabilities
    capabilities = StreamCapabilities(
        input_dimensions=16,
        output_dimensions=4,
        input_labels=[f"sensor_{i}" for i in range(16)],
        output_labels=[f"motor_{i}" for i in range(4)]
    )
    
    field_brain.negotiate_stream_capabilities(capabilities)
    
    # Process through the brain
    output_stream, brain_state = field_brain.process_input_stream(test_input)
    
    # Log brain state values
    if brain_state:
        tracer.log_step("brain_state_energy", brain_state.get('field_total_energy', 0.0), 
                       "Total field energy from brain state")
        
        field_coords = brain_state.get('field_coordinates')
        if field_coords is not None:
            tracer.log_step("field_coords_norm", torch.norm(field_coords).item(), 
                           "L2 norm of field coordinates")
            tracer.log_step("field_coords_max", torch.max(torch.abs(field_coords)).item(), 
                           "Maximum absolute field coordinate")
    
    return field_brain

def test_experience_creation(field_brain):
    """Test direct experience creation and imprinting."""
    print("\n🧪 Testing Experience Creation...")
    
    # Create test coordinates
    test_coords = torch.randn(37) * 0.5  # Start with small values
    tracer.log_step("test_coords_norm", torch.norm(test_coords).item(), 
                   "L2 norm of test coordinates")
    
    # Calculate intensity the way generic_brain does it
    raw_intensity = torch.norm(test_coords).item()
    tracer.log_step("raw_intensity", raw_intensity, "Raw intensity from torch.norm")
    
    normalized_intensity = raw_intensity / np.sqrt(len(test_coords))
    tracer.log_step("normalized_intensity", normalized_intensity, "Normalized intensity")
    
    final_intensity = min(normalized_intensity, 1.0)
    tracer.log_step("final_intensity", final_intensity, "Final clamped intensity")
    
    # Create experience
    experience = UnifiedFieldExperience(
        timestamp=time.time(),
        field_coordinates=test_coords,
        raw_input_stream=torch.zeros(16),
        field_intensity=final_intensity,
        dynamics_family_activations={
            FieldDynamicsFamily.ENERGY: final_intensity,
            FieldDynamicsFamily.TOPOLOGY: final_intensity * 0.7,
            FieldDynamicsFamily.EMERGENCE: final_intensity * 0.5
        }
    )
    
    # Log experience values
    tracer.log_step("experience_intensity", experience.field_intensity, 
                   "Experience field intensity")
    
    # Get field stats before imprinting
    stats_before = field_brain.field_impl.get_field_statistics()
    tracer.log_step("field_energy_before", stats_before.get('total_activation', 0.0), 
                   "Field energy before imprinting")
    
    # Imprint experience
    field_brain.field_impl.imprint_experience(experience)
    
    # Get field stats after imprinting
    stats_after = field_brain.field_impl.get_field_statistics()
    tracer.log_step("field_energy_after", stats_after.get('total_activation', 0.0), 
                   "Field energy after imprinting")
    
    # Calculate energy delta
    energy_delta = stats_after.get('total_activation', 0.0) - stats_before.get('total_activation', 0.0)
    tracer.log_step("energy_delta", energy_delta, "Energy added by single imprint")
    
    return experience

def test_enhanced_dynamics(field_brain):
    """Test enhanced dynamics behavior."""
    print("\n🧪 Testing Enhanced Dynamics...")
    
    if not hasattr(field_brain, 'enhanced_dynamics') or field_brain.enhanced_dynamics is None:
        print("⚠️ Enhanced dynamics not enabled - skipping test")
        return
    
    enhanced = field_brain.enhanced_dynamics
    
    # Get initial energy
    initial_energy = enhanced.global_energy_level
    tracer.log_step("enhanced_initial_energy", initial_energy, "Enhanced dynamics initial energy")
    
    # Run evolution cycles
    for cycle in range(5):
        enhanced.evolve_with_enhancements(dt=0.1)
        
        current_energy = enhanced.global_energy_level
        tracer.log_step(f"enhanced_cycle_{cycle}_energy", current_energy, 
                       f"Enhanced dynamics energy after cycle {cycle}")
        
        # Check for attractor creation
        if enhanced.active_attractors:
            for i, attractor in enumerate(enhanced.active_attractors):
                attractor_intensity = attractor.get('intensity', 0.0)
                tracer.log_step(f"attractor_{i}_intensity", attractor_intensity, 
                               f"Attractor {i} intensity")

def test_multiple_imprints(field_brain):
    """Test behavior with multiple experience imprints."""
    print("\n🧪 Testing Multiple Imprints...")
    
    initial_stats = field_brain.field_impl.get_field_statistics()
    initial_energy = initial_stats.get('total_activation', 0.0)
    tracer.log_step("multi_initial_energy", initial_energy, "Initial energy before multiple imprints")
    
    # Imprint multiple experiences
    for i in range(10):
        # Create varied experiences
        coords = torch.randn(37) * 0.3  # Small random coordinates
        intensity = 0.1 + i * 0.05  # Gradually increasing intensity
        
        tracer.log_step(f"imprint_{i}_input_intensity", intensity, f"Input intensity for imprint {i}")
        
        experience = UnifiedFieldExperience(
            timestamp=time.time(),
            field_coordinates=coords,
            raw_input_stream=torch.zeros(16),
            field_intensity=intensity,
            dynamics_family_activations={
                FieldDynamicsFamily.ENERGY: intensity,
            }
        )
        
        field_brain.field_impl.imprint_experience(experience)
        
        # Check energy after each imprint
        stats = field_brain.field_impl.get_field_statistics()
        current_energy = stats.get('total_activation', 0.0)
        tracer.log_step(f"imprint_{i}_field_energy", current_energy, 
                       f"Field energy after imprint {i}")

def main():
    """Run comprehensive field energy pipeline trace."""
    print("🔍 Field Energy Pipeline Tracer")
    print("=" * 80)
    print("Tracing energy values through the complete field processing pipeline...")
    print()
    
    try:
        # Test 1: Input stream processing
        field_brain = test_input_stream_processing()
        
        # Test 2: Experience creation
        test_experience_creation(field_brain)
        
        # Test 3: Enhanced dynamics
        test_enhanced_dynamics(field_brain)
        
        # Test 4: Multiple imprints
        test_multiple_imprints(field_brain)
        
        # Print comprehensive summary
        tracer.print_summary()
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Still print partial results
        tracer.print_summary()

if __name__ == "__main__":
    main()