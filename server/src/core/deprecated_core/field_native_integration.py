#!/usr/bin/env python3
"""
Field-Native Sensor Integration

Shows how our current architecture ALREADY supports direct field injection!
The brain makes decisions based on field state, not buffer contents.
This means sensors could write directly to field regions.

This is a natural evolution, not a rewrite!
"""

import torch
import threading
import time
import numpy as np
from typing import Dict, Tuple


class FieldNativeSensorIntegration:
    """
    Sensors can inject directly into the field instead of through buffers.
    
    Current flow:
        Sensor → Buffer → Sensory Vector → Injection → Field → Motor
    
    Possible flow:
        Sensor → Direct Field Injection → Motor
        
    The brain already only cares about field state!
    """
    
    def __init__(self, brain):
        """
        Args:
            brain: PureFieldBrain instance
        """
        self.brain = brain
        
        # Each sensor gets a designated field region
        # Like cortical areas in biological brains
        self.sensor_regions = self._allocate_sensor_regions()
        
        # Optional: Keep buffers for fallback/comparison
        self.use_direct_injection = True
        
    def _allocate_sensor_regions(self) -> Dict:
        """
        Allocate specific field regions for each sensor type.
        
        Just like how the brain has:
        - Visual cortex (occipital lobe)
        - Auditory cortex (temporal lobe)  
        - Somatosensory cortex (parietal lobe)
        """
        # Get field dimensions from first level
        field = self.brain.levels[0].field
        size = field.shape[0]  # Spatial dimension
        channels = field.shape[3]  # Channel dimension
        
        # Divide field into sensor regions
        regions = {
            'vision': {
                'spatial': (slice(0, size//2), slice(0, size//2), slice(0, size//2)),
                'channels': slice(0, min(32, channels//2)),  # Visual features
                'injection_strength': 0.1
            },
            'ultrasonic': {
                'spatial': (slice(size//2, size), slice(0, size//4), slice(0, size//4)),
                'channels': slice(channels//2, channels//2 + 4),  # Distance
                'injection_strength': 0.2
            },
            'battery': {
                'spatial': (slice(0, 1), slice(0, 1), slice(0, 1)),  # Single point
                'channels': slice(channels-4, channels-3),  # One channel
                'injection_strength': 0.05
            },
            'proprioception': {
                'spatial': (slice(size//4, 3*size//4), slice(size//4, 3*size//4), slice(0, size//2)),
                'channels': slice(channels//2 + 4, channels//2 + 12),  # Body state
                'injection_strength': 0.15
            }
        }
        
        return regions
    
    def inject_sensor_direct(self, sensor_type: str, data: np.ndarray):
        """
        Inject sensor data directly into designated field region.
        
        This bypasses the buffer → vector → injection pipeline!
        The sensor writes directly to its "cortical area".
        """
        if sensor_type not in self.sensor_regions:
            return
        
        region = self.sensor_regions[sensor_type]
        field = self.brain.levels[0].field
        
        # Convert sensor data to field perturbation
        perturbation = self._sensor_to_field(sensor_type, data, region)
        
        # Direct injection into field!
        # No synchronization - conflicts are "neural noise"
        spatial = region['spatial']
        channels = region['channels']
        strength = region['injection_strength']
        
        # The magic: Direct field modification
        with torch.no_grad():
            # Decay old activation slightly
            field[spatial][..., channels] *= 0.98
            
            # Inject new activation
            field[spatial][..., channels] += perturbation * strength
            
            # Natural field dynamics will integrate this!
            # No need to wait for next process() call
    
    def _sensor_to_field(self, sensor_type: str, data: np.ndarray, region: Dict) -> torch.Tensor:
        """
        Convert raw sensor data to field-compatible tensor.
        
        This is where sensor-specific processing happens.
        """
        # Get region dimensions
        spatial = region['spatial']
        spatial_shape = tuple(
            s.stop - s.start if hasattr(s, 'start') else 1
            for s in spatial
        )
        
        channels = region['channels']
        n_channels = channels.stop - channels.start
        
        if sensor_type == 'vision':
            # Vision: Complex processing
            # Could do edge detection, motion, etc.
            # For now, just reshape and normalize
            if len(data) > np.prod(spatial_shape) * n_channels:
                # Downsample if needed
                data = data[:np.prod(spatial_shape) * n_channels]
            
            field_data = torch.tensor(data, dtype=torch.float32)
            field_data = field_data.reshape(*spatial_shape, n_channels)
            
        elif sensor_type == 'ultrasonic':
            # Distance: Create gradient field
            distance = float(data) if np.isscalar(data) else data[0]
            
            # Create distance gradient in field
            field_data = torch.zeros(*spatial_shape, n_channels)
            
            # Near = high activation, far = low
            activation = 1.0 / (1.0 + distance / 100.0)
            field_data[..., 0] = activation
            
        elif sensor_type == 'battery':
            # Battery: Simple scalar injection
            voltage = float(data) if np.isscalar(data) else data[0]
            
            # Map voltage to activation
            activation = (voltage - 6.0) / 2.4  # Normalize 6-8.4V to 0-1
            field_data = torch.full((*spatial_shape, n_channels), activation)
            
        else:
            # Generic: Just normalize and reshape
            field_data = torch.tensor(data, dtype=torch.float32)
            field_data = field_data.reshape(*spatial_shape, n_channels)
        
        return field_data


class ParallelFieldInjection:
    """
    Multiple sensors injecting into field simultaneously.
    
    Since the brain only reads field state, not buffers,
    we can have truly parallel sensor processing!
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.integrator = FieldNativeSensorIntegration(brain)
        self.threads = {}
        self.running = False
        
    def start_sensor_thread(self, sensor_type: str, sensor_stream):
        """
        Start a thread that injects sensor data directly into field.
        
        No buffers! No synchronization! Just direct field writes!
        """
        def sensor_loop():
            while self.running:
                # Get latest sensor data
                data = sensor_stream.get()
                
                if data is not None:
                    # Inject directly into field
                    self.integrator.inject_sensor_direct(sensor_type, data)
                
                # Sensor's natural rate
                time.sleep(sensor_stream.period)
        
        thread = threading.Thread(target=sensor_loop, daemon=True)
        thread.start()
        self.threads[sensor_type] = thread
        
        print(f"Started {sensor_type} → field injection thread")


class EvolutionContinues:
    """
    The beautiful part: Field evolution continues regardless!
    
    The brain's process() loop:
    1. Field evolves (diffusion, decay, dynamics)
    2. Motor extracted from field gradients
    3. Repeat
    
    It doesn't care HOW data got into the field!
    - Traditional: buffer → vector → injection
    - Direct: sensor → field region
    - Parallel: multiple sensors → multiple regions simultaneously
    
    The field integrates everything naturally!
    """
    pass


def demonstrate_compatibility():
    """Show how this fits with current architecture."""
    
    print("FIELD-NATIVE SENSOR INTEGRATION")
    print("=" * 60)
    print()
    print("Current PureFieldBrain architecture:")
    print("  • Brain reads field state")
    print("  • Extracts motor from field gradients")
    print("  • Field evolves continuously")
    print()
    print("This means sensors could inject DIRECTLY into field!")
    print()
    print("Traditional flow:")
    print("  Sensor → Buffer → Vector → Injection → Field → Motor")
    print()
    print("Possible parallel flow:")
    print("  Vision ──→ Visual cortex region    ┐")
    print("  Audio ───→ Auditory region         ├→ Field → Motor")
    print("  Touch ───→ Somatosensory region    ┘")
    print()
    print("Benefits:")
    print("  ✓ True parallel processing")
    print("  ✓ No synchronization needed")
    print("  ✓ Conflicts become 'neural noise'")
    print("  ✓ More biological")
    print()
    print("The brain already doesn't care about buffers!")
    print("It only cares about field state!")
    print()
    print("This is a NATURAL EVOLUTION of current architecture,")
    print("not a rewrite!")


if __name__ == "__main__":
    demonstrate_compatibility()