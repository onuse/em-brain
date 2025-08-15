#!/usr/bin/env python3
"""
Field Injection Threads - Parallel sensor processing with direct field writes.

Starting with battery as proof of concept.
Each sensor gets its own thread that injects directly into designated field regions.
No buffers, no synchronization with main loop.
"""

import socket
import struct
import threading
import time
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple

# Import sensor injectors
try:
    from .ultrasonic_field_injector import UltrasonicFieldInjector
except ImportError:
    UltrasonicFieldInjector = None

try:
    from .vision_field_injector import VisionFieldInjector
except ImportError:
    VisionFieldInjector = None

try:
    from .audio_field_injector import AudioFieldInjector
except ImportError:
    AudioFieldInjector = None


class BatteryFieldInjector:
    """
    Battery sensor thread with direct field injection.
    Simplest possible test of parallel architecture.
    """
    
    def __init__(self, brain_field: torch.Tensor, port: int = 10004):
        """
        Args:
            brain_field: Reference to brain's field tensor (first level)
            port: UDP port to listen on
        """
        self.field = brain_field
        self.port = port
        self.running = False
        self.thread = None
        
        # Battery gets a tiny region in the field
        # Like a "homeostatic monitoring" nucleus
        self.field_region = self._allocate_region()
        
        # Stats
        self.injection_count = 0
        self.last_voltage = 7.4
        self.last_injection_time = 0
        
    def _allocate_region(self) -> Dict:
        """Allocate a small field region for battery status."""
        # Just one voxel in the corner, one channel
        # Minimal interference with other processes
        return {
            'spatial': (0, 0, 0),  # Single point in corner
            'channel': -1,         # Last channel (homeostatic)
            'decay_rate': 0.99,    # Slow decay (battery changes slowly)
            'injection_strength': 0.1
        }
    
    def start(self):
        """Start the battery injection thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()
        print(f"🔋 Battery field injector started on port {self.port}")
    
    def stop(self):
        """Stop the injection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"🔋 Battery injector stopped. Total injections: {self.injection_count}")
    
    def _injection_loop(self):
        """
        Main loop - receives battery data and injects into field.
        Runs completely independently of main brain loop!
        """
        # Setup UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(0.1)  # Non-blocking with timeout
        
        print(f"🔋 Battery injector listening on UDP:{self.port}")
        
        while self.running:
            try:
                # Receive battery voltage
                data, addr = sock.recvfrom(1024)
                if len(data) == 4:
                    voltage = struct.unpack('!f', data)[0]
                    
                    # Inject directly into field!
                    self._inject_into_field(voltage)
                    
            except socket.timeout:
                # No data received - that's fine
                # Maybe inject decay or prediction
                pass
            except Exception as e:
                print(f"⚠️  Battery injector error: {e}")
        
        sock.close()
    
    def _inject_into_field(self, voltage: float):
        """
        Inject battery status directly into field.
        No synchronization with main loop!
        """
        self.last_voltage = voltage
        self.last_injection_time = time.time()
        
        # Convert voltage to field activation
        # 6.0V = low (0.0), 8.4V = full (1.0)
        activation = (voltage - 6.0) / 2.4
        activation = max(0.0, min(1.0, activation))
        
        # Get field location
        x, y, z = self.field_region['spatial']
        c = self.field_region['channel']
        decay = self.field_region['decay_rate']
        strength = self.field_region['injection_strength']
        
        # Direct field modification!
        # No locks, no synchronization - conflicts are "neural noise"
        with torch.no_grad():
            # Decay old activation
            self.field[x, y, z, c] *= decay
            
            # Inject new activation
            self.field[x, y, z, c] += activation * strength
            
            # Clamp to reasonable range (prevent explosion)
            self.field[x, y, z, c] = torch.clamp(self.field[x, y, z, c], -10, 10)
        
        self.injection_count += 1
        
        # Log every 10 injections
        if self.injection_count % 10 == 0:
            field_value = self.field[x, y, z, c].item()
            print(f"🔋 Injection #{self.injection_count}: {voltage:.2f}V → "
                  f"field[{x},{y},{z},{c}] = {field_value:.3f}")
        
        # Warn on low battery (high priority injection)
        if voltage < 6.5:
            # Stronger injection for critical states
            self.field[x, y, z, c] += 0.5  # Emergency signal!
            print(f"⚠️  LOW BATTERY: {voltage:.2f}V - Strong injection!")


class SensorFieldInjectionManager:
    """
    Manages field injection threads for all sensors.
    Will expand to handle dynamic spawning based on handshake.
    """
    
    def __init__(self, brain):
        """
        Args:
            brain: PureFieldBrain instance
        """
        self.brain = brain
        self.injectors = {}
        
        # Get reference to first level field (where injection happens)
        if hasattr(brain, 'levels') and len(brain.levels) > 0:
            self.field = brain.levels[0].field
        else:
            # Fallback for single-level brain
            self.field = brain.field
        
        print(f"📍 Field injection ready. Field shape: {self.field.shape}")
    
    def start_battery_injector(self, port: int = 10004):
        """Start battery field injection thread (proof of concept)."""
        if 'battery' not in self.injectors:
            self.injectors['battery'] = BatteryFieldInjector(self.field, port)
            self.injectors['battery'].start()
            return True
        return False
    
    def start_ultrasonic_injector(self, port: int = 10003):
        """Start ultrasonic field injection thread."""
        if UltrasonicFieldInjector and 'ultrasonic' not in self.injectors:
            self.injectors['ultrasonic'] = UltrasonicFieldInjector(self.field, port)
            self.injectors['ultrasonic'].start()
            return True
        return False
    
    def start_vision_injector(self, port: int = 10002, resolution: Tuple[int, int] = (320, 240)):
        """Start vision field injection thread - THE CRITICAL ONE!"""
        if VisionFieldInjector and 'vision' not in self.injectors:
            self.injectors['vision'] = VisionFieldInjector(self.field, port, resolution)
            self.injectors['vision'].start()
            return True
        return False
    
    def start_audio_injector(self, port: int = 10006, sample_rate: int = 16000):
        """Start audio field injection thread."""
        if AudioFieldInjector and 'audio' not in self.injectors:
            self.injectors['audio'] = AudioFieldInjector(self.field, port, sample_rate)
            self.injectors['audio'].start()
            return True
        return False
    
    def start_all_basic_sensors(self):
        """Start battery and ultrasonic injectors for testing."""
        started = []
        if self.start_battery_injector():
            started.append('battery')
        if self.start_ultrasonic_injector():
            started.append('ultrasonic')
        return started
    
    def start_all_sensors(self):
        """Start ALL sensor injectors including vision and audio."""
        started = []
        if self.start_battery_injector():
            started.append('battery')
        if self.start_ultrasonic_injector():
            started.append('ultrasonic')
        if self.start_vision_injector():
            started.append('vision')
        if self.start_audio_injector():
            started.append('audio')
        return started
    
    def start_from_handshake(self, sensor_declaration: Dict):
        """
        Future: Spawn threads based on handshake.
        
        Args:
            sensor_declaration: Robot's declared sensors
                {
                    'battery': {'port': 10004, 'rate_hz': 1},
                    'ultrasonic': {'port': 10003, 'rate_hz': 20},
                    'vision': {'port': 10002, 'rate_hz': 15},
                    ...
                }
        """
        # TODO: Implement dynamic spawning
        # For now, just start battery as proof of concept
        if 'battery' in sensor_declaration:
            self.start_battery_injector(sensor_declaration['battery'].get('port', 10004))
    
    def stop_all(self):
        """Stop all injection threads."""
        for name, injector in self.injectors.items():
            print(f"Stopping {name} injector...")
            injector.stop()
        self.injectors.clear()
    
    def get_stats(self) -> Dict:
        """Get statistics from all injectors."""
        stats = {}
        for name, injector in self.injectors.items():
            if hasattr(injector, 'injection_count'):
                stats[name] = {
                    'injections': injector.injection_count,
                    'last_value': getattr(injector, 'last_voltage', None),
                    'last_time': getattr(injector, 'last_injection_time', 0)
                }
        return stats


class FieldCoherenceMonitor:
    """
    Monitors field coherence during parallel injection.
    Helps verify that parallel writes don't break field dynamics.
    """
    
    def __init__(self, field: torch.Tensor):
        self.field = field
        self.baseline_stats = self._compute_stats()
        
    def _compute_stats(self) -> Dict:
        """Compute field statistics."""
        with torch.no_grad():
            return {
                'mean': self.field.mean().item(),
                'std': self.field.std().item(),
                'min': self.field.min().item(),
                'max': self.field.max().item(),
                'nan_count': torch.isnan(self.field).sum().item(),
                'inf_count': torch.isinf(self.field).sum().item()
            }
    
    def check_coherence(self) -> Tuple[bool, str]:
        """
        Check if field is still coherent.
        
        Returns:
            (is_coherent, message)
        """
        stats = self._compute_stats()
        
        # Check for NaN/Inf
        if stats['nan_count'] > 0:
            return False, f"Field has {stats['nan_count']} NaN values!"
        if stats['inf_count'] > 0:
            return False, f"Field has {stats['inf_count']} Inf values!"
        
        # Check for explosion
        if abs(stats['mean']) > 100:
            return False, f"Field mean exploded: {stats['mean']}"
        if stats['std'] > 1000:
            return False, f"Field variance exploded: {stats['std']}"
        
        # Check for death
        if stats['std'] < 0.0001:
            return False, f"Field died (no variance): {stats['std']}"
        
        return True, "Field coherent"


# Test script
if __name__ == "__main__":
    print("FIELD INJECTION TEST - Battery Sensor")
    print("=" * 60)
    
    # Create mock field
    field = torch.randn(16, 16, 16, 64) * 0.1
    print(f"Created test field: {field.shape}")
    
    # Create monitor
    monitor = FieldCoherenceMonitor(field)
    
    # Start battery injector
    injector = BatteryFieldInjector(field, port=10004)
    injector.start()
    
    print("\nSimulating battery updates...")
    print("Run battery streamer to see real injection")
    print("Or simulate with: echo '7.4' | nc -u localhost 10004")
    
    try:
        for i in range(30):
            time.sleep(1)
            
            # Check coherence
            coherent, msg = monitor.check_coherence()
            if not coherent:
                print(f"❌ Field coherence lost: {msg}")
                break
            
            # Show field value at battery location
            battery_value = field[0, 0, 0, -1].item()
            print(f"Field battery location: {battery_value:.3f} - {msg}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    
    injector.stop()
    
    # Final stats
    print("\nFinal statistics:")
    print(f"  Injections: {injector.injection_count}")
    print(f"  Last voltage: {injector.last_voltage:.2f}V")
    print(f"  Field coherent: {monitor.check_coherence()[1]}")
    
    print("\n✓ Test complete!")