#!/usr/bin/env python3
"""
Ultrasonic Field Injector - Second sensor for parallel architecture.

Distance sensor that injects obstacle proximity directly into field.
Higher frequency than battery (20Hz vs 1Hz).
"""

import socket
import struct
import threading
import time
import torch
import numpy as np
from typing import Dict, Optional


class UltrasonicFieldInjector:
    """
    Ultrasonic sensor thread with direct field injection.
    Tests higher-frequency sensor in parallel architecture.
    """
    
    def __init__(self, brain_field: torch.Tensor, port: int = 10003):
        """
        Args:
            brain_field: Reference to brain's field tensor
            port: UDP port to listen on
        """
        self.field = brain_field
        self.port = port
        self.running = False
        self.thread = None
        
        # Ultrasonic gets a spatial region for distance representation
        # Like a "spatial awareness" region
        self.field_region = self._allocate_region()
        
        # Stats
        self.injection_count = 0
        self.last_distance = 100.0  # cm
        self.last_injection_time = 0
        self.obstacle_warnings = 0
        
    def _allocate_region(self) -> Dict:
        """
        Allocate field region for distance/obstacle representation.
        
        Unlike battery (single point), ultrasonic uses a small region
        to encode distance as a spatial gradient.
        """
        return {
            'spatial': (slice(0, 2), slice(0, 2), slice(0, 2)),  # 2x2x2 region
            'channels': slice(28, 32),  # 4 channels for distance encoding
            'decay_rate': 0.95,  # Faster decay than battery (distance changes quickly)
            'injection_strength': 0.2,
            'critical_distance': 10.0  # cm - stronger injection when close
        }
    
    def start(self):
        """Start the ultrasonic injection thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()
        print(f"📏 Ultrasonic field injector started on port {self.port}")
    
    def stop(self):
        """Stop the injection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"📏 Ultrasonic injector stopped. Injections: {self.injection_count}, "
              f"Warnings: {self.obstacle_warnings}")
    
    def _injection_loop(self):
        """
        Main loop - receives distance data and injects into field.
        Runs at ~20Hz independently of main brain loop.
        """
        # Setup UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(0.05)  # 50ms timeout for 20Hz operation
        
        print(f"📏 Ultrasonic injector listening on UDP:{self.port}")
        
        while self.running:
            try:
                # Receive distance measurement
                data, addr = sock.recvfrom(1024)
                if len(data) == 4:
                    distance = struct.unpack('!f', data)[0]
                    
                    # Inject into field with spatial encoding
                    self._inject_into_field(distance)
                    
            except socket.timeout:
                # No data - maybe inject prediction or decay
                self._inject_decay()
                
            except Exception as e:
                print(f"⚠️  Ultrasonic injector error: {e}")
        
        sock.close()
    
    def _inject_into_field(self, distance: float):
        """
        Inject distance into field with spatial gradient encoding.
        
        Close obstacles create strong, localized activation.
        Far obstacles create weak, diffuse activation.
        """
        self.last_distance = distance
        self.last_injection_time = time.time()
        
        # Get region parameters
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        decay = self.field_region['decay_rate']
        strength = self.field_region['injection_strength']
        critical = self.field_region['critical_distance']
        
        # Calculate activation based on distance
        # Close = high activation, far = low activation
        if distance < critical:
            # OBSTACLE! Strong, focused injection
            activation = 1.0
            self.obstacle_warnings += 1
            focus = 1.0  # Sharp focus
            
            if self.obstacle_warnings % 5 == 1:
                print(f"⚠️  OBSTACLE at {distance:.1f}cm!")
        else:
            # Normal navigation - weaker, diffuse injection
            activation = critical / (distance + 1e-6)  # Inverse distance
            activation = min(1.0, activation)
            focus = 0.5  # More diffuse
        
        # Create spatial gradient in the region
        with torch.no_grad():
            # Decay old activation
            self.field[spatial][..., channels] *= decay
            
            # Inject with spatial gradient
            # Near obstacles: sharp gradient
            # Far obstacles: smooth gradient
            region_shape = (
                spatial[0].stop - spatial[0].start if hasattr(spatial[0], 'stop') else 1,
                spatial[1].stop - spatial[1].start if hasattr(spatial[1], 'stop') else 1,
                spatial[2].stop - spatial[2].start if hasattr(spatial[2], 'stop') else 1,
            )
            
            # Create gradient tensor
            gradient = torch.zeros(region_shape + (4,))  # 4 channels
            
            # Channel 0: Raw distance
            gradient[..., 0] = activation
            
            # Channel 1: Proximity gradient (stronger at [0,0,0])
            for i in range(region_shape[0]):
                for j in range(region_shape[1]):
                    for k in range(region_shape[2]):
                        dist_from_origin = (i**2 + j**2 + k**2) ** 0.5
                        gradient[i, j, k, 1] = activation * np.exp(-dist_from_origin * focus)
            
            # Channel 2: Obstacle indicator
            gradient[..., 2] = 1.0 if distance < critical else 0.1
            
            # Channel 3: Change rate (difference from last)
            if hasattr(self, '_last_activation'):
                gradient[..., 3] = abs(activation - self._last_activation)
            
            # Inject gradient into field
            self.field[spatial][..., channels] += gradient * strength
            
            # Clamp to prevent explosion
            self.field[spatial][..., channels] = torch.clamp(
                self.field[spatial][..., channels], -10, 10
            )
        
        self._last_activation = activation
        self.injection_count += 1
        
        # Log periodically
        if self.injection_count % 20 == 0:
            mean_activation = self.field[spatial][..., channels].mean().item()
            print(f"📏 Injection #{self.injection_count}: {distance:.1f}cm → "
                  f"activation={activation:.2f}, field_mean={mean_activation:.3f}")
    
    def _inject_decay(self):
        """
        When no data received, apply decay to represent uncertainty.
        """
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        
        with torch.no_grad():
            # Faster decay when no data (uncertainty grows)
            self.field[spatial][..., channels] *= 0.9


# Standalone test
if __name__ == "__main__":
    import random
    
    print("ULTRASONIC FIELD INJECTOR TEST")
    print("=" * 50)
    
    # Create test field
    field = torch.zeros(16, 16, 16, 64)
    print(f"Created field: {field.shape}")
    
    # Create injector
    injector = UltrasonicFieldInjector(field, port=10003)
    injector.start()
    
    # Simulate distance measurements
    def send_distances():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Simulate approaching and receding obstacle
        distances = [100, 80, 60, 40, 20, 10, 8, 10, 20, 40, 60, 80, 100]
        
        for d in distances:
            # Add noise
            noisy_d = d + random.gauss(0, 2)
            packet = struct.pack('!f', noisy_d)
            sock.sendto(packet, ('localhost', 10003))
            print(f"  Sent: {noisy_d:.1f}cm")
            time.sleep(0.05)  # 20Hz
        
        sock.close()
    
    print("\nSimulating obstacle approach...")
    sender = threading.Thread(target=send_distances, daemon=True)
    sender.start()
    
    # Monitor field changes
    print("\nMonitoring field region...")
    for i in range(20):
        region = field[0:2, 0:2, 0:2, 28:32]
        mean_val = region.mean().item()
        max_val = region.max().item()
        print(f"  t={i*0.1:.1f}s: mean={mean_val:.3f}, max={max_val:.3f}")
        time.sleep(0.1)
    
    injector.stop()
    
    print(f"\nFinal stats:")
    print(f"  Total injections: {injector.injection_count}")
    print(f"  Obstacle warnings: {injector.obstacle_warnings}")
    print(f"  Last distance: {injector.last_distance:.1f}cm")
    
    if injector.injection_count > 0 and injector.obstacle_warnings > 0:
        print("\n✅ Ultrasonic injector works!")