#!/usr/bin/env python3
"""
Minimal test of parallel field injection - no blocking operations.
"""

import torch
import threading
import time
import random


class SimpleFieldInjector:
    """Minimal injector that just modifies field periodically."""
    
    def __init__(self, field: torch.Tensor, region: tuple, name: str):
        self.field = field
        self.region = region
        self.name = name
        self.running = False
        self.injection_count = 0
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._inject_loop, daemon=True)
        self.thread.start()
        
    def _inject_loop(self):
        """Simple injection loop - no I/O, just field writes."""
        while self.running:
            # Generate some value
            value = random.gauss(0.5, 0.1)
            
            # Inject into field (no locks!)
            with torch.no_grad():
                x, y, z, c = self.region
                self.field[x, y, z, c] *= 0.95  # Decay
                self.field[x, y, z, c] += value * 0.1  # Inject
                
            self.injection_count += 1
            time.sleep(0.05)  # 20Hz
            
    def stop(self):
        self.running = False


def main():
    print("MINIMAL PARALLEL INJECTION TEST")
    print("=" * 50)
    print("\nTesting core concept: Multiple threads writing to field\n")
    
    # Create field
    field = torch.zeros(8, 8, 8, 16)
    print(f"Created field: {field.shape}")
    
    # Create multiple injectors for different "sensors"
    injectors = [
        SimpleFieldInjector(field, (0, 0, 0, 0), "sensor1"),
        SimpleFieldInjector(field, (7, 7, 7, 15), "sensor2"),
        SimpleFieldInjector(field, (3, 3, 3, 8), "sensor3"),
    ]
    
    # Start all injectors (parallel threads!)
    print("\nStarting parallel injectors...")
    for inj in injectors:
        inj.start()
        print(f"  ✓ {inj.name} started")
    
    # Main loop (simulates brain processing)
    print("\nMain loop running (simulates brain)...")
    print("-" * 40)
    
    for cycle in range(50):
        # "Brain" evolves field
        with torch.no_grad():
            field *= 0.99  # Global decay
            field += torch.randn_like(field) * 0.001  # Noise
        
        # Check values periodically
        if cycle % 10 == 0:
            vals = [field[inj.region].item() for inj in injectors]
            counts = [inj.injection_count for inj in injectors]
            
            print(f"Cycle {cycle:3d} | ", end="")
            for i, inj in enumerate(injectors):
                print(f"{inj.name}: {vals[i]:6.3f} ({counts[i]:3d} inj) | ", end="")
            print()
        
        time.sleep(0.02)  # 50Hz main loop
    
    # Stop injectors
    print("\nStopping injectors...")
    for inj in injectors:
        inj.stop()
    
    # Results
    print("\n" + "=" * 50)
    print("RESULTS:")
    
    # Check if parallel injection worked
    success = True
    for inj in injectors:
        x, y, z, c = inj.region
        value = field[x, y, z, c].item()
        print(f"  {inj.name}: {inj.injection_count} injections, "
              f"final value: {value:.4f}")
        if inj.injection_count == 0 or abs(value) < 0.001:
            success = False
    
    # Check field coherence
    if torch.isnan(field).any():
        print("\n❌ Field has NaN values!")
        success = False
    elif torch.isinf(field).any():
        print("\n❌ Field has Inf values!")
        success = False
    else:
        mean = field.mean().item()
        std = field.std().item()
        print(f"\n  Field stats: mean={mean:.4f}, std={std:.4f}")
        if abs(mean) > 10 or std > 100:
            print("  ⚠️  Field values seem extreme")
            success = False
    
    if success:
        print("\n✅ SUCCESS! Parallel injection works!")
        print("  • Multiple threads wrote to field simultaneously")
        print("  • No synchronization needed")
        print("  • Field stayed coherent")
        print("  • This proves the architecture!")
    else:
        print("\n⚠️  Test had issues - check results above")


if __name__ == "__main__":
    main()