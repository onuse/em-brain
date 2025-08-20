#!/usr/bin/env python3
"""
Minimal Constrained Brain: Testing the Core Hypothesis
Intelligence emerges from constrained field dynamics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class MinimalConstrainedBrain:
    """
    Absolute minimum implementation to test core hypothesis:
    Intelligence emerges from constrained field dynamics
    
    This takes the best of UnifiedFieldBrain (field dynamics, discomfort drive)
    but adds severe constraints that force intelligent behavior to emerge.
    """
    
    def __init__(self, device='cuda', verbose=False):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        # Field (from current working code, but smaller for testing)
        self.field = torch.randn(32, 32, 32, 64, device=self.device)
        
        # CONSTRAINTS - These force intelligence to emerge
        self.memory_slots = 8  # Severe memory pressure (0.01% of field)
        self.energy_budget = 100  # Limited compute per cycle
        self.attention_bandwidth = 10  # Can only focus on 10 patterns
        
        # Discomfort drive (proven innovation from current code)
        self.discomfort_threshold = 0.3
        self.boredom_rate = 0.99
        self.discomfort_history = []
        
        # Emergent pattern tracking
        self.resonances = {}  # Frequency -> pattern mapping
        self.preferences = torch.zeros_like(self.field)  # Learned preferences
        
        # Metrics for observing emergence
        self.metrics = {
            'cycles': 0,
            'abstractions_formed': 0,
            'attention_focuses': [],
            'preference_strength': 0,
            'compression_ratio': 0,
        }
        
    def process(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        Main processing cycle with constraints forcing emergence
        """
        self.metrics['cycles'] += 1
        
        # 1. Input disturbs field (proven to work)
        self._inject_sensory_input(sensory_input)
        
        # 2. Apply constraints (this forces intelligence)
        original_field = self.field.clone()
        self.field = self._apply_memory_constraint(self.field)
        self.field = self._apply_energy_constraint(self.field)
        self.field = self._apply_attention_constraint(self.field)
        
        # 3. Measure compression (are abstractions forming?)
        self.metrics['compression_ratio'] = self._measure_compression(
            original_field, self.field
        )
        
        # 4. Update preferences based on discomfort change
        discomfort = self._compute_discomfort()
        self._update_preferences(discomfort)
        
        # 5. Apply intrinsic drives (proven innovation)
        self._apply_intrinsic_drive()
        
        # 6. Extract motor commands
        action = self._field_to_motor()
        
        if self.verbose:
            self._print_emergence_metrics()
        
        return action
    
    def _inject_sensory_input(self, sensory_input: np.ndarray):
        """Inject sensory data into field"""
        input_tensor = torch.tensor(
            sensory_input, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # Reshape to field dimensions and inject
        if len(sensory_input) <= self.field.shape[-1]:
            injection = torch.zeros(1, 1, 1, self.field.shape[-1], device=self.device)
            injection[0, 0, 0, :len(sensory_input)] = input_tensor
            self.field += injection * 0.1
        
    def _apply_memory_constraint(self, field: torch.Tensor) -> torch.Tensor:
        """
        Force abstraction through severe compression.
        Only 8 frequencies can be remembered - forces generalization.
        """
        # FFT to frequency domain (GPU-native operation)
        spectrum = torch.fft.fftn(field, dim=(0, 1, 2))
        
        # Find dominant frequencies (these become abstractions)
        magnitudes = spectrum.abs()
        flat_mags = magnitudes.flatten()
        
        # Keep only top-k frequencies (brutal compression)
        k = min(self.memory_slots, flat_mags.numel())
        top_k = torch.topk(flat_mags, k)
        
        # Store these as our "concepts"
        for idx in top_k.indices[:self.memory_slots]:
            freq = idx.item()
            self.resonances[freq] = spectrum.flatten()[idx]
        
        self.metrics['abstractions_formed'] = len(self.resonances)
        
        # Zero out everything except top frequencies
        mask = torch.zeros_like(flat_mags)
        mask[top_k.indices] = 1
        masked_spectrum = spectrum.flatten() * mask
        
        # Back to spatial domain (lossy but abstracted)
        compressed = torch.fft.ifftn(
            masked_spectrum.reshape(spectrum.shape),
            dim=(0, 1, 2)
        )
        
        return compressed.real
    
    def _apply_energy_constraint(self, field: torch.Tensor) -> torch.Tensor:
        """
        Force selective processing through energy limits.
        Can't process everything - must choose what's important.
        """
        # Calculate energy per region
        energy = field.abs().sum(dim=-1)
        
        # Only process top energy regions (limited budget)
        energy_flat = energy.flatten()
        k = min(self.energy_budget, energy_flat.numel())
        top_k_energy = torch.topk(energy_flat, k)
        
        # Create attention mask
        threshold = top_k_energy.values[-1] if k > 0 else 0
        mask = (energy >= threshold).unsqueeze(-1).float()
        
        # Apply mask (unprocessed regions decay)
        processed = field * mask
        unprocessed = field * (1 - mask) * 0.9  # Decay unattended regions
        
        return processed + unprocessed
    
    def _apply_attention_constraint(self, field: torch.Tensor) -> torch.Tensor:
        """
        Force focus through attention bandwidth limit.
        Can only maintain 10 active patterns simultaneously.
        """
        # Find active patterns (local maxima)
        pooled = F.max_pool3d(
            field.abs().mean(dim=-1, keepdim=True),
            kernel_size=3, stride=1, padding=1
        )
        is_peak = (field.abs().mean(dim=-1, keepdim=True) == pooled)
        
        # Get strengths of all peaks
        peak_strengths = field.abs().mean(dim=-1) * is_peak.squeeze(-1)
        peak_flat = peak_strengths.flatten()
        
        # Keep only top attention_bandwidth patterns
        k = min(self.attention_bandwidth, (peak_flat > 0).sum().item())
        if k > 0:
            top_k = torch.topk(peak_flat, k)
            threshold = top_k.values[-1]
            
            # Create attention mask
            attention_mask = (peak_strengths >= threshold).unsqueeze(-1).float()
            
            # Track what we're attending to
            self.metrics['attention_focuses'].append(k)
            
            # Amplify attended patterns, suppress others
            attended = field * (1 + attention_mask)  # Amplify
            unattended = field * (1 - attention_mask * 0.5)  # Partial suppression
            
            return attended * attention_mask + unattended * (1 - attention_mask)
        
        return field
    
    def _compute_discomfort(self) -> float:
        """
        Compute field discomfort (from proven current implementation).
        High uniformity = high discomfort (boring)
        """
        # Local variance as comfort measure
        variance = self.field.var()
        
        # High variance = comfortable (interesting)
        # Low variance = uncomfortable (boring)
        discomfort = 1.0 / (1.0 + variance)
        
        self.discomfort_history.append(discomfort.item())
        
        return discomfort
    
    def _update_preferences(self, current_discomfort: torch.Tensor):
        """
        Learn preferences from discomfort changes.
        Patterns that reduce discomfort are preferred.
        """
        if len(self.discomfort_history) > 1:
            discomfort_delta = current_discomfort - self.discomfort_history[-2]
            
            if discomfort_delta < 0:  # Discomfort decreased (good!)
                # Strengthen current patterns (these are good)
                self.preferences = 0.9 * self.preferences + 0.1 * self.field
            else:  # Discomfort increased (bad!)
                # Weaken current patterns (avoid these)
                self.preferences = 0.9 * self.preferences - 0.1 * self.field
        
        # Track preference strength
        self.metrics['preference_strength'] = self.preferences.abs().mean().item()
    
    def _apply_intrinsic_drive(self):
        """
        Apply intrinsic tension from discomfort (proven innovation).
        Boredom drives exploration.
        """
        if len(self.discomfort_history) > 0:
            current_discomfort = self.discomfort_history[-1]
            
            if current_discomfort > self.discomfort_threshold:
                # Too uniform/boring - inject noise for exploration
                noise = torch.randn_like(self.field) * 0.1
                self.field += noise
                
                # Also move toward preferences if we have them
                if self.metrics['preference_strength'] > 0.01:
                    self.field += self.preferences * 0.01
    
    def _field_to_motor(self) -> np.ndarray:
        """
        Extract motor commands from field state.
        Simplified but functional.
        """
        # Pool field down to motor dimensions
        motor_field = F.adaptive_avg_pool3d(
            self.field.mean(dim=-1, keepdim=True),
            (2, 3, 1)
        )
        
        # Extract 6 motor values
        motor_values = motor_field.flatten()[:6]
        
        # Add preference influence
        if self.metrics['preference_strength'] > 0.01:
            preference_motor = F.adaptive_avg_pool3d(
                self.preferences.mean(dim=-1, keepdim=True),
                (2, 3, 1)
            ).flatten()[:6]
            motor_values = motor_values * 0.8 + preference_motor * 0.2
        
        return motor_values.cpu().numpy()
    
    def _measure_compression(self, original: torch.Tensor, compressed: torch.Tensor) -> float:
        """
        Measure how much information was compressed.
        High compression = good abstraction.
        """
        original_info = original.abs().sum()
        compressed_info = compressed.abs().sum()
        
        if original_info > 0:
            return (1 - compressed_info / original_info).item()
        return 0
    
    def _print_emergence_metrics(self):
        """Print metrics showing what's emerging"""
        print(f"\n=== Cycle {self.metrics['cycles']} ===")
        print(f"Abstractions formed: {self.metrics['abstractions_formed']}")
        print(f"Compression ratio: {self.metrics['compression_ratio']:.2%}")
        print(f"Attention focuses: {self.metrics['attention_focuses'][-1] if self.metrics['attention_focuses'] else 0}")
        print(f"Preference strength: {self.metrics['preference_strength']:.3f}")
        print(f"Current discomfort: {self.discomfort_history[-1] if self.discomfort_history else 0:.3f}")
    
    def dream(self, cycles: int = 100):
        """
        Let field evolve without input.
        See what patterns emerge from pure dynamics.
        """
        dream_patterns = []
        
        for _ in range(cycles):
            # Pure evolution without sensory input
            self.field = self._apply_memory_constraint(self.field)
            self.field = self._apply_energy_constraint(self.field)
            self.field = self._apply_attention_constraint(self.field)
            
            # Record emergent patterns
            dream_patterns.append({
                'resonances': list(self.resonances.keys()),
                'compression': self.metrics['compression_ratio'],
                'preference_drift': self.preferences.mean().item()
            })
        
        return dream_patterns
    
    def get_telemetry(self) -> Dict:
        """Get current brain state for monitoring"""
        return {
            'cycles': self.metrics['cycles'],
            'abstractions': self.metrics['abstractions_formed'],
            'compression_ratio': self.metrics['compression_ratio'],
            'attention_bandwidth_used': len(self.metrics['attention_focuses']),
            'preference_strength': self.metrics['preference_strength'],
            'discomfort': self.discomfort_history[-1] if self.discomfort_history else 0,
            'resonance_count': len(self.resonances),
            'field_energy': self.field.abs().mean().item(),
            'field_variance': self.field.var().item(),
        }


def test_minimal_brain():
    """Quick test of the minimal constrained brain"""
    print("Testing Minimal Constrained Brain")
    print("=" * 50)
    
    brain = MinimalConstrainedBrain(verbose=True)
    
    # Test with varying sensory input
    for i in range(10):
        # Generate sensory input (5 sensors with varying values)
        sensory_input = np.array([
            np.sin(i * 0.1),  # Oscillating input
            np.cos(i * 0.1),
            0.5,  # Constant
            np.random.randn() * 0.1,  # Noise
            i * 0.1  # Ramping
        ])
        
        # Process and get motor output
        motor_output = brain.process(sensory_input)
        
        print(f"\nInput: {sensory_input[:3].round(2)}...")
        print(f"Motor: {motor_output[:3].round(2)}...")
    
    # Test dreaming
    print("\n" + "=" * 50)
    print("Testing Dream Mode (no input)")
    dream_patterns = brain.dream(cycles=10)
    print(f"Dream generated {len(dream_patterns)} patterns")
    print(f"Final resonances: {dream_patterns[-1]['resonances'][:5]}...")
    
    # Final telemetry
    print("\n" + "=" * 50)
    print("Final Brain State:")
    telemetry = brain.get_telemetry()
    for key, value in telemetry.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_minimal_brain()