#!/usr/bin/env python3
"""
Biological Oscillator System

Implements authentic neural oscillation patterns for natural synchronization
and cross-stream coordination without explicit locks or barriers.

Key biological frequencies:
- Gamma (40Hz/25ms): Real-time binding, conscious perception, cross-modal integration
- Theta (6Hz/167ms): Memory consolidation, learning integration, session coordination

This replaces artificial 50ms cycles with biologically accurate gamma-frequency processing.
"""

import time
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OscillationPhase(Enum):
    """Biological oscillation phases within gamma cycles."""
    SENSORY_WINDOW = "sensory_window"      # Early gamma: sensory processing peak
    INTEGRATION_WINDOW = "integration"     # Mid gamma: cross-modal binding
    MOTOR_WINDOW = "motor_window"         # Late gamma: motor output peak
    CONSOLIDATION = "consolidation"        # Between cycles: brief consolidation


@dataclass
class BiologicalTiming:
    """Current biological timing state for coordination."""
    gamma_phase: float           # 0.0-1.0 within current gamma cycle
    theta_phase: float           # 0.0-1.0 within current theta cycle
    cycle_start_time: float      # Absolute start time of current gamma cycle
    oscillation_phase: OscillationPhase
    binding_window_active: bool  # True during cross-stream binding windows
    consolidation_active: bool   # True during theta-driven consolidation


class BiologicalOscillator:
    """
    Biological neural oscillation system for natural coordination.
    
    Provides gamma-frequency cycles (25ms) for real-time processing
    and theta-frequency cycles (167ms) for memory consolidation.
    
    No explicit synchronization - coordination emerges from shared timing.
    """
    
    def __init__(self, gamma_freq: float = 40.0, theta_freq: float = 6.0, quiet_mode: bool = False):
        """
        Initialize biological oscillator system.
        
        Args:
            gamma_freq: Gamma frequency in Hz (default 40Hz = 25ms cycles)
            theta_freq: Theta frequency in Hz (default 6Hz = 167ms cycles)
            quiet_mode: Suppress debug output
        """
        self.gamma_freq = gamma_freq
        self.theta_freq = theta_freq
        self.quiet_mode = quiet_mode
        
        # Cycle timing
        self.gamma_period = 1.0 / gamma_freq  # 0.025s = 25ms
        self.theta_period = 1.0 / theta_freq  # 0.167s = 167ms
        
        # Biological phase windows (within gamma cycle)
        self.sensory_window = (0.0, 0.3)      # 0-30% of cycle: sensory processing
        self.integration_window = (0.3, 0.7)  # 30-70% of cycle: cross-modal binding
        self.motor_window = (0.7, 1.0)        # 70-100% of cycle: motor output
        
        # Binding window timing (critical for cross-stream coordination)
        self.binding_window_duration = 0.005  # 5ms binding windows
        self.binding_window_start = 0.4       # Start at 40% of gamma cycle
        
        # Consolidation timing (theta-driven)
        self.consolidation_window_duration = 0.010  # 10ms consolidation windows
        
        # Oscillator state
        self.start_time = time.time()
        self.cycle_count = 0
        self.theta_cycle_count = 0
        
        if not quiet_mode:
            print(f"🧠 BiologicalOscillator initialized:")
            print(f"   Gamma: {gamma_freq}Hz ({self.gamma_period*1000:.1f}ms cycles)")
            print(f"   Theta: {theta_freq}Hz ({self.theta_period*1000:.1f}ms cycles)")
            print(f"   Binding windows: {self.binding_window_duration*1000:.1f}ms every {self.gamma_period*1000:.1f}ms")
    
    def get_current_timing(self) -> BiologicalTiming:
        """
        Get current biological timing state for stream coordination.
        
        Returns:
            BiologicalTiming object with all current oscillation state
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate gamma cycle position
        gamma_cycle_elapsed = elapsed % self.gamma_period
        gamma_phase = gamma_cycle_elapsed / self.gamma_period
        
        # Calculate theta cycle position  
        theta_cycle_elapsed = elapsed % self.theta_period
        theta_phase = theta_cycle_elapsed / self.theta_period
        
        # Determine current oscillation phase
        if gamma_phase < self.sensory_window[1]:
            oscillation_phase = OscillationPhase.SENSORY_WINDOW
        elif gamma_phase < self.integration_window[1]:
            oscillation_phase = OscillationPhase.INTEGRATION_WINDOW
        elif gamma_phase < self.motor_window[1]:
            oscillation_phase = OscillationPhase.MOTOR_WINDOW
        else:
            oscillation_phase = OscillationPhase.CONSOLIDATION
        
        # Check if we're in a binding window (critical for cross-stream coordination)
        binding_window_active = (
            self.binding_window_start <= gamma_phase <= 
            self.binding_window_start + (self.binding_window_duration / self.gamma_period)
        )
        
        # Check if we're in theta-driven consolidation
        consolidation_active = (
            theta_cycle_elapsed < self.consolidation_window_duration
        )
        
        # Calculate cycle start time
        cycle_start_time = current_time - gamma_cycle_elapsed
        
        return BiologicalTiming(
            gamma_phase=gamma_phase,
            theta_phase=theta_phase,
            cycle_start_time=cycle_start_time,
            oscillation_phase=oscillation_phase,
            binding_window_active=binding_window_active,
            consolidation_active=consolidation_active
        )
    
    def wait_for_next_gamma_cycle(self) -> BiologicalTiming:
        """
        Wait for the start of the next gamma cycle.
        
        This enforces biological timing constraints while allowing
        parallel processing within each cycle.
        
        Returns:
            BiologicalTiming for the new cycle
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate time to next gamma cycle
        current_cycle_elapsed = elapsed % self.gamma_period
        time_to_next_cycle = self.gamma_period - current_cycle_elapsed
        
        # Sleep until next cycle (biological constraint enforcement)
        if time_to_next_cycle > 0:
            time.sleep(time_to_next_cycle)
        
        # Update cycle counters
        self.cycle_count += 1
        
        # Check if we completed a theta cycle
        if (self.cycle_count * self.gamma_period) >= self.theta_period:
            self.theta_cycle_count += 1
            if not self.quiet_mode and self.theta_cycle_count % 10 == 0:  # Log every 10 theta cycles
                print(f"🌊 Theta cycle {self.theta_cycle_count} completed ({self.theta_period*1000:.0f}ms)")
        
        return self.get_current_timing()
    
    def should_process_in_phase(self, phase: OscillationPhase) -> bool:
        """
        Check if current timing allows processing in specified phase.
        
        This enables phase-specific processing without explicit locks.
        
        Args:
            phase: The oscillation phase to check
            
        Returns:
            True if currently in the specified phase
        """
        current_timing = self.get_current_timing()
        return current_timing.oscillation_phase == phase
    
    def get_coordination_signal(self) -> Dict[str, Any]:
        """
        Get coordination signals for stream synchronization.
        
        Returns timing signals that streams can use for natural coordination
        without explicit synchronization primitives.
        
        Returns:
            Dictionary with coordination signals
        """
        timing = self.get_current_timing()
        
        return {
            'gamma_phase': timing.gamma_phase,
            'theta_phase': timing.theta_phase,
            'sensory_peak': timing.oscillation_phase == OscillationPhase.SENSORY_WINDOW,
            'integration_peak': timing.oscillation_phase == OscillationPhase.INTEGRATION_WINDOW,
            'motor_peak': timing.oscillation_phase == OscillationPhase.MOTOR_WINDOW,
            'binding_window': timing.binding_window_active,
            'consolidation_window': timing.consolidation_active,
            'cycle_count': self.cycle_count,
            'theta_cycle_count': self.theta_cycle_count
        }
    
    def estimate_processing_budget(self) -> float:
        """
        Estimate remaining processing time in current gamma cycle.
        
        This allows streams to adapt their processing depth based on
        available time within the biological constraint.
        
        Returns:
            Remaining time in seconds within current gamma cycle
        """
        timing = self.get_current_timing()
        elapsed_in_cycle = timing.gamma_phase * self.gamma_period
        remaining = self.gamma_period - elapsed_in_cycle
        
        return max(0.001, remaining)  # Minimum 1ms for safety
    
    def get_oscillator_stats(self) -> Dict[str, Any]:
        """Get oscillator statistics for monitoring."""
        timing = self.get_current_timing()
        uptime = time.time() - self.start_time
        
        return {
            'gamma_freq': self.gamma_freq,
            'theta_freq': self.theta_freq,
            'gamma_period_ms': self.gamma_period * 1000,
            'theta_period_ms': self.theta_period * 1000,
            'uptime_seconds': uptime,
            'gamma_cycles_completed': self.cycle_count,
            'theta_cycles_completed': self.theta_cycle_count,
            'current_gamma_phase': timing.gamma_phase,
            'current_theta_phase': timing.theta_phase,
            'current_phase': timing.oscillation_phase.value,
            'binding_window_active': timing.binding_window_active,
            'consolidation_active': timing.consolidation_active,
            'avg_gamma_frequency': self.cycle_count / uptime if uptime > 0 else 0
        }


def create_biological_oscillator(config: Dict[str, Any] = None, quiet_mode: bool = False) -> BiologicalOscillator:
    """
    Factory function to create biological oscillator with configuration.
    
    Args:
        config: Configuration dictionary with 'gamma_freq' and 'theta_freq' keys
        quiet_mode: Suppress debug output
        
    Returns:
        Configured BiologicalOscillator instance
    """
    if config is None:
        config = {}
    
    gamma_freq = config.get('gamma_freq', 40.0)
    theta_freq = config.get('theta_freq', 6.0)
    
    return BiologicalOscillator(gamma_freq, theta_freq, quiet_mode)


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Testing Biological Oscillator System")
    
    # Create oscillator
    oscillator = BiologicalOscillator(quiet_mode=False)
    
    # Test timing coordination
    print(f"\n🔬 Testing phase coordination...")
    for i in range(5):
        timing = oscillator.get_current_timing()
        coordination = oscillator.get_coordination_signal()
        
        print(f"Cycle {i}: {timing.oscillation_phase.value} "
              f"(gamma: {timing.gamma_phase:.2f}, theta: {timing.theta_phase:.2f})")
        print(f"  Binding: {coordination['binding_window']}, "
              f"Processing budget: {oscillator.estimate_processing_budget()*1000:.1f}ms")
        
        # Wait for next gamma cycle
        oscillator.wait_for_next_gamma_cycle()
    
    # Display final stats
    stats = oscillator.get_oscillator_stats()
    print(f"\n📊 Oscillator Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")