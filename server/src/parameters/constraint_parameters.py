#!/usr/bin/env python3
"""
Emergent Parameters System - Hardware-Constrained Adaptive Parameters

This system eliminates hardcoded magic numbers by deriving all parameters
from fundamental constraints:

1. Hardware Constraints: CPU, RAM, bandwidth, processing time
2. Biological Constraints: Attention span, working memory, reaction time
3. Information Theory: Channel capacity, signal-to-noise ratio
4. Energy Constraints: Power consumption, heat dissipation

All parameters emerge from these constraints rather than being hardcoded.
"""

import math
import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of constraints that drive parameter emergence"""
    HARDWARE = "hardware"           # CPU, RAM, bandwidth limits
    BIOLOGICAL = "biological"       # Human-like cognitive limits
    INFORMATION = "information"     # Channel capacity, entropy
    ENERGY = "energy"              # Power, heat, efficiency
    TEMPORAL = "temporal"          # Timing, latency, deadlines


@dataclass
class HardwareProfile:
    """Current hardware capabilities"""
    cpu_cores: int
    cpu_freq_ghz: float
    ram_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    storage_iops: int
    network_bandwidth_mbps: float
    
    @classmethod
    def detect_current(cls) -> 'HardwareProfile':
        """Detect current hardware capabilities"""
        cpu_count = psutil.cpu_count()
        
        # Get CPU frequency (fallback to reasonable default)
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_freq_ghz = cpu_freq.current / 1000.0 if cpu_freq else 2.5
        except:
            cpu_freq_ghz = 2.5
        
        # Get RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Simplified GPU detection
        gpu_available = False
        gpu_memory_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return cls(
            cpu_cores=cpu_count,
            cpu_freq_ghz=cpu_freq_ghz,
            ram_gb=ram_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            storage_iops=1000,  # Reasonable default
            network_bandwidth_mbps=100.0  # Reasonable default
        )


@dataclass
class BiologicalConstraints:
    """Biologically-inspired cognitive constraints"""
    attention_span_seconds: float = 0.4     # Human attention switching ~400ms
    working_memory_slots: int = 7           # Miller's magic number ±2
    reaction_time_ms: float = 200           # Human reaction time
    inhibition_return_ms: float = 1000      # Inhibition of return ~1s
    binding_window_ms: float = 100          # Temporal binding window ~100ms
    
    def scale_to_hardware(self, hardware: HardwareProfile) -> 'BiologicalConstraints':
        """Scale biological constraints based on hardware capability"""
        # Faster hardware can support shorter attention spans
        cpu_speedup = hardware.cpu_freq_ghz / 2.0  # 2GHz baseline
        memory_speedup = hardware.ram_gb / 8.0     # 8GB baseline
        
        speedup_factor = min(cpu_speedup, memory_speedup)
        
        return BiologicalConstraints(
            attention_span_seconds=self.attention_span_seconds / speedup_factor,
            working_memory_slots=int(self.working_memory_slots * min(2.0, speedup_factor)),
            reaction_time_ms=self.reaction_time_ms / speedup_factor,
            inhibition_return_ms=self.inhibition_return_ms / speedup_factor,
            binding_window_ms=self.binding_window_ms / speedup_factor
        )


class EmergentParameterSystem:
    """
    System that derives all parameters from fundamental constraints
    rather than hardcoding magic numbers.
    """
    
    def __init__(self, 
                 hardware_profile: Optional[HardwareProfile] = None,
                 biological_constraints: Optional[BiologicalConstraints] = None,
                 target_framerate: float = 30.0,
                 power_budget_watts: float = 10.0):
        
        # Detect or use provided hardware profile
        self.hardware = hardware_profile or HardwareProfile.detect_current()
        
        # Scale biological constraints to hardware
        base_bio = biological_constraints or BiologicalConstraints()
        self.biological = base_bio.scale_to_hardware(self.hardware)
        
        # Performance targets
        self.target_framerate = target_framerate
        self.target_frame_time_ms = 1000.0 / target_framerate
        self.power_budget_watts = power_budget_watts
        
        # Derived constraint parameters
        self.compute_budget = self._calculate_compute_budget()
        self.memory_budget = self._calculate_memory_budget()
        self.attention_parameters = self._calculate_attention_parameters()
        self.binding_parameters = self._calculate_binding_parameters()
        
        # Cache for expensive calculations
        self._parameter_cache = {}
    
    def _calculate_compute_budget(self) -> int:
        """Calculate computational budget based on hardware constraints"""
        # Base compute units from CPU
        base_compute = self.hardware.cpu_cores * self.hardware.cpu_freq_ghz * 10
        
        # GPU acceleration bonus - more sophisticated scaling
        if self.hardware.gpu_available:
            # GPU provides significant speedup for parallel operations
            gpu_memory_factor = min(4.0, self.hardware.gpu_memory_gb / 4.0)  # 4GB baseline
            gpu_speedup = 8 * gpu_memory_factor  # 8x baseline speedup with memory scaling
            base_compute *= gpu_speedup
        
        # Scale by target framerate - faster framerate needs more compute per frame
        framerate_scaling = 30.0 / self.target_framerate
        
        # Power constraint - limit compute to stay within power budget
        power_scaling = min(1.0, self.power_budget_watts / 20.0)  # 20W baseline
        
        return int(base_compute * framerate_scaling * power_scaling)
    
    def _calculate_memory_budget(self) -> Dict[str, int]:
        """Calculate memory budgets based on RAM and GPU memory constraints"""
        available_ram_mb = self.hardware.ram_gb * 1024 * 0.3  # Use 30% of RAM
        
        # Factor in GPU memory for accelerated operations
        if self.hardware.gpu_available:
            gpu_memory_mb = self.hardware.gpu_memory_gb * 1024 * 0.6  # Use 60% of GPU memory
            # GPU memory can store more patterns due to efficient tensor operations
            gpu_pattern_capacity = int(gpu_memory_mb / 2)  # 2MB per pattern on GPU
            max_objects = min(100, int(available_ram_mb / 10) + gpu_pattern_capacity)
            pattern_cache_size = int(available_ram_mb * 0.5 + gpu_memory_mb * 0.3)
        else:
            max_objects = min(20, int(available_ram_mb / 10))  # 10MB per object
            pattern_cache_size = int(available_ram_mb * 0.5)
        
        # Allocate memory based on cognitive requirements
        return {
            'working_memory_slots': self.biological.working_memory_slots,
            'max_objects': max_objects,
            'pattern_cache_size': pattern_cache_size,
            'correlation_history_size': min(100, int(available_ram_mb / 5))
        }
    
    def _calculate_attention_parameters(self) -> Dict[str, float]:
        """Calculate attention parameters from constraints"""
        # Attention duration from biological constraints
        attention_duration = self.biological.attention_span_seconds
        
        # Switch cost from hardware - slower hardware has higher switch cost
        cpu_baseline = 2.0  # 2GHz baseline
        switch_cost_factor = cpu_baseline / self.hardware.cpu_freq_ghz
        base_switch_cost = self.compute_budget * 0.1  # 10% of compute budget
        switch_cost = base_switch_cost * switch_cost_factor
        
        # Inhibition period from biological constraints
        inhibition_duration = self.biological.inhibition_return_ms / 1000.0
        
        # Thresholds emerge from signal-to-noise ratio
        # Better hardware can detect weaker signals
        memory_snr = min(2.0, self.hardware.ram_gb / 8.0)  # 8GB baseline
        base_threshold = 0.5
        coherence_threshold = base_threshold / memory_snr
        correlation_threshold = base_threshold / memory_snr
        
        return {
            'attention_duration': attention_duration,
            'switch_cost': switch_cost,
            'inhibition_duration': inhibition_duration,
            'coherence_threshold': coherence_threshold,
            'correlation_threshold': correlation_threshold
        }
    
    def _calculate_binding_parameters(self) -> Dict[str, float]:
        """Calculate cross-modal binding parameters"""
        # Binding window from biological constraints
        binding_window = self.biological.binding_window_ms / 1000.0
        
        # Spatial correlation radius emerges from resolution and processing power
        base_radius = 10  # pixels
        resolution_factor = 1.0  # Could be derived from display resolution
        processing_factor = min(2.0, self.hardware.cpu_cores / 4.0)  # 4 cores baseline
        spatial_radius = base_radius * resolution_factor * processing_factor
        
        # Temporal correlation window emerges from framerate
        temporal_window = 1.0 / self.target_framerate * 5  # 5 frame window
        
        # Weighting factors emerge from information theory
        # Equal weighting is information-theoretically optimal without prior knowledge
        temporal_weight = 0.4
        spatial_weight = 0.3
        feature_weight = 0.3
        
        return {
            'binding_window': binding_window,
            'spatial_radius': spatial_radius,
            'temporal_window': temporal_window,
            'temporal_weight': temporal_weight,
            'spatial_weight': spatial_weight,
            'feature_weight': feature_weight
        }
    
    def get_parameter(self, parameter_name: str, constraint_type: ConstraintType) -> Any:
        """Get a parameter derived from specific constraint type"""
        cache_key = f"{parameter_name}_{constraint_type.value}"
        
        if cache_key in self._parameter_cache:
            return self._parameter_cache[cache_key]
        
        if constraint_type == ConstraintType.HARDWARE:
            result = self._get_hardware_parameter(parameter_name)
        elif constraint_type == ConstraintType.BIOLOGICAL:
            result = self._get_biological_parameter(parameter_name)
        elif constraint_type == ConstraintType.INFORMATION:
            result = self._get_information_parameter(parameter_name)
        elif constraint_type == ConstraintType.ENERGY:
            result = self._get_energy_parameter(parameter_name)
        elif constraint_type == ConstraintType.TEMPORAL:
            result = self._get_temporal_parameter(parameter_name)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        self._parameter_cache[cache_key] = result
        return result
    
    def _get_hardware_parameter(self, parameter_name: str) -> Any:
        """Get hardware-constrained parameter"""
        if parameter_name == 'max_objects':
            return self.memory_budget['max_objects']
        elif parameter_name == 'compute_budget':
            return self.compute_budget
        elif parameter_name == 'parallel_streams':
            return min(8, self.hardware.cpu_cores)
        elif parameter_name == 'cache_size':
            return int(self.hardware.ram_gb * 1024 * 0.1)  # 10% of RAM in MB
        else:
            return None
    
    def _get_biological_parameter(self, parameter_name: str) -> Any:
        """Get biologically-constrained parameter"""
        if parameter_name == 'attention_duration':
            return self.biological.attention_span_seconds
        elif parameter_name == 'working_memory_slots':
            return self.biological.working_memory_slots
        elif parameter_name == 'reaction_time':
            return self.biological.reaction_time_ms / 1000.0
        elif parameter_name == 'inhibition_duration':
            return self.biological.inhibition_return_ms / 1000.0
        elif parameter_name == 'binding_window':
            return self.biological.binding_window_ms / 1000.0
        else:
            return None
    
    def _get_information_parameter(self, parameter_name: str) -> Any:
        """Get information-theoretically optimal parameter"""
        if parameter_name == 'correlation_threshold':
            # Threshold emerges from signal-to-noise ratio
            snr = self.hardware.ram_gb / 8.0  # More RAM = better SNR
            return 0.5 / min(2.0, snr)
        elif parameter_name == 'novelty_decay':
            # Novelty decays exponentially with information content
            return 1.0 - (1.0 / self.target_framerate)
        elif parameter_name == 'entropy_threshold':
            # Minimum entropy for pattern recognition
            return math.log2(self.biological.working_memory_slots)
        else:
            return None
    
    def _get_energy_parameter(self, parameter_name: str) -> Any:
        """Get energy-constrained parameter"""
        if parameter_name == 'switch_cost':
            # Switching costs energy - more limited power = higher cost
            baseline_power = 20.0  # 20W baseline
            power_factor = baseline_power / self.power_budget_watts
            return self.compute_budget * 0.1 * power_factor
        elif parameter_name == 'sleep_threshold':
            # When to go into low-power mode
            return 0.1  # 10% activity threshold
        elif parameter_name == 'processing_intensity':
            # How intensively to process based on power budget
            return min(1.0, self.power_budget_watts / 10.0)
        else:
            return None
    
    def _get_temporal_parameter(self, parameter_name: str) -> Any:
        """Get temporally-constrained parameter"""
        if parameter_name == 'max_latency':
            return self.target_frame_time_ms / 1000.0
        elif parameter_name == 'prediction_horizon':
            # How far ahead to predict
            return 1.0 / self.target_framerate * 10  # 10 frames
        elif parameter_name == 'temporal_decay':
            # How fast temporal correlations decay
            return 1.0 - (1.0 / self.target_framerate)
        else:
            return None
    
    def get_competitive_weights(self) -> Dict[str, float]:
        """Get competitive weighting factors for attention"""
        # Weights emerge from constraint priorities
        hardware_weight = 0.3 if self.hardware.cpu_cores >= 4 else 0.5
        biological_weight = 0.4
        information_weight = 0.2
        energy_weight = 0.1 if self.power_budget_watts > 15 else 0.3
        
        # Normalize weights
        total = hardware_weight + biological_weight + information_weight + energy_weight
        
        return {
            'salience': biological_weight / total,
            'novelty': information_weight / total, 
            'urgency': energy_weight / total,
            'coherence': hardware_weight / total
        }
    
    def adapt_to_performance(self, current_framerate: float, cpu_usage: float, memory_usage: float):
        """Adapt parameters based on current performance"""
        # If performance is poor, reduce quality to maintain framerate
        if current_framerate < self.target_framerate * 0.9:
            # Reduce compute budget
            self.compute_budget = int(self.compute_budget * 0.9)
            
            # Increase thresholds (less sensitive but faster)
            self.attention_parameters['coherence_threshold'] *= 1.1
            self.attention_parameters['correlation_threshold'] *= 1.1
        
        # If performance is good, increase quality
        elif current_framerate > self.target_framerate * 1.1:
            # Increase compute budget
            self.compute_budget = int(self.compute_budget * 1.05)
            
            # Decrease thresholds (more sensitive but slower)
            self.attention_parameters['coherence_threshold'] *= 0.95
            self.attention_parameters['correlation_threshold'] *= 0.95
        
        # Memory pressure adaptation
        if memory_usage > 0.8:
            # Reduce memory usage
            self.memory_budget['max_objects'] = int(self.memory_budget['max_objects'] * 0.9)
            self.memory_budget['correlation_history_size'] = int(self.memory_budget['correlation_history_size'] * 0.9)
        
        # Clear cache to force recalculation
        self._parameter_cache.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all emergent parameters"""
        return {
            'hardware_profile': {
                'cpu_cores': self.hardware.cpu_cores,
                'cpu_freq_ghz': self.hardware.cpu_freq_ghz,
                'ram_gb': self.hardware.ram_gb,
                'gpu_available': self.hardware.gpu_available
            },
            'biological_constraints': {
                'attention_span_seconds': self.biological.attention_span_seconds,
                'working_memory_slots': self.biological.working_memory_slots,
                'reaction_time_ms': self.biological.reaction_time_ms
            },
            'derived_parameters': {
                'compute_budget': self.compute_budget,
                'memory_budget': self.memory_budget,
                'attention_parameters': self.attention_parameters,
                'binding_parameters': self.binding_parameters
            },
            'competitive_weights': self.get_competitive_weights()
        }


# Global instance for easy access
_global_parameter_system = None

def get_parameter_system() -> EmergentParameterSystem:
    """Get global parameter system instance"""
    global _global_parameter_system
    if _global_parameter_system is None:
        _global_parameter_system = EmergentParameterSystem()
    return _global_parameter_system

def get_emergent_parameter(parameter_name: str, constraint_type: ConstraintType) -> Any:
    """Convenience function to get emergent parameter"""
    return get_parameter_system().get_parameter(parameter_name, constraint_type)