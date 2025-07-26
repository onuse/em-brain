#!/usr/bin/env python3
"""
Unified Hardware Configuration & Adaptation System

Combines startup configuration with runtime adaptation into a single,
coherent system with one source of truth.
"""

import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HardwareProfile:
    """Complete hardware profile including static capabilities and dynamic performance."""
    # Static Hardware Info
    device_type: str = 'cpu'  # 'cpu', 'mps', 'cuda'
    cpu_cores: int = 1
    system_memory_gb: float = 0.0
    gpu_memory_gb: float = 0.0
    
    # Performance Measurements
    benchmark_score: float = 0.0  # Startup benchmark result
    current_cycle_time_ms: float = 150.0
    avg_cycle_time_ms: float = 150.0
    cycle_time_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Adaptive Settings (start conservative, adapt up/down)
    spatial_resolution: int = 4
    working_memory_limit: int = 100
    similarity_search_limit: int = 1000
    batch_processing_threshold: int = 50
    
    # Performance State
    performance_tier: str = 'unknown'
    last_adaptation_time: float = field(default_factory=time.time)
    adaptation_count: int = 0


class UnifiedHardwareConfig:
    """
    Single source of truth for all hardware-related configuration and adaptation.
    
    Responsibilities:
    1. Detect hardware capabilities at startup
    2. Load/save hardware configuration
    3. Provide initial settings based on hardware
    4. Monitor and adapt performance at runtime
    5. Make GPU usage decisions
    """
    
    # Performance tier thresholds (ms)
    TIER_THRESHOLDS = {
        'datacenter': 10,     # <10ms cycles
        'workstation': 25,    # <25ms cycles  
        'desktop': 50,        # <50ms cycles
        'laptop': 100,        # <100ms cycles
        'embedded': 200,      # <200ms cycles
        'slow': float('inf')  # >200ms cycles
    }
    
    # Spatial resolution by tier
    RESOLUTION_BY_TIER = {
        'datacenter': 8,
        'workstation': 6,
        'desktop': 5,
        'laptop': 4,
        'embedded': 3,
        'slow': 3
    }
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.profile = HardwareProfile()
        
        # Detect hardware
        self._detect_hardware()
        
        # Run benchmark
        self._run_benchmark()
        
        # Load saved config or use defaults
        self._load_or_initialize_config()
        
        # Start adaptation tracking
        self._adaptation_enabled = True
        self._min_samples_for_adaptation = 20
    
    def _detect_hardware(self):
        """Detect static hardware capabilities."""
        # CPU and memory
        self.profile.cpu_cores = psutil.cpu_count(logical=False) or 1
        self.profile.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.profile.device_type = 'cuda'
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    self.profile.gpu_memory_gb = props.total_memory / (1024**3)
            elif torch.backends.mps.is_available():
                self.profile.device_type = 'mps'
                # MPS uses unified memory - estimate conservatively
                self.profile.gpu_memory_gb = min(self.profile.system_memory_gb * 0.6, 32.0)
            else:
                self.profile.device_type = 'cpu'
    
    def _run_benchmark(self):
        """Run a quick benchmark to estimate processing speed."""
        if not TORCH_AVAILABLE:
            self.profile.benchmark_score = 100.0
            return
        
        try:
            device = self._get_device()
            
            # Small benchmark computation
            size = 100
            start = time.perf_counter()
            
            # Create test tensors
            a = torch.randn(size, size, size, device=device)
            b = torch.randn(size, size, size, device=device)
            
            # Perform operations
            for _ in range(10):
                c = torch.matmul(a.reshape(size, -1), b.reshape(-1, size))
                c = torch.nn.functional.relu(c)
            
            # Ensure completion
            if device.type != 'cpu':
                torch.cuda.synchronize() if device.type == 'cuda' else None
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            self.profile.benchmark_score = elapsed / 10  # ms per operation
            
        except Exception as e:
            print(f"⚠️ Benchmark failed: {e}")
            self.profile.benchmark_score = 100.0
    
    def _load_or_initialize_config(self):
        """Load saved configuration or initialize from benchmark."""
        # Try to load saved config
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved = json.load(f)
                    # Only load adaptive settings, not hardware info
                    self.profile.spatial_resolution = saved.get('spatial_resolution', 4)
                    self.profile.working_memory_limit = saved.get('working_memory_limit', 100)
                    self.profile.similarity_search_limit = saved.get('similarity_search_limit', 1000)
                    print(f"📂 Loaded hardware config from {self.config_file}")
                    return
            except Exception as e:
                print(f"⚠️ Failed to load config: {e}")
        
        # Initialize from benchmark
        self._initialize_from_benchmark()
    
    def _initialize_from_benchmark(self):
        """Initialize settings based on benchmark results."""
        # Determine performance tier
        for tier, threshold in self.TIER_THRESHOLDS.items():
            if self.profile.benchmark_score < threshold:
                self.profile.performance_tier = tier
                break
        
        # Set initial resolution
        self.profile.spatial_resolution = self.RESOLUTION_BY_TIER[self.profile.performance_tier]
        
        # Set cognitive limits based on available memory
        memory_factor = min(self.profile.system_memory_gb / 8.0, 2.0)  # 8GB baseline
        
        if self.profile.performance_tier in ['datacenter', 'workstation']:
            self.profile.working_memory_limit = int(1000 * memory_factor)
            self.profile.similarity_search_limit = int(20000 * memory_factor)
        elif self.profile.performance_tier in ['desktop', 'laptop']:
            self.profile.working_memory_limit = int(500 * memory_factor)
            self.profile.similarity_search_limit = int(10000 * memory_factor)
        else:  # embedded/slow
            self.profile.working_memory_limit = int(200 * memory_factor)
            self.profile.similarity_search_limit = int(5000 * memory_factor)
        
        self.profile.batch_processing_threshold = max(10, self.profile.cpu_cores * 5)
    
    def record_cycle_performance(self, cycle_time_ms: float):
        """Record actual cycle performance and adapt if needed."""
        self.profile.current_cycle_time_ms = cycle_time_ms
        self.profile.cycle_time_history.append(cycle_time_ms)
        
        # Update average
        if len(self.profile.cycle_time_history) > 0:
            self.profile.avg_cycle_time_ms = np.mean(list(self.profile.cycle_time_history))
        
        # Check if we should adapt
        if (self._adaptation_enabled and 
            len(self.profile.cycle_time_history) >= self._min_samples_for_adaptation and
            time.time() - self.profile.last_adaptation_time > 30):  # Adapt at most every 30s
            
            self._adapt_to_performance()
    
    def _adapt_to_performance(self):
        """Adapt settings based on actual performance."""
        avg_time = self.profile.avg_cycle_time_ms
        
        # Determine new tier based on actual performance
        old_tier = self.profile.performance_tier
        for tier, threshold in self.TIER_THRESHOLDS.items():
            if avg_time < threshold:
                self.profile.performance_tier = tier
                break
        
        # Only adapt if tier changed or performance is concerning
        if old_tier != self.profile.performance_tier or avg_time > 500:
            print(f"🔧 Adapting to {self.profile.performance_tier} tier (avg: {avg_time:.1f}ms)")
            
            # Adjust spatial resolution
            target_resolution = self.RESOLUTION_BY_TIER[self.profile.performance_tier]
            if avg_time > 500 and self.profile.spatial_resolution > 3:
                # Emergency reduction
                self.profile.spatial_resolution = max(3, self.profile.spatial_resolution - 1)
                print(f"   ⚠️ Reducing spatial resolution to {self.profile.spatial_resolution}³")
            elif avg_time < 50 and self.profile.spatial_resolution < target_resolution:
                # Can increase
                self.profile.spatial_resolution = min(target_resolution, self.profile.spatial_resolution + 1)
                print(f"   ⬆️ Increasing spatial resolution to {self.profile.spatial_resolution}³")
            
            # Adjust cognitive limits
            if avg_time > 300:
                # Reduce limits
                self.profile.working_memory_limit = int(self.profile.working_memory_limit * 0.8)
                self.profile.similarity_search_limit = int(self.profile.similarity_search_limit * 0.8)
                print(f"   ⬇️ Reducing cognitive limits")
            elif avg_time < 100:
                # Increase limits
                self.profile.working_memory_limit = int(self.profile.working_memory_limit * 1.2)
                self.profile.similarity_search_limit = int(self.profile.similarity_search_limit * 1.2)
                print(f"   ⬆️ Increasing cognitive limits")
            
            self.profile.last_adaptation_time = time.time()
            self.profile.adaptation_count += 1
            
            # Save adapted config
            self._save_config()
    
    def should_use_gpu(self, operation_type: str, data_size: int) -> bool:
        """Decide whether to use GPU for a specific operation."""
        if self.profile.device_type == 'cpu':
            return False
        
        # GPU overhead thresholds
        thresholds = {
            'similarity_search': 1000,  # Use GPU if >1000 patterns
            'pattern_formation': 500,
            'field_evolution': 10000,   # Use GPU if field >10k elements
            'gradient_computation': 5000
        }
        
        min_size = thresholds.get(operation_type, 1000)
        return data_size >= min_size
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for brain initialization."""
        return {
            'brain': {
                'field_spatial_resolution': self.profile.spatial_resolution,
                'target_cycle_time_ms': 150,  # Always target biological timescale
                'sensory_dim': 24,
                'motor_dim': 4
            },
            'cognitive': {
                'working_memory_limit': self.profile.working_memory_limit,
                'similarity_search_limit': self.profile.similarity_search_limit,
                'batch_processing_threshold': self.profile.batch_processing_threshold
            },
            'hardware': {
                'device_type': self.profile.device_type,
                'cpu_cores': self.profile.cpu_cores,
                'system_memory_gb': self.profile.system_memory_gb,
                'gpu_memory_gb': self.profile.gpu_memory_gb,
                'performance_tier': self.profile.performance_tier
            }
        }
    
    def _get_device(self) -> torch.device:
        """Get PyTorch device."""
        if not TORCH_AVAILABLE:
            return None
        
        if self.profile.device_type == 'cuda':
            return torch.device('cuda')
        elif self.profile.device_type == 'mps':
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _save_config(self):
        """Save current configuration."""
        if not self.config_file:
            return
        
        try:
            config = {
                'spatial_resolution': self.profile.spatial_resolution,
                'working_memory_limit': self.profile.working_memory_limit,
                'similarity_search_limit': self.profile.similarity_search_limit,
                'performance_tier': self.profile.performance_tier,
                'avg_cycle_time_ms': self.profile.avg_cycle_time_ms
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")
    
    def print_summary(self):
        """Print hardware configuration summary."""
        print("\n" + "="*60)
        print("🔧 UNIFIED HARDWARE CONFIGURATION")
        print("="*60)
        
        print(f"\n💻 Hardware:")
        print(f"   Device: {self.profile.device_type.upper()}")
        print(f"   CPU cores: {self.profile.cpu_cores}")
        print(f"   RAM: {self.profile.system_memory_gb:.1f}GB")
        if self.profile.gpu_memory_gb > 0:
            print(f"   GPU memory: {self.profile.gpu_memory_gb:.1f}GB")
        
        print(f"\n📊 Performance:")
        print(f"   Benchmark: {self.profile.benchmark_score:.1f}ms")
        print(f"   Tier: {self.profile.performance_tier}")
        if self.profile.avg_cycle_time_ms > 0:
            print(f"   Avg cycle: {self.profile.avg_cycle_time_ms:.1f}ms")
        
        print(f"\n🧠 Adaptive Settings:")
        print(f"   Spatial resolution: {self.profile.spatial_resolution}³")
        print(f"   Working memory: {self.profile.working_memory_limit}")
        print(f"   Search limit: {self.profile.similarity_search_limit}")
        
        print("="*60 + "\n")


# Global instance for easy access
_hardware_config: Optional[UnifiedHardwareConfig] = None


def get_hardware_config() -> UnifiedHardwareConfig:
    """Get or create the global hardware configuration."""
    global _hardware_config
    if _hardware_config is None:
        _hardware_config = UnifiedHardwareConfig()
    return _hardware_config


def initialize_hardware_config(config_file: Optional[str] = None) -> UnifiedHardwareConfig:
    """Initialize the hardware configuration system."""
    global _hardware_config
    _hardware_config = UnifiedHardwareConfig(config_file)
    _hardware_config.print_summary()
    return _hardware_config


if __name__ == "__main__":
    # Test the unified system
    hw = initialize_hardware_config()
    config = hw.get_config_dict()
    print("\nGenerated config:")
    print(json.dumps(config, indent=2))