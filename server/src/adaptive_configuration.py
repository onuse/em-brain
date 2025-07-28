#!/usr/bin/env python3
"""
Adaptive Configuration System - The ONLY configuration system.

This is the single source of truth for all brain configuration.
No other configuration files or systems should exist.
"""

import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AdaptiveConfiguration:
    """The complete brain configuration with hardware adaptation."""
    
    # Core settings (from settings.json)
    brain_type: str = "field"
    sensory_dim: int = 24
    motor_dim: int = 4
    spatial_resolution: Optional[int] = None
    
    # Features
    enhanced_dynamics: bool = True
    attention_guidance: bool = True
    hierarchical_processing: bool = True
    attention_super_resolution: bool = False
    
    # Memory
    persistent_memory_path: str = "./robot_memory"
    enable_persistence: bool = True
    
    # Network
    host: str = "0.0.0.0"
    port: int = 9999
    monitoring_port: int = 9998
    
    # Logging
    log_directory: str = "./logs"
    log_level: str = "info"
    enable_async_logging: bool = True
    
    # Hardware info (detected)
    device_type: str = "cpu"
    cpu_cores: int = 1
    system_memory_gb: float = 4.0
    gpu_memory_gb: float = 0.0
    
    # Performance (adaptive)
    target_cycle_time_ms: int = 150
    working_memory_limit: int = 100
    similarity_search_limit: int = 1000
    
    # Overrides
    force_spatial_resolution: Optional[int] = None
    force_device: Optional[str] = None
    disable_adaptation: bool = False


class AdaptiveConfigurationManager:
    """
    The ONE AND ONLY configuration manager for the brain.
    
    Responsibilities:
    1. Load settings.json
    2. Detect hardware capabilities
    3. Determine optimal configuration
    4. Provide configuration to all components
    5. NO OTHER CONFIGURATION SYSTEMS SHOULD EXIST
    """
    
    def __init__(self, settings_file: str = "settings.json", suppress_output: bool = False):
        self.settings_file = Path(settings_file)
        self.config = AdaptiveConfiguration()
        self.suppress_output = suppress_output
        
        # Load user settings
        self._load_settings()
        
        # Detect hardware
        self._detect_hardware()
        
        # Determine adaptive settings
        self._determine_adaptive_settings()
    
    def _load_settings(self):
        """Load settings.json - the ONLY settings file."""
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                data = json.load(f)
                
                # Brain settings
                brain = data.get('brain', {})
                self.config.brain_type = brain.get('type', 'field')
                self.config.sensory_dim = brain.get('sensory_dim', 24)
                self.config.motor_dim = brain.get('motor_dim', 4)
                self.config.spatial_resolution = brain.get('spatial_resolution')
                
                # Features
                features = brain.get('features', {})
                self.config.enhanced_dynamics = features.get('enhanced_dynamics', True)
                self.config.attention_guidance = features.get('attention_guidance', True)
                self.config.hierarchical_processing = features.get('hierarchical_processing', True)
                self.config.attention_super_resolution = features.get('attention_super_resolution', False)
                
                # Memory
                memory = data.get('memory', {})
                self.config.persistent_memory_path = memory.get('persistent_memory_path', './robot_memory')
                self.config.enable_persistence = memory.get('enable_persistence', True)
                
                # Network
                network = data.get('network', {})
                self.config.host = network.get('host', '0.0.0.0')
                self.config.port = network.get('port', 9999)
                self.config.monitoring_port = network.get('monitoring_port', 9998)
                
                # Logging
                logging = data.get('logging', {})
                self.config.log_directory = logging.get('log_directory', './logs')
                self.config.log_level = logging.get('log_level', 'info')
                self.config.enable_async_logging = logging.get('enable_async_logging', True)
                
                # Overrides
                overrides = data.get('overrides', {})
                self.config.force_spatial_resolution = overrides.get('force_spatial_resolution')
                self.config.force_device = overrides.get('force_device')
                self.config.disable_adaptation = overrides.get('disable_adaptation', False)
    
    def _detect_hardware(self):
        """Detect hardware capabilities."""
        # CPU and memory
        self.config.cpu_cores = psutil.cpu_count(logical=False) or 1
        self.config.system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU detection
        if TORCH_AVAILABLE and not self.config.force_device:
            if torch.cuda.is_available():
                self.config.device_type = 'cuda'
                if torch.cuda.device_count() > 0:
                    props = torch.cuda.get_device_properties(0)
                    self.config.gpu_memory_gb = props.total_memory / (1024**3)
            elif torch.backends.mps.is_available():
                self.config.device_type = 'mps'
                # MPS unified memory estimate
                self.config.gpu_memory_gb = min(self.config.system_memory_gb * 0.6, 32.0)
            else:
                self.config.device_type = 'cpu'
        elif self.config.force_device:
            self.config.device_type = self.config.force_device
    
    def _determine_adaptive_settings(self):
        """Determine adaptive settings based on hardware."""
        if self.config.disable_adaptation:
            return
        
        # Spatial resolution
        if self.config.force_spatial_resolution is not None:
            self.config.spatial_resolution = self.config.force_spatial_resolution
        elif self.config.spatial_resolution is None:
            # Auto-determine based on hardware
            self.config.spatial_resolution = self._calculate_optimal_resolution()
        
        # Cognitive limits based on memory
        memory_factor = min(self.config.system_memory_gb / 8.0, 2.0)  # 8GB baseline
        
        if self.config.device_type == 'cuda' and self.config.gpu_memory_gb > 8:
            self.config.working_memory_limit = int(1000 * memory_factor)
            self.config.similarity_search_limit = int(20000 * memory_factor)
        elif self.config.device_type in ['cuda', 'mps']:
            self.config.working_memory_limit = int(500 * memory_factor)
            self.config.similarity_search_limit = int(10000 * memory_factor)
        else:  # CPU
            self.config.working_memory_limit = int(200 * memory_factor)
            self.config.similarity_search_limit = int(5000 * memory_factor)
    
    def _calculate_optimal_resolution(self) -> int:
        """Calculate optimal spatial resolution based on hardware."""
        # Quick benchmark
        if TORCH_AVAILABLE:
            try:
                device = self.get_device()
                
                # Test computation speed
                size = 64
                test_tensor = torch.randn(size, size, size, device=device)
                
                start = time.perf_counter()
                for _ in range(10):
                    result = torch.nn.functional.conv3d(
                        test_tensor.unsqueeze(0).unsqueeze(0),
                        torch.ones(1, 1, 3, 3, 3, device=device),
                        padding=1
                    )
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed_ms = (time.perf_counter() - start) * 100  # ms per op
                
                # Choose resolution based on speed
                if elapsed_ms < 5:
                    return 6  # Very fast
                elif elapsed_ms < 20:
                    return 5  # Fast
                elif elapsed_ms < 50:
                    return 4  # Medium
                else:
                    return 3  # Slow
                    
            except Exception:
                pass
        
        # Fallback based on device type
        if self.config.device_type == 'cuda':
            return 5
        elif self.config.device_type == 'mps':
            return 4
        else:
            return 3
    
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        if not TORCH_AVAILABLE:
            return None
            
        return torch.device(self.config.device_type)
    
    def get_brain_config(self) -> Dict[str, Any]:
        """Get configuration for brain initialization."""
        return {
            'type': self.config.brain_type,
            'sensory_dim': self.config.sensory_dim,
            'motor_dim': self.config.motor_dim,
            'field_spatial_resolution': self.config.spatial_resolution,
            'target_cycle_time_ms': self.config.target_cycle_time_ms,
            'enhanced_dynamics': self.config.enhanced_dynamics,
            'attention_guidance': self.config.attention_guidance,
            'hierarchical_processing': self.config.hierarchical_processing,
            'attention_super_resolution': self.config.attention_super_resolution,
            'working_memory_limit': self.config.working_memory_limit,
            'similarity_search_limit': self.config.similarity_search_limit
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            'brain': self.get_brain_config(),
            'memory': {
                'persistent_memory_path': self.config.persistent_memory_path,
                'enable_persistence': self.config.enable_persistence
            },
            'network': {
                'host': self.config.host,
                'port': self.config.port,
                'monitoring_port': self.config.monitoring_port
            },
            'logging': {
                'log_directory': self.config.log_directory,
                'log_level': self.config.log_level,
                'enable_async_logging': self.config.enable_async_logging
            },
            'system': {
                'device_type': self.config.device_type,
                'cpu_cores': self.config.cpu_cores,
                'system_memory_gb': self.config.system_memory_gb,
                'gpu_memory_gb': self.config.gpu_memory_gb
            }
        }
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("🔧 ADAPTIVE CONFIGURATION")
        print("="*60)
        
        print(f"\n💻 Hardware:")
        print(f"   Device: {self.config.device_type.upper()}")
        print(f"   CPU cores: {self.config.cpu_cores}")
        print(f"   RAM: {self.config.system_memory_gb:.1f}GB")
        if self.config.gpu_memory_gb > 0:
            print(f"   GPU memory: {self.config.gpu_memory_gb:.1f}GB")
        
        print(f"\n🧠 Brain Configuration:")
        print(f"   Type: {self.config.brain_type}")
        # Handle dynamic dimensions gracefully
        if self.config.sensory_dim is not None and self.config.motor_dim is not None:
            print(f"   Sensory/Motor: {self.config.sensory_dim}D → {self.config.motor_dim}D")
        else:
            print(f"   Sensory/Motor: Dynamic (determined by robot)")
        print(f"   Spatial resolution: {self.config.spatial_resolution}³")
        
        if self.config.force_spatial_resolution is not None:
            print(f"   ⚠️  Resolution forced to {self.config.force_spatial_resolution}³")
        
        print(f"\n✅ Features:")
        if self.config.enhanced_dynamics: print("   - Enhanced Dynamics")
        if self.config.attention_guidance: print("   - Attention Guidance")
        if self.config.hierarchical_processing: print("   - Hierarchical Processing")
        if self.config.attention_super_resolution: print("   - Super Resolution")
        
        print("="*60 + "\n")


# The ONE global configuration instance
_config_manager: Optional[AdaptiveConfigurationManager] = None


def load_adaptive_configuration(settings_file: str = "settings.json", suppress_output: bool = False) -> Dict[str, Any]:
    """Load configuration - This is the ONLY way to get configuration."""
    global _config_manager
    _config_manager = AdaptiveConfigurationManager(settings_file, suppress_output)
    if not suppress_output:
        _config_manager.print_summary()
    return _config_manager.get_full_config()


def get_configuration() -> AdaptiveConfiguration:
    """Get the current configuration object."""
    global _config_manager
    if _config_manager is None:
        load_adaptive_configuration()
    return _config_manager.config


def get_device() -> torch.device:
    """Get the PyTorch device from configuration."""
    global _config_manager
    if _config_manager is None:
        load_adaptive_configuration()
    return _config_manager.get_device()


if __name__ == "__main__":
    # Test the configuration system
    config = load_adaptive_configuration()
    print("\nGenerated configuration:")
    print(json.dumps(config, indent=2))