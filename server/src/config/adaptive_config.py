#!/usr/bin/env python3
"""
Adaptive Configuration System

Three-tier configuration management:
1. Hardware-Adaptive: Automatically determined from hardware detection
2. User Settings: Feature flags and paths that users care about
3. Developer Overrides: Advanced parameters for development/tuning

Eliminates the need for users to understand technical parameters while
providing full control for developers.
"""

import json
import os
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

# GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        DEVICE_TYPE = 'cuda'
        DEVICE_COUNT = torch.cuda.device_count()
        if DEVICE_COUNT > 0:
            DEVICE_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            DEVICE_MEMORY_GB = 0
    elif torch.backends.mps.is_available():
        DEVICE_TYPE = 'mps'
        DEVICE_COUNT = 1
        # MPS uses unified memory - estimate conservatively
        system_memory = psutil.virtual_memory().total / (1024**3)
        DEVICE_MEMORY_GB = min(system_memory * 0.8, 64.0)
    else:
        DEVICE_TYPE = 'cpu'
        DEVICE_COUNT = 0
        DEVICE_MEMORY_GB = 0
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE_TYPE = 'cpu'
    DEVICE_COUNT = 0
    DEVICE_MEMORY_GB = 0


class AdaptiveConfigManager:
    """
    Manages three-tier configuration with hardware adaptation.
    """
    
    def __init__(self, user_config_path: str = "settings_simple.json"):
        self.base_dir = Path(__file__).parent.parent.parent
        self.user_config_path = self.base_dir / user_config_path
        self.hardware_defaults_path = Path(__file__).parent / "hardware_defaults.json"
        self.dev_overrides_path = Path(__file__).parent / "developer_overrides.json"
        
        # System info
        self.cpu_cores = psutil.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.device_type = DEVICE_TYPE
        self.device_memory_gb = DEVICE_MEMORY_GB
        
        # Load configurations
        self.user_config = self._load_user_config()
        self.hardware_defaults = self._load_hardware_defaults()
        self.dev_overrides = self._load_dev_overrides()
        
        # Generate final config
        self.final_config = self._generate_adaptive_config()
    
    def _load_user_config(self) -> Dict[str, Any]:
        """Load user-facing configuration."""
        if self.user_config_path.exists():
            with open(self.user_config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  User config not found at {self.user_config_path}, using defaults")
            return {
                "brain": {"type": "field", "sensory_dim": 16, "motor_dim": 4},
                "memory": {"persistent_memory_path": "./server/robot_memory", "enable_persistence": True},
                "network": {"host": "0.0.0.0", "port": 9999},
                "logging": {"log_directory": "./logs", "log_level": "info"}
            }
    
    def _load_hardware_defaults(self) -> Dict[str, Any]:
        """Load hardware-adaptive defaults."""
        with open(self.hardware_defaults_path, 'r') as f:
            return json.load(f)
    
    def _load_dev_overrides(self) -> Dict[str, Any]:
        """Load developer overrides if available."""
        if self.dev_overrides_path.exists():
            with open(self.dev_overrides_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _generate_adaptive_config(self) -> Dict[str, Any]:
        """Generate final configuration with hardware adaptation."""
        config = {}
        
        # Start with user configuration
        config.update(self.user_config)
        
        # Add hardware-adaptive brain parameters
        brain_config = config.setdefault("brain", {})
        self._apply_hardware_adaptive_brain_config(brain_config)
        
        # Add hardware-adaptive system parameters
        config["system"] = self._generate_system_config()
        
        # Apply developer overrides
        self._apply_dev_overrides(config)
        
        return config
    
    def _apply_hardware_adaptive_brain_config(self, brain_config: Dict[str, Any]):
        """Apply hardware-adaptive brain parameters."""
        brain_type = brain_config.get("type", "field")
        hardware_defaults = self.hardware_defaults.get("field_brain", {})
        scaling = self.hardware_defaults.get("hardware_scaling", {})
        
        # Base parameters from hardware defaults
        brain_config.setdefault("temporal_dim", 4)
        brain_config.setdefault("prediction_threshold", hardware_defaults.get("prediction_threshold", 0.7))
        brain_config.setdefault("novelty_threshold", hardware_defaults.get("novelty_threshold", 0.3))
        
        # Hardware-scaled spatial resolution
        base_resolution = hardware_defaults.get("base_spatial_resolution", 20)
        device_multiplier = scaling.get("spatial_resolution_scaling", {}).get(f"{self.device_type}_multiplier", 1.0)
        
        # Scale by available memory (more VRAM = higher resolution)
        if self.device_memory_gb > 0:
            memory_multiplier = min(self.device_memory_gb / 4.0, 3.0)  # 4GB baseline, max 3x
        else:
            memory_multiplier = 0.5  # CPU fallback
            
        adaptive_resolution = int(base_resolution * device_multiplier * memory_multiplier)
        brain_config["field_spatial_resolution"] = adaptive_resolution
        
        # Hardware-scaled cycle time
        cycle_base = scaling.get("cycle_time_scaling", {}).get(f"{self.device_type}_base_ms", 200)
        min_cycle = hardware_defaults.get("min_cycle_time_ms", 50)
        brain_config["target_cycle_time_ms"] = max(cycle_base, min_cycle)
        
        # Hardware-scaled evolution rate
        base_evolution_rate = hardware_defaults.get("base_field_evolution_rate", 0.1)
        evolution_multiplier = scaling.get("evolution_rate_scaling", {}).get(f"{self.device_type}_multiplier", 1.0)
        brain_config["field_evolution_rate"] = base_evolution_rate * evolution_multiplier
        
        # Other base parameters
        brain_config.setdefault("field_temporal_window", hardware_defaults.get("base_temporal_window", 10.0))
        brain_config.setdefault("constraint_discovery_rate", hardware_defaults.get("base_constraint_discovery_rate", 0.15))
        brain_config.setdefault("learning_rate", hardware_defaults.get("base_learning_rate", 0.1))
        
        # Attention parameters if enabled
        if brain_config.get("attention_guidance", False):
            brain_config.setdefault("attention_base_resolution", hardware_defaults.get("attention_base_resolution", 50))
            brain_config.setdefault("attention_focus_resolution", hardware_defaults.get("attention_focus_resolution", 100))
        
        # Hierarchical parameters if enabled
        if brain_config.get("hierarchical_processing", False):
            brain_config.setdefault("hierarchical_max_time_ms", hardware_defaults.get("hierarchical_max_time_ms", 40.0))
    
    def _generate_system_config(self) -> Dict[str, Any]:
        """Generate hardware-adaptive system configuration."""
        scaling = self.hardware_defaults.get("hardware_scaling", {})
        
        # Use all available resources aggressively
        vram_percent = scaling.get("vram_usage_percent", 100)
        ram_percent = scaling.get("ram_usage_percent", 90)
        
        # Calculate actual memory limits
        if self.device_memory_gb > 0:
            gpu_memory_limit_mb = int(self.device_memory_gb * 1024 * (vram_percent / 100))
        else:
            gpu_memory_limit_mb = 0
            
        system_memory_limit_mb = int(self.total_memory_gb * 1024 * (ram_percent / 100))
        
        return {
            "device_type": self.device_type,
            "cpu_cores": self.cpu_cores,
            "total_memory_gb": self.total_memory_gb,
            "device_memory_gb": self.device_memory_gb,
            "gpu_memory_limit_mb": gpu_memory_limit_mb,
            "system_memory_limit_mb": system_memory_limit_mb,
            "max_worker_threads": max(1, self.cpu_cores - 1),  # Reserve one core
            "hardware_adaptation_enabled": True
        }
    
    def _apply_dev_overrides(self, config: Dict[str, Any]):
        """Apply developer overrides."""
        if not self.dev_overrides:
            return
            
        # Apply testing overrides
        testing = self.dev_overrides.get("testing_overrides", {})
        if testing.get("force_cycle_time_ms"):
            config["brain"]["target_cycle_time_ms"] = testing["force_cycle_time_ms"]
        if testing.get("mock_gpu_memory_gb"):
            config["system"]["device_memory_gb"] = testing["mock_gpu_memory_gb"]
        if testing.get("disable_persistence"):
            config["memory"]["enable_persistence"] = False
            
        # Apply development overrides
        dev = self.dev_overrides.get("development_overrides", {})
        if dev.get("force_cpu_mode"):
            config["system"]["device_type"] = "cpu"
        if dev.get("disable_hardware_adaptation"):
            config["system"]["hardware_adaptation_enabled"] = False
    
    def get_config(self) -> Dict[str, Any]:
        """Get the final adaptive configuration."""
        return self.final_config
    
    def print_configuration_summary(self):
        """Print a summary of the configuration decisions."""
        print("\\n" + "="*60)
        print("🔧 ADAPTIVE CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"\\n💻 Hardware Detection:")
        print(f"   Device: {self.device_type.upper()}")
        print(f"   CPU cores: {self.cpu_cores}")
        print(f"   System RAM: {self.total_memory_gb:.1f}GB")
        if self.device_memory_gb > 0:
            print(f"   GPU memory: {self.device_memory_gb:.1f}GB")
        
        brain = self.final_config["brain"]
        system = self.final_config["system"]
        
        print(f"\\n🧠 Adaptive Brain Configuration:")
        print(f"   Spatial resolution: {brain.get('field_spatial_resolution', 'N/A')}")
        print(f"   Target cycle time: {brain.get('target_cycle_time_ms', 'N/A')}ms")
        print(f"   Evolution rate: {brain.get('field_evolution_rate', 'N/A')}")
        
        print(f"\\n⚡ System Resource Allocation:")
        if system.get("gpu_memory_limit_mb", 0) > 0:
            print(f"   GPU memory limit: {system['gpu_memory_limit_mb']}MB (100% of available)")
        print(f"   System memory limit: {system.get('system_memory_limit_mb', 0)}MB (90% of available)")
        print(f"   Worker threads: {system.get('max_worker_threads', 1)}")
        
        features = []
        if brain.get("enhanced_dynamics"): features.append("enhanced_dynamics")
        if brain.get("attention_guidance"): features.append("attention_guidance")
        if brain.get("hierarchical_processing"): features.append("hierarchical_processing")
        if brain.get("attention_super_resolution"): features.append("attention_super_resolution")
        
        print(f"\\n🎯 Enabled Features: {', '.join(features) if features else 'None'}")
        
        print("="*60 + "\\n")


def create_adaptive_config(user_config_path: str = "settings_simple.json") -> Dict[str, Any]:
    """Create adaptive configuration from user settings and hardware detection."""
    manager = AdaptiveConfigManager(user_config_path)
    manager.print_configuration_summary()
    return manager.get_config()


if __name__ == "__main__":
    # Test the adaptive configuration system
    config = create_adaptive_config()
    
    print("Generated configuration:")
    print(json.dumps(config, indent=2))