#!/usr/bin/env python3
"""
Unified Configuration System

Single source of truth for brain configuration with clear hierarchy:
1. Command-line arguments (highest priority)
2. Environment variables 
3. settings_unified.json (user configuration)
4. Hardware adaptation (automatic defaults)

No more competing configuration files!
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from ..utils.hardware_adaptation import HardwareAdaptation


class UnifiedConfig:
    """
    Unified configuration manager with single source of truth.
    """
    
    def __init__(self, config_file: str = "settings_unified.json", override_args: Dict[str, Any] = None):
        self.config_file = Path(__file__).parent.parent.parent / config_file
        self.override_args = override_args or {}
        
        # Load base configuration
        self.config = self._load_base_config()
        
        # Apply environment variables
        self._apply_env_overrides()
        
        # Apply command-line overrides
        self._apply_arg_overrides()
        
        # Apply hardware adaptation for unset values
        self._apply_hardware_adaptation()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from settings_unified.json"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                # Remove comment lines
                content = '\n'.join(line for line in f if not line.strip().startswith('//'))
                return json.loads(content)
        else:
            # Minimal defaults if no config file
            return {
                "brain": {
                    "type": "field",
                    "sensory_dim": 24,
                    "motor_dim": 4,
                    "spatial_resolution": None,
                    "enhanced_dynamics": True,
                    "attention_guidance": True,
                    "hierarchical_processing": True
                },
                "memory": {
                    "persistent_memory_path": "./robot_memory",
                    "enable_persistence": True
                },
                "network": {
                    "host": "0.0.0.0",
                    "port": 9999,
                    "monitoring_port": 9998
                },
                "logging": {
                    "log_directory": "./logs",
                    "log_level": "info"
                },
                "performance": {
                    "target_cycle_time_ms": 150,
                    "force_spatial_resolution": None
                }
            }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # BRAIN_SPATIAL_RESOLUTION=3 python brain_server.py
        if os.environ.get('BRAIN_SPATIAL_RESOLUTION'):
            self.config['performance']['force_spatial_resolution'] = int(os.environ['BRAIN_SPATIAL_RESOLUTION'])
        
        # BRAIN_DISABLE_PERSISTENCE=1 python brain_server.py
        if os.environ.get('BRAIN_DISABLE_PERSISTENCE'):
            self.config['memory']['enable_persistence'] = False
        
        # BRAIN_LOG_LEVEL=debug python brain_server.py
        if os.environ.get('BRAIN_LOG_LEVEL'):
            self.config['logging']['log_level'] = os.environ['BRAIN_LOG_LEVEL']
    
    def _apply_arg_overrides(self):
        """Apply command-line argument overrides"""
        for key, value in self.override_args.items():
            if key == 'spatial_resolution' and value is not None:
                self.config['performance']['force_spatial_resolution'] = value
            elif key == 'disable_persistence':
                self.config['memory']['enable_persistence'] = not value
            elif key == 'port':
                self.config['network']['port'] = value
    
    def _apply_hardware_adaptation(self):
        """Apply hardware adaptation for unset values"""
        # Only adapt if not explicitly set
        if (self.config['brain'].get('spatial_resolution') is None and 
            self.config['performance'].get('force_spatial_resolution') is None):
            
            # Get hardware recommendation
            hw_profile = HardwareAdaptation.analyze_hardware()
            
            # Use conservative setting (recommended - 1)
            recommended = hw_profile.recommended_spatial_resolution
            self.config['brain']['spatial_resolution'] = max(3, recommended - 1)
            
            print(f"🔧 Hardware-adapted spatial resolution: {self.config['brain']['spatial_resolution']}³")
        
        # Use forced value if set
        elif self.config['performance'].get('force_spatial_resolution') is not None:
            self.config['brain']['spatial_resolution'] = self.config['performance']['force_spatial_resolution']
            print(f"⚠️  Using forced spatial resolution: {self.config['brain']['spatial_resolution']}³")
    
    def get_brain_config(self) -> Dict[str, Any]:
        """Get brain-specific configuration"""
        brain_config = self.config['brain'].copy()
        
        # Map spatial_resolution to field_spatial_resolution for compatibility
        if 'spatial_resolution' in brain_config:
            brain_config['field_spatial_resolution'] = brain_config['spatial_resolution']
        
        # Add performance settings
        brain_config['target_cycle_time_ms'] = self.config['performance']['target_cycle_time_ms']
        
        return brain_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("📋 UNIFIED CONFIGURATION")
        print("="*60)
        
        brain = self.config['brain']
        perf = self.config['performance']
        
        print(f"\n🧠 Brain Configuration:")
        print(f"   Type: {brain['type']}")
        print(f"   Sensory dimensions: {brain['sensory_dim']}")
        print(f"   Motor dimensions: {brain['motor_dim']}")
        print(f"   Spatial resolution: {brain.get('spatial_resolution', 'auto')}³")
        
        print(f"\n⚡ Performance Settings:")
        print(f"   Target cycle time: {perf['target_cycle_time_ms']}ms")
        if perf.get('force_spatial_resolution'):
            print(f"   Forced resolution: {perf['force_spatial_resolution']}³")
        
        print(f"\n✅ Features:")
        features = []
        if brain.get('enhanced_dynamics'): features.append('Enhanced Dynamics')
        if brain.get('attention_guidance'): features.append('Attention')
        if brain.get('hierarchical_processing'): features.append('Hierarchical')
        for feature in features:
            print(f"   - {feature}")
        
        print("="*60 + "\n")


def load_unified_config(config_file: str = "settings_unified.json", **overrides) -> Dict[str, Any]:
    """Load configuration with unified system"""
    config_manager = UnifiedConfig(config_file, overrides)
    config_manager.print_summary()
    return config_manager.get_config()


if __name__ == "__main__":
    # Test unified configuration
    config = load_unified_config()
    print("\nBrain config:")
    print(json.dumps(UnifiedConfig().get_brain_config(), indent=2))