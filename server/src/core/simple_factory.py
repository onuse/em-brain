"""
Simple Brain Factory - Creates field-based brains.

Automatically configures optimal parameters based on hardware.
Supports multiple brain architectures for testing emergence.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.unified_field_brain import UnifiedFieldBrain
from src.brains.field.critical_mass_field_brain import CriticalMassFieldBrain, EmergenceConfig
from src.brains.field.enhanced_critical_mass_brain import EnhancedCriticalMassBrain
from src.brains.field.auto_config import get_optimal_config


def create_brain(brain_type='unified', target='balanced', quiet=False):
    """
    Create a brain with optimal settings.
    
    Args:
        brain_type: 'unified', 'critical_mass', 'emergence', or 'enhanced'
        target: 'speed', 'balanced', or 'intelligence'
        quiet: Suppress output
    
    Returns:
        Brain instance (UnifiedFieldBrain, CriticalMassFieldBrain, or EnhancedCriticalMassBrain)
    """
    
    if brain_type == 'enhanced':
        # Create Enhanced Critical Mass Brain with full learning capabilities
        if not quiet:
            print("Creating Enhanced Critical Mass Brain (Full Learning Systems)")
            print("Features: Causal Learning + Semantic Grounding + Temporal Memory + Curiosity")
        
        # Configure based on target
        if target == 'speed':
            config = EmergenceConfig(
                field_size=(32, 32, 32, 64),  # Smaller for speed
                swarm_size=500,
                superposition_branches=50
            )
        elif target == 'intelligence':
            config = EmergenceConfig(
                field_size=(64, 64, 64, 128),  # Larger for more emergence
                swarm_size=2000,
                superposition_branches=200
            )
        else:  # balanced
            config = EmergenceConfig()  # Use defaults
        
        if not quiet:
            print(f"Field size: {config.field_size}")
            print(f"Total parameters: {sum(config.field_size) * config.field_size[-1]:,}")
            print("Learning systems: Predictive chains, semantic grounding, temporal memory, curiosity")
        
        brain = EnhancedCriticalMassBrain(config=config)
        
    elif brain_type in ['critical_mass', 'emergence']:
        # Create Critical Mass Brain for testing emergence hypothesis
        if not quiet:
            print("Creating Critical Mass Brain (Emergence Testing)")
        
        # Configure based on target
        if target == 'speed':
            config = EmergenceConfig(
                field_size=(32, 32, 32, 64),  # Smaller for speed
                swarm_size=500,
                superposition_branches=50
            )
        elif target == 'intelligence':
            config = EmergenceConfig(
                field_size=(64, 64, 64, 128),  # Larger for more emergence
                swarm_size=2000,
                superposition_branches=200
            )
        else:  # balanced
            config = EmergenceConfig()  # Use defaults
        
        if not quiet:
            print(f"Field size: {config.field_size}")
            print(f"Total parameters: {sum(config.field_size) * config.field_size[-1]:,}")
            print(f"Swarm agents: {config.swarm_size}")
            print(f"Superposition branches: {config.superposition_branches}")
            print("Hypothesis: Intelligence emerges from constrained dynamics")
        
        brain = CriticalMassFieldBrain(config=config)
        
    else:  # default to unified
        # Create original UnifiedFieldBrain
        config = get_optimal_config(target)
        
        if not quiet:
            print(f"Creating Unified Field Brain: {config['spatial_size']}³×{config['channels']}")
            print(f"Parameters: {config['parameters']:,}")
            print(f"Device: {config['device']}")
        
        brain = UnifiedFieldBrain(
            spatial_size=config['spatial_size'],
            channels=config['channels'],
            device=config['device'],
            quiet_mode=quiet
        )
    
    return brain