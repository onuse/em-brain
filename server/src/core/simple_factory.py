"""
Simple Brain Factory - Creates the unified field brain.

Automatically configures optimal parameters based on hardware.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.unified_field_brain import UnifiedFieldBrain
from src.brains.field.auto_config import get_optimal_config


def create_brain(target='balanced', quiet=False):
    """
    Create the brain with optimal settings.
    
    Args:
        target: 'speed', 'balanced', or 'intelligence'
        quiet: Suppress output
    
    Returns:
        UnifiedFieldBrain instance
    """
    config = get_optimal_config(target)
    
    if not quiet:
        print(f"Creating brain: {config['spatial_size']}³×{config['channels']}")
        print(f"Parameters: {config['parameters']:,}")
        print(f"Device: {config['device']}")
    
    brain = UnifiedFieldBrain(
        spatial_size=config['spatial_size'],
        channels=config['channels'],
        device=config['device'],
        quiet_mode=quiet
    )
    
    return brain