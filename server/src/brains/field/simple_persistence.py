"""
Simple Persistence - Just save and load the field.

No complex formats. Just torch.save/load.
"""

import torch
import os
from pathlib import Path
from typing import Optional


class SimplePersistence:
    """Dead simple persistence for the field brain."""
    
    def __init__(self, save_dir: str = "brain_states"):
        """
        Initialize persistence.
        
        Args:
            save_dir: Directory to save brain states
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def save(self, brain, name: Optional[str] = None) -> str:
        """
        Save brain state to disk.
        
        Args:
            brain: UnifiedFieldBrain instance
            name: Optional name for the save file
            
        Returns:
            Path to saved file
        """
        if name is None:
            name = f"brain_cycle_{brain.cycle}.pt"
        
        filepath = self.save_dir / name
        
        # Save just the essentials
        state = {
            'field': brain.field.cpu(),
            'cycle': brain.cycle,
            'spatial_size': brain.spatial_size,
            'channels': brain.channels,
            # Save learnable parameters
            'prediction_projection': brain.prediction.projection.cpu(),
            'motor_regions': brain.motor.motor_regions.cpu(),
        }
        
        torch.save(state, filepath)
        print(f"💾 Saved brain state to {filepath}")
        return str(filepath)
    
    def load(self, brain, name: str) -> bool:
        """
        Load brain state from disk.
        
        Args:
            brain: UnifiedFieldBrain instance to load into
            name: Name of the save file
            
        Returns:
            True if successful
        """
        filepath = self.save_dir / name
        
        if not filepath.exists():
            print(f"❌ Save file not found: {filepath}")
            return False
        
        try:
            state = torch.load(filepath, map_location=brain.device)
            
            # Verify dimensions match
            if state['spatial_size'] != brain.spatial_size or state['channels'] != brain.channels:
                print(f"❌ Dimension mismatch: saved {state['spatial_size']}³×{state['channels']}, "
                      f"current {brain.spatial_size}³×{brain.channels}")
                return False
            
            # Load field
            brain.field = state['field'].to(brain.device)
            brain.cycle = state['cycle']
            
            # Load learned parameters
            brain.prediction.projection = state['prediction_projection'].to(brain.device)
            brain.motor.motor_regions = state['motor_regions'].to(brain.device)
            
            print(f"✅ Loaded brain state from {filepath} (cycle {brain.cycle})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            return False
    
    def list_saves(self) -> list:
        """List available save files."""
        saves = list(self.save_dir.glob("*.pt"))
        return sorted([s.name for s in saves])
    
    def auto_save(self, brain, interval: int = 1000):
        """
        Auto-save if enough cycles have passed.
        
        Args:
            brain: Brain to save
            interval: Save every N cycles
        """
        if brain.cycle % interval == 0 and brain.cycle > 0:
            self.save(brain, f"autosave_cycle_{brain.cycle}.pt")
    
    def cleanup_old_saves(self, keep_recent: int = 5):
        """Keep only the N most recent saves."""
        saves = sorted(self.save_dir.glob("autosave_*.pt"), 
                      key=lambda x: x.stat().st_mtime)
        
        if len(saves) > keep_recent:
            for old_save in saves[:-keep_recent]:
                old_save.unlink()
                print(f"🗑️ Deleted old save: {old_save.name}")