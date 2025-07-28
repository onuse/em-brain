"""
Integrated Persistence for Dynamic Brain Architecture

This integrates the persistence system with the current dynamic brain architecture,
providing automatic saves, recovery, and session continuity.
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .dynamic_persistence_adapter import DynamicPersistenceAdapter, DynamicBrainState
from .binary_persistence import BinaryPersistence
from ..core.interfaces import IBrain


class IntegratedPersistence:
    """
    Simplified persistence manager for the dynamic brain architecture.
    
    Features:
    - Automatic periodic saves
    - Save on shutdown
    - Recovery on startup
    - Session tracking
    - Backward compatibility
    """
    
    def __init__(self, 
                 memory_path: str = "./brain_memory",
                 save_interval_cycles: int = 1000,
                 auto_save: bool = True):
        """
        Initialize integrated persistence.
        
        Args:
            memory_path: Directory for brain state files
            save_interval_cycles: Save every N brain cycles
            auto_save: Enable automatic periodic saves
        """
        self.memory_path = Path(memory_path)
        self.save_interval_cycles = save_interval_cycles
        self.auto_save = auto_save
        
        # Create adapter with compression disabled until it's implemented
        self.adapter = DynamicPersistenceAdapter(compression_enabled=False)
        
        # State tracking
        self.last_save_cycle = 0
        self.session_id = f"session_{int(time.time())}"
        self.save_count = 0
        
        # Threading
        self._lock = threading.Lock()
        self._save_thread = None
        self._shutdown = threading.Event()
        
        # Ensure directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🗄️ Integrated persistence initialized")
        print(f"   Memory path: {self.memory_path}")
        print(f"   Auto-save: {'enabled' if auto_save else 'disabled'}")
        print(f"   Save interval: {save_interval_cycles} cycles")
    
    def get_latest_state_file(self) -> Optional[Path]:
        """Find the most recent brain state file."""
        state_files = list(self.memory_path.glob("brain_state_*.json"))
        if not state_files:
            return None
        
        # Sort by modification time
        state_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return state_files[0]
    
    def recover_brain_state(self, brain: IBrain) -> bool:
        """
        Recover brain state from disk on startup.
        
        Args:
            brain: The brain instance to restore state to
            
        Returns:
            True if state was recovered successfully
        """
        latest_file = self.get_latest_state_file()
        if not latest_file:
            print("📝 No existing brain state found - starting fresh")
            return False
        
        try:
            print(f"🔄 Recovering brain state from {latest_file.name}")
            
            # Load state from file
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Deserialize
            state = self.adapter.deserialize_from_dict(data)
            
            # Increment session count
            state.session_count += 1
            
            # Restore to brain
            success = self.adapter.restore_brain_state(brain, state)
            
            if success:
                print(f"✅ Brain state recovered successfully")
                print(f"   Session: {state.session_count}")
                print(f"   Brain cycles: {state.brain_cycles}")
                print(f"   Total experiences: {state.total_experiences}")
                print(f"   Last saved: {datetime.fromtimestamp(state.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Update tracking
                self.last_save_cycle = state.brain_cycles
                
            return success
            
        except Exception as e:
            print(f"❌ Failed to recover brain state: {e}")
            return False
    
    def save_brain_state(self, brain: IBrain, blocking: bool = False) -> bool:
        """
        Save brain state to disk.
        
        Args:
            brain: The brain instance to save
            blocking: If True, save synchronously. If False, save in background.
            
        Returns:
            True if save was successful (or queued successfully)
        """
        if blocking:
            return self._save_brain_state_sync(brain)
        else:
            # Queue for background save
            if self._save_thread and self._save_thread.is_alive():
                print("⏳ Save already in progress, skipping")
                return False
            
            self._save_thread = threading.Thread(
                target=self._save_brain_state_sync,
                args=(brain,),
                daemon=True
            )
            self._save_thread.start()
            return True
    
    def _save_brain_state_sync(self, brain: IBrain) -> bool:
        """Synchronous save implementation."""
        with self._lock:
            try:
                start_time = time.perf_counter()
                
                # Extract state
                state = self.adapter.extract_brain_state(brain)
                
                # Update session info
                state.session_count = getattr(state, 'session_count', 0)
                state.timestamp = time.time()
                
                # Serialize
                data = self.adapter.serialize_to_dict(state)
                
                # Generate filename
                filename = f"brain_state_{self.session_id}_{state.brain_cycles}.json"
                filepath = self.memory_path / filename
                
                # Write to file with numpy array handling
                with open(filepath, 'w') as f:
                    # Use custom encoder for numpy arrays
                    import numpy as np
                    
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            if isinstance(obj, np.integer):
                                return int(obj)
                            if isinstance(obj, np.floating):
                                return float(obj)
                            return super().default(obj)
                    
                    json.dump(data, f, indent=2, cls=NumpyEncoder)
                
                # Clean up old files (keep last 10)
                self._cleanup_old_files()
                
                duration = (time.perf_counter() - start_time) * 1000
                print(f"💾 Brain state saved: {filename} ({duration:.1f}ms)")
                print(f"   Brain cycles: {state.brain_cycles}")
                print(f"   Field energy: {state.field_energy:.4f}")
                print(f"   Memory regions: {len(state.topology_regions)}")
                
                self.save_count += 1
                self.last_save_cycle = state.brain_cycles
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to save brain state: {e}")
                return False
    
    def _cleanup_old_files(self, keep_count: int = 10):
        """Remove old state files, keeping the most recent ones."""
        state_files = list(self.memory_path.glob("brain_state_*.json"))
        if len(state_files) <= keep_count:
            return
        
        # Sort by modification time
        state_files.sort(key=lambda f: f.stat().st_mtime)
        
        # Remove oldest files
        for f in state_files[:-keep_count]:
            try:
                f.unlink()
                print(f"🗑️ Removed old state file: {f.name}")
            except:
                pass
    
    def check_auto_save(self, brain: IBrain) -> bool:
        """
        Check if auto-save should be triggered based on cycle count.
        
        Args:
            brain: The brain instance to check
            
        Returns:
            True if save was triggered
        """
        if not self.auto_save:
            return False
        
        # Get current cycle count
        brain_obj = brain.brain if hasattr(brain, 'brain') else brain
        current_cycle = brain_obj.brain_cycles
        
        # Check if interval has passed
        if current_cycle - self.last_save_cycle >= self.save_interval_cycles:
            return self.save_brain_state(brain, blocking=False)
        
        return False
    
    def shutdown_save(self, brain: IBrain) -> bool:
        """
        Perform a final save before shutdown.
        
        Args:
            brain: The brain instance to save
            
        Returns:
            True if save was successful
        """
        print("🔚 Performing shutdown save...")
        
        # Signal shutdown
        self._shutdown.set()
        
        # Wait for any pending saves
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5.0)
        
        # Perform final blocking save
        return self.save_brain_state(brain, blocking=True)
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return {
            'session_id': self.session_id,
            'save_count': self.save_count,
            'last_save_cycle': self.last_save_cycle,
            'auto_save_enabled': self.auto_save,
            'save_interval': self.save_interval_cycles,
            'memory_path': str(self.memory_path),
            'state_files': len(list(self.memory_path.glob("brain_state_*.json")))
        }


# Global persistence instance
_global_persistence: Optional[IntegratedPersistence] = None


def initialize_persistence(memory_path: str = "./brain_memory",
                         save_interval_cycles: int = 1000,
                         auto_save: bool = True) -> IntegratedPersistence:
    """Initialize the global persistence instance."""
    global _global_persistence
    _global_persistence = IntegratedPersistence(memory_path, save_interval_cycles, auto_save)
    return _global_persistence


def get_persistence() -> Optional[IntegratedPersistence]:
    """Get the global persistence instance."""
    return _global_persistence