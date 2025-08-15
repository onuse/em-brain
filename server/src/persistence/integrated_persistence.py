"""
Integrated Persistence with Binary Format

Fast and efficient persistence for dynamic brain architecture using binary format.
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
    Integrated persistence with efficient binary storage.
    
    This replaces JSON with binary format, providing:
    - ~20-50x smaller files
    - ~10-100x faster save/load
    - Support for terabyte-scale brains
    """
    
    def __init__(self, 
                 memory_path: str = "./brain_memory",
                 save_interval_cycles: int = 1000,
                 auto_save: bool = True,
                 use_binary: bool = True):
        """
        Initialize integrated persistence.
        
        Args:
            memory_path: Directory for brain state files
            save_interval_cycles: Save every N brain cycles
            auto_save: Enable automatic periodic saves
            use_binary: Use efficient binary format (recommended)
        """
        self.memory_path = Path(memory_path)
        self.save_interval_cycles = save_interval_cycles
        self.auto_save = auto_save
        self.use_binary = use_binary
        
        # Create persistence backend
        self.binary_persistence = BinaryPersistence(memory_path) if use_binary else None
        
        # Create adapter
        self.adapter = DynamicPersistenceAdapter(compression_enabled=False)
        
        # State tracking
        self.last_save_cycle = 0
        self.session_id = f"session_{int(time.time())}"
        self.save_count = 0
        self.session_count = 0  # Track session count for recovery
        
        # Threading
        self._lock = threading.Lock()
        self._save_thread = None
        self._shutdown = threading.Event()
        
        # Ensure directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Print statements removed - brain_service.py handles this
    
    def get_latest_state_file(self) -> Optional[Path]:
        """Find the most recent brain state file."""
        if self.use_binary:
            # Binary format uses metadata files
            state_files = list(self.memory_path.glob("brain_state_*_meta.json"))
        else:
            # Legacy JSON format
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
        if self.use_binary and self.binary_persistence:
            # Use fast binary loading
            state_dict = self.binary_persistence.load_brain_state()
            if not state_dict:
                return False
                
            try:
                # Convert dict back to brain state using correct method
                if hasattr(self.adapter, 'deserialize_from_dict'):
                    brain_state = self.adapter.deserialize_from_dict(state_dict)
                else:
                    brain_state = self.adapter.dict_to_brain_state(state_dict)
                
                # Apply to brain
                self.adapter.restore_brain_state(brain, brain_state)
                
                # Update tracking
                brain_obj = brain.brain if hasattr(brain, 'brain') else brain
                self.last_save_cycle = brain_obj.brain_cycles
                self.session_id = state_dict.get('session_id', self.session_id)
                
                # Show recovery info
                print(f"✅ Brain state recovered successfully")
                print(f"   Session: {self.session_count + 1}")
                print(f"   Brain cycles: {brain_obj.brain_cycles}")
                print(f"   Total experiences: {len(brain_obj.experiences) if hasattr(brain_obj, 'experiences') else 0}")
                print(f"   Last saved: {datetime.fromtimestamp(state_dict.get('save_timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.session_count += 1
                return True
                
            except Exception as e:
                print(f"❌ Failed to restore brain state: {e}")
                return False
        else:
            # Fall back to legacy JSON loading
            return self._recover_json_state(brain)
    
    def _recover_json_state(self, brain: IBrain) -> bool:
        """Legacy JSON recovery for compatibility"""
        latest_file = self.get_latest_state_file()
        if not latest_file:
            print("📄 No saved brain state found")
            return False
        
        try:
            print(f"🔄 Recovering brain state from {latest_file.name}")
            start_time = time.time()
            
            with open(latest_file, 'r') as f:
                state_dict = json.load(f)
            
            # Convert to brain state using correct method
            if hasattr(self.adapter, 'deserialize_from_dict'):
                brain_state = self.adapter.deserialize_from_dict(state_dict)
            else:
                brain_state = self.adapter.dict_to_brain_state(state_dict)
            
            # Apply to brain
            self.adapter.restore_brain_state(brain, brain_state)
            
            load_time = time.time() - start_time
            print(f"   Load time: {load_time:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to recover brain state: {e}")
            return False
    
    def save_brain_state(self, brain: IBrain, blocking: bool = True) -> bool:
        """
        Save current brain state.
        
        Args:
            brain: The brain instance to save
            blocking: If False, save in background thread
            
        Returns:
            True if save was initiated successfully
        """
        if not blocking:
            # Start background save
            if self._save_thread and self._save_thread.is_alive():
                print("⚠️  Previous save still in progress")
                return False
            
            self._save_thread = threading.Thread(
                target=self._save_brain_state,
                args=(brain,),
                daemon=True
            )
            self._save_thread.start()
            return True
        else:
            # Blocking save
            return self._save_brain_state(brain)
    
    def _save_brain_state(self, brain: IBrain) -> bool:
        """Internal save method"""
        with self._lock:
            try:
                # Extract state
                state = self.adapter.extract_brain_state(brain)
                state_dict = self.adapter.serialize_to_dict(state)
                
                # Add metadata
                state_dict['session_id'] = self.session_id
                state_dict['save_timestamp'] = time.time()
                state_dict['save_count'] = self.save_count
                
                # Save based on format
                if self.use_binary and self.binary_persistence:
                    # Fast binary save
                    save_time = self.binary_persistence.save_brain_state(
                        state_dict,
                        self.session_id,
                        state.brain_cycles
                    )
                else:
                    # Legacy JSON save
                    save_time = self._save_json_state(state_dict, state.brain_cycles)
                
                # Update tracking
                self.save_count += 1
                self.last_save_cycle = state.brain_cycles
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to save brain state: {e}")
                return False
    
    def _save_json_state(self, state_dict: Dict[str, Any], cycles: int) -> float:
        """Legacy JSON save for compatibility"""
        start_time = time.time()
        
        filename = f"brain_state_{self.session_id}_{cycles}.json"
        filepath = self.memory_path / filename
        
        # Write to temp file first
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        # Atomic rename
        temp_path.rename(filepath)
        
        save_time = time.time() - start_time
        file_size = filepath.stat().st_size / 1e6
        
        print(f"💾 Brain state saved: {filename} ({save_time:.1f}s)")
        print(f"   Brain cycles: {cycles}")
        print(f"   File size: {file_size:.1f} MB")
        
        # Cleanup old files
        self._cleanup_old_files()
        
        return save_time
    
    def _cleanup_old_files(self, keep_count: int = 10):
        """Remove old state files"""
        if self.use_binary and self.binary_persistence:
            # Binary persistence handles its own cleanup
            return
            
        state_files = list(self.memory_path.glob("brain_state_*.json"))
        if len(state_files) <= keep_count:
            return
        
        state_files.sort(key=lambda f: f.stat().st_mtime)
        
        for f in state_files[:-keep_count]:
            try:
                f.unlink()
                print(f"🗑️ Removed old state file: {f.name}")
            except:
                pass
    
    def check_auto_save(self, brain: IBrain) -> bool:
        """Check if auto-save should be triggered"""
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
        """Perform final save on shutdown"""
        print("🔚 Performing shutdown save...")
        
        # Set shutdown flag
        self._shutdown.set()
        
        # Wait for any background save to complete
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=30.0)
        
        # Perform final save
        return self.save_brain_state(brain, blocking=True)
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get persistence statistics"""
        stats = {
            'session_id': self.session_id,
            'save_count': self.save_count,
            'last_save_cycle': self.last_save_cycle,
            'format': 'binary' if self.use_binary else 'json'
        }
        
        if self.use_binary and self.binary_persistence:
            stats.update(self.binary_persistence.get_storage_stats())
        
        return stats


# Module-level initialization
_persistence_instance: Optional[IntegratedPersistence] = None


def initialize_persistence(memory_path: str = "./brain_memory",
                         save_interval_cycles: int = 1000,
                         auto_save: bool = True,
                         use_binary: bool = True) -> IntegratedPersistence:
    """Initialize the global persistence instance"""
    global _persistence_instance
    
    _persistence_instance = IntegratedPersistence(
        memory_path=memory_path,
        save_interval_cycles=save_interval_cycles,
        auto_save=auto_save,
        use_binary=use_binary
    )
    
    return _persistence_instance


def get_persistence() -> Optional[IntegratedPersistence]:
    """Get the global persistence instance"""
    return _persistence_instance