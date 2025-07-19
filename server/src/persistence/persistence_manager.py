"""
Persistence Manager

Central coordinator for the production-grade brain persistence subsystem.
Orchestrates all persistence operations: incremental saves, consolidation, and recovery.
"""

import os
import time
import threading
from typing import Dict, Any, Optional

from .brain_serializer import BrainSerializer, SerializedBrainState
from .incremental_engine import IncrementalEngine
from .consolidation_engine import ConsolidationEngine
from .recovery_manager import RecoveryManager, RecoveryResult
from .storage_backend import StorageBackend
from .persistence_config import PersistenceConfig, create_default_config


class PersistenceManager:
    """
    Central persistence manager coordinating all brain state persistence operations.
    
    Features:
    - Incremental saves with full brain content every N cycles
    - Automatic consolidation when incremental files accumulate
    - Robust recovery from consolidated + incremental files
    - Crash resistance and corruption handling
    - Background threads for non-blocking operation
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None, memory_path: str = "./robot_memory"):
        # Initialize configuration
        if config is None:
            config = create_default_config(memory_path)
        self.config = config
        
        # Initialize subsystem components
        self.brain_serializer = BrainSerializer()
        self.storage = StorageBackend(config)
        self.incremental_engine = IncrementalEngine(config)
        self.consolidation_engine = ConsolidationEngine(config)
        self.recovery_manager = RecoveryManager(config)
        
        # Manager state
        self.is_initialized = False
        self.current_session_id = None
        self.startup_recovery_result: Optional[RecoveryResult] = None
        
        # Performance tracking
        self.manager_stats = {
            'initialization_time_ms': 0.0,
            'total_cycles_processed': 0,
            'total_saves_requested': 0,
            'total_consolidations_triggered': 0,
            'session_start_time': time.time()
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize the persistence subsystem
        self._initialize()
    
    def _initialize(self):
        """Initialize the persistence subsystem."""
        start_time = time.perf_counter()
        
        try:
            print("🚀 Initializing production persistence subsystem...")
            
            # Ensure all directories exist
            self.storage.ensure_directories_exist()
            
            # Start background engines
            if not self.incremental_engine.is_running:
                self.incremental_engine.start_background_thread()
            
            if not self.consolidation_engine.is_running:
                self.consolidation_engine.start_background_thread()
            
            # Generate session ID
            self.current_session_id = f"session_{int(time.time())}"
            
            self.is_initialized = True
            
            init_time_ms = (time.perf_counter() - start_time) * 1000
            self.manager_stats['initialization_time_ms'] = init_time_ms
            
            print(f"✅ Persistence subsystem initialized ({init_time_ms:.1f}ms)")
            
        except Exception as e:
            print(f"❌ Failed to initialize persistence subsystem: {e}")
            raise
    
    def recover_brain_state_at_startup(self) -> Optional[SerializedBrainState]:
        """
        Recover complete brain state at startup using the full recovery pipeline.
        
        Returns the recovered brain state or None if no state exists.
        """
        print("🔄 Starting brain state recovery at startup...")
        
        try:
            # Use recovery manager to handle the complex recovery logic
            self.startup_recovery_result = self.recovery_manager.recover_brain_state()
            
            if self.startup_recovery_result.success:
                brain_state = self.startup_recovery_result.brain_state
                
                # Only increment session count if this is a real recovery (not fresh state)
                if self.startup_recovery_result.recovery_method != "fresh_state":
                    brain_state.session_count += 1
                
                print(f"🧠 Brain state recovered successfully:")
                print(f"   Session: {brain_state.session_count}")
                print(f"   Patterns: {len(brain_state.patterns)}")
                print(f"   Experiences: {brain_state.total_experiences}")
                print(f"   Recovery method: {self.startup_recovery_result.recovery_method}")
                
                return brain_state
            else:
                print("📝 No existing brain state found - will start fresh")
                return None
                
        except Exception as e:
            print(f"⚠️ Brain state recovery failed: {e}")
            return None
    
    def process_brain_cycle(self, brain):
        """
        Process a brain cycle for persistence (called every brain cycle).
        
        This is the main entry point for ongoing persistence operations.
        """
        with self._lock:
            self.manager_stats['total_cycles_processed'] += 1
            
            # Check if incremental save should be triggered
            self.incremental_engine.maybe_save_incremental(brain)
            
            # Check if consolidation should be triggered
            if self.consolidation_engine.check_consolidation_needed():
                self.manager_stats['total_consolidations_triggered'] += 1
                self.consolidation_engine.trigger_consolidation()
    
    def save_brain_state_incremental(self, brain, priority: int = 0) -> bool:
        """
        Save brain state incrementally (non-blocking).
        
        Args:
            brain: The brain instance to save
            priority: Save priority (higher = more urgent)
            
        Returns:
            True if save was queued successfully
        """
        with self._lock:
            self.manager_stats['total_saves_requested'] += 1
            
            return self.incremental_engine.queue_incremental_save(brain, priority)
    
    def save_brain_state_blocking(self, brain) -> bool:
        """
        Save brain state immediately (blocking - for shutdown).
        
        Args:
            brain: The brain instance to save
            
        Returns:
            True if save was successful
        """
        print("💾 Performing blocking brain state save...")
        
        try:
            # Force an immediate incremental save
            success = self.incremental_engine.force_incremental_save(brain)
            
            if success:
                print("✅ Blocking save completed successfully")
            else:
                print("❌ Blocking save failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Blocking save failed with exception: {e}")
            return False
    
    def trigger_consolidation(self, force: bool = False) -> bool:
        """
        Trigger consolidation of incremental files.
        
        Args:
            force: Force consolidation even if not needed
            
        Returns:
            True if consolidation was triggered
        """
        return self.consolidation_engine.trigger_consolidation(force)
    
    def cleanup_old_files(self):
        """Clean up old files based on configured retention policies."""
        try:
            print("🗑️ Starting cleanup of old persistence files...")
            
            # Clean up old incremental files
            self.incremental_engine.cleanup_old_incremental_files()
            
            # Clean up old backups
            self.storage.cleanup_old_backups()
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
    
    def get_persistence_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the persistence subsystem."""
        consolidation_status = self.consolidation_engine.get_consolidation_status()
        recovery_status = self.recovery_manager.get_recovery_status()
        
        # Calculate total disk usage
        total_disk_usage_mb = (
            self.storage.get_directory_size(self.config.get_consolidated_dir()) +
            self.storage.get_directory_size(self.config.get_incremental_dir()) +
            self.storage.get_directory_size(self.config.get_metadata_dir()) +
            self.storage.get_directory_size(self.config.get_recovery_dir())
        )
        
        return {
            'is_initialized': self.is_initialized,
            'current_session_id': self.current_session_id,
            'total_disk_usage_mb': total_disk_usage_mb,
            'startup_recovery': {
                'success': self.startup_recovery_result.success if self.startup_recovery_result else False,
                'method': self.startup_recovery_result.recovery_method if self.startup_recovery_result else None,
                'patterns_recovered': self.startup_recovery_result.patterns_recovered if self.startup_recovery_result else 0
            },
            'consolidation': consolidation_status,
            'recovery': recovery_status,
            'incremental_engine': {
                'is_running': self.incremental_engine.is_running,
                'queue_size': self.incremental_engine.save_queue.qsize() if self.incremental_engine.is_running else 0,
                'cycles_since_last_save': self.incremental_engine.cycles_since_last_save
            },
            'manager_stats': self.manager_stats
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all subsystem components."""
        return {
            'manager': self.manager_stats,
            'brain_serializer': self.brain_serializer.get_stats(),
            'incremental_engine': self.incremental_engine.get_stats(),
            'consolidation_engine': self.consolidation_engine.get_stats(),
            'recovery_manager': self.recovery_manager.get_stats(),
            'storage_backend': self.storage.get_stats(),
            'configuration': {
                'memory_root_path': self.config.memory_root_path,
                'incremental_save_interval_cycles': self.config.incremental_save_interval_cycles,
                'max_incremental_files': self.config.consolidation_policy.max_incremental_files,
                'enable_compression': self.config.enable_compression,
                'enable_corruption_detection': self.config.enable_corruption_detection
            }
        }
    
    def validate_subsystem_health(self) -> Dict[str, Any]:
        """Validate that all subsystem components are healthy."""
        health_report = {
            'overall_healthy': True,
            'issues': [],
            'warnings': []
        }
        
        try:
            # Check if directories exist and are writable
            required_dirs = [
                self.config.get_consolidated_dir(),
                self.config.get_incremental_dir(),
                self.config.get_metadata_dir(),
                self.config.get_recovery_dir()
            ]
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    health_report['issues'].append(f"Directory missing: {directory}")
                    health_report['overall_healthy'] = False
                elif not os.access(directory, os.W_OK):
                    health_report['issues'].append(f"Directory not writable: {directory}")
                    health_report['overall_healthy'] = False
            
            # Check background threads
            if not self.incremental_engine.is_running:
                health_report['issues'].append("Incremental save engine not running")
                health_report['overall_healthy'] = False
            
            if not self.consolidation_engine.is_running:
                health_report['warnings'].append("Consolidation engine not running")
            
            # Check disk usage
            total_usage_mb = sum([
                self.storage.get_directory_size(d) for d in required_dirs
            ])
            
            if total_usage_mb > 1000:  # > 1GB
                health_report['warnings'].append(f"High disk usage: {total_usage_mb:.1f}MB")
            
            # Check for failed operations
            stats = self.get_comprehensive_stats()
            
            if stats['incremental_engine']['failed_saves'] > 0:
                health_report['warnings'].append(f"Incremental save failures: {stats['incremental_engine']['failed_saves']}")
            
            if stats['consolidation_engine']['failed_consolidations'] > 0:
                health_report['warnings'].append(f"Consolidation failures: {stats['consolidation_engine']['failed_consolidations']}")
            
            if stats['storage_backend']['atomic_write_failures'] > 0:
                health_report['warnings'].append(f"Storage write failures: {stats['storage_backend']['atomic_write_failures']}")
            
        except Exception as e:
            health_report['issues'].append(f"Health check failed: {e}")
            health_report['overall_healthy'] = False
        
        return health_report
    
    def shutdown(self, timeout: float = 10.0):
        """Shutdown the persistence subsystem gracefully."""
        print("🛑 Shutting down persistence subsystem...")
        
        try:
            # Stop background engines
            self.incremental_engine.stop_background_thread(timeout=timeout/2)
            self.consolidation_engine.stop_background_thread(timeout=timeout/2)
            
            # Perform final cleanup
            self.cleanup_old_files()
            
            print("✅ Persistence subsystem shutdown complete")
            
        except Exception as e:
            print(f"⚠️ Error during persistence shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with graceful shutdown."""
        self.shutdown()
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'is_initialized') and self.is_initialized:
            self.shutdown(timeout=2.0)