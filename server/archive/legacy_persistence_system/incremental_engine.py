"""
Incremental Save Engine

Handles frequent saves of complete brain content to incremental files.
Unlike the previous system, these saves contain full brain patterns, not just metadata.
"""

import os
import json
import time
import threading
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional
from queue import Queue, Empty
from dataclasses import asdict

from .brain_serializer import BrainSerializer, SerializedBrainState
from .storage_backend import StorageBackend
from .persistence_config import PersistenceConfig


class IncrementalSaveRequest:
    """A request to save brain state incrementally."""
    
    def __init__(self, brain, request_id: str, priority: int = 0):
        self.brain = brain
        self.request_id = request_id
        self.priority = priority
        self.timestamp = time.time()
        self.cycles_at_request = brain.total_cycles


class IncrementalEngine:
    """Engine for frequent incremental saves of complete brain content."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.brain_serializer = BrainSerializer()
        self.storage = StorageBackend(config)
        
        # Background save system
        self.save_queue = Queue(maxsize=config.save_queue_max_size)
        self.save_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.is_running = False
        
        # Save tracking
        self.cycles_since_last_save = 0
        self.last_save_time = time.time()
        self.save_counter = 0
        
        # Performance stats
        self.stats = {
            'total_incremental_saves': 0,
            'total_patterns_saved': 0,
            'avg_save_time_ms': 0.0,
            'queue_overflows': 0,
            'successful_saves': 0,
            'failed_saves': 0,
            'largest_save_size_mb': 0.0,
            'total_save_size_mb': 0.0
        }
        
        # Ensure directories exist
        self.storage.ensure_directories_exist()
        
        # Start background thread if enabled
        if config.background_save_thread:
            self.start_background_thread()
    
    def start_background_thread(self):
        """Start the background save thread."""
        if self.save_thread and self.save_thread.is_alive():
            return
        
        self.is_running = True
        self.save_thread = threading.Thread(
            target=self._background_save_worker,
            name="IncrementalSaveEngine",
            daemon=True
        )
        self.save_thread.start()
        print("🔄 Incremental save engine started")
    
    def stop_background_thread(self, timeout: float = 5.0):
        """Stop the background save thread gracefully."""
        if not self.is_running:
            return
        
        print("🛑 Stopping incremental save engine...")
        self.shutdown_event.set()
        self.is_running = False
        
        if self.save_thread:
            self.save_thread.join(timeout=timeout)
            if self.save_thread.is_alive():
                print("⚠️ Background save thread did not shutdown cleanly")
        
        # Process any remaining saves synchronously
        remaining_saves = 0
        while not self.save_queue.empty():
            try:
                save_request = self.save_queue.get_nowait()
                self._execute_save_request(save_request)
                remaining_saves += 1
            except Empty:
                break
        
        if remaining_saves > 0:
            print(f"💾 Processed {remaining_saves} remaining saves during shutdown")
    
    def maybe_save_incremental(self, brain):
        """Check if incremental save should be triggered and queue if needed."""
        self.cycles_since_last_save += 1
        current_time = time.time()
        
        # Check trigger conditions
        should_save_cycles = (
            self.cycles_since_last_save >= self.config.incremental_save_interval_cycles
        )
        should_save_time = (
            current_time - self.last_save_time >= self.config.incremental_save_interval_seconds
        )
        
        if should_save_cycles or should_save_time:
            self.queue_incremental_save(brain)
            self.cycles_since_last_save = 0
            self.last_save_time = current_time
    
    def queue_incremental_save(self, brain, priority: int = 0) -> bool:
        """Queue an incremental save request (non-blocking)."""
        if not self.is_running:
            return False
        
        request_id = f"incremental_{self.save_counter:06d}_{int(time.time())}"
        self.save_counter += 1
        
        save_request = IncrementalSaveRequest(brain, request_id, priority)
        
        try:
            # Non-blocking queue add
            self.save_queue.put_nowait(save_request)
            return True
        except:
            # Queue is full - record overflow and skip this save
            self.stats['queue_overflows'] += 1
            print(f"⚠️ Save queue overflow - skipping incremental save {request_id}")
            return False
    
    def force_incremental_save(self, brain) -> bool:
        """Force an immediate incremental save (blocking)."""
        request_id = f"forced_{int(time.time())}"
        save_request = IncrementalSaveRequest(brain, request_id, priority=999)
        
        return self._execute_save_request(save_request)
    
    def _background_save_worker(self):
        """Background thread worker for processing save requests."""
        print("🔄 Background incremental save worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for save request with timeout
                save_request = self.save_queue.get(timeout=1.0)
                
                # Execute the save
                self._execute_save_request(save_request)
                
                # Mark task as done
                self.save_queue.task_done()
                
            except Empty:
                # Timeout - check shutdown and continue
                continue
            except Exception as e:
                print(f"⚠️ Background save worker error: {e}")
                self.stats['failed_saves'] += 1
    
    def _execute_save_request(self, save_request: IncrementalSaveRequest) -> bool:
        """Execute a save request synchronously."""
        start_time = time.perf_counter()
        
        try:
            # Serialize brain state (this extracts all patterns)
            brain_state = self.brain_serializer.serialize_brain_state(save_request.brain)
            
            # Add incremental metadata
            incremental_metadata = {
                'request_id': save_request.request_id,
                'save_type': 'incremental',
                'cycles_at_save': save_request.cycles_at_request,
                'request_timestamp': save_request.timestamp,
                'save_timestamp': time.time()
            }
            
            # Convert to dictionary for storage
            brain_dict = self.brain_serializer.serialize_to_dict(brain_state)
            brain_dict['incremental_metadata'] = incremental_metadata
            
            # Save to incremental file
            filename = f"delta_{save_request.request_id}.json"
            if self.config.enable_compression:
                filename += ".gz"
            
            filepath = os.path.join(self.config.get_incremental_dir(), filename)
            
            # Save using storage backend
            save_size_mb = self.storage.save_json(brain_dict, filepath, compress=self.config.enable_compression)
            
            # Update statistics
            save_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_save_stats(brain_state, save_time_ms, save_size_mb)
            
            print(f"💾 Incremental save: {save_request.request_id} ({save_size_mb:.1f}MB, {save_time_ms:.1f}ms)")
            
            return True
            
        except Exception as e:
            print(f"❌ Incremental save failed {save_request.request_id}: {e}")
            self.stats['failed_saves'] += 1
            return False
    
    def _update_save_stats(self, brain_state: SerializedBrainState, save_time_ms: float, save_size_mb: float):
        """Update save performance statistics."""
        stats = self.stats
        stats['total_incremental_saves'] += 1
        stats['successful_saves'] += 1
        # UnifiedFieldBrain doesn't have patterns - track cycles instead
        stats['total_patterns_saved'] = brain_state.brain_cycles
        stats['total_save_size_mb'] += save_size_mb
        
        # Update averages
        total_saves = stats['total_incremental_saves']
        stats['avg_save_time_ms'] = (
            (stats['avg_save_time_ms'] * (total_saves - 1) + save_time_ms) / total_saves
        )
        
        # Track largest save
        if save_size_mb > stats['largest_save_size_mb']:
            stats['largest_save_size_mb'] = save_size_mb
    
    def get_incremental_files(self) -> List[Dict[str, Any]]:
        """Get list of all incremental save files with metadata."""
        incremental_dir = Path(self.config.get_incremental_dir())
        
        if not incremental_dir.exists():
            return []
        
        files = []
        for filepath in incremental_dir.glob("delta_*.json.gz"):
            # Skip integrity files - only process actual brain state files
            if filepath.name.endswith('.integrity'):
                continue
            try:
                stat = filepath.stat()
                file_info = {
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'size_mb': stat.st_size / (1024 * 1024),
                    'created_time': stat.st_ctime,
                    'modified_time': stat.st_mtime,
                    'compressed': filepath.suffix == '.gz'
                }
                
                # Try to extract request ID from filename
                parts = filepath.stem.replace('.json', '').split('_')
                if len(parts) >= 2:
                    file_info['request_id'] = '_'.join(parts[1:])
                
                files.append(file_info)
                
            except Exception as e:
                print(f"⚠️ Error reading incremental file {filepath}: {e}")
        
        # Sort by creation time
        files.sort(key=lambda x: x['created_time'])
        return files
    
    def load_incremental_file(self, filepath: str) -> Optional[SerializedBrainState]:
        """Load a specific incremental save file."""
        try:
            # Load JSON data
            brain_dict = self.storage.load_json(filepath)
            
            if not brain_dict:
                return None
            
            # Remove incremental metadata if present
            brain_dict.pop('incremental_metadata', None)
            # Remove consolidation metadata if present
            brain_dict.pop('consolidation_metadata', None)
            
            # Convert back to SerializedBrainState
            return self.brain_serializer.from_dict(brain_dict)
            
        except Exception as e:
            print(f"⚠️ Failed to load incremental file {filepath}: {e}")
            return None
    
    def cleanup_old_incremental_files(self, keep_days: int = None):
        """Clean up old incremental files based on age."""
        if keep_days is None:
            keep_days = self.config.keep_incremental_history_days
        
        cutoff_time = time.time() - (keep_days * 24 * 3600)
        incremental_files = self.get_incremental_files()
        
        removed_count = 0
        for file_info in incremental_files:
            if file_info['created_time'] < cutoff_time:
                try:
                    os.remove(file_info['filepath'])
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to remove old incremental file {file_info['filename']}: {e}")
        
        if removed_count > 0:
            print(f"🗑️ Cleaned up {removed_count} old incremental files (>{keep_days} days)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get incremental save engine statistics."""
        stats = self.stats.copy()
        stats['queue_size'] = self.save_queue.qsize()
        stats['is_running'] = self.is_running
        stats['cycles_since_last_save'] = self.cycles_since_last_save
        
        # Add file system stats
        incremental_files = self.get_incremental_files()
        stats['total_incremental_files'] = len(incremental_files)
        stats['total_disk_usage_mb'] = sum(f['size_mb'] for f in incremental_files)
        
        return stats
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_background_thread(timeout=1.0)