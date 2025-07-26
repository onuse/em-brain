"""
Consolidation Engine

Merges incremental save files into consolidated snapshots when accumulation triggers are met.
This prevents the incremental directory from growing unbounded while maintaining all learned content.
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from .brain_serializer import BrainSerializer, SerializedBrainState
from .storage_backend import StorageBackend
from .persistence_config import PersistenceConfig


class ConsolidationEngine:
    """Engine for merging incremental saves into consolidated snapshots."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.brain_serializer = BrainSerializer()
        self.storage = StorageBackend(config)
        
        # Consolidation tracking
        self.last_consolidation_time = time.time()
        self.consolidation_counter = 0
        
        # Background consolidation
        self.consolidation_thread: Optional[threading.Thread] = None
        self.consolidation_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.is_running = False
        
        # Performance stats
        self.stats = {
            'total_consolidations': 0,
            'total_files_consolidated': 0,
            'total_patterns_merged': 0,
            'avg_consolidation_time_ms': 0.0,
            'largest_consolidation_mb': 0.0,
            'space_saved_mb': 0.0,
            'failed_consolidations': 0,
            'pattern_conflicts_resolved': 0
        }
        
        # Start background thread if enabled
        if config.consolidation_thread:
            self.start_background_thread()
    
    def start_background_thread(self):
        """Start background consolidation monitoring thread."""
        if self.consolidation_thread and self.consolidation_thread.is_alive():
            return
        
        self.is_running = True
        self.consolidation_thread = threading.Thread(
            target=self._background_consolidation_worker,
            name="ConsolidationEngine",
            daemon=True
        )
        self.consolidation_thread.start()
        print("🔄 Consolidation engine started")
    
    def stop_background_thread(self, timeout: float = 10.0):
        """Stop background consolidation thread gracefully."""
        if not self.is_running:
            return
        
        print("🛑 Stopping consolidation engine...")
        self.shutdown_event.set()
        self.is_running = False
        
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=timeout)
            if self.consolidation_thread.is_alive():
                print("⚠️ Consolidation thread did not shutdown cleanly")
    
    def check_consolidation_needed(self) -> bool:
        """Check if consolidation should be triggered based on current conditions."""
        # Get current metrics
        incremental_files = self.storage.list_files(
            self.config.get_incremental_dir(), 
            "delta_*.json*"
        )
        
        incremental_count = len(incremental_files)
        incremental_size_mb = sum(f['size_mb'] for f in incremental_files)
        hours_since_last = (time.time() - self.last_consolidation_time) / 3600.0
        
        # Check against policy
        return self.config.should_consolidate(
            incremental_count, 
            incremental_size_mb, 
            hours_since_last
        )
    
    def trigger_consolidation(self, force: bool = False) -> bool:
        """Trigger consolidation (can be called from any thread)."""
        if force or self.check_consolidation_needed():
            if self.is_running:
                # Signal background thread
                self.consolidation_event.set()
                return True
            else:
                # Execute immediately
                return self.consolidate_incremental_files()
        return False
    
    def consolidate_incremental_files(self) -> bool:
        """Perform consolidation of incremental files into a new snapshot."""
        start_time = time.perf_counter()
        
        try:
            print("🔄 Starting consolidation of incremental files...")
            
            # Get all incremental files
            incremental_files = self._get_incremental_files_sorted()
            
            if not incremental_files:
                print("📝 No incremental files to consolidate")
                return True
            
            # Get latest consolidated snapshot as base
            base_state = self._load_latest_consolidated_state()
            
            # Merge all incremental changes
            consolidated_state = self._merge_incremental_states(base_state, incremental_files)
            
            if not consolidated_state:
                print("❌ Failed to merge incremental states")
                self.stats['failed_consolidations'] += 1
                return False
            
            # Save new consolidated snapshot
            snapshot_success = self._save_consolidated_snapshot(consolidated_state)
            
            if not snapshot_success:
                print("❌ Failed to save consolidated snapshot")
                self.stats['failed_consolidations'] += 1
                return False
            
            # Calculate space savings
            original_size_mb = sum(f['size_mb'] for f in incremental_files)
            
            # Archive old incremental files (don't delete immediately)
            archived_count = self._archive_incremental_files(incremental_files)
            
            # Update statistics
            consolidation_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_consolidation_stats(
                len(incremental_files), 
                1,  # Simplified - just one state
                consolidation_time_ms,
                original_size_mb
            )
            
            self.last_consolidation_time = time.time()
            
            print(f"✅ Consolidation complete:")
            print(f"   Merged {len(incremental_files)} incremental files")
            print(f"   Field dimensions: {consolidated_state.field_dimensions}D")
            print(f"   Archived {archived_count} files")
            print(f"   Time: {consolidation_time_ms:.1f}ms")
            
            return True
            
        except Exception as e:
            print(f"❌ Consolidation failed: {e}")
            self.stats['failed_consolidations'] += 1
            return False
    
    def _background_consolidation_worker(self):
        """Background worker for monitoring consolidation triggers."""
        print("🔄 Background consolidation worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for consolidation trigger or timeout
                triggered = self.consolidation_event.wait(timeout=3600)  # Check hourly
                
                if self.shutdown_event.is_set():
                    break
                
                if triggered or self.check_consolidation_needed():
                    self.consolidate_incremental_files()
                    self.consolidation_event.clear()
                
            except Exception as e:
                print(f"⚠️ Background consolidation worker error: {e}")
    
    def _get_incremental_files_sorted(self) -> List[Dict[str, Any]]:
        """Get all incremental files sorted by creation time."""
        incremental_files = self.storage.list_files(
            self.config.get_incremental_dir(),
            "delta_*.json*"
        )
        
        # Filter out .integrity files - they are handled automatically by storage backend
        incremental_files = [f for f in incremental_files if not f['filename'].endswith('.integrity')]
        
        # Sort by creation time (oldest first)
        incremental_files.sort(key=lambda x: x['created_time'])
        return incremental_files
    
    def _load_latest_consolidated_state(self) -> Optional[SerializedBrainState]:
        """Load the most recent consolidated snapshot as base state."""
        consolidated_files = self.storage.list_files(
            self.config.get_consolidated_dir(),
            "brain_state_*.json*"
        )
        
        if not consolidated_files:
            print("📝 No existing consolidated snapshots found")
            return None
        
        # Get most recent consolidated file
        latest_file = max(consolidated_files, key=lambda x: x['created_time'])
        
        print(f"📖 Loading base state from {latest_file['filename']}")
        
        # Load the consolidated state
        brain_dict = self.storage.load_json(latest_file['filepath'])
        if brain_dict:
            # Remove any incremental metadata
            brain_dict.pop('incremental_metadata', None)
            # Remove any consolidation metadata
            brain_dict.pop('consolidation_metadata', None)
            return self.brain_serializer.from_dict(brain_dict)
        
        return None
    
    def _merge_incremental_states(self, base_state: Optional[SerializedBrainState], 
                                 incremental_files: List[Dict[str, Any]]) -> Optional[SerializedBrainState]:
        """Merge incremental states with base state to create consolidated state."""
        
        # Start with base state or create new one
        if base_state:
            merged_state = base_state
            print(f"📖 Base state: {base_state.field_dimensions}D field, {base_state.brain_cycles} cycles")
        else:
            # Create initial state from first incremental file
            if not incremental_files:
                return None
            
            first_file = incremental_files[0]
            brain_dict = self.storage.load_json(first_file['filepath'])
            if not brain_dict:
                return None
            
            brain_dict.pop('incremental_metadata', None)
            brain_dict.pop('consolidation_metadata', None)
            
            try:
                merged_state = self.brain_serializer.from_dict(brain_dict)
                incremental_files = incremental_files[1:]  # Skip first file
                print(f"📖 Initial state: {merged_state.field_dimensions}D field")
            except Exception as e:
                print(f"❌ Failed to parse initial file {first_file['filename']}: {e}")
                print(f"   File keys: {list(brain_dict.keys()) if brain_dict else 'None'}")
                return None
        
        # Merge each incremental file
        for file_info in incremental_files:
            print(f"🔀 Merging {file_info['filename']}...")
            
            # Load incremental state
            brain_dict = self.storage.load_json(file_info['filepath'])
            if not brain_dict:
                print(f"⚠️ Failed to load {file_info['filename']}")
                continue
            
            brain_dict.pop('incremental_metadata', None)
            brain_dict.pop('consolidation_metadata', None)
            
            try:
                incremental_state = self.brain_serializer.from_dict(brain_dict)
                
                # Merge this incremental state (simplified - just use latest state)
                merged_state = incremental_state
            except Exception as e:
                print(f"❌ Failed to parse incremental file {file_info['filename']}: {e}")
                print(f"   File keys: {list(brain_dict.keys()) if brain_dict else 'None'}")
                continue  # Skip this file and continue with others
        
        # Update consolidation metadata
        merged_state.timestamp = time.time()
        merged_state.version = "1.0"
        
        print(f"🔀 Final merged state: {merged_state.field_dimensions}D field, {merged_state.brain_cycles} cycles")
        
        return merged_state
    
    def _merge_two_states(self, base_state: SerializedBrainState, 
                         incremental_state: SerializedBrainState) -> SerializedBrainState:
        """Merge two brain states - simplified for UnifiedFieldBrain."""
        # With the simplified brain, just use the more recent state
        return incremental_state
    
    # Pattern merging removed - not needed with simplified UnifiedFieldBrain
    
    def _save_consolidated_snapshot(self, consolidated_state: SerializedBrainState) -> bool:
        """Save consolidated state as new snapshot."""
        try:
            # Generate snapshot filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"brain_state_{timestamp}.json"
            
            if self.config.enable_compression:
                snapshot_filename += ".gz"
            
            snapshot_filepath = os.path.join(
                self.config.get_consolidated_dir(),
                snapshot_filename
            )
            
            # Convert to dictionary
            brain_dict = self.brain_serializer.serialize_to_dict(consolidated_state)
            
            # Add consolidation metadata
            consolidation_metadata = {
                'consolidation_id': f"consolidation_{self.consolidation_counter:06d}",
                'consolidation_timestamp': time.time(),
                'consolidation_type': 'incremental_merge',
                'field_dimensions': consolidated_state.field_dimensions,
                'source_files_count': 'unknown'  # Would need to pass this through
            }
            brain_dict['consolidation_metadata'] = consolidation_metadata
            
            # Save the snapshot
            snapshot_size_mb = self.storage.save_json(
                brain_dict, 
                snapshot_filepath, 
                compress=self.config.enable_compression
            )
            
            print(f"💾 Saved consolidated snapshot: {snapshot_filename} ({snapshot_size_mb:.1f}MB)")
            
            # Update latest symlink
            self._update_latest_consolidated_link(snapshot_filepath)
            
            # Clean up old snapshots if needed
            self._cleanup_old_snapshots()
            
            self.consolidation_counter += 1
            return True
            
        except Exception as e:
            print(f"❌ Failed to save consolidated snapshot: {e}")
            return False
    
    def _update_latest_consolidated_link(self, snapshot_filepath: str):
        """Update symlink to point to latest consolidated snapshot."""
        try:
            # Match symlink extension to target file extension
            if snapshot_filepath.endswith('.gz'):
                symlink_name = "latest_consolidated.json.gz"
            else:
                symlink_name = "latest_consolidated.json"
                
            latest_link = os.path.join(
                self.config.get_consolidated_dir(),
                symlink_name
            )
            
            # Remove existing symlinks (both .json and .json.gz versions)
            old_json_link = os.path.join(self.config.get_consolidated_dir(), "latest_consolidated.json")
            old_gz_link = os.path.join(self.config.get_consolidated_dir(), "latest_consolidated.json.gz")
            
            for old_link in [old_json_link, old_gz_link]:
                if os.path.islink(old_link):
                    os.unlink(old_link)
                elif os.path.exists(old_link):
                    os.remove(old_link)
            
            # Create new symlink
            os.symlink(os.path.basename(snapshot_filepath), latest_link)
            
        except Exception as e:
            print(f"⚠️ Failed to update latest consolidated link: {e}")
    
    def _archive_incremental_files(self, incremental_files: List[Dict[str, Any]]) -> int:
        """Archive incremental files instead of deleting them immediately."""
        archive_dir = os.path.join(self.config.get_recovery_dir(), "archived_incrementals")
        
        try:
            os.makedirs(archive_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Failed to create archive directory {archive_dir}: {e}")
            return 0
        
        archived_count = 0
        failed_archives = []
        
        for file_info in incremental_files:
            try:
                src_path = file_info['filepath']
                filename = file_info['filename']
                
                # Check if source file still exists
                if not os.path.exists(src_path):
                    print(f"⚠️ Source file missing during archive: {src_path}")
                    failed_archives.append(filename)
                    continue
                
                # Generate unique destination filename
                timestamp = int(time.time())
                dst_filename = f"archived_{timestamp}_{filename}"
                dst_path = os.path.join(archive_dir, dst_filename)
                
                # Ensure unique destination path
                counter = 1
                while os.path.exists(dst_path):
                    dst_filename = f"archived_{timestamp}_{counter}_{filename}"
                    dst_path = os.path.join(archive_dir, dst_filename)
                    counter += 1
                
                if self.storage.move_file(src_path, dst_path):
                    archived_count += 1
                    print(f"📦 Archived: {filename} -> {dst_filename}")
                else:
                    failed_archives.append(filename)
                
            except Exception as e:
                print(f"⚠️ Failed to archive {file_info.get('filename', 'unknown')}: {e}")
                failed_archives.append(file_info.get('filename', 'unknown'))
        
        if failed_archives:
            print(f"⚠️ Failed to archive {len(failed_archives)} files: {failed_archives[:3]}{'...' if len(failed_archives) > 3 else ''}")
        
        return archived_count
    
    def _cleanup_old_snapshots(self):
        """Clean up old consolidated snapshots, keeping only recent ones."""
        try:
            consolidated_files = self.storage.list_files(
                self.config.get_consolidated_dir(),
                "brain_state_*.json*"
            )
            
            if len(consolidated_files) <= self.config.keep_backup_snapshots:
                return
            
            # Sort by creation time (newest first)
            consolidated_files.sort(key=lambda x: x['created_time'], reverse=True)
            
            # Remove old snapshots
            files_to_remove = consolidated_files[self.config.keep_backup_snapshots:]
            removed_count = 0
            
            for file_info in files_to_remove:
                if self.storage.delete_file(file_info['filepath']):
                    removed_count += 1
            
            if removed_count > 0:
                print(f"🗑️ Cleaned up {removed_count} old consolidated snapshots")
                
        except Exception as e:
            print(f"⚠️ Error during snapshot cleanup: {e}")
    
    def _update_consolidation_stats(self, files_count: int, patterns_count: int, 
                                   time_ms: float, original_size_mb: float):
        """Update consolidation performance statistics."""
        stats = self.stats
        stats['total_consolidations'] += 1
        stats['total_files_consolidated'] += files_count
        stats['total_patterns_merged'] += patterns_count
        
        # Update average time
        total_consolidations = stats['total_consolidations']
        stats['avg_consolidation_time_ms'] = (
            (stats['avg_consolidation_time_ms'] * (total_consolidations - 1) + time_ms) / 
            total_consolidations
        )
        
        # Track space savings (approximate)
        stats['space_saved_mb'] += original_size_mb * 0.3  # Assume 30% space savings
        
        # Track largest consolidation
        if original_size_mb > stats['largest_consolidation_mb']:
            stats['largest_consolidation_mb'] = original_size_mb
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status and recommendations."""
        incremental_files = self.storage.list_files(
            self.config.get_incremental_dir(),
            "delta_*.json*"
        )
        
        incremental_count = len(incremental_files)
        incremental_size_mb = sum(f['size_mb'] for f in incremental_files)
        hours_since_last = (time.time() - self.last_consolidation_time) / 3600.0
        
        consolidation_needed = self.config.should_consolidate(
            incremental_count,
            incremental_size_mb,
            hours_since_last
        )
        
        return {
            'consolidation_needed': consolidation_needed,
            'incremental_files_count': incremental_count,
            'incremental_files_size_mb': incremental_size_mb,
            'hours_since_last_consolidation': hours_since_last,
            'last_consolidation_time': self.last_consolidation_time,
            'is_running': self.is_running,
            'policy_limits': {
                'max_files': self.config.consolidation_policy.max_incremental_files,
                'max_size_mb': self.config.consolidation_policy.max_incremental_size_mb,
                'max_hours': self.config.consolidation_policy.max_hours_since_consolidation
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation engine statistics."""
        return {
            **self.stats,
            **self.get_consolidation_status()
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_background_thread(timeout=2.0)