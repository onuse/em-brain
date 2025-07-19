"""
Storage Backend Abstraction

Provides file I/O abstraction for the persistence subsystem with support for:
- JSON serialization with compression
- Atomic writes (write to temp, then rename)
- Corruption detection and recovery
- Performance monitoring
"""

import os
import json
import gzip
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .persistence_config import PersistenceConfig


@dataclass
class FileIntegrityInfo:
    """Information about file integrity and checksums."""
    filepath: str
    size_bytes: int
    checksum_md5: str
    created_time: float
    modified_time: float
    is_compressed: bool


class StorageBackend:
    """File storage backend with atomic writes and integrity checking."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.stats = {
            'total_writes': 0,
            'total_reads': 0,
            'total_bytes_written': 0,
            'total_bytes_read': 0,
            'avg_write_time_ms': 0.0,
            'avg_read_time_ms': 0.0,
            'checksum_failures': 0,
            'corruption_recoveries': 0,
            'atomic_write_failures': 0
        }
    
    def ensure_directories_exist(self):
        """Create all necessary directories."""
        directories = [
            self.config.get_consolidated_dir(),
            self.config.get_incremental_dir(),
            self.config.get_metadata_dir(),
            self.config.get_recovery_dir(),
            os.path.join(self.config.get_recovery_dir(), "corruption_backups")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_json(self, data: Dict[str, Any], filepath: str, compress: bool = False) -> float:
        """
        Save JSON data to file with atomic write and optional compression.
        
        Returns the size of the saved file in MB.
        """
        start_time = time.perf_counter()
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Atomic write: write to temp file first
            temp_filepath = filepath + ".tmp"
            
            if compress:
                with gzip.open(temp_filepath, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                with open(temp_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            os.rename(temp_filepath, filepath)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)
            
            # Update statistics
            write_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_write_stats(file_size, write_time_ms)
            
            # Save integrity info if enabled
            if self.config.enable_corruption_detection:
                self._save_integrity_info(filepath, file_size, compress)
            
            return file_size_mb
            
        except Exception as e:
            # Clean up temp file if it exists
            temp_filepath = filepath + ".tmp"
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass
            
            self.stats['atomic_write_failures'] += 1
            raise Exception(f"Failed to save JSON to {filepath}: {e}")
    
    def load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON data from file with corruption detection.
        
        Returns None if file doesn't exist or is corrupted.
        """
        if not os.path.exists(filepath):
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Check integrity if enabled
            if self.config.enable_corruption_detection:
                if not self._verify_file_integrity(filepath):
                    print(f"⚠️ File integrity check failed for {filepath}")
                    self.stats['checksum_failures'] += 1
                    
                    # Try to recover from backup
                    recovered_data = self._attempt_corruption_recovery(filepath)
                    if recovered_data:
                        return recovered_data
                    else:
                        return None
            
            # Determine if file is compressed
            is_compressed = filepath.endswith('.gz')
            
            # Load the file
            if is_compressed:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Update statistics
            file_size = os.path.getsize(filepath)
            read_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_read_stats(file_size, read_time_ms)
            
            return data
            
        except Exception as e:
            print(f"⚠️ Failed to load JSON from {filepath}: {e}")
            return None
    
    def copy_file(self, src_filepath: str, dst_filepath: str) -> bool:
        """Copy file with integrity preservation."""
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_filepath, dst_filepath)
            
            # Copy integrity info if it exists
            if self.config.enable_corruption_detection:
                src_integrity_path = self._get_integrity_filepath(src_filepath)
                if os.path.exists(src_integrity_path):
                    dst_integrity_path = self._get_integrity_filepath(dst_filepath)
                    shutil.copy2(src_integrity_path, dst_integrity_path)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to copy file {src_filepath} to {dst_filepath}: {e}")
            return False
    
    def move_file(self, src_filepath: str, dst_filepath: str) -> bool:
        """Move file with integrity preservation."""
        try:
            # Check if source file exists
            if not os.path.exists(src_filepath):
                print(f"⚠️ Source file does not exist: {src_filepath}")
                return False
            
            # Ensure destination directory exists
            dst_dir = os.path.dirname(dst_filepath)
            os.makedirs(dst_dir, exist_ok=True)
            
            # Check if destination already exists and remove it
            if os.path.exists(dst_filepath):
                print(f"⚠️ Destination file already exists, removing: {dst_filepath}")
                try:
                    os.remove(dst_filepath)
                except Exception as remove_e:
                    print(f"⚠️ Failed to remove existing destination file: {remove_e}")
                    return False
            
            # Move main file
            shutil.move(src_filepath, dst_filepath)
            
            # Move integrity info if it exists
            if self.config.enable_corruption_detection:
                src_integrity_path = self._get_integrity_filepath(src_filepath)
                if os.path.exists(src_integrity_path):
                    dst_integrity_path = self._get_integrity_filepath(dst_filepath)
                    try:
                        # Remove destination integrity file if it exists
                        if os.path.exists(dst_integrity_path):
                            os.remove(dst_integrity_path)
                        shutil.move(src_integrity_path, dst_integrity_path)
                    except Exception as integrity_e:
                        print(f"⚠️ Failed to move integrity file (main move succeeded): {integrity_e}")
                        # Don't fail the whole operation if only integrity move fails
            
            # Verify the move succeeded
            if not os.path.exists(dst_filepath):
                print(f"⚠️ Move operation reported success but destination file missing: {dst_filepath}")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to move file {src_filepath} to {dst_filepath}: {e}")
            # If move partially succeeded, try to clean up
            try:
                if os.path.exists(dst_filepath) and os.path.exists(src_filepath):
                    # Both files exist - remove destination to avoid duplicates
                    os.remove(dst_filepath)
            except:
                pass  # Ignore cleanup failures
            return False
    
    def delete_file(self, filepath: str) -> bool:
        """Delete file and its integrity info."""
        try:
            # Delete main file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Delete integrity info if it exists
            if self.config.enable_corruption_detection:
                integrity_path = self._get_integrity_filepath(filepath)
                if os.path.exists(integrity_path):
                    os.remove(integrity_path)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to delete file {filepath}: {e}")
            return False
    
    def list_files(self, directory: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in directory with metadata."""
        if not os.path.exists(directory):
            return []
        
        files = []
        for filepath in Path(directory).glob(pattern):
            if filepath.is_file():
                try:
                    stat = filepath.stat()
                    file_info = {
                        'filename': filepath.name,
                        'filepath': str(filepath),
                        'size_bytes': stat.st_size,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'created_time': stat.st_ctime,
                        'modified_time': stat.st_mtime,
                        'is_compressed': filepath.suffix == '.gz'
                    }
                    
                    # Add integrity status if enabled
                    if self.config.enable_corruption_detection:
                        file_info['integrity_ok'] = self._verify_file_integrity(str(filepath))
                    
                    files.append(file_info)
                    
                except Exception as e:
                    print(f"⚠️ Error reading file info for {filepath}: {e}")
        
        return files
    
    def get_directory_size(self, directory: str) -> float:
        """Get total size of directory in MB."""
        if not os.path.exists(directory):
            return 0.0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        
        return total_size / (1024 * 1024)
    
    def _save_integrity_info(self, filepath: str, file_size: int, is_compressed: bool):
        """Save file integrity information."""
        try:
            # Calculate MD5 checksum
            checksum = self._calculate_file_checksum(filepath)
            
            # Create integrity info
            stat = os.stat(filepath)
            integrity_info = FileIntegrityInfo(
                filepath=filepath,
                size_bytes=file_size,
                checksum_md5=checksum,
                created_time=stat.st_ctime,
                modified_time=stat.st_mtime,
                is_compressed=is_compressed
            )
            
            # Save integrity file
            integrity_filepath = self._get_integrity_filepath(filepath)
            with open(integrity_filepath, 'w') as f:
                json.dump(integrity_info.__dict__, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save integrity info for {filepath}: {e}")
    
    def _verify_file_integrity(self, filepath: str) -> bool:
        """Verify file integrity using saved checksum."""
        try:
            integrity_filepath = self._get_integrity_filepath(filepath)
            
            if not os.path.exists(integrity_filepath):
                # No integrity info available - assume ok
                return True
            
            # Load integrity info
            with open(integrity_filepath, 'r') as f:
                integrity_data = json.load(f)
            
            integrity_info = FileIntegrityInfo(**integrity_data)
            
            # Check file size
            current_size = os.path.getsize(filepath)
            if current_size != integrity_info.size_bytes:
                return False
            
            # Check checksum
            current_checksum = self._calculate_file_checksum(filepath)
            if current_checksum != integrity_info.checksum_md5:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error verifying integrity for {filepath}: {e}")
            return False
    
    def _calculate_file_checksum(self, filepath: str) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        
        # Handle compressed files
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_integrity_filepath(self, filepath: str) -> str:
        """Get filepath for integrity information."""
        return filepath + ".integrity"
    
    def _attempt_corruption_recovery(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover from file corruption using backups."""
        try:
            # Look for backup files
            backup_dir = os.path.join(self.config.get_recovery_dir(), "corruption_backups")
            filename = os.path.basename(filepath)
            
            # Find backup files for this file
            backup_pattern = f"{filename}.backup.*"
            backup_files = list(Path(backup_dir).glob(backup_pattern))
            
            if not backup_files:
                return None
            
            # Try most recent backup first
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for backup_file in backup_files:
                try:
                    # Try to load backup
                    backup_data = self.load_json(str(backup_file))
                    if backup_data:
                        print(f"🔄 Recovered from backup: {backup_file}")
                        self.stats['corruption_recoveries'] += 1
                        
                        # Create new backup of corrupted file for analysis
                        corrupted_backup = f"{filepath}.corrupted.{int(time.time())}"
                        shutil.copy2(filepath, corrupted_backup)
                        
                        return backup_data
                        
                except Exception as e:
                    print(f"⚠️ Backup {backup_file} also corrupted: {e}")
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Corruption recovery failed for {filepath}: {e}")
            return None
    
    def create_backup(self, filepath: str) -> bool:
        """Create a backup copy of a file."""
        try:
            if not os.path.exists(filepath):
                return False
            
            backup_dir = os.path.join(self.config.get_recovery_dir(), "corruption_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            filename = os.path.basename(filepath)
            backup_filename = f"{filename}.backup.{int(time.time())}"
            backup_filepath = os.path.join(backup_dir, backup_filename)
            
            return self.copy_file(filepath, backup_filepath)
            
        except Exception as e:
            print(f"⚠️ Failed to create backup for {filepath}: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count: int = 5):
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            backup_dir = os.path.join(self.config.get_recovery_dir(), "corruption_backups")
            if not os.path.exists(backup_dir):
                return
            
            # Group backups by original filename
            backups_by_file = {}
            for backup_file in Path(backup_dir).glob("*.backup.*"):
                # Extract original filename (everything before .backup.)
                parts = backup_file.name.split('.backup.')
                if len(parts) >= 2:
                    original_name = parts[0]
                    if original_name not in backups_by_file:
                        backups_by_file[original_name] = []
                    backups_by_file[original_name].append(backup_file)
            
            # Clean up old backups for each file
            total_removed = 0
            for original_name, backup_files in backups_by_file.items():
                if len(backup_files) > keep_count:
                    # Sort by modification time (newest first)
                    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Remove old backups
                    for old_backup in backup_files[keep_count:]:
                        try:
                            old_backup.unlink()
                            total_removed += 1
                        except Exception as e:
                            print(f"⚠️ Failed to remove old backup {old_backup}: {e}")
            
            if total_removed > 0:
                print(f"🗑️ Cleaned up {total_removed} old backup files")
                
        except Exception as e:
            print(f"⚠️ Error during backup cleanup: {e}")
    
    def _update_write_stats(self, bytes_written: int, time_ms: float):
        """Update write performance statistics."""
        self.stats['total_writes'] += 1
        self.stats['total_bytes_written'] += bytes_written
        
        # Update average time
        total_writes = self.stats['total_writes']
        self.stats['avg_write_time_ms'] = (
            (self.stats['avg_write_time_ms'] * (total_writes - 1) + time_ms) / total_writes
        )
    
    def _update_read_stats(self, bytes_read: int, time_ms: float):
        """Update read performance statistics."""
        self.stats['total_reads'] += 1
        self.stats['total_bytes_read'] += bytes_read
        
        # Update average time
        total_reads = self.stats['total_reads']
        self.stats['avg_read_time_ms'] = (
            (self.stats['avg_read_time_ms'] * (total_reads - 1) + time_ms) / total_reads
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage backend statistics."""
        stats = self.stats.copy()
        
        # Add directory size information
        stats['consolidated_size_mb'] = self.get_directory_size(self.config.get_consolidated_dir())
        stats['incremental_size_mb'] = self.get_directory_size(self.config.get_incremental_dir())
        stats['metadata_size_mb'] = self.get_directory_size(self.config.get_metadata_dir())
        stats['recovery_size_mb'] = self.get_directory_size(self.config.get_recovery_dir())
        
        # Calculate throughput
        if self.stats['total_writes'] > 0:
            stats['avg_write_throughput_mbps'] = (
                (self.stats['total_bytes_written'] / (1024 * 1024)) / 
                (self.stats['avg_write_time_ms'] / 1000) if self.stats['avg_write_time_ms'] > 0 else 0
            )
        
        if self.stats['total_reads'] > 0:
            stats['avg_read_throughput_mbps'] = (
                (self.stats['total_bytes_read'] / (1024 * 1024)) / 
                (self.stats['avg_read_time_ms'] / 1000) if self.stats['avg_read_time_ms'] > 0 else 0
            )
        
        return stats