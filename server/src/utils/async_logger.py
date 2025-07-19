#!/usr/bin/env python3
"""
Async Logging Framework - Phase 6 Implementation

High-performance logging system that ensures brain cognition never waits for I/O.
Core principle: <1ms overhead per brain cycle regardless of data volume.

Architecture:
- LoggableObject: Interface for objects that can serialize themselves
- AsyncLogger: Background thread managing queue and disk writes  
- BrainThread: Only creates objects and queues them (<1ms)
- LoggerThread: Handles all heavy serialization and I/O work
"""

import json
import time
import threading
import queue
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import atexit


class LogLevel(Enum):
    """Log levels for production observability."""
    VITAL = 1      # Always-on production logging (<0.1ms overhead)
    DEBUG = 2      # Conditional detailed logging (<0.5ms overhead)  
    TRACE = 3      # Full investigation logging (<1ms overhead)


class LoggableObject(ABC):
    """
    Base class for objects that can be logged asynchronously.
    
    Key design: Heavy serialization work happens on background thread,
    brain thread only creates object and queues it.
    """
    
    def __init__(self, timestamp: float = None, log_level: LogLevel = LogLevel.DEBUG):
        self.timestamp = timestamp or time.time()
        self.log_level = log_level
    
    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize object to dictionary for JSON logging.
        
        This method runs on the background logger thread,
        NOT on the brain thread, so it can be expensive.
        """
        pass
    
    def get_log_category(self) -> str:
        """Return log category for file organization."""
        return "general"


@dataclass
class LoggerConfig:
    """Configuration for async logger."""
    log_directory: Path
    max_queue_size: int = 10000
    batch_size: int = 50
    flush_interval_seconds: float = 1.0
    max_log_files: int = 100
    rotate_size_mb: int = 100
    compression_enabled: bool = True
    
    # Backpressure handling
    drop_policy: str = "drop_oldest"  # "drop_oldest", "drop_newest", "block"
    
    # Performance monitoring
    enable_performance_tracking: bool = True


class AsyncLogger:
    """
    High-performance async logger with <1ms brain thread overhead.
    
    Design principles:
    1. Brain thread never waits for I/O
    2. Batched writes for efficiency
    3. Graceful degradation under pressure
    4. Production-ready reliability
    """
    
    def __init__(self, config: LoggerConfig, quiet_mode: bool = False):
        self.config = config
        self.quiet_mode = quiet_mode
        
        # Ensure log directory exists
        self.config.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Background processing
        self.log_queue = queue.Queue(maxsize=config.max_queue_size)
        self.shutdown_event = threading.Event()
        self.logger_thread = None
        
        # File management
        self.log_files = {}  # category -> file handle
        self.log_counters = {}  # category -> write count
        
        # Performance tracking
        self.total_logs_queued = 0
        self.total_logs_written = 0
        self.total_logs_dropped = 0
        self.queue_full_events = 0
        self.last_performance_report = time.time()
        
        # Thread safety
        self.stats_lock = threading.Lock()
        
        # Start background thread
        self._start_logger_thread()
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        if not quiet_mode:
            print(f"📊 AsyncLogger initialized")
            print(f"   Queue size: {config.max_queue_size}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Log directory: {config.log_directory}")
            print(f"   Drop policy: {config.drop_policy}")
    
    def please_log(self, obj: LoggableObject) -> bool:
        """
        Queue object for async logging.
        
        This is the main interface for brain threads.
        Designed to complete in <1ms regardless of object complexity.
        
        Returns:
            True if successfully queued, False if dropped due to backpressure
        """
        try:
            # Non-blocking queue put with backpressure handling
            if self.log_queue.full():
                return self._handle_queue_full(obj)
            
            self.log_queue.put_nowait(obj)
            
            with self.stats_lock:
                self.total_logs_queued += 1
            
            return True
            
        except queue.Full:
            return self._handle_queue_full(obj)
        except Exception as e:
            if not self.quiet_mode:
                print(f"⚠️ AsyncLogger.please_log error: {e}")
            return False
    
    def _handle_queue_full(self, obj: LoggableObject) -> bool:
        """Handle queue full condition based on drop policy."""
        with self.stats_lock:
            self.queue_full_events += 1
        
        if self.config.drop_policy == "drop_oldest":
            try:
                # Drop oldest item to make room
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(obj)
                with self.stats_lock:
                    self.total_logs_dropped += 1
                    self.total_logs_queued += 1
                return True
            except queue.Empty:
                # Queue became empty between full check and get
                return self.please_log(obj)
                
        elif self.config.drop_policy == "drop_newest":
            # Drop the new object
            with self.stats_lock:
                self.total_logs_dropped += 1
            return False
            
        elif self.config.drop_policy == "block":
            # Block brain thread (NOT recommended for production)
            try:
                self.log_queue.put(obj, timeout=0.001)  # 1ms max block
                with self.stats_lock:
                    self.total_logs_queued += 1
                return True
            except queue.Full:
                with self.stats_lock:
                    self.total_logs_dropped += 1
                return False
        
        return False
    
    def _start_logger_thread(self):
        """Start background logger thread."""
        self.logger_thread = threading.Thread(
            target=self._logger_worker,
            name="AsyncLogger",
            daemon=True
        )
        self.logger_thread.start()
    
    def _logger_worker(self):
        """
        Background thread worker for processing log queue.
        
        Handles all expensive operations:
        - Object serialization
        - JSON encoding  
        - Disk I/O
        - File rotation
        - Batching for efficiency
        """
        batch = []
        last_flush = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Collect batch of log objects
                while len(batch) < self.config.batch_size:
                    try:
                        # Get log object with timeout
                        obj = self.log_queue.get(timeout=0.1)
                        batch.append(obj)
                    except queue.Empty:
                        break
                
                # Process batch if we have items or it's time to flush
                current_time = time.time()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_flush >= self.config.flush_interval_seconds)
                )
                
                if should_flush and batch:
                    batch_size = len(batch)  # Store size before clearing
                    self._process_batch(batch)
                    batch = []
                    last_flush = current_time
                    
                    with self.stats_lock:
                        self.total_logs_written += batch_size
                
                # Performance reporting
                if self.config.enable_performance_tracking:
                    self._maybe_report_performance()
                    
            except Exception as e:
                if not self.quiet_mode:
                    print(f"⚠️ AsyncLogger worker error: {e}")
                # Continue processing to maintain reliability
        
        # Flush remaining logs on shutdown
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[LoggableObject]):
        """Process a batch of log objects."""
        # Group by category for efficient file handling
        categorized = {}
        for obj in batch:
            category = obj.get_log_category()
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(obj)
        
        # Write each category
        for category, objects in categorized.items():
            self._write_category_batch(category, objects)
    
    def _write_category_batch(self, category: str, objects: List[LoggableObject]):
        """Write batch of objects to category log file."""
        try:
            # Get or create file handle
            file_handle = self._get_log_file(category)
            
            # Serialize and write each object
            for obj in objects:
                try:
                    # Heavy serialization happens here (background thread)
                    serialized = obj.serialize()
                    
                    # Add metadata
                    log_entry = {
                        "timestamp": obj.timestamp,
                        "log_level": obj.log_level.name,
                        "category": category,
                        "data": serialized
                    }
                    
                    # Write JSON line
                    json_line = json.dumps(log_entry, separators=(',', ':'))
                    file_handle.write(json_line + '\n')
                    
                except Exception as e:
                    if not self.quiet_mode:
                        print(f"⚠️ Failed to serialize {type(obj).__name__}: {e}")
            
            # Flush for reliability
            file_handle.flush()
            
            # Check for rotation
            self._maybe_rotate_file(category, file_handle)
            
        except Exception as e:
            if not self.quiet_mode:
                print(f"⚠️ Failed to write {category} batch: {e}")
    
    def _get_log_file(self, category: str):
        """Get or create log file handle for category."""
        if category not in self.log_files:
            log_file_path = self.config.log_directory / f"{category}.jsonl"
            self.log_files[category] = open(log_file_path, 'a', encoding='utf-8')
            self.log_counters[category] = 0
        
        return self.log_files[category]
    
    def _maybe_rotate_file(self, category: str, file_handle):
        """Check if file needs rotation and rotate if necessary."""
        self.log_counters[category] += 1
        
        # Simple rotation based on write count (could be enhanced with size checking)
        if self.log_counters[category] % 10000 == 0:
            # Close current file
            file_handle.close()
            
            # Create new file with timestamp
            timestamp = int(time.time())
            new_path = self.config.log_directory / f"{category}_{timestamp}.jsonl"
            old_path = self.config.log_directory / f"{category}.jsonl"
            
            # Rename current to timestamped
            if old_path.exists():
                old_path.rename(new_path)
            
            # Open new current file
            self.log_files[category] = open(old_path, 'a', encoding='utf-8')
            self.log_counters[category] = 0
    
    def _maybe_report_performance(self):
        """Report performance statistics periodically."""
        current_time = time.time()
        if current_time - self.last_performance_report > 60.0:  # Every minute
            with self.stats_lock:
                if not self.quiet_mode:
                    print(f"📊 AsyncLogger Performance:")
                    print(f"   In queue: {self.log_queue.qsize()}")
                    print(f"   Lifetime queued: {self.total_logs_queued}")
                    print(f"   Written: {self.total_logs_written}")
                    print(f"   Dropped: {self.total_logs_dropped}")
                    print(f"   Queue full events: {self.queue_full_events}")
                
                self.last_performance_report = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current logging statistics."""
        with self.stats_lock:
            return {
                "current_queue_size": self.log_queue.qsize(),
                "total_logs_queued": self.total_logs_queued,
                "total_logs_written": self.total_logs_written,
                "total_logs_dropped": self.total_logs_dropped,
                "queue_full_events": self.queue_full_events,
                "drop_rate": self.total_logs_dropped / max(1, self.total_logs_queued),
                "logger_thread_alive": self.logger_thread.is_alive() if self.logger_thread else False
            }
    
    def shutdown(self, timeout: float = 5.0):
        """Gracefully shutdown async logger."""
        if not self.quiet_mode:
            print("🔄 Shutting down AsyncLogger...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for worker thread to finish
        if self.logger_thread and self.logger_thread.is_alive():
            self.logger_thread.join(timeout=timeout)
        
        # Close all log files
        for file_handle in self.log_files.values():
            try:
                file_handle.close()
            except Exception:
                pass
        
        # Final stats
        stats = self.get_stats()
        if not self.quiet_mode:
            print(f"✅ AsyncLogger shutdown complete")
            print(f"   Final stats: {stats['total_logs_written']} written, {stats['total_logs_dropped']} dropped")


# Global async logger instance
_global_async_logger: Optional[AsyncLogger] = None


def initialize_global_async_logger(config: LoggerConfig, quiet_mode: bool = False):
    """Initialize global async logger."""
    global _global_async_logger
    _global_async_logger = AsyncLogger(config, quiet_mode)


def get_global_async_logger() -> Optional[AsyncLogger]:
    """Get global async logger instance."""
    return _global_async_logger


def please_log_global(obj: LoggableObject) -> bool:
    """Convenience function to log to global async logger."""
    if _global_async_logger:
        return _global_async_logger.please_log(obj)
    return False