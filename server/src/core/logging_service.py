"""
Dynamic Brain Logging Service

Provides comprehensive logging for the dynamic brain architecture.
Tracks brain evolution, performance metrics, and emergent behaviors.
"""

import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from ..utils.brain_logger import BrainLogger
from ..utils.loggable_objects import (
    LoggableBrainState,
    LoggablePerformanceMetrics,
    LoggableSystemEvent
)


class LoggingService:
    """
    Centralized logging service for the dynamic brain architecture.
    
    Features:
    - Per-session logging
    - Performance metrics tracking
    - Brain state evolution
    - System events
    - Async logging support
    """
    
    def __init__(self, log_dir: str = "logs", enable_async: bool = True, quiet_mode: bool = False):
        """Initialize logging service."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.enable_async = enable_async
        self.quiet_mode = quiet_mode
        
        # Session loggers
        self.session_loggers: Dict[str, BrainLogger] = {}
        
        # Global logger for system events
        self.system_logger = BrainLogger(
            session_name="system",
            log_dir=str(self.log_dir),
            enable_async=enable_async,
            quiet_mode=quiet_mode
        )
        
        # Performance tracking
        self.performance_window = 100  # Track last N cycles
        self.session_metrics: Dict[str, List[float]] = {}
        
        if not quiet_mode:
            print(f"📝 Logging service initialized at {self.log_dir}")
    
    def create_session_logger(self, session_id: str, robot_type: str) -> BrainLogger:
        """Create a logger for a specific brain session."""
        
        # Create unique session name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{robot_type}_{session_id}_{timestamp}"
        
        # Create logger
        logger = BrainLogger(
            session_name=session_name,
            log_dir=str(self.log_dir),
            enable_async=self.enable_async,
            quiet_mode=self.quiet_mode
        )
        
        self.session_loggers[session_id] = logger
        
        # Log session creation
        self.log_system_event(
            event_type="session_created",
            details={
                "session_id": session_id,
                "robot_type": robot_type,
                "timestamp": time.time()
            }
        )
        
        return logger
    
    def log_brain_cycle(self, session_id: str, brain_state: Dict[str, Any], 
                       cycle_time_ms: float, cycle_num: int):
        """Log a brain processing cycle."""
        
        logger = self.session_loggers.get(session_id)
        if not logger:
            return
        
        # The LoggableBrainState expects a brain object and experience_count
        # Since we don't have the actual brain object here, we'll use the fallback sync method
        self._log_brain_state_sync(session_id, brain_state, cycle_time_ms, cycle_num)
        
        # Track performance metrics
        self._track_performance(session_id, cycle_time_ms)
    
    def log_experience(self, session_id: str, experience_id: str, 
                      sensory_dim: int, motor_dim: int):
        """Log an experience storage event."""
        
        logger = self.session_loggers.get(session_id)
        if not logger:
            return
        
        event = LoggableSystemEvent(
            event_name="experience_stored",
            details={
                "timestamp": time.time(),
                "session_id": session_id,
                "experience_id": experience_id,
                "sensory_dimensions": sensory_dim,
                "motor_dimensions": motor_dim
            }
        )
        
        # Use the async logger if available
        if logger.async_logger and logger.enable_async:
            logger.async_logger.please_log(event)
        else:
            # Fallback to synchronous logging  
            with open(logger.emergence_log, 'a') as f:
                f.write(json.dumps({
                    "type": "system_event",
                    "timestamp": time.time(),
                    "event_name": "experience_stored",
                    "details": event.details
                }) + '\n')
    
    def log_performance_metrics(self, session_id: str):
        """Log performance metrics for a session."""
        
        logger = self.session_loggers.get(session_id)
        if not logger or session_id not in self.session_metrics:
            return
        
        cycle_times = self.session_metrics[session_id]
        if not cycle_times:
            return
        
        # Log performance metrics as system event
        window_times = cycle_times[-self.performance_window:]
        metrics_data = {
            "type": "performance_metrics",
            "timestamp": time.time(),
            "session_id": session_id,
            "avg_cycle_time_ms": sum(window_times) / len(window_times),
            "max_cycle_time_ms": max(window_times),
            "min_cycle_time_ms": min(window_times),
            "total_cycles": len(cycle_times)
        }
        
        # Write to log file
        with open(logger.brain_state_log, 'a') as f:
            f.write(json.dumps(metrics_data) + '\n')
    
    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log a system-wide event."""
        
        # Add timestamp and session_id to details
        enhanced_details = {
            "timestamp": time.time(),
            "session_id": "system",
            **details
        }
        
        event = LoggableSystemEvent(
            event_name=event_type,
            details=enhanced_details
        )
        
        # Use the async logger if available, otherwise fallback to sync
        if self.system_logger.async_logger and self.system_logger.enable_async:
            self.system_logger.async_logger.please_log(event)
        else:
            # Fallback to synchronous logging
            with open(self.system_logger.emergence_log, 'a') as f:
                f.write(json.dumps({
                    "type": "system_event",
                    "timestamp": time.time(),
                    "event_name": event_type,
                    "details": enhanced_details
                }) + '\n')
    
    def close_session(self, session_id: str):
        """Close a session logger."""
        
        logger = self.session_loggers.get(session_id)
        if logger:
            # Log final metrics
            self.log_performance_metrics(session_id)
            
            # Log session closure
            self.log_system_event(
                event_type="session_closed",
                details={
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "total_cycles": len(self.session_metrics.get(session_id, []))
                }
            )
            
            # Close logger
            logger.close_session()
            
            # Clean up
            del self.session_loggers[session_id]
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]
    
    def shutdown(self):
        """Shutdown all loggers."""
        
        # Close all session loggers
        session_ids = list(self.session_loggers.keys())
        for session_id in session_ids:
            self.close_session(session_id)
        
        # Log shutdown
        self.log_system_event(
            event_type="logging_service_shutdown",
            details={
                "timestamp": time.time()
            }
        )
        
        # Close system logger
        self.system_logger.close_session()
        
        if not self.quiet_mode:
            print("📝 Logging service shutdown complete")
    
    def _log_brain_state_sync(self, session_id: str, brain_state: Dict[str, Any], 
                             cycle_time_ms: float, cycle_num: int):
        """Synchronous brain state logging."""
        
        logger = self.session_loggers.get(session_id)
        if not logger:
            return
            
        # Write brain state to log file
        timestamp = time.time()
        state_entry = {
            "type": "brain_state",
            "timestamp": timestamp,
            "session_id": session_id,
            "cycle_number": cycle_num,
            "field_dimensions": brain_state.get('field_dimensions', 0),
            "prediction_confidence": brain_state.get('prediction_confidence', 0.5),
            "cycle_time_ms": cycle_time_ms,
            "active_patterns": brain_state.get('active_patterns', 0)
        }
        
        # Write to brain state log file
        brain_state_log = Path(logger.brain_state_log)
        with open(brain_state_log, 'a') as f:
            f.write(json.dumps(state_entry) + '\n')
    
    def _track_performance(self, session_id: str, cycle_time_ms: float):
        """Track performance metrics for a session."""
        
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = []
        
        self.session_metrics[session_id].append(cycle_time_ms)
        
        # Log performance warning if needed
        if cycle_time_ms > 750:
            self.log_system_event(
                event_type="performance_warning",
                details={
                    "session_id": session_id,
                    "cycle_time_ms": cycle_time_ms,
                    "threshold_ms": 750
                }
            )
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a session."""
        
        if session_id not in self.session_metrics:
            return None
        
        cycle_times = self.session_metrics[session_id]
        if not cycle_times:
            return None
        
        return {
            "session_id": session_id,
            "total_cycles": len(cycle_times),
            "avg_cycle_time_ms": sum(cycle_times) / len(cycle_times),
            "max_cycle_time_ms": max(cycle_times),
            "min_cycle_time_ms": min(cycle_times),
            "recent_avg_ms": sum(cycle_times[-10:]) / len(cycle_times[-10:]) if len(cycle_times) >= 10 else None
        }