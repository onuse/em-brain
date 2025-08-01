"""
Maintenance Scheduler for Dynamic Brain Architecture

Schedules and manages maintenance operations across all active brains,
including field maintenance, memory cleanup, and performance optimization.
"""

import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import gc

from .interfaces import IBrainPool, IBrainService
from ..config.enhanced_gpu_memory_manager import (
    check_gpu_memory_pressure, cleanup_gpu_memory, get_gpu_memory_stats
)


class MaintenanceScheduler:
    """
    Schedules maintenance operations for the dynamic brain architecture.
    
    Features:
    - Periodic field maintenance
    - Memory pressure monitoring
    - Performance optimization
    - Garbage collection coordination
    """
    
    def __init__(self, brain_pool: IBrainPool, brain_service: IBrainService,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize maintenance scheduler."""
        self.brain_pool = brain_pool
        self.brain_service = brain_service
        self.config = config or {}
        
        # Maintenance intervals (in seconds)
        self.field_maintenance_interval = self.config.get('field_maintenance_interval', 300)  # 5 minutes
        self.memory_check_interval = self.config.get('memory_check_interval', 60)  # 1 minute
        self.performance_check_interval = self.config.get('performance_check_interval', 120)  # 2 minutes
        
        # State
        self.running = False
        self.maintenance_thread = None
        self.last_field_maintenance = {}
        self.last_memory_check = time.time()
        self.last_performance_check = time.time()
        
        # Thresholds
        self.memory_pressure_threshold = self.config.get('memory_pressure_threshold', 0.85)  # 85% memory usage
        self.performance_degradation_threshold = self.config.get('performance_degradation_threshold', 5.0)  # 5x slower
        
    def start(self):
        """Start the maintenance scheduler."""
        if self.running:
            return
            
        self.running = True
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="MaintenanceScheduler",
            daemon=True
        )
        self.maintenance_thread.start()
        print("🔧 Maintenance scheduler started")
        
    def stop(self):
        """Stop the maintenance scheduler."""
        if not self.running:
            return
            
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)
        print("🔧 Maintenance scheduler stopped")
        
    def _maintenance_loop(self):
        """Main maintenance loop."""
        while self.running:
            try:
                current_time = time.time()
                
                # Check memory pressure
                if current_time - self.last_memory_check > self.memory_check_interval:
                    self._check_memory_pressure()
                    self.last_memory_check = current_time
                
                # Check performance
                if current_time - self.last_performance_check > self.performance_check_interval:
                    self._check_performance()
                    self.last_performance_check = current_time
                
                # Schedule field maintenance
                self._schedule_field_maintenance(current_time)
                
                # Sleep briefly to avoid busy waiting
                time.sleep(10)
                
            except Exception as e:
                print(f"⚠️ Maintenance error: {e}")
                time.sleep(60)  # Back off on error
                
    def _check_memory_pressure(self):
        """Check system and GPU memory pressure and trigger cleanup if needed."""
        try:
            import psutil
            
            # Check system memory
            mem = psutil.virtual_memory()
            memory_usage = mem.percent / 100.0
            
            # Check GPU memory pressure
            gpu_pressure = check_gpu_memory_pressure()
            gpu_stats = get_gpu_memory_stats()
            
            if memory_usage > self.memory_pressure_threshold:
                print(f"⚠️ High system memory pressure: {memory_usage:.1%}")
                self._trigger_memory_cleanup()
            elif gpu_pressure:
                device_type = gpu_stats.get('device', 'unknown')
                if 'cuda_allocated_mb' in gpu_stats:
                    allocated = gpu_stats['cuda_allocated_mb']
                    total = gpu_stats['cuda_total_mb']
                    usage_pct = (allocated / total) * 100
                    print(f"⚠️ High GPU memory pressure: {usage_pct:.1f}% ({allocated:.0f}/{total:.0f}MB)")
                else:
                    print(f"⚠️ High GPU memory pressure on {device_type}")
                self._trigger_memory_cleanup()
                
        except ImportError:
            # psutil not available, skip memory check
            pass
        except Exception as e:
            print(f"⚠️ Memory check error: {e}")
            
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup across all brains."""
        print("🧹 Triggering memory cleanup...")
        
        # Get all active brains
        active_brains = self.brain_pool.get_active_brains()
        
        for profile, brain in active_brains.items():
            try:
                # Each brain can implement its own cleanup
                if hasattr(brain, 'cleanup_memory'):
                    brain.cleanup_memory()
                    
            except Exception as e:
                print(f"⚠️ Cleanup error for {profile}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Clean up GPU memory
        cleanup_gpu_memory()
        
        print("✅ Memory cleanup complete (system + GPU)")
        
    def _check_performance(self):
        """Check performance metrics and optimize if needed."""
        # Get all active sessions
        sessions = self.brain_service.get_all_sessions()
        
        for session_id, stats in sessions.items():
            avg_cycle_time = stats.get('average_cycle_time_ms', 0)
            
            # Check if performance is degraded
            target_cycle_time = 250  # Target 250ms per cycle
            if avg_cycle_time > target_cycle_time * self.performance_degradation_threshold:
                print(f"⚠️ Performance degradation in session {session_id}: {avg_cycle_time:.1f}ms")
                self._optimize_session_performance(session_id)
                
    def _optimize_session_performance(self, session_id: str):
        """Optimize performance for a specific session."""
        # This could involve:
        # - Reducing field resolution temporarily
        # - Disabling non-essential features
        # - Adjusting processing priorities
        print(f"🔧 Optimizing performance for session {session_id}")
        
    def _schedule_field_maintenance(self, current_time: float):
        """Schedule field maintenance for active brains."""
        active_brains = self.brain_pool.get_active_brains()
        
        for profile, brain in active_brains.items():
            # Check if maintenance is due
            last_maintenance = self.last_field_maintenance.get(profile, 0)
            if current_time - last_maintenance > self.field_maintenance_interval:
                self._perform_field_maintenance(profile, brain)
                self.last_field_maintenance[profile] = current_time
                
    def _perform_field_maintenance(self, profile: str, brain):
        """Perform field maintenance on a brain."""
        try:
            if hasattr(brain, 'perform_maintenance'):
                print(f"🔧 Performing field maintenance for {profile}")
                
                start_time = time.time()
                brain.perform_maintenance()
                duration = (time.time() - start_time) * 1000
                
                print(f"✅ Field maintenance complete for {profile} ({duration:.1f}ms)")
            
        except Exception as e:
            print(f"⚠️ Field maintenance error for {profile}: {e}")
            
    def get_status(self) -> Dict[str, Any]:
        """Get maintenance scheduler status."""
        active_brains = self.brain_pool.get_active_brains()
        
        return {
            'running': self.running,
            'active_brains': len(active_brains),
            'maintenance_schedule': {
                profile: {
                    'last_maintenance': self.last_field_maintenance.get(profile, 0),
                    'next_maintenance': self.last_field_maintenance.get(profile, 0) + self.field_maintenance_interval,
                    'overdue': time.time() > self.last_field_maintenance.get(profile, 0) + self.field_maintenance_interval
                }
                for profile in active_brains
            },
            'intervals': {
                'field_maintenance': self.field_maintenance_interval,
                'memory_check': self.memory_check_interval,
                'performance_check': self.performance_check_interval
            }
        }