#!/usr/bin/env python3
"""
Brain Maintenance Interface

Unified maintenance interface for all brain implementations providing
standardized light maintenance, heavy maintenance, and deep consolidation
operations during downtime periods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time


class BrainMaintenanceInterface(ABC):
    """Unified maintenance interface for all brain implementations."""
    
    def __init__(self):
        """Initialize maintenance tracking."""
        self.last_light_maintenance = 0.0
        self.last_heavy_maintenance = 0.0
        self.last_deep_consolidation = 0.0
        self.maintenance_stats = {
            'light_maintenance_count': 0,
            'heavy_maintenance_count': 0,
            'deep_consolidation_count': 0,
            'total_light_time_ms': 0.0,
            'total_heavy_time_ms': 0.0,
            'total_consolidation_time_ms': 0.0
        }
    
    @abstractmethod
    def light_maintenance(self) -> None:
        """
        Quick cleanup operations (<1ms) - safe during active processing.
        
        Typical operations:
        - Cleanup expired temporary data
        - Update simple metrics
        - Clear small buffers
        - Basic housekeeping
        
        Should be safe to call frequently without performance impact.
        """
        pass
    
    @abstractmethod  
    def heavy_maintenance(self) -> None:
        """
        Moderate maintenance (10-50ms) - during short idle periods.
        
        Typical operations:
        - Rebalance data structures
        - Optimize search indices
        - Redistribute resources
        - Medium-complexity cleanup
        
        Called during brief idle periods (5-30 seconds).
        """
        pass
    
    @abstractmethod
    def deep_consolidation(self) -> None:
        """
        Intensive consolidation (100-500ms) - during extended idle periods.
        
        Typical operations:
        - Full data structure optimization
        - Memory defragmentation
        - Pattern consolidation
        - Heavy reorganization
        
        Called during extended idle periods (60+ seconds).
        """
        pass
    
    def get_maintenance_status(self) -> Dict[str, Any]:
        """
        Return maintenance metrics for monitoring.
        
        Returns:
            Dictionary containing maintenance statistics and timing info.
        """
        current_time = time.time()
        return {
            'maintenance_stats': self.maintenance_stats.copy(),
            'time_since_light_maintenance': current_time - self.last_light_maintenance,
            'time_since_heavy_maintenance': current_time - self.last_heavy_maintenance,
            'time_since_deep_consolidation': current_time - self.last_deep_consolidation,
            'implementation_type': self.__class__.__name__
        }
    
    def _track_maintenance_timing(self, maintenance_type: str, start_time: float, end_time: float) -> None:
        """Track timing statistics for maintenance operations."""
        duration_ms = (end_time - start_time) * 1000.0
        
        if maintenance_type == 'light':
            self.maintenance_stats['light_maintenance_count'] += 1
            self.maintenance_stats['total_light_time_ms'] += duration_ms
            self.last_light_maintenance = end_time
        elif maintenance_type == 'heavy':
            self.maintenance_stats['heavy_maintenance_count'] += 1
            self.maintenance_stats['total_heavy_time_ms'] += duration_ms
            self.last_heavy_maintenance = end_time
        elif maintenance_type == 'deep':
            self.maintenance_stats['deep_consolidation_count'] += 1
            self.maintenance_stats['total_consolidation_time_ms'] += duration_ms
            self.last_deep_consolidation = end_time
    
    def safe_light_maintenance(self) -> None:
        """Safe wrapper for light maintenance with timing tracking."""
        start_time = time.time()
        try:
            self.light_maintenance()
        finally:
            end_time = time.time()
            self._track_maintenance_timing('light', start_time, end_time)
    
    def safe_heavy_maintenance(self) -> None:
        """Safe wrapper for heavy maintenance with timing tracking."""
        start_time = time.time()
        try:
            self.heavy_maintenance()
        finally:
            end_time = time.time()
            self._track_maintenance_timing('heavy', start_time, end_time)
    
    def safe_deep_consolidation(self) -> None:
        """Safe wrapper for deep consolidation with timing tracking."""
        start_time = time.time()
        try:
            self.deep_consolidation()
        finally:
            end_time = time.time()
            self._track_maintenance_timing('deep', start_time, end_time)
    
    def get_maintenance_recommendations(self, idle_time_seconds: float) -> Dict[str, bool]:
        """
        Get maintenance recommendations based on idle time and last maintenance.
        
        Args:
            idle_time_seconds: How long the brain has been idle (cognitive load < 0.3)
            
        Returns:
            Dictionary indicating which maintenance operations are recommended
        """
        current_time = time.time()
        
        # Time since last maintenance operations
        time_since_light = current_time - self.last_light_maintenance
        time_since_heavy = current_time - self.last_heavy_maintenance
        time_since_deep = current_time - self.last_deep_consolidation
        
        # Enhanced recommendations for continuous processing scenarios
        return {
            'light_maintenance': (
                # Cognitive idle periods (preferred)
                (idle_time_seconds > 10.0 and time_since_light > 30.0) or  
                # Time-based fallback for continuous processing
                time_since_light > 300.0     # Force light maintenance every 5 minutes
            ),
            'heavy_maintenance': (
                # Cognitive idle periods (preferred) 
                (idle_time_seconds > 30.0 and time_since_heavy > 60.0) or  
                # Time-based fallback for continuous processing
                time_since_heavy > 1200.0    # Force heavy maintenance every 20 minutes
            ),
            'deep_consolidation': (
                # Cognitive idle periods (preferred)
                (idle_time_seconds > 120.0 and time_since_deep > 300.0) or  
                # Time-based fallback for continuous processing
                time_since_deep > 3600.0     # Force deep consolidation every hour
            )
        }


class MaintenanceScheduler:
    """
    Scheduler for coordinating maintenance operations across brain implementations.
    """
    
    def __init__(self, brain: BrainMaintenanceInterface):
        """Initialize scheduler with brain instance."""
        self.brain = brain
        self.last_activity_time = time.time()
        self.last_maintenance_check = 0.0  # Throttle maintenance checks
    
    def mark_activity(self, cognitive_load: float = 1.0) -> None:
        """
        Mark brain activity with cognitive load indicator.
        
        Args:
            cognitive_load: 0.0-1.0 where:
                0.0-0.3 = routine/background processing (sensors, but nothing interesting)
                0.3-0.7 = moderate cognitive activity (learning, adaptation) 
                0.7-1.0 = high cognitive activity (novel situations, problem solving)
        """
        current_time = time.time()
        
        # Only reset idle timer for significant cognitive activity
        if cognitive_load > 0.3:  # Moderate to high cognitive load
            self.last_activity_time = current_time
        # Low cognitive load (routine processing) doesn't reset idle timer
    
    def get_idle_time(self) -> float:
        """Get current idle time in seconds."""
        return time.time() - self.last_activity_time
    
    def run_recommended_maintenance(self) -> Dict[str, bool]:
        """
        Run maintenance operations recommended for current idle time.
        
        Returns:
            Dictionary indicating which maintenance operations were performed
        """
        current_time = time.time()
        
        # Throttle maintenance checks to prevent excessive calls
        # Reduced throttle for continuous processing scenarios
        if current_time - self.last_maintenance_check < 2.0:  # Minimum 2 seconds between checks
            return {
                'light_maintenance': False,
                'heavy_maintenance': False,
                'deep_consolidation': False
            }
        
        self.last_maintenance_check = current_time
        idle_time = self.get_idle_time()
        recommendations = self.brain.get_maintenance_recommendations(idle_time)
        
        performed = {
            'light_maintenance': False,
            'heavy_maintenance': False,
            'deep_consolidation': False
        }
        
        if recommendations['light_maintenance']:
            start_time = time.time()
            self.brain.safe_light_maintenance()
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine trigger reason
            time_since_light = current_time - self.brain.last_light_maintenance
            if idle_time > 10.0 and time_since_light > 30.0:
                trigger = f"cognitive idle: {idle_time:.1f}s"
            else:
                trigger = f"time-based: {time_since_light/60:.1f}min since last"
            
            print(f"🔧 SCHEDULER: Light maintenance completed ({duration_ms:.1f}ms, {trigger})")
            performed['light_maintenance'] = True
        
        if recommendations['heavy_maintenance']:
            start_time = time.time()
            self.brain.safe_heavy_maintenance() 
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine trigger reason
            time_since_heavy = current_time - self.brain.last_heavy_maintenance
            if idle_time > 30.0 and time_since_heavy > 60.0:
                trigger = f"cognitive idle: {idle_time:.1f}s"
            else:
                trigger = f"time-based: {time_since_heavy/60:.1f}min since last"
            
            print(f"🔧 SCHEDULER: Heavy maintenance completed ({duration_ms:.1f}ms, {trigger})")
            performed['heavy_maintenance'] = True
            
        if recommendations['deep_consolidation']:
            start_time = time.time()
            self.brain.safe_deep_consolidation()
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine trigger reason  
            time_since_deep = current_time - self.brain.last_deep_consolidation
            if idle_time > 120.0 and time_since_deep > 300.0:
                trigger = f"cognitive idle: {idle_time:.1f}s"
            else:
                trigger = f"time-based: {time_since_deep/60:.1f}min since last"
            
            print(f"🔧 SCHEDULER: Deep consolidation completed ({duration_ms:.1f}ms, {trigger})")
            performed['deep_consolidation'] = True
        
        return performed