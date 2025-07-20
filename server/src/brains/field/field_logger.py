#!/usr/bin/env python3
"""
Field Brain Logger

Specialized logging for field brain operations, focusing on field-specific
metrics rather than discrete experience/pattern metrics.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FieldMetrics:
    """Field-specific performance metrics."""
    field_evolution_cycles: int = 0
    field_consolidation_cycles: int = 0
    sparse_attention_regions: int = 0
    field_compression_ratio: float = 0.0
    field_memory_usage_mb: float = 0.0
    average_field_update_time_ms: float = 0.0


class FieldBrainLogger:
    """
    Logger specialized for field brain operations.
    
    Focuses on field-specific metrics like field evolution, sparse attention,
    and continuous field dynamics rather than discrete patterns/experiences.
    """
    
    def __init__(self, quiet_mode: bool = False):
        """Initialize field brain logger."""
        self.quiet_mode = quiet_mode
        self.metrics = FieldMetrics()
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.report_interval = 60.0  # Report every 60 seconds
        
        # Performance tracking
        self.cycle_times = []
        self.field_update_times = []
        
        if not quiet_mode:
            print("🧠 Field Brain Logger initialized")
            print("   Tracking: field evolution, sparse attention, consolidation")
    
    def log_initialization(self, config: Dict[str, Any]):
        """Log field brain initialization details."""
        if self.quiet_mode:
            return
            
        print("🧠 Field Brain Architecture:")
        spatial_res = config.get('spatial_resolution', 20)
        temporal_window = config.get('temporal_window', 10.0)
        evolution_rate = config.get('field_evolution_rate', 0.1)
        constraint_rate = config.get('constraint_discovery_rate', 0.15)
        
        print(f"   Field space: {spatial_res}³ spatial resolution")
        print(f"   Temporal window: {temporal_window}s")
        print(f"   Evolution rate: {evolution_rate}")
        print(f"   Constraint discovery: {constraint_rate}")
        print(f"   Sparse attention: ✅ Biological optimization")
        print(f"   Field consolidation: ✅ Intrinsic memory")
    
    def log_field_cycle(self, cycle_time_ms: float, sparse_regions: int = 0, 
                       field_update_time_ms: float = 0.0):
        """Log a field processing cycle."""
        self.cycle_times.append(cycle_time_ms)
        if field_update_time_ms > 0:
            self.field_update_times.append(field_update_time_ms)
        
        self.metrics.sparse_attention_regions = sparse_regions
        if field_update_time_ms > 0:
            # Rolling average for field update time
            total_updates = len(self.field_update_times)
            self.metrics.average_field_update_time_ms = (
                (self.metrics.average_field_update_time_ms * (total_updates - 1) + field_update_time_ms) 
                / total_updates
            )
        
        # Keep recent history only
        if len(self.cycle_times) > 100:
            self.cycle_times = self.cycle_times[-50:]
        if len(self.field_update_times) > 100:
            self.field_update_times = self.field_update_times[-50:]
    
    def log_field_evolution(self):
        """Log field evolution cycle."""
        self.metrics.field_evolution_cycles += 1
    
    def log_field_consolidation(self, compression_ratio: float = 0.0, 
                               memory_usage_mb: float = 0.0):
        """Log field consolidation/decay cycle."""
        self.metrics.field_consolidation_cycles += 1
        if compression_ratio > 0:
            self.metrics.field_compression_ratio = compression_ratio
        if memory_usage_mb > 0:
            self.metrics.field_memory_usage_mb = memory_usage_mb
    
    def maybe_report_performance(self):
        """Report field performance metrics periodically."""
        current_time = time.time()
        if current_time - self.last_report_time < self.report_interval:
            return
        
        if len(self.cycle_times) < 5:  # Need some samples
            self.last_report_time = current_time
            return
        
        # Calculate field-specific metrics
        recent_cycles = self.cycle_times[-20:]  # Last 20 cycles
        avg_cycle_time = sum(recent_cycles) / len(recent_cycles)
        min_cycle_time = min(recent_cycles)
        max_cycle_time = max(recent_cycles)
        
        # Field processing frequency
        field_frequency = 1000.0 / avg_cycle_time if avg_cycle_time > 0 else 0
        
        print(f"🧠 Field Brain Performance Report:")
        print(f"   Cycle time: {avg_cycle_time:.1f}ms avg ({min_cycle_time:.1f}-{max_cycle_time:.1f}ms)")
        print(f"   Field frequency: {field_frequency:.1f}Hz")
        print(f"   Field evolution cycles: {self.metrics.field_evolution_cycles}")
        print(f"   Field consolidation cycles: {self.metrics.field_consolidation_cycles}")
        
        if self.metrics.sparse_attention_regions > 0:
            print(f"   Active attention regions: {self.metrics.sparse_attention_regions}")
        
        if self.metrics.average_field_update_time_ms > 0:
            print(f"   Field update time: {self.metrics.average_field_update_time_ms:.1f}ms avg")
        
        if self.metrics.field_compression_ratio > 0:
            print(f"   Field compression: {self.metrics.field_compression_ratio:.2f}x")
        
        if self.metrics.field_memory_usage_mb > 0:
            print(f"   Field memory: {self.metrics.field_memory_usage_mb:.1f}MB")
        
        print(f"   Uptime: {(current_time - self.start_time) / 60:.1f}min")
        
        self.last_report_time = current_time
    
    def log_error(self, error_msg: str, context: str = ""):
        """Log field brain errors."""
        print(f"❌ Field Brain Error: {error_msg}")
        if context:
            print(f"   Context: {context}")
    
    def log_warning(self, warning_msg: str, context: str = ""):
        """Log field brain warnings."""
        print(f"⚠️ Field Brain Warning: {warning_msg}")
        if context:
            print(f"   Context: {context}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current field brain metrics."""
        recent_cycles = self.cycle_times[-20:] if self.cycle_times else []
        
        return {
            'field_evolution_cycles': self.metrics.field_evolution_cycles,
            'field_consolidation_cycles': self.metrics.field_consolidation_cycles,
            'sparse_attention_regions': self.metrics.sparse_attention_regions,
            'field_compression_ratio': self.metrics.field_compression_ratio,
            'field_memory_usage_mb': self.metrics.field_memory_usage_mb,
            'average_cycle_time_ms': sum(recent_cycles) / len(recent_cycles) if recent_cycles else 0,
            'average_field_update_time_ms': self.metrics.average_field_update_time_ms,
            'total_cycles': len(self.cycle_times),
            'uptime_minutes': (time.time() - self.start_time) / 60.0
        }