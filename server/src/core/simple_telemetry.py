"""
Simple Telemetry - Just the essentials.

No complex hierarchies, no 50+ metrics. Just what matters.
"""

import time
from typing import Dict, Any


class SimpleTelemetry:
    """Dead simple telemetry for one brain."""
    
    def __init__(self):
        self.start_time = time.time()
        self.cycles = 0
        self.last_update = time.time()
        
    def update(self, brain_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update telemetry with latest brain state.
        
        Returns simplified metrics that actually matter.
        """
        self.cycles += 1
        now = time.time()
        
        # Calculate rates
        dt = now - self.last_update
        hz = 1.0 / dt if dt > 0 else 0
        self.last_update = now
        
        # Essential metrics only
        return {
            'uptime': now - self.start_time,
            'cycles': self.cycles,
            'hz': hz,
            'energy': brain_telemetry.get('energy', 0),
            'comfort': brain_telemetry.get('comfort', 0),
            'motivation': brain_telemetry.get('motivation', 'unknown'),
            'exploring': brain_telemetry.get('exploring', False),
        }
    
    def get_summary(self) -> str:
        """One-line summary for logging."""
        uptime_min = (time.time() - self.start_time) / 60
        avg_hz = self.cycles / (time.time() - self.start_time)
        return f"Uptime: {uptime_min:.1f}m, Cycles: {self.cycles}, Avg Hz: {avg_hz:.1f}"