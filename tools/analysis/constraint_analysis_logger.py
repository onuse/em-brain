#!/usr/bin/env python3
"""
Constraint Analysis Logger

Logs and analyzes constraint propagation patterns over longer sessions.
Designed to detect emergent constraint behaviors that may only appear
after extended operation.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ConstraintEvent:
    """A single constraint propagation event."""
    timestamp: float
    cycle: int
    source_stream: str
    constraint_type: str
    intensity: float
    target_streams: List[str]
    metadata: Dict[str, Any]


@dataclass
class ConstraintAnalysis:
    """Analysis results for constraint patterns."""
    total_events: int
    constraint_types_seen: List[str]
    most_active_source: str
    most_affected_target: str
    avg_intensity: float
    peak_simultaneous_constraints: int
    emergent_patterns: List[str]
    temporal_patterns: Dict[str, Any]


class ConstraintAnalysisLogger:
    """
    Logs constraint propagation events and analyzes emergent patterns.
    
    This is designed for longer test sessions where subtle constraint
    emergence patterns need to be tracked and analyzed.
    """
    
    def __init__(self, session_name: str = None):
        self.session_name = session_name or f"constraint_session_{int(time.time())}"
        self.events: List[ConstraintEvent] = []
        self.cycle_count = 0
        self.start_time = time.time()
        
        # Pattern tracking
        self.source_counts = defaultdict(int)
        self.target_counts = defaultdict(int)
        self.constraint_type_counts = defaultdict(int)
        self.intensity_history = []
        self.simultaneous_constraint_peaks = []
        
        print(f"🔍 ConstraintAnalysisLogger started: {self.session_name}")
    
    def log_cycle(self, brain_state: Dict[str, Any], cycle: int = None):
        """Log constraint information from a brain cycle."""
        if cycle is None:
            cycle = self.cycle_count
        self.cycle_count = cycle
        
        # Extract constraint propagation information
        if 'constraint_propagation' in brain_state:
            constraint_info = brain_state['constraint_propagation']
            
            # Log active constraints
            if 'propagation_stats' in constraint_info:
                stats = constraint_info['propagation_stats']
                active_count = stats.get('active_constraints', 0)
                if active_count > 0:
                    self.simultaneous_constraint_peaks.append(active_count)
            
            # Log constraint pressures for each stream
            if 'constraint_pressures' in constraint_info:
                pressures = constraint_info['constraint_pressures']
                current_time = time.time()
                
                for stream_name, pressure_info in pressures.items():
                    total_pressure = pressure_info.get('total_pressure', 0.0)
                    constraint_breakdown = pressure_info.get('constraint_breakdown', {})
                    
                    # Log individual constraint types
                    for constraint_type, intensity in constraint_breakdown.items():
                        if intensity > 0.05:  # Only log significant constraints
                            # Infer this came from cross-stream propagation
                            event = ConstraintEvent(
                                timestamp=current_time,
                                cycle=cycle,
                                source_stream="unknown",  # Would need to track source
                                constraint_type=constraint_type,
                                intensity=intensity,
                                target_streams=[stream_name],
                                metadata={'total_pressure': total_pressure}
                            )
                            self.events.append(event)
                            
                            # Update tracking
                            self.target_counts[stream_name] += 1
                            self.constraint_type_counts[constraint_type] += 1
                            self.intensity_history.append(intensity)
    
    def log_constraint_event(self, source_stream: str, constraint_type: str, 
                           intensity: float, target_streams: List[str], 
                           metadata: Dict[str, Any] = None):
        """Manually log a constraint propagation event."""
        event = ConstraintEvent(
            timestamp=time.time(),
            cycle=self.cycle_count,
            source_stream=source_stream,
            constraint_type=constraint_type,
            intensity=intensity,
            target_streams=target_streams,
            metadata=metadata or {}
        )
        self.events.append(event)
        
        # Update tracking
        self.source_counts[source_stream] += 1
        for target in target_streams:
            self.target_counts[target] += 1
        self.constraint_type_counts[constraint_type] += 1
        self.intensity_history.append(intensity)
    
    def analyze(self) -> ConstraintAnalysis:
        """Analyze logged constraint events for emergent patterns."""
        if not self.events:
            return ConstraintAnalysis(
                total_events=0,
                constraint_types_seen=[],
                most_active_source="none",
                most_affected_target="none",
                avg_intensity=0.0,
                peak_simultaneous_constraints=0,
                emergent_patterns=[],
                temporal_patterns={}
            )
        
        # Basic statistics
        total_events = len(self.events)
        constraint_types = list(self.constraint_type_counts.keys())
        avg_intensity = sum(self.intensity_history) / len(self.intensity_history)
        peak_simultaneous = max(self.simultaneous_constraint_peaks) if self.simultaneous_constraint_peaks else 0
        
        # Find most active streams
        most_active_source = max(self.source_counts.items(), key=lambda x: x[1])[0] if self.source_counts else "none"
        most_affected_target = max(self.target_counts.items(), key=lambda x: x[1])[0] if self.target_counts else "none"
        
        # Detect emergent patterns
        emergent_patterns = self._detect_emergent_patterns()
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns()
        
        return ConstraintAnalysis(
            total_events=total_events,
            constraint_types_seen=constraint_types,
            most_active_source=most_active_source,
            most_affected_target=most_affected_target,
            avg_intensity=avg_intensity,
            peak_simultaneous_constraints=peak_simultaneous,
            emergent_patterns=emergent_patterns,
            temporal_patterns=temporal_patterns
        )
    
    def _detect_emergent_patterns(self) -> List[str]:
        """Detect emergent constraint propagation patterns."""
        patterns = []
        
        # Pattern 1: Cascade effects (one constraint leading to others)
        if len(self.events) > 10:
            # Group events by time windows
            time_windows = defaultdict(list)
            for event in self.events:
                window = int(event.timestamp * 10) / 10  # 100ms windows
                time_windows[window].append(event)
            
            # Look for cascades (multiple constraints in same window)
            cascade_count = sum(1 for events in time_windows.values() if len(events) > 1)
            if cascade_count > len(time_windows) * 0.3:  # >30% of windows have cascades
                patterns.append("cascade_propagation")
        
        # Pattern 2: Oscillatory constraints (periodic appearance)
        if len(self.constraint_type_counts) > 1:
            # Check if different constraint types alternate
            type_sequence = [event.constraint_type for event in self.events[-20:]]  # Last 20 events
            if len(set(type_sequence)) > 1 and len(type_sequence) > 10:
                patterns.append("constraint_oscillation")
        
        # Pattern 3: Load balancing (constraints distribute across streams)
        if len(self.target_counts) > 2:
            target_values = list(self.target_counts.values())
            if max(target_values) / min(target_values) < 3:  # Relatively balanced
                patterns.append("load_balancing")
        
        # Pattern 4: Resource competition (scarcity constraints)
        scarcity_events = sum(1 for event in self.events if 'scarcity' in event.constraint_type.lower())
        if scarcity_events > len(self.events) * 0.5:  # >50% scarcity constraints
            patterns.append("resource_competition")
        
        return patterns
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in constraint propagation."""
        if len(self.events) < 5:
            return {}
        
        # Calculate event frequency
        session_duration = time.time() - self.start_time
        event_frequency = len(self.events) / session_duration if session_duration > 0 else 0
        
        # Intensity trends
        recent_intensities = self.intensity_history[-10:] if len(self.intensity_history) >= 10 else self.intensity_history
        early_intensities = self.intensity_history[:10] if len(self.intensity_history) >= 10 else []
        
        intensity_trend = "stable"
        if early_intensities and recent_intensities:
            avg_recent = sum(recent_intensities) / len(recent_intensities)
            avg_early = sum(early_intensities) / len(early_intensities)
            if avg_recent > avg_early * 1.2:
                intensity_trend = "increasing"
            elif avg_recent < avg_early * 0.8:
                intensity_trend = "decreasing"
        
        return {
            'event_frequency_per_second': event_frequency,
            'session_duration_seconds': session_duration,
            'intensity_trend': intensity_trend,
            'peak_periods': len(self.simultaneous_constraint_peaks),
            'constraint_type_diversity': len(self.constraint_type_counts),
        }
    
    def save_analysis(self, filename: str = None):
        """Save analysis results to a JSON file."""
        if filename is None:
            filename = f"logs/constraint_analysis_{self.session_name}_{int(time.time())}.json"
        
        analysis = self.analyze()
        
        report = {
            'session_info': {
                'name': self.session_name,
                'start_time': self.start_time,
                'duration_seconds': time.time() - self.start_time,
                'total_cycles': self.cycle_count
            },
            'analysis': asdict(analysis),
            'raw_events': [asdict(event) for event in self.events[-100:]]  # Last 100 events
        }
        
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Constraint analysis saved to: {filename}")
            return filename
        except Exception as e:
            print(f"⚠️ Could not save analysis: {e}")
            return None
    
    def print_summary(self):
        """Print a summary of constraint analysis."""
        analysis = self.analyze()
        
        print(f"\n🔍 Constraint Analysis Summary - {self.session_name}")
        print("=" * 60)
        print(f"Total constraint events: {analysis.total_events}")
        print(f"Constraint types observed: {', '.join(analysis.constraint_types_seen)}")
        print(f"Most active source stream: {analysis.most_active_source}")
        print(f"Most affected target stream: {analysis.most_affected_target}")
        print(f"Average constraint intensity: {analysis.avg_intensity:.3f}")
        print(f"Peak simultaneous constraints: {analysis.peak_simultaneous_constraints}")
        
        if analysis.emergent_patterns:
            print(f"🌟 Emergent patterns detected:")
            for pattern in analysis.emergent_patterns:
                print(f"  - {pattern}")
        else:
            print("🔍 No emergent patterns detected (may need longer observation)")
        
        if analysis.temporal_patterns:
            print(f"⏱️ Temporal patterns:")
            for key, value in analysis.temporal_patterns.items():
                print(f"  - {key}: {value}")


# Factory function
def create_constraint_logger(session_name: str = None) -> ConstraintAnalysisLogger:
    """Create a constraint analysis logger for a session."""
    return ConstraintAnalysisLogger(session_name)


if __name__ == "__main__":
    # Example usage
    logger = create_constraint_logger("test_session")
    
    # Simulate some constraint events
    logger.log_constraint_event("sensory", "processing_load", 0.8, ["motor", "temporal"])
    logger.log_constraint_event("motor", "resource_scarcity", 0.6, ["sensory"])
    logger.log_constraint_event("temporal", "urgency_signal", 0.9, ["attention"])
    
    # Print analysis
    logger.print_summary()