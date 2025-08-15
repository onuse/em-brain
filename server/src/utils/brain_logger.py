"""
Comprehensive Brain Logging System

Rich logging for emergence analysis and debugging. When growing minds rather than
building them, we need detailed logs to understand what patterns emerge during
long learning phases.

Logs are structured for analysis:
- Real-time brain state snapshots
- Adaptation event logs  
- Similarity learning evolution
- Prediction pattern tracking
- Emergent behavior detection
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
import numpy as np
from pathlib import Path

# Async logging imports
from .async_logger import AsyncLogger, LoggerConfig, LogLevel
from .loggable_objects import (
    LoggableBrainState, 
    LoggableEmergenceEvent, 
    LoggablePerformanceMetrics,
    LoggableVitalSigns,
    LoggableSystemEvent
)


class BrainLogger:
    """
    Comprehensive logging system for brain emergence analysis.
    
    Captures detailed information about:
    - Brain state evolution over time
    - Adaptation events and triggers  
    - Similarity learning patterns
    - Prediction accuracy trends
    - Emergent behavior detection
    """
    
    def __init__(self, session_name: str = None, log_dir: str = "logs", config: dict = None, 
                 enable_async: bool = True, quiet_mode: bool = False):
        """
        Initialize brain logging system.
        
        Args:
            session_name: Name for this learning session
            log_dir: Directory to store log files (overridden by config if provided)
            config: Configuration dict that may contain logging settings
            enable_async: Use async logging for performance (default: True)
            quiet_mode: Suppress console output for async logger
        """
        self.session_name = session_name or f"brain_session_{int(time.time())}"
        
        # Use config log directory if provided, otherwise use parameter
        if config and 'logging' in config:
            self.log_dir = config['logging'].get('log_directory', log_dir)
        else:
            self.log_dir = log_dir
            
        self.session_start = time.time()
        self.enable_async = enable_async
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize async logger if enabled
        self.async_logger = None
        if enable_async:
            async_config = LoggerConfig(
                log_directory=Path(self.log_dir),
                max_queue_size=5000,
                batch_size=25,
                flush_interval_seconds=1.0,
                drop_policy="drop_oldest"
            )
            self.async_logger = AsyncLogger(async_config, quiet_mode=quiet_mode)
        
        # Log file paths
        self.brain_state_log = os.path.join(self.log_dir, f"{self.session_name}_brain_state.jsonl")
        self.adaptation_log = os.path.join(self.log_dir, f"{self.session_name}_adaptations.jsonl")
        self.similarity_log = os.path.join(self.log_dir, f"{self.session_name}_similarity.jsonl")
        self.emergence_log = os.path.join(self.log_dir, f"{self.session_name}_emergence.jsonl")
        self.session_summary = os.path.join(self.log_dir, f"{self.session_name}_summary.json")
        
        # In-memory buffers for pattern detection
        self.recent_states = deque(maxlen=100)  # Recent brain states
        self.recent_adaptations = deque(maxlen=50)  # Recent adaptation events
        self.prediction_history = deque(maxlen=200)  # Prediction accuracy history
        
        # Pattern detection state
        self.last_similarity_stats = None
        self.adaptation_triggers = defaultdict(int)
        self.emergence_events = []
        
        # Performance tracking
        self.total_experiences_logged = 0
        self.total_adaptations_logged = 0
        
        # Initialize session
        self._initialize_session()
        
        print(f"BrainLogger initialized for session: {self.session_name}")
        print(f"Log directory: {log_dir}")
    
    def _initialize_session(self):
        """Initialize logging session with metadata."""
        session_metadata = {
            "session_name": self.session_name,
            "start_time": self.session_start,
            "start_datetime": datetime.fromtimestamp(self.session_start).isoformat(),
            "log_files": {
                "brain_state": self.brain_state_log,
                "adaptations": self.adaptation_log, 
                "similarity": self.similarity_log,
                "emergence": self.emergence_log,
                "summary": self.session_summary
            },
            "logging_version": "1.0"
        }
        
        # Write session metadata
        with open(self.session_summary, 'w') as f:
            json.dump(session_metadata, f, indent=2)
    
    def log_brain_state(self, brain, experience_count: int, additional_data: Dict = None):
        """
        Log comprehensive brain state snapshot.
        
        Args:
            brain: MinimalBrain instance
            experience_count: Current experience count
            additional_data: Additional context to log
        """
        if self.async_logger and self.enable_async:
            # Async logging - create LoggableObject quickly (<1ms)
            loggable_state = LoggableBrainState(
                brain=brain,
                experience_count=experience_count,
                additional_data=additional_data,
                log_level=LogLevel.DEBUG
            )
            
            # Queue for background processing (should be <1ms)
            success = self.async_logger.please_log(loggable_state)
            if not success:
                # Fallback to synchronous if async fails
                self._log_brain_state_sync(brain, experience_count, additional_data)
        else:
            # Synchronous fallback
            self._log_brain_state_sync(brain, experience_count, additional_data)
        
        self.total_experiences_logged = experience_count
    
    def _log_brain_state_sync(self, brain, experience_count: int, additional_data: Dict = None):
        """Synchronous brain state logging (fallback)."""
        timestamp = time.time()
        
        # Get comprehensive brain stats
        brain_stats = brain.get_brain_stats()
        
        # Compute current intrinsic reward
        current_reward = brain.compute_intrinsic_reward()
        
        # Brain state snapshot
        state_entry = {
            "timestamp": timestamp,
            "session_time": timestamp - self.session_start,
            "experience_count": experience_count,
            "brain_state": {
                "total_experiences": brain.total_experiences,
                "total_predictions": brain.total_predictions,
                "optimal_prediction_error": brain.optimal_prediction_error,
                "current_intrinsic_reward": current_reward,
                "learning_outcomes_tracked": len(brain.recent_learning_outcomes)
            },
            "system_stats": brain_stats,
            "additional_data": additional_data or {}
        }
        
        # Add to in-memory buffer
        self.recent_states.append(state_entry)
        
        # Write to log file
        with open(self.brain_state_log, 'a') as f:
            f.write(json.dumps(state_entry) + '\n')
        
        # Check for emergence patterns
        self._detect_emergence_patterns(state_entry)
    
    def log_adaptation_event(self, adaptation_type: str, trigger_reason: str, 
                           before_stats: Dict, after_stats: Dict, 
                           adaptation_details: Dict = None):
        """
        Log an adaptation event with detailed before/after comparison.
        
        Args:
            adaptation_type: Type of adaptation (similarity, activation, optimal_error)
            trigger_reason: What triggered this adaptation
            before_stats: System state before adaptation
            after_stats: System state after adaptation  
            adaptation_details: Additional adaptation-specific data
        """
        timestamp = time.time()
        
        adaptation_entry = {
            "timestamp": timestamp,
            "session_time": timestamp - self.session_start,
            "adaptation_type": adaptation_type,
            "trigger_reason": trigger_reason,
            "before_stats": before_stats,
            "after_stats": after_stats,
            "adaptation_details": adaptation_details or {},
            "change_magnitude": self._compute_change_magnitude(before_stats, after_stats)
        }
        
        # Track adaptation triggers
        self.adaptation_triggers[f"{adaptation_type}_{trigger_reason}"] += 1
        
        # Add to in-memory buffer
        self.recent_adaptations.append(adaptation_entry)
        
        # Write to log file
        with open(self.adaptation_log, 'a') as f:
            f.write(json.dumps(adaptation_entry) + '\n')
        
        self.total_adaptations_logged += 1
        
        print(f"Adaptation logged: {adaptation_type} ({trigger_reason}) - "
              f"magnitude: {adaptation_entry['change_magnitude']:.3f}")
    
    def log_similarity_evolution(self, similarity_stats: Dict, experience_count: int):
        """
        Log similarity function evolution patterns.
        
        Args:
            similarity_stats: Current similarity learning statistics
            experience_count: Current experience count
        """
        timestamp = time.time()
        
        similarity_entry = {
            "timestamp": timestamp,
            "session_time": timestamp - self.session_start,
            "experience_count": experience_count,
            "similarity_stats": similarity_stats
        }
        
        # Compute evolution metrics if we have previous stats
        if self.last_similarity_stats:
            evolution_metrics = self._compute_similarity_evolution(
                self.last_similarity_stats, similarity_stats
            )
            similarity_entry["evolution_metrics"] = evolution_metrics
        
        # Write to log file
        with open(self.similarity_log, 'a') as f:
            f.write(json.dumps(similarity_entry) + '\n')
        
        self.last_similarity_stats = similarity_stats.copy()
    
    def log_prediction_outcome(self, predicted_action: List[float], 
                             actual_outcome: List[float],
                             prediction_confidence: float,
                             similar_experiences_used: int):
        """
        Log prediction outcome for accuracy tracking.
        
        Args:
            predicted_action: What was predicted
            actual_outcome: What actually happened
            prediction_confidence: Confidence in prediction
            similar_experiences_used: Number of similar experiences used
        """
        # Compute prediction error
        if len(predicted_action) == len(actual_outcome):
            pred_array = np.array(predicted_action)
            actual_array = np.array(actual_outcome)
            error = np.linalg.norm(pred_array - actual_array)
            max_error = np.linalg.norm(pred_array) + np.linalg.norm(actual_array)
            normalized_error = error / max_error if max_error > 0 else 0.0
            accuracy = 1.0 - min(1.0, normalized_error)
        else:
            accuracy = 0.0
        
        prediction_entry = {
            "timestamp": time.time(),
            "prediction_confidence": prediction_confidence,
            "prediction_accuracy": accuracy,
            "similar_experiences_used": similar_experiences_used,
            "prediction_error": 1.0 - accuracy
        }
        
        self.prediction_history.append(prediction_entry)
    
    def log_emergence_event(self, event_type: str, description: str, 
                          evidence: Dict, significance: float):
        """
        Log detected emergence event.
        
        Args:
            event_type: Type of emergent behavior detected
            description: Human-readable description
            evidence: Data supporting the emergence claim
            significance: Significance score (0.0-1.0)
        """
        if self.async_logger and self.enable_async:
            # Async logging - create LoggableObject quickly
            loggable_event = LoggableEmergenceEvent(
                event_type=event_type,
                description=description,
                evidence=evidence,
                significance=significance,
                experience_count=self.total_experiences_logged,
                log_level=LogLevel.VITAL
            )
            
            success = self.async_logger.please_log(loggable_event)
            if not success:
                self._log_emergence_sync(event_type, description, evidence, significance)
        else:
            self._log_emergence_sync(event_type, description, evidence, significance)
        
        # Always add to memory for pattern detection
        emergence_entry = {
            "timestamp": time.time(),
            "session_time": time.time() - self.session_start,
            "event_type": event_type,
            "description": description,
            "evidence": evidence,
            "significance": significance,
            "experience_count": self.total_experiences_logged
        }
        self.emergence_events.append(emergence_entry)
        
        print(f"Emergence detected: {event_type} - {description} (significance: {significance:.3f})")
    
    def _log_emergence_sync(self, event_type: str, description: str, evidence: Dict, significance: float):
        """Synchronous emergence event logging (fallback)."""
        timestamp = time.time()
        
        emergence_entry = {
            "timestamp": timestamp,
            "session_time": timestamp - self.session_start,
            "event_type": event_type,
            "description": description,
            "evidence": evidence,
            "significance": significance,
            "experience_count": self.total_experiences_logged
        }
        
        # Write to log file
        with open(self.emergence_log, 'a') as f:
            f.write(json.dumps(emergence_entry) + '\n')
    
    def _compute_change_magnitude(self, before: Dict, after: Dict) -> float:
        """Compute magnitude of change between before/after states."""
        try:
            # Simple heuristic - count significant numerical changes
            changes = 0
            total_comparisons = 0
            
            for key in before:
                if key in after:
                    total_comparisons += 1
                    if isinstance(before[key], (int, float)) and isinstance(after[key], (int, float)):
                        if abs(before[key] - after[key]) > 0.01:  # Threshold for significant change
                            changes += 1
            
            return changes / max(1, total_comparisons)
        except:
            return 0.0
    
    def _compute_similarity_evolution(self, before: Dict, after: Dict) -> Dict:
        """Compute similarity function evolution metrics."""
        evolution = {}
        
        # Track key evolution metrics
        if "adaptations_performed" in before and "adaptations_performed" in after:
            evolution["new_adaptations"] = after["adaptations_performed"] - before["adaptations_performed"]
        
        if "similarity_success_correlation" in before and "similarity_success_correlation" in after:
            evolution["correlation_change"] = after["similarity_success_correlation"] - before["similarity_success_correlation"]
        
        if "feature_weight_variance" in before and "feature_weight_variance" in after:
            evolution["specialization_change"] = after["feature_weight_variance"] - before["feature_weight_variance"]
        
        return evolution
    
    def _detect_emergence_patterns(self, current_state: Dict):
        """Detect emergent patterns in brain behavior."""
        
        if len(self.recent_states) < 10:
            return  # Need sufficient history
        
        # Pattern 1: Sudden improvement in prediction accuracy
        if len(self.prediction_history) >= 20:
            recent_accuracy = [p["prediction_accuracy"] for p in list(self.prediction_history)[-10:]]
            older_accuracy = [p["prediction_accuracy"] for p in list(self.prediction_history)[-20:-10]]
            
            if len(recent_accuracy) > 0 and len(older_accuracy) > 0:
                recent_avg = np.mean(recent_accuracy)
                older_avg = np.mean(older_accuracy)
                improvement = recent_avg - older_avg
                
                if improvement > 0.2:  # Significant improvement threshold
                    self.log_emergence_event(
                        "prediction_breakthrough",
                        f"Sudden prediction accuracy improvement: {improvement:.3f}",
                        {"recent_accuracy": recent_avg, "older_accuracy": older_avg},
                        min(1.0, improvement * 2)
                    )
        
        # Pattern 2: Adaptation frequency changes
        if len(self.recent_adaptations) >= 5:
            recent_adaptations = list(self.recent_adaptations)[-5:]
            adaptation_times = [a["timestamp"] for a in recent_adaptations]
            time_span = adaptation_times[-1] - adaptation_times[0]
            
            if time_span > 0:
                adaptation_rate = len(recent_adaptations) / time_span
                
                # Log high adaptation activity
                if adaptation_rate > 0.1:  # More than 0.1 adaptations per second
                    self.log_emergence_event(
                        "rapid_adaptation_phase",
                        f"High adaptation activity: {adaptation_rate:.3f} adaptations/sec",
                        {"adaptation_rate": adaptation_rate, "time_span": time_span},
                        min(1.0, adaptation_rate * 5)
                    )
    
    def generate_session_report(self) -> Dict:
        """Generate comprehensive session analysis report."""
        session_end = time.time()
        session_duration = session_end - self.session_start
        
        # Compute session statistics
        report = {
            "session_summary": {
                "session_name": self.session_name,
                "duration_seconds": session_duration,
                "duration_hours": session_duration / 3600,
                "total_experiences": self.total_experiences_logged,
                "total_adaptations": self.total_adaptations_logged,
                "experiences_per_hour": self.total_experiences_logged / (session_duration / 3600),
                "adaptations_per_hour": self.total_adaptations_logged / (session_duration / 3600)
            },
            "adaptation_analysis": {
                "adaptation_triggers": dict(self.adaptation_triggers),
                "most_common_trigger": max(self.adaptation_triggers.items(), key=lambda x: x[1])[0] if self.adaptation_triggers else None
            },
            "emergence_analysis": {
                "total_emergence_events": len(self.emergence_events),
                "emergence_events": self.emergence_events,
                "high_significance_events": [e for e in self.emergence_events if e["significance"] > 0.7]
            },
            "prediction_analysis": self._analyze_prediction_trends(),
            "generated_at": session_end
        }
        
        # Write final report
        with open(self.session_summary, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_prediction_trends(self) -> Dict:
        """Analyze prediction accuracy trends."""
        if len(self.prediction_history) < 10:
            return {"insufficient_data": True}
        
        predictions = list(self.prediction_history)
        accuracies = [p["prediction_accuracy"] for p in predictions]
        confidences = [p["prediction_confidence"] for p in predictions]
        
        # Compute trends
        early_accuracy = np.mean(accuracies[:len(accuracies)//3]) if len(accuracies) >= 3 else 0
        late_accuracy = np.mean(accuracies[-len(accuracies)//3:]) if len(accuracies) >= 3 else 0
        
        return {
            "total_predictions": len(predictions),
            "overall_accuracy": np.mean(accuracies),
            "overall_confidence": np.mean(confidences),
            "early_accuracy": early_accuracy,
            "late_accuracy": late_accuracy,
            "learning_improvement": late_accuracy - early_accuracy,
            "accuracy_variance": np.var(accuracies),
            "confidence_variance": np.var(confidences)
        }
    
    def close_session(self):
        """Close logging session and generate final report."""
        print(f"\nClosing logging session: {self.session_name}")
        
        # Shutdown async logger if enabled
        if self.async_logger:
            stats = self.async_logger.get_stats()
            print(f"Async logging stats: {stats['total_logs_written']} written, {stats['total_logs_dropped']} dropped")
            self.async_logger.shutdown()
        
        report = self.generate_session_report()
        
        print(f"Session duration: {report['session_summary']['duration_hours']:.2f} hours")
        print(f"Total experiences logged: {report['session_summary']['total_experiences']}")
        print(f"Total adaptations logged: {report['session_summary']['total_adaptations']}")
        print(f"Emergence events detected: {report['emergence_analysis']['total_emergence_events']}")
        
        if report['emergence_analysis']['high_significance_events']:
            print("High-significance emergence events:")
            for event in report['emergence_analysis']['high_significance_events']:
                print(f"  - {event['event_type']}: {event['description']}")
        
        print(f"Final report saved to: {self.session_summary}")
        
        return report