"""
Decision Logger - Comprehensive logging system for analyzing robot brain behavior.

This logger captures every decision made by the robot with full context,
making it easy to analyze problematic behaviors, drive interactions,
and system performance over time.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class DecisionLogger:
    """
    Logs every robot decision with complete context for analysis.
    
    Creates detailed logs that can be analyzed to understand:
    - Why the robot made specific decisions
    - How drives interact over time
    - When and why problems occur
    - System performance trends
    """
    
    def __init__(self, log_dir: str = "decision_logs", session_name: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create session-specific log file
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.log_file = self.log_dir / f"{session_name}.jsonl"  # JSON Lines format
        self.summary_file = self.log_dir / f"{session_name}_summary.json"
        
        # Session tracking
        self.session_start_time = time.time()
        self.decision_count = 0
        self.session_metadata = {
            "session_name": session_name,
            "start_time": datetime.now().isoformat(),
            "log_file": str(self.log_file),
            "summary_file": str(self.summary_file)
        }
        
        # Performance tracking
        self.drive_dominance_history = []
        self.total_pressure_history = []
        self.homeostatic_rest_events = []
        self.problem_indicators = []
        
        print(f"📊 Decision logging started: {self.log_file}")
    
    def log_decision(self, decision_context: Dict[str, Any]):
        """
        Log a complete decision with full context.
        
        Args:
            decision_context: Dictionary containing all decision information
        """
        self.decision_count += 1
        timestamp = time.time()
        
        # Create comprehensive log entry
        log_entry = {
            # Basic metadata
            "decision_id": self.decision_count,
            "timestamp": timestamp,
            "time_since_start": timestamp - self.session_start_time,
            "datetime": datetime.now().isoformat(),
            
            # Copy all provided context
            **decision_context,
            
            # Add analysis flags
            "analysis": self._analyze_decision(decision_context)
        }
        
        # Write to log file (JSON Lines format for easy streaming)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(self._make_serializable(log_entry)) + '\n')
        
        # Update session tracking
        self._update_session_tracking(log_entry)
        
        # Check for problems and log them
        self._check_for_problems(log_entry)
    
    def log_brain_decision(self, motivation_result, drive_context, brain_stats: Dict[str, Any], 
                          step_count: int, frame_info: Dict[str, Any] = None):
        """
        Convenience method for logging brain interface decisions.
        
        Args:
            motivation_result: MotivationResult from motivation system
            drive_context: DriveContext used for decision
            brain_stats: Current brain statistics
            step_count: Current simulation step
            frame_info: Optional frame/performance information
        """
        # Extract drive weights and pain/pleasure information
        drive_weights = {}
        drive_contributions = getattr(motivation_result, 'drive_contributions', {})
        drive_pain_pleasure_predictions = {}
        
        # Try to get current drive weights and pain/pleasure predictions if available
        if hasattr(motivation_result, 'motivation_system'):
            for drive_name, drive in motivation_result.motivation_system.drives.items():
                drive_weights[drive_name] = getattr(drive, 'current_weight', 0.0)
                
                # Get drive-specific pain/pleasure predictions if available
                if hasattr(drive, 'evaluate_action_pain_pleasure'):
                    try:
                        pain, pleasure = drive.evaluate_action_pain_pleasure(
                            getattr(motivation_result, 'chosen_action', {}), drive_context
                        )
                        drive_pain_pleasure_predictions[drive_name] = {
                            'expected_pain': pain,
                            'expected_pleasure': pleasure,
                            'net_valence': pleasure + pain  # pain is negative
                        }
                    except Exception:
                        # If evaluation fails, skip this drive
                        pass
        
        # Extract mood and satisfaction data
        mood_data = {}
        if hasattr(motivation_result, 'motivation_system'):
            try:
                mood_data = motivation_result.motivation_system.calculate_robot_mood(drive_context)
            except Exception as e:
                # If mood calculation fails, continue without mood data
                mood_data = {"error": f"Failed to calculate mood: {str(e)}"}
        
        decision_context = {
            # Step and timing info
            "step_count": step_count,
            "frame_info": frame_info or {},
            
            # Decision results
            "chosen_action": getattr(motivation_result, 'chosen_action', {}),
            "total_score": getattr(motivation_result, 'total_score', 0.0),
            "dominant_drive": getattr(motivation_result, 'dominant_drive', 'unknown'),
            "confidence": getattr(motivation_result, 'confidence', 0.0),
            "reasoning": getattr(motivation_result, 'reasoning', ''),
            "urgency": getattr(motivation_result, 'urgency', 0.0),
            
            # Drive information
            "drive_contributions": drive_contributions,
            "drive_weights": drive_weights,
            "total_drive_pressure": sum(drive_weights.values()) if drive_weights else 0.0,
            
            # Mood and satisfaction data
            "mood_data": mood_data,
            "overall_satisfaction": mood_data.get('overall_satisfaction', 0.0),
            "overall_urgency": mood_data.get('overall_urgency', 0.0),
            "overall_confidence": mood_data.get('overall_confidence', 0.0),
            "mood_descriptor": mood_data.get('mood_descriptor', 'unknown'),
            "dominant_emotion": mood_data.get('dominant_emotion', 'unknown'),
            "drive_specific_moods": mood_data.get('drive_moods', {}),
            "drive_pain_pleasure_predictions": drive_pain_pleasure_predictions,
            
            # Robot state
            "robot_health": getattr(drive_context, 'robot_health', 0.0),
            "robot_energy": getattr(drive_context, 'robot_energy', 0.0),
            "robot_position": getattr(drive_context, 'robot_position', (0, 0)),
            "robot_orientation": getattr(drive_context, 'robot_orientation', 0),
            "threat_level": getattr(drive_context, 'threat_level', 'unknown'),
            
            # Learning state
            "prediction_errors": getattr(drive_context, 'prediction_errors', []),
            "recent_prediction_error": getattr(drive_context, 'prediction_errors', [0])[-1] if getattr(drive_context, 'prediction_errors', []) else 0.0,
            "time_since_last_food": getattr(drive_context, 'time_since_last_food', 0),
            "time_since_last_damage": getattr(drive_context, 'time_since_last_damage', 0),
            
            # Brain statistics
            "brain_stats": brain_stats,
            "total_nodes": brain_stats.get('total_nodes', 0),
            "average_prediction_error": brain_stats.get('graph_stats', {}).get('avg_prediction_error', 0.0),
            
            # Memory and consolidation
            "consolidation_stats": brain_stats.get('consolidation_stats', {}),
            "novelty_stats": brain_stats.get('novelty_detection_stats', {}),
        }
        
        self.log_decision(decision_context)
    
    def _analyze_decision(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decision for potential issues and interesting patterns."""
        analysis = {
            "flags": [],
            "indicators": {}
        }
        
        # Check for oscillation patterns
        if len(self.drive_dominance_history) >= 10:
            recent_drives = self.drive_dominance_history[-10:]
            if len(set(recent_drives)) <= 2 and len(recent_drives) == 10:
                analysis["flags"].append("potential_oscillation")
        
        # Check for stuck behavior
        total_score = decision_context.get('total_score', 0.0)
        if total_score < 0.1:
            analysis["flags"].append("very_low_motivation")
        elif total_score > 0.9:
            analysis["flags"].append("very_high_urgency")
        
        # Check for drive balance issues
        drive_weights = decision_context.get('drive_weights', {})
        if drive_weights:
            max_weight = max(drive_weights.values())
            min_weight = min(drive_weights.values())
            if max_weight > 0.8 and min_weight < 0.05:
                analysis["flags"].append("extreme_drive_imbalance")
        
        # Check for mood/satisfaction issues
        satisfaction = decision_context.get('overall_satisfaction', 0.0)
        urgency = decision_context.get('overall_urgency', 0.0)
        if satisfaction < -0.6:
            analysis["flags"].append("very_low_satisfaction")
        elif satisfaction > 0.8:
            analysis["flags"].append("very_high_satisfaction")
        
        if urgency > 0.8:
            analysis["flags"].append("high_urgency_state")
        elif urgency < 0.1:
            analysis["flags"].append("low_urgency_state")
        
        # Check for learning stagnation
        prediction_errors = decision_context.get('prediction_errors', [])
        if len(prediction_errors) >= 20:
            recent_variance = self._calculate_variance(prediction_errors[-20:])
            if recent_variance < 0.001:
                analysis["flags"].append("learning_stagnation")
        
        # Check for homeostatic rest
        total_pressure = decision_context.get('total_drive_pressure', 1.0)
        if total_pressure < 0.1:
            analysis["flags"].append("near_homeostatic_rest")
        
        # Performance indicators
        analysis["indicators"] = {
            "decision_complexity": len(drive_weights),
            "drive_diversity": len([w for w in drive_weights.values() if w > 0.1]) if drive_weights else 0,
            "learning_activity": len(prediction_errors) if prediction_errors else 0,
            "robot_wellness": (decision_context.get('robot_health', 0) + decision_context.get('robot_energy', 0)) / 2.0,
            "emotional_state": decision_context.get('mood_descriptor', 'unknown'),
            "satisfaction_level": decision_context.get('overall_satisfaction', 0.0),
            "urgency_level": decision_context.get('overall_urgency', 0.0)
        }
        
        return analysis
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _make_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._make_serializable(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            # Convert objects with __dict__ to dictionaries
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert everything else to string
            return str(obj)
    
    def _update_session_tracking(self, log_entry: Dict[str, Any]):
        """Update session-level tracking information."""
        # Track drive dominance
        dominant_drive = log_entry.get('dominant_drive', 'unknown')
        self.drive_dominance_history.append(dominant_drive)
        if len(self.drive_dominance_history) > 100:
            self.drive_dominance_history = self.drive_dominance_history[-50:]
        
        # Track total pressure
        total_pressure = log_entry.get('total_drive_pressure', 0.0)
        self.total_pressure_history.append(total_pressure)
        if len(self.total_pressure_history) > 100:
            self.total_pressure_history = self.total_pressure_history[-50:]
        
        # Track homeostatic rest events
        if 'near_homeostatic_rest' in log_entry.get('analysis', {}).get('flags', []):
            self.homeostatic_rest_events.append(log_entry['step_count'])
    
    def _check_for_problems(self, log_entry: Dict[str, Any]):
        """Check for and log potential problems."""
        flags = log_entry.get('analysis', {}).get('flags', [])
        
        for flag in flags:
            if flag in ['potential_oscillation', 'extreme_drive_imbalance', 'learning_stagnation']:
                problem_entry = {
                    "step_count": log_entry['step_count'],
                    "problem_type": flag,
                    "context": {
                        "dominant_drive": log_entry.get('dominant_drive'),
                        "total_score": log_entry.get('total_score'),
                        "drive_weights": log_entry.get('drive_weights'),
                        "reasoning": log_entry.get('reasoning')
                    }
                }
                self.problem_indicators.append(problem_entry)
    
    def generate_session_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive session summary."""
        session_duration = time.time() - self.session_start_time
        
        # Drive dominance analysis
        drive_counts = {}
        for drive in self.drive_dominance_history:
            drive_counts[drive] = drive_counts.get(drive, 0) + 1
        
        # Problem analysis
        problem_counts = {}
        for problem in self.problem_indicators:
            ptype = problem['problem_type']
            problem_counts[ptype] = problem_counts.get(ptype, 0) + 1
        
        summary = {
            "session_metadata": self.session_metadata,
            "session_duration_seconds": session_duration,
            "total_decisions": self.decision_count,
            "decisions_per_second": self.decision_count / session_duration if session_duration > 0 else 0,
            
            "drive_analysis": {
                "dominance_counts": drive_counts,
                "most_dominant_drive": max(drive_counts.keys(), key=lambda k: drive_counts[k]) if drive_counts else None,
                "drive_switches": len([i for i in range(1, len(self.drive_dominance_history)) 
                                     if self.drive_dominance_history[i] != self.drive_dominance_history[i-1]])
            },
            
            "pressure_analysis": {
                "average_pressure": sum(self.total_pressure_history) / len(self.total_pressure_history) if self.total_pressure_history else 0,
                "min_pressure": min(self.total_pressure_history) if self.total_pressure_history else 0,
                "max_pressure": max(self.total_pressure_history) if self.total_pressure_history else 0,
                "rest_events": len(self.homeostatic_rest_events)
            },
            
            "problem_analysis": {
                "problem_counts": problem_counts,
                "total_problems": len(self.problem_indicators),
                "problem_rate": len(self.problem_indicators) / self.decision_count if self.decision_count > 0 else 0,
                "recent_problems": self.problem_indicators[-5:] if self.problem_indicators else []
            },
            
            "recommendations": self._generate_recommendations(drive_counts, problem_counts)
        }
        
        # Save summary to file
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _generate_recommendations(self, drive_counts: Dict[str, int], problem_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on session analysis."""
        recommendations = []
        
        if 'potential_oscillation' in problem_counts:
            recommendations.append("High oscillation detected - consider adjusting drive update rates or thresholds")
        
        if 'extreme_drive_imbalance' in problem_counts:
            recommendations.append("Extreme drive imbalances detected - review drive weight calculations")
        
        if 'learning_stagnation' in problem_counts:
            recommendations.append("Learning stagnation detected - robot may need new challenges or environment changes")
        
        if drive_counts and max(drive_counts.values()) / sum(drive_counts.values()) > 0.8:
            dominant_drive = max(drive_counts.keys(), key=lambda k: drive_counts[k])
            recommendations.append(f"{dominant_drive} drive is dominating - check balance with other drives")
        
        if not recommendations:
            recommendations.append("No major issues detected - system appears to be functioning well")
        
        return recommendations
    
    def close_session(self):
        """Close the logging session and generate final summary."""
        summary = self.generate_session_summary()
        
        print(f"📊 Decision logging completed:")
        print(f"   Total decisions: {self.decision_count}")
        print(f"   Session duration: {summary['session_duration_seconds']:.1f}s")
        print(f"   Log file: {self.log_file}")
        print(f"   Summary file: {self.summary_file}")
        
        if summary['problem_analysis']['total_problems'] > 0:
            print(f"   ⚠️  Problems detected: {summary['problem_analysis']['total_problems']}")
        else:
            print(f"   ✅ No major problems detected")
        
        return summary


# Global logger instance for easy access
_global_logger: Optional[DecisionLogger] = None


def get_decision_logger() -> Optional[DecisionLogger]:
    """Get the global decision logger instance."""
    return _global_logger


def start_decision_logging(session_name: Optional[str] = None, log_dir: str = "decision_logs") -> DecisionLogger:
    """Start decision logging for the current session."""
    global _global_logger
    _global_logger = DecisionLogger(log_dir=log_dir, session_name=session_name)
    return _global_logger


def stop_decision_logging() -> Optional[Dict[str, Any]]:
    """Stop decision logging and return session summary."""
    global _global_logger
    if _global_logger:
        summary = _global_logger.close_session()
        _global_logger = None
        return summary
    return None


def log_brain_decision(motivation_result, drive_context, brain_stats: Dict[str, Any], 
                      step_count: int, frame_info: Dict[str, Any] = None):
    """Log a brain decision if logging is active."""
    global _global_logger
    
    # Check if logging is disabled via environment variable
    if os.getenv('DISABLE_DECISION_LOGGING', '').lower() in ['true', '1', 'yes']:
        return
    
    if _global_logger:
        _global_logger.log_brain_decision(
            motivation_result, drive_context, brain_stats, step_count, frame_info
        )