"""
Log Analyzer - Tool for analyzing decision logs to identify patterns and problems.

This tool can analyze decision logs to help understand:
- Why robots get stuck or oscillate
- How drives interact over time
- When performance degrades
- System behavior patterns
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter

# Optional dependencies for advanced analysis
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class LogAnalyzer:
    """
    Analyzes decision logs to identify patterns, problems, and insights.
    """
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.data = []
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load decision log data from file."""
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
        
        print(f"📊 Loading decision log: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    self.data.append(entry)
                except json.JSONDecodeError:
                    continue
        
        if not self.data:
            raise ValueError("No valid log entries found")
        
        # Convert to DataFrame for analysis (if pandas available)
        if HAS_PANDAS:
            self.df = pd.DataFrame(self.data)
            print(f"   Loaded {len(self.data)} decisions")
            print(f"   Time range: {self.df['step_count'].min()} - {self.df['step_count'].max()} steps")
        else:
            self.df = None
            print(f"   Loaded {len(self.data)} decisions")
            if self.data:
                steps = [d['step_count'] for d in self.data]
                print(f"   Time range: {min(steps)} - {max(steps)} steps")
    
    def analyze_oscillation(self, window_size: int = 20) -> Dict[str, Any]:
        """Analyze for oscillation patterns in drive dominance."""
        if len(self.data) < window_size:
            return {"oscillation_detected": False, "reason": "Insufficient data"}
        
        # Look for repetitive patterns in drive dominance
        dominance_sequence = [entry.get('dominant_drive', 'unknown') for entry in self.data]
        
        oscillation_patterns = []
        for i in range(len(dominance_sequence) - window_size + 1):
            window = dominance_sequence[i:i + window_size]
            unique_drives = set(window)
            
            if len(unique_drives) <= 3:  # Only 2-3 drives switching
                pattern_length = self._find_pattern_length(window)
                if pattern_length and pattern_length <= 6:  # Short repeating pattern
                    oscillation_patterns.append({
                        "start_step": self.data[i]['step_count'],
                        "pattern_length": pattern_length,
                        "drives_involved": list(unique_drives),
                        "pattern": window[:pattern_length * 2]
                    })
        
        return {
            "oscillation_detected": len(oscillation_patterns) > 0,
            "oscillation_patterns": oscillation_patterns,
            "total_patterns": len(oscillation_patterns),
            "analysis": self._analyze_oscillation_patterns(oscillation_patterns)
        }
    
    def _find_pattern_length(self, sequence: List[str]) -> Optional[int]:
        """Find the length of a repeating pattern in a sequence."""
        for pattern_len in range(2, min(8, len(sequence) // 2)):
            pattern = sequence[:pattern_len]
            repeats = len(sequence) // pattern_len
            
            if repeats >= 2:
                reconstructed = (pattern * repeats)[:len(sequence)]
                if reconstructed == sequence:
                    return pattern_len
        return None
    
    def _analyze_oscillation_patterns(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze oscillation patterns for insights."""
        if not patterns:
            return {"no_patterns": True}
        
        # Find most common pattern length
        pattern_lengths = [p['pattern_length'] for p in patterns]
        most_common_length = Counter(pattern_lengths).most_common(1)[0][0]
        
        # Find drives most involved in oscillations
        all_drives = []
        for p in patterns:
            all_drives.extend(p['drives_involved'])
        drive_involvement = Counter(all_drives)
        
        return {
            "most_common_pattern_length": most_common_length,
            "drives_most_involved": drive_involvement.most_common(3),
            "oscillation_intensity": len(patterns) / len(self.data),
            "longest_oscillation": max(patterns, key=lambda p: p['pattern_length'])
        }
    
    def analyze_drive_behavior(self) -> Dict[str, Any]:
        """Analyze how drives behave over time."""
        drive_data = defaultdict(list)
        
        for entry in self.data:
            step = entry['step_count']
            drive_weights = entry.get('drive_weights', {})
            
            for drive_name, weight in drive_weights.items():
                drive_data[drive_name].append({
                    'step': step,
                    'weight': weight,
                    'dominant': entry.get('dominant_drive') == drive_name
                })
        
        analysis = {}
        for drive_name, data in drive_data.items():
            weights = [d['weight'] for d in data]
            dominance_count = sum(1 for d in data if d['dominant'])
            
            analysis[drive_name] = {
                "average_weight": np.mean(weights),
                "weight_std": np.std(weights),
                "min_weight": np.min(weights),
                "max_weight": np.max(weights),
                "dominance_count": dominance_count,
                "dominance_rate": dominance_count / len(data) if data else 0,
                "can_reach_zero": np.min(weights) < 0.01,
                "weight_stability": 1.0 / (1.0 + np.std(weights))  # Higher = more stable
            }
        
        return analysis
    
    def analyze_learning_progression(self) -> Dict[str, Any]:
        """Analyze how learning progresses over time."""
        steps = []
        errors = []
        node_counts = []
        
        for entry in self.data:
            steps.append(entry['step_count'])
            errors.append(entry.get('recent_prediction_error', 0.0))
            node_counts.append(entry.get('total_nodes', 0))
        
        # Calculate learning rate (improvement in prediction error)
        learning_rate = 0.0
        if len(errors) > 20:
            early_errors = np.mean(errors[:10])
            late_errors = np.mean(errors[-10:])
            learning_rate = (early_errors - late_errors) / len(errors)
        
        return {
            "initial_error": errors[0] if errors else 0,
            "final_error": errors[-1] if errors else 0,
            "average_error": np.mean(errors),
            "error_improvement": errors[0] - errors[-1] if len(errors) > 1 else 0,
            "learning_rate": learning_rate,
            "node_growth": node_counts[-1] - node_counts[0] if len(node_counts) > 1 else 0,
            "final_node_count": node_counts[-1] if node_counts else 0,
            "learning_plateau_detected": self._detect_learning_plateau(errors)
        }
    
    def _detect_learning_plateau(self, errors: List[float]) -> bool:
        """Detect if learning has plateaued."""
        if len(errors) < 50:
            return False
        
        recent_errors = errors[-50:]
        variance = np.var(recent_errors)
        return variance < 0.001  # Very low variance indicates plateau
    
    def analyze_homeostatic_rest(self) -> Dict[str, Any]:
        """Analyze homeostatic rest patterns."""
        rest_events = []
        total_pressures = []
        
        for entry in self.data:
            total_pressure = entry.get('total_drive_pressure', 0.0)
            total_pressures.append(total_pressure)
            
            if entry.get('dominant_drive') == 'homeostatic_rest':
                rest_events.append({
                    'step': entry['step_count'],
                    'total_pressure': total_pressure,
                    'reasoning': entry.get('reasoning', '')
                })
        
        return {
            "rest_events_count": len(rest_events),
            "rest_frequency": len(rest_events) / len(self.data) if self.data else 0,
            "average_pressure": np.mean(total_pressures),
            "min_pressure": np.min(total_pressures),
            "rest_threshold_estimate": np.mean([e['total_pressure'] for e in rest_events]) if rest_events else 0,
            "rest_achieved": len(rest_events) > 0,
            "pressure_trend": self._calculate_pressure_trend(total_pressures)
        }
    
    def _calculate_pressure_trend(self, pressures: List[float]) -> str:
        """Calculate if pressure is trending up, down, or stable."""
        if len(pressures) < 10:
            return "insufficient_data"
        
        early_avg = np.mean(pressures[:len(pressures)//3])
        late_avg = np.mean(pressures[-len(pressures)//3:])
        
        if late_avg < early_avg - 0.05:
            return "decreasing"
        elif late_avg > early_avg + 0.05:
            return "increasing"
        else:
            return "stable"
    
    def identify_problem_periods(self) -> List[Dict[str, Any]]:
        """Identify periods where the robot had problems."""
        problems = []
        
        for i, entry in enumerate(self.data):
            flags = entry.get('analysis', {}).get('flags', [])
            
            if any(flag in ['potential_oscillation', 'extreme_drive_imbalance', 'learning_stagnation'] for flag in flags):
                problems.append({
                    'step': entry['step_count'],
                    'problem_types': flags,
                    'dominant_drive': entry.get('dominant_drive'),
                    'total_score': entry.get('total_score'),
                    'reasoning': entry.get('reasoning'),
                    'drive_weights': entry.get('drive_weights', {}),
                    'context': {
                        'recent_error': entry.get('recent_prediction_error', 0),
                        'total_nodes': entry.get('total_nodes', 0),
                        'robot_health': entry.get('robot_health', 0),
                        'robot_energy': entry.get('robot_energy', 0)
                    }
                })
        
        return problems
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        print("📊 Generating comprehensive analysis report...")
        
        report = {
            "metadata": {
                "log_file": str(self.log_file),
                "total_decisions": len(self.data),
                "step_range": f"{min(d['step_count'] for d in self.data)} - {max(d['step_count'] for d in self.data)}",
                "analysis_timestamp": (pd.Timestamp.now() if HAS_PANDAS else __import__('datetime').datetime.now()).isoformat()
            },
            
            "oscillation_analysis": self.analyze_oscillation(),
            "drive_behavior": self.analyze_drive_behavior(),
            "learning_progression": self.analyze_learning_progression(),
            "homeostatic_rest": self.analyze_homeostatic_rest(),
            "problem_periods": self.identify_problem_periods(),
        }
        
        # Add summary insights
        report["insights"] = self._generate_insights(report)
        
        return report
    
    def _generate_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate key insights from the analysis."""
        insights = []
        
        # Oscillation insights
        if report["oscillation_analysis"]["oscillation_detected"]:
            patterns = report["oscillation_analysis"]["oscillation_patterns"]
            insights.append(f"🔄 Oscillation detected: {len(patterns)} patterns found")
            
            most_involved = report["oscillation_analysis"]["analysis"].get("drives_most_involved", [])
            if most_involved:
                insights.append(f"   Most involved drives: {', '.join(drive for drive, count in most_involved)}")
        
        # Drive behavior insights
        drive_behavior = report["drive_behavior"]
        zero_capable_drives = [name for name, data in drive_behavior.items() if data["can_reach_zero"]]
        if zero_capable_drives:
            insights.append(f"✅ Drives that can reach zero: {', '.join(zero_capable_drives)}")
        
        stuck_drives = [name for name, data in drive_behavior.items() if not data["can_reach_zero"] and data["min_weight"] > 0.1]
        if stuck_drives:
            insights.append(f"⚠️  Drives stuck above 0.1: {', '.join(stuck_drives)}")
        
        # Learning insights
        learning = report["learning_progression"]
        if learning["learning_plateau_detected"]:
            insights.append(f"📈 Learning plateau detected at error {learning['final_error']:.3f}")
        
        if learning["error_improvement"] > 0.1:
            insights.append(f"📚 Good learning progress: error improved by {learning['error_improvement']:.3f}")
        
        # Rest insights
        rest = report["homeostatic_rest"]
        if rest["rest_achieved"]:
            insights.append(f"😴 Homeostatic rest achieved {rest['rest_events_count']} times")
        else:
            insights.append(f"❌ No homeostatic rest achieved (min pressure: {rest['min_pressure']:.3f})")
        
        # Problem insights
        problems = report["problem_periods"]
        if problems:
            problem_types = Counter()
            for p in problems:
                problem_types.update(p['problem_types'])
            insights.append(f"🚨 {len(problems)} problem periods detected")
            for ptype, count in problem_types.most_common(3):
                insights.append(f"   {ptype}: {count} occurrences")
        
        return insights
    
    def save_report(self, report: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Save the analysis report to a file."""
        if output_file is None:
            output_file = self.log_file.with_suffix('.analysis.json')
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Analysis report saved: {output_file}")
        return str(output_file)
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the analysis."""
        print("\n" + "="*60)
        print("🧠 DECISION LOG ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Dataset: {report['metadata']['total_decisions']} decisions")
        print(f"🎯 Steps: {report['metadata']['step_range']}")
        
        print(f"\n🔍 Key Insights:")
        for insight in report['insights']:
            print(f"  {insight}")
        
        print(f"\n📈 Learning Progress:")
        learning = report['learning_progression']
        print(f"  Initial error: {learning['initial_error']:.3f}")
        print(f"  Final error: {learning['final_error']:.3f}")
        print(f"  Node count: {learning['final_node_count']}")
        
        print(f"\n🎮 Drive Behavior:")
        for drive_name, data in report['drive_behavior'].items():
            print(f"  {drive_name}: avg={data['average_weight']:.3f}, min={data['min_weight']:.3f}, max={data['max_weight']:.3f}")
        
        if report['problem_periods']:
            print(f"\n⚠️  Problem Periods: {len(report['problem_periods'])}")
            for problem in report['problem_periods'][:3]:  # Show first 3
                print(f"  Step {problem['step']}: {', '.join(problem['problem_types'])}")
        
        print("="*60)


def analyze_log_file(log_file: str, save_report: bool = True) -> Dict[str, Any]:
    """Convenience function to analyze a log file."""
    analyzer = LogAnalyzer(log_file)
    report = analyzer.generate_comprehensive_report()
    
    if save_report:
        analyzer.save_report(report)
    
    analyzer.print_summary(report)
    return report