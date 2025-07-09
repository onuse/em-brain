# 📊 Decision Logging Guide

This guide explains how to enable comprehensive decision logging for analyzing robot brain behavior.

## 🚀 Quick Start

### 1. Enable Logging
```bash
python enable_decision_logging.py patch
```

### 2. Run Demo with Logging
```bash
python demo_robot_brain.py
```

### 3. Analyze Results
```bash
python enable_decision_logging.py analyze
```

### 4. Disable Logging (Optional)
```bash
python enable_decision_logging.py unpatch
```

## 🎛️ Logging Control

### Check Current Status
```bash
python enable_decision_logging.py status
```

### Enable/Disable Logging
```bash
python enable_decision_logging.py patch    # Enable logging
python enable_decision_logging.py unpatch  # Disable logging
```

### Temporary Disable (Environment Variable)
```bash
# Disable logging for one run without unpatching
DISABLE_DECISION_LOGGING=true python demo_robot_brain.py

# Or export for multiple runs
export DISABLE_DECISION_LOGGING=true
python demo_robot_brain.py  # No logging
unset DISABLE_DECISION_LOGGING
```

## 📋 What Gets Logged

Every robot decision includes:
- **Step count** and timing information
- **Drive weights** and contributions
- **Chosen action** and reasoning
- **Robot state** (health, energy, position)
- **Brain statistics** (nodes, prediction errors)
- **Problem detection** (oscillation, stagnation, etc.)

## 🔍 Log Analysis Features

### Automatic Problem Detection
- **Oscillation patterns** - Repetitive drive switching
- **Drive imbalances** - One drive dominating too much
- **Learning stagnation** - Prediction errors not improving
- **Homeostatic rest** - When robot achieves natural rest

### Behavioral Analysis
- **Drive behavior over time** - How each drive evolves
- **Learning progression** - Prediction accuracy improvements
- **Spatial exploration** - Position and movement patterns
- **Decision quality** - Confidence and reasoning trends

## 📁 Log Files

Logs are saved in `decision_logs/` directory:
- `session_YYYYMMDD_HHMMSS.jsonl` - Raw decision data (JSON Lines format)
- `session_YYYYMMDD_HHMMSS_summary.json` - Session summary
- `session_YYYYMMDD_HHMMSS.analysis.json` - Detailed analysis report

## 🔧 Manual Analysis

### Load and Analyze Specific Log
```python
from core.log_analyzer import analyze_log_file

report = analyze_log_file("decision_logs/session_20250109_143022.jsonl")
```

### Custom Analysis
```python
from core.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer("decision_logs/session_20250109_143022.jsonl")

# Check for oscillation
oscillation = analyzer.analyze_oscillation()
print(f"Oscillation detected: {oscillation['oscillation_detected']}")

# Analyze drive behavior
drives = analyzer.analyze_drive_behavior()
for drive_name, data in drives.items():
    print(f"{drive_name}: min={data['min_weight']:.3f}, max={data['max_weight']:.3f}")

# Check homeostatic rest
rest = analyzer.analyze_homeostatic_rest()
print(f"Rest achieved: {rest['rest_achieved']} ({rest['rest_events_count']} times)")
```

## 📊 Example Analysis Output

```
🧠 DECISION LOG ANALYSIS SUMMARY
====================================
📊 Dataset: 1247 decisions
🎯 Steps: 1 - 1247

🔍 Key Insights:
  🔄 Oscillation detected: 3 patterns found
     Most involved drives: Curiosity, Exploration
  ✅ Drives that can reach zero: Exploration, Survival
  ⚠️  Drives stuck above 0.1: Curiosity
  ❌ No homeostatic rest achieved (min pressure: 0.087)

📈 Learning Progress:
  Initial error: 0.842
  Final error: 0.156
  Node count: 1247

🎮 Drive Behavior:
  Curiosity: avg=0.234, min=0.098, max=0.456
  Exploration: avg=0.123, min=0.000, max=0.287
  Survival: avg=0.045, min=0.000, max=0.234

⚠️  Problem Periods: 23
  Step 234: potential_oscillation
  Step 456: extreme_drive_imbalance
  Step 789: learning_stagnation
```

## 🎯 Debugging Common Issues

### Robot Oscillating
Look for:
- `oscillation_detected: true` in analysis
- Short repetitive patterns in drive dominance
- High drive switching frequency

### Robot Stuck
Look for:
- `learning_stagnation` flags
- Drives that can't reach zero (`can_reach_zero: false`)
- Very low total drive pressure without rest

### Poor Learning
Look for:
- `learning_plateau_detected: true`
- No improvement in prediction errors
- High variance in drive weights

## 📝 Integration with Your Code

### Manual Logging
```python
from core.decision_logger import start_decision_logging, stop_decision_logging, log_brain_decision

# Start logging
logger = start_decision_logging("my_session")

# Log decisions (done automatically in patched demo)
log_brain_decision(motivation_result, drive_context, brain_stats, step_count)

# Stop and analyze
summary = stop_decision_logging()
```

### Custom Log Analysis
```python
from core.log_analyzer import LogAnalyzer

analyzer = LogAnalyzer("my_log.jsonl")
report = analyzer.generate_comprehensive_report()

# Check specific issues
if report['oscillation_analysis']['oscillation_detected']:
    print("Found oscillation patterns!")
    
if not report['homeostatic_rest']['rest_achieved']:
    print("Robot never achieved rest")
```

## 🔧 Advanced Usage

### Filter Decisions
```python
# Load raw data
analyzer = LogAnalyzer("my_log.jsonl")

# Filter by step range
recent_decisions = [d for d in analyzer.data if d['step_count'] > 1000]

# Filter by drive dominance
curiosity_decisions = [d for d in analyzer.data if d['dominant_drive'] == 'Curiosity']

# Filter by problem periods
problem_decisions = [d for d in analyzer.data if d['analysis']['flags']]
```

### Custom Metrics
```python
# Calculate custom metrics
def calculate_exploration_efficiency(data):
    positions = [(d['robot_position'][0], d['robot_position'][1]) for d in data]
    unique_positions = len(set(positions))
    return unique_positions / len(positions)

efficiency = calculate_exploration_efficiency(analyzer.data)
```

This logging system will help you identify exactly what's causing the undesirable behavior in your robot!