# Brain Visualization Tools

Real-time monitoring clients that connect to the brain's monitoring socket (port 9998).

## Overview

These visualization tools are standalone applications that can connect to any running brain server. They're designed to be "third party" clients - you can start/stop them at any time without affecting the brain or validation experiments.

## Available Tools

### 1. Real-time Brain Monitor (GUI)
`realtime_brain_monitor.py` - Full-featured graphical interface with live plots

**Features:**
- Real-time plots of key metrics (energy, confidence, evolution strength)
- Behavior state visualization
- Session selection for multiple brains
- Historical data tracking

**Usage:**
```bash
python3 tools/visualization/realtime_brain_monitor.py
```

### 2. Terminal Brain Monitor
`terminal_brain_monitor.py` - Lightweight terminal-based monitor for SSH/headless use

**Features:**
- ASCII sparkline graphs
- Color-coded status indicators
- Keyboard navigation
- Minimal resource usage

**Usage:**
```bash
python3 tools/visualization/terminal_brain_monitor.py

# Connect to remote server
python3 tools/visualization/terminal_brain_monitor.py --host 192.168.1.100
```

**Controls:**
- `c` - Connect to monitoring server
- `d` - Disconnect
- `r` - Refresh session list
- `1-9` - Select session by number
- `q` - Quit

## How It Works

1. Start any brain server or validation experiment
2. Launch a visualization tool
3. Click "Connect" (GUI) or press 'c' (terminal)
4. Select the active brain session
5. Watch real-time telemetry data

The monitoring server (port 9998) supports multiple simultaneous clients, so you can have several visualization tools running at once.

## Requirements

- Python 3.7+
- GUI Monitor: tkinter, matplotlib, numpy
- Terminal Monitor: curses (built-in)