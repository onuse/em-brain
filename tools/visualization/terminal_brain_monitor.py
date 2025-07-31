#!/usr/bin/env python3
"""
Terminal Brain Monitor

A lightweight, terminal-based visualization client for brain telemetry.
Perfect for SSH sessions or headless environments.
"""

import sys
import os
import socket
import json
import time
import threading
from collections import deque
from datetime import datetime
import curses


class TerminalBrainMonitor:
    """Terminal-based brain telemetry monitor"""
    
    def __init__(self, host='localhost', port=9998):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.running = False
        self.selected_session = None
        
        # Data buffers
        self.energy_history = deque(maxlen=50)
        self.confidence_history = deque(maxlen=50)
        self.evolution_history = deque(maxlen=50)
        self.cycle_time_history = deque(maxlen=50)
        
    def connect(self):
        """Connect to monitoring server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            
            # Receive welcome message
            response = self._receive_json()
            if response and response.get('status') == 'connected':
                self.connected = True
                return True
        except:
            pass
        return False
    
    def disconnect(self):
        """Disconnect from server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        self.socket = None
    
    def _send_json(self, data):
        """Send JSON data to server"""
        if not self.socket:
            return False
        try:
            message = json.dumps(data).encode('utf-8')
            self.socket.sendall(message + b'\n')
            return True
        except:
            return False
    
    def _receive_json(self):
        """Receive JSON data from server"""
        if not self.socket:
            return None
        try:
            data = b''
            while not data.endswith(b'\n'):
                chunk = self.socket.recv(4096)
                if not chunk:
                    return None
                data += chunk
            return json.loads(data.decode('utf-8').strip())
        except:
            return None
    
    def get_active_sessions(self):
        """Get list of active brain sessions"""
        if not self.connected:
            return []
        
        self._send_json({'command': 'active_brains'})
        response = self._receive_json()
        
        if response and response.get('status') == 'success':
            return response.get('active_brains', [])
        return []
    
    def get_brain_telemetry(self, session_id):
        """Get telemetry data for a specific session"""
        if not self.connected:
            return None
        
        self._send_json({
            'command': 'brain_stats',
            'session_id': session_id
        })
        
        response = self._receive_json()
        if response and response.get('status') == 'success':
            return response.get('data', {})
        return None
    
    def draw_bar(self, value, max_value, width=20, char='█'):
        """Draw a text-based progress bar"""
        if max_value == 0:
            filled = 0
        else:
            filled = int((value / max_value) * width)
        
        bar = char * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def draw_sparkline(self, data, width=30, height=5):
        """Draw a simple sparkline graph"""
        if not data:
            return [""] * height
        
        # Normalize data to height
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            normalized = [height // 2] * len(data)
        else:
            normalized = [int((v - min_val) / (max_val - min_val) * (height - 1)) 
                         for v in data]
        
        # Sample to fit width
        if len(normalized) > width:
            step = len(normalized) / width
            sampled = [normalized[int(i * step)] for i in range(width)]
        else:
            sampled = normalized
        
        # Create lines
        lines = []
        for h in range(height - 1, -1, -1):
            line = ""
            for v in sampled:
                if v >= h:
                    line += "▄"
                else:
                    line += " "
            lines.append(line)
        
        return lines
    
    def run_curses(self, stdscr):
        """Main curses loop"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(100) # Refresh timeout
        
        # Define colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        
        self.running = True
        last_update = 0
        telemetry = {}
        
        while self.running:
            height, width = stdscr.getmaxyx()
            stdscr.clear()
            
            # Header
            header = "🧠 Terminal Brain Monitor"
            stdscr.addstr(0, (width - len(header)) // 2, header, curses.A_BOLD)
            
            # Connection status
            y = 2
            if self.connected:
                status = f"Connected to {self.host}:{self.port}"
                stdscr.addstr(y, 2, "●", curses.color_pair(1))
                stdscr.addstr(y, 4, status, curses.color_pair(1))
            else:
                status = "Disconnected"
                stdscr.addstr(y, 2, "●", curses.color_pair(3))
                stdscr.addstr(y, 4, status, curses.color_pair(3))
            
            y += 2
            
            # Session info
            if self.selected_session:
                stdscr.addstr(y, 2, f"Session: {self.selected_session}", curses.A_BOLD)
                y += 2
                
                # Update telemetry
                current_time = time.time()
                if current_time - last_update > 1.0:  # Update every second
                    new_telemetry = self.get_brain_telemetry(self.selected_session)
                    if new_telemetry:
                        telemetry = new_telemetry
                        
                        # Update history
                        self.energy_history.append(telemetry.get('field_energy', 0))
                        self.confidence_history.append(telemetry.get('prediction_confidence', 0.5))
                        
                        evo_state = telemetry.get('evolution_state', {})
                        self.evolution_history.append(evo_state.get('self_modification_strength', 0.01))
                        
                        self.cycle_time_history.append(telemetry.get('cycle_time_ms', 0))
                    
                    last_update = current_time
                
                if telemetry:
                    # Field Energy
                    energy = telemetry.get('field_energy', 0)
                    stdscr.addstr(y, 2, "Field Energy:    ", curses.color_pair(4))
                    stdscr.addstr(y, 20, f"{energy:.3f} ")
                    stdscr.addstr(y, 28, self.draw_bar(energy, 1.0))
                    y += 1
                    
                    # Sparkline for energy
                    spark_lines = self.draw_sparkline(list(self.energy_history), width=50, height=3)
                    for line in spark_lines:
                        stdscr.addstr(y, 28, line, curses.color_pair(4))
                        y += 1
                    y += 1
                    
                    # Prediction Confidence
                    confidence = telemetry.get('prediction_confidence', 0.5)
                    stdscr.addstr(y, 2, "Confidence:      ", curses.color_pair(1))
                    stdscr.addstr(y, 20, f"{confidence:.3f} ")
                    stdscr.addstr(y, 28, self.draw_bar(confidence, 1.0))
                    y += 1
                    
                    # Sparkline for confidence
                    spark_lines = self.draw_sparkline(list(self.confidence_history), width=50, height=3)
                    for line in spark_lines:
                        stdscr.addstr(y, 28, line, curses.color_pair(1))
                        y += 1
                    y += 1
                    
                    # Evolution State
                    evo_state = telemetry.get('evolution_state', {})
                    sm_strength = evo_state.get('self_modification_strength', 0.01)
                    evo_cycles = evo_state.get('evolution_cycles', 0)
                    
                    stdscr.addstr(y, 2, "Self-Modification:", curses.color_pair(5))
                    stdscr.addstr(y, 20, f"{sm_strength:.3f} ")
                    stdscr.addstr(y, 28, self.draw_bar(sm_strength, 0.2))
                    y += 1
                    
                    stdscr.addstr(y, 2, f"Evolution Cycles: {evo_cycles}")
                    y += 2
                    
                    # Working Memory
                    wm = evo_state.get('working_memory', {})
                    patterns = wm.get('n_patterns', 0)
                    stdscr.addstr(y, 2, f"Working Memory Patterns: {patterns}")
                    y += 2
                    
                    # Behavior State
                    behavior = telemetry.get('behavior_state', 'unknown')
                    color = curses.color_pair(2)  # Default yellow
                    if 'exploring' in behavior.lower():
                        color = curses.color_pair(4)  # Cyan
                    elif 'exploiting' in behavior.lower():
                        color = curses.color_pair(1)  # Green
                    elif 'uncertain' in behavior.lower():
                        color = curses.color_pair(3)  # Red
                    
                    stdscr.addstr(y, 2, f"Behavior State: {behavior}", color | curses.A_BOLD)
                    y += 2
                    
                    # Cycle Time
                    cycle_time = telemetry.get('cycle_time_ms', 0)
                    if cycle_time < 150:
                        ct_color = curses.color_pair(1)  # Green
                    elif cycle_time < 300:
                        ct_color = curses.color_pair(2)  # Yellow
                    else:
                        ct_color = curses.color_pair(3)  # Red
                    
                    stdscr.addstr(y, 2, f"Cycle Time: {cycle_time:.0f}ms", ct_color)
                    y += 2
                    
                    # Topology
                    topology = telemetry.get('topology_regions', {})
                    active = topology.get('active', 0)
                    total = topology.get('total', 0)
                    stdscr.addstr(y, 2, f"Active Regions: {active}/{total}")
                    y += 1
            
            else:
                # Show session list
                stdscr.addstr(y, 2, "Available Sessions:", curses.A_BOLD)
                y += 2
                
                sessions = self.get_active_sessions()
                if sessions:
                    for i, session in enumerate(sessions):
                        session_id = session.get('session_id', 'unknown')
                        robot_type = session.get('robot_type', 'unknown')
                        stdscr.addstr(y, 4, f"{i+1}. {session_id} ({robot_type})")
                        y += 1
                    
                    y += 1
                    stdscr.addstr(y, 2, "Press number to select session")
                else:
                    stdscr.addstr(y, 4, "No active sessions")
            
            # Instructions
            bottom_y = height - 3
            stdscr.addstr(bottom_y, 2, "Commands: [c]onnect [d]isconnect [r]efresh [q]uit", 
                         curses.A_DIM)
            
            # Handle input
            key = stdscr.getch()
            if key == ord('q'):
                self.running = False
            elif key == ord('c'):
                if not self.connected:
                    self.connect()
            elif key == ord('d'):
                if self.connected:
                    self.disconnect()
                    self.selected_session = None
            elif key == ord('r'):
                self.selected_session = None
            elif ord('1') <= key <= ord('9'):
                # Select session
                sessions = self.get_active_sessions()
                idx = key - ord('1')
                if 0 <= idx < len(sessions):
                    self.selected_session = sessions[idx].get('session_id')
                    # Clear history
                    self.energy_history.clear()
                    self.confidence_history.clear()
                    self.evolution_history.clear()
                    self.cycle_time_history.clear()
            
            stdscr.refresh()
    
    def run(self):
        """Run the terminal monitor"""
        try:
            curses.wrapper(self.run_curses)
        finally:
            self.disconnect()


def main():
    """Run the terminal brain monitor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terminal-based brain monitor")
    parser.add_argument('--host', default='localhost', 
                       help='Brain server host (default: localhost)')
    parser.add_argument('--port', type=int, default=9998,
                       help='Monitoring port (default: 9998)')
    
    args = parser.parse_args()
    
    print("Starting Terminal Brain Monitor...")
    print(f"Connecting to {args.host}:{args.port}")
    print("Press any key to continue...")
    
    monitor = TerminalBrainMonitor(host=args.host, port=args.port)
    monitor.run()


if __name__ == "__main__":
    main()