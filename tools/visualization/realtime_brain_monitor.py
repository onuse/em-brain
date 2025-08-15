#!/usr/bin/env python3
"""
Real-time Brain Monitor

A standalone visualization client that connects to the brain monitoring socket (port 9998)
and displays live telemetry data. Can be started/stopped at any time without affecting
the running brain or validation experiments.
"""

import sys
import os
from pathlib import Path
import socket
import json
import time
import threading
from collections import deque
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np


class BrainMonitorClient:
    """Client for connecting to brain monitoring server"""
    
    def __init__(self, host='localhost', port=9998):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.data_buffer = deque(maxlen=1000)
        self.session_id = None
        
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
                print(f"✅ Connected to {response.get('server', 'monitoring server')}")
                return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
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
        
        # Get all telemetry data to find active sessions
        self._send_json({'command': 'telemetry'})
        response = self._receive_json()
        
        if response and response.get('status') == 'success':
            telemetry_data = response.get('data', {})
            if telemetry_data:
                # Extract session IDs from telemetry
                sessions = []
                for session_id in telemetry_data.keys():
                    sessions.append({
                        'session_id': session_id,
                        'robot_type': 'active',
                        'cycles': 0  # Will be updated from telemetry
                    })
                return sessions
        
        # Fallback to session_info
        self._send_json({'command': 'session_info'})
        response = self._receive_json()
        
        if response and response.get('status') == 'success':
            sessions = response.get('data', [])
            if sessions:
                return sessions
        
        return []
    
    def get_brain_telemetry(self, session_id):
        """Get telemetry data for a specific session"""
        if not self.connected:
            return None
        
        # Try telemetry command with session ID
        self._send_json({
            'command': f'telemetry {session_id}'
        })
        
        response = self._receive_json()
        if response and response.get('status') == 'success':
            telemetry_data = response.get('data', {})
            # If data is nested under session ID, extract it
            if isinstance(telemetry_data, dict) and session_id in telemetry_data:
                return telemetry_data[session_id]
            return telemetry_data
        
        # Fallback to brain_stats
        self._send_json({
            'command': 'brain_stats',
            'session_id': session_id
        })
        
        response = self._receive_json()
        if response and response.get('status') == 'success':
            return response.get('data', {})
        
        return None


class RealTimeBrainMonitor:
    """Real-time visualization of brain telemetry"""
    
    def __init__(self):
        self.client = BrainMonitorClient()
        self.root = tk.Tk()
        self.root.title("Brain Monitor - Real-time Telemetry")
        self.root.geometry("1200x800")
        
        # Data storage
        self.time_data = deque(maxlen=300)  # 5 minutes at 1Hz
        self.energy_data = deque(maxlen=300)
        self.confidence_data = deque(maxlen=300)
        self.evolution_data = deque(maxlen=300)
        self.prediction_error_data = deque(maxlen=300)
        self.cycle_time_data = deque(maxlen=300)
        
        # UI state
        self.selected_session = None
        self.monitoring_active = False
        self.update_thread = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")
        
        # Connection status
        self.status_label = ttk.Label(control_frame, text="Status: Disconnected", 
                                     font=("Arial", 10, "bold"))
        self.status_label.grid(row=0, column=0, padx=5)
        
        # Connect button
        self.connect_btn = ttk.Button(control_frame, text="Connect", 
                                     command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=1, padx=5)
        
        # Session selector
        ttk.Label(control_frame, text="Session:").grid(row=0, column=2, padx=5)
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(control_frame, textvariable=self.session_var,
                                         state="readonly", width=30)
        self.session_combo.grid(row=0, column=3, padx=5)
        self.session_combo.bind('<<ComboboxSelected>>', self.on_session_selected)
        
        # Refresh sessions button
        self.refresh_btn = ttk.Button(control_frame, text="Refresh Sessions",
                                     command=self.refresh_sessions, state="disabled")
        self.refresh_btn.grid(row=0, column=4, padx=5)
        
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 7))
        self.fig.tight_layout(pad=3.0)
        
        # Configure subplots
        self.ax_energy = self.axes[0, 0]
        self.ax_confidence = self.axes[0, 1]
        self.ax_evolution = self.axes[0, 2]
        self.ax_prediction = self.axes[1, 0]
        self.ax_cycle_time = self.axes[1, 1]
        self.ax_behavior = self.axes[1, 2]
        
        # Setup axes
        self.setup_axes()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Info panel
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.grid(row=2, column=0, sticky="ew")
        
        self.info_text = tk.Text(info_frame, height=5, width=100, 
                                font=("Courier", 9))
        self.info_text.grid(row=0, column=0, sticky="ew")
        
        # Start animation (will be initialized after mainloop starts)
        self.ani = None
        
    def setup_axes(self):
        """Configure plot axes"""
        # Energy plot
        self.ax_energy.set_title("Field Energy")
        self.ax_energy.set_xlabel("Time (s)")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.set_ylim(0, 1)
        self.ax_energy.grid(True, alpha=0.3)
        
        # Confidence plot
        self.ax_confidence.set_title("Prediction Confidence")
        self.ax_confidence.set_xlabel("Time (s)")
        self.ax_confidence.set_ylabel("Confidence")
        self.ax_confidence.set_ylim(0, 1)
        self.ax_confidence.grid(True, alpha=0.3)
        
        # Evolution plot
        self.ax_evolution.set_title("Self-Modification Strength")
        self.ax_evolution.set_xlabel("Time (s)")
        self.ax_evolution.set_ylabel("SM Strength")
        self.ax_evolution.set_ylim(0, 0.2)
        self.ax_evolution.grid(True, alpha=0.3)
        
        # Prediction error plot
        self.ax_prediction.set_title("Prediction Error")
        self.ax_prediction.set_xlabel("Time (s)")
        self.ax_prediction.set_ylabel("Error")
        self.ax_prediction.set_ylim(0, 1)
        self.ax_prediction.grid(True, alpha=0.3)
        
        # Cycle time plot
        self.ax_cycle_time.set_title("Cycle Time")
        self.ax_cycle_time.set_xlabel("Time (s)")
        self.ax_cycle_time.set_ylabel("Time (ms)")
        self.ax_cycle_time.set_ylim(0, 500)
        self.ax_cycle_time.grid(True, alpha=0.3)
        
        # Behavior state plot
        self.ax_behavior.set_title("Behavior State")
        self.ax_behavior.set_xlabel("Energy")
        self.ax_behavior.set_ylabel("Confidence")
        self.ax_behavior.set_xlim(0, 1)
        self.ax_behavior.set_ylim(0, 1)
        self.ax_behavior.grid(True, alpha=0.3)
        
        # Add behavior state regions
        self.ax_behavior.axhline(0.6, color='gray', linestyle='--', alpha=0.5)
        self.ax_behavior.axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        self.ax_behavior.axvline(0.7, color='gray', linestyle='--', alpha=0.5)
        
        # Label regions
        self.ax_behavior.text(0.15, 0.8, "Exploring", ha='center', alpha=0.5)
        self.ax_behavior.text(0.5, 0.8, "Learning", ha='center', alpha=0.5)
        self.ax_behavior.text(0.85, 0.8, "Exploiting", ha='center', alpha=0.5)
        self.ax_behavior.text(0.5, 0.3, "Uncertain", ha='center', alpha=0.5)
        
    def toggle_connection(self):
        """Connect or disconnect from monitoring server"""
        if self.client.connected:
            self.disconnect()
        else:
            self.connect()
    
    def connect(self):
        """Connect to monitoring server"""
        if self.client.connect():
            self.status_label.config(text="Status: Connected", foreground="green")
            self.connect_btn.config(text="Disconnect")
            self.refresh_btn.config(state="normal")
            self.refresh_sessions()
        else:
            self.status_label.config(text="Status: Connection Failed", foreground="red")
    
    def disconnect(self):
        """Disconnect from monitoring server"""
        self.monitoring_active = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        
        self.client.disconnect()
        self.status_label.config(text="Status: Disconnected", foreground="gray")
        self.connect_btn.config(text="Connect")
        self.refresh_btn.config(state="disabled")
        self.session_combo.set("")
        self.session_combo['values'] = []
    
    def refresh_sessions(self):
        """Refresh list of active sessions"""
        sessions = self.client.get_active_sessions()
        
        if sessions:
            session_list = []
            for session in sessions:
                session_id = session.get('session_id', 'unknown')
                robot_type = session.get('robot_type', 'unknown')
                cycles = session.get('cycles', 0)
                
                # Format session info
                if cycles > 0:
                    session_str = f"{session_id} ({robot_type}) - {cycles} cycles"
                else:
                    session_str = f"{session_id} ({robot_type})"
                
                session_list.append(session_str)
            
            self.session_combo['values'] = session_list
            
            # Auto-select first session if none selected
            if not self.selected_session and session_list:
                self.session_combo.current(0)
                self.on_session_selected()
        else:
            self.session_combo['values'] = ["No active sessions"]
            self.session_combo.set("No active sessions")
    
    def on_session_selected(self, event=None):
        """Handle session selection"""
        selection = self.session_var.get()
        if selection and "No active" not in selection:
            # Extract session ID
            self.selected_session = selection.split(' ')[0]
            
            # Clear data buffers
            self.time_data.clear()
            self.energy_data.clear()
            self.confidence_data.clear()
            self.evolution_data.clear()
            self.prediction_error_data.clear()
            self.cycle_time_data.clear()
            
            # Start monitoring
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start monitoring selected session"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.update_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
        self.update_thread.start()
    
    def monitoring_loop(self):
        """Background thread for fetching telemetry data"""
        while self.monitoring_active and self.client.connected:
            try:
                if self.selected_session:
                    telemetry = self.client.get_brain_telemetry(self.selected_session)
                    
                    if telemetry:
                        # Add timestamp
                        current_time = len(self.time_data)
                        self.time_data.append(current_time)
                        
                        # Extract data
                        self.energy_data.append(telemetry.get('field_energy', 0))
                        self.confidence_data.append(telemetry.get('prediction_confidence', 0.5))
                        
                        # Evolution state
                        evo_state = telemetry.get('evolution_state', {})
                        self.evolution_data.append(evo_state.get('self_modification_strength', 0.01))
                        
                        # Performance metrics
                        self.prediction_error_data.append(telemetry.get('prediction_error', 0))
                        self.cycle_time_data.append(telemetry.get('cycle_time_ms', 0))
                        
                        # Update info text
                        self.update_info_text(telemetry)
                
                time.sleep(1.0)  # Update rate
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def update_info_text(self, telemetry):
        """Update info panel with latest telemetry"""
        info_lines = []
        
        # Evolution state
        evo_state = telemetry.get('evolution_state', {})
        info_lines.append(f"Evolution Cycles: {evo_state.get('evolution_cycles', 0)}")
        info_lines.append(f"Working Memory: {evo_state.get('working_memory', {}).get('n_patterns', 0)} patterns")
        
        # Behavior state
        behavior = telemetry.get('behavior_state', 'unknown')
        info_lines.append(f"Behavior: {behavior}")
        
        # Topology
        topology = telemetry.get('topology_regions', {})
        info_lines.append(f"Active Regions: {topology.get('active', 0)}/{topology.get('total', 0)}")
        
        # Sensory organization
        sensory = telemetry.get('sensory_organization', {})
        info_lines.append(f"Sensory Patterns: {sensory.get('unique_patterns', 0)}")
        
        # Update text widget in main thread
        self.root.after(0, lambda: self._update_info_widget(info_lines))
    
    def _update_info_widget(self, lines):
        """Update info widget (must be called from main thread)"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, '\n'.join(lines))
    
    def update_plots(self, frame):
        """Update all plots"""
        if not self.time_data:
            return []
        
        # Convert to numpy arrays
        time_array = np.array(self.time_data)
        
        # Clear and redraw each plot
        # Energy
        self.ax_energy.clear()
        self.ax_energy.plot(time_array, self.energy_data, 'b-', linewidth=2)
        self.ax_energy.set_title("Field Energy")
        self.ax_energy.set_ylabel("Energy")
        self.ax_energy.set_ylim(0, 1)
        self.ax_energy.grid(True, alpha=0.3)
        
        # Confidence
        self.ax_confidence.clear()
        self.ax_confidence.plot(time_array, self.confidence_data, 'g-', linewidth=2)
        self.ax_confidence.set_title("Prediction Confidence")
        self.ax_confidence.set_ylabel("Confidence")
        self.ax_confidence.set_ylim(0, 1)
        self.ax_confidence.grid(True, alpha=0.3)
        
        # Evolution
        self.ax_evolution.clear()
        self.ax_evolution.plot(time_array, self.evolution_data, 'm-', linewidth=2)
        self.ax_evolution.set_title("Self-Modification Strength")
        self.ax_evolution.set_ylabel("SM Strength")
        self.ax_evolution.set_ylim(0, max(0.2, max(self.evolution_data) * 1.2) if self.evolution_data else 0.2)
        self.ax_evolution.grid(True, alpha=0.3)
        
        # Prediction error
        self.ax_prediction.clear()
        self.ax_prediction.plot(time_array, self.prediction_error_data, 'r-', linewidth=2)
        self.ax_prediction.set_title("Prediction Error")
        self.ax_prediction.set_ylabel("Error")
        self.ax_prediction.set_ylim(0, 1)
        self.ax_prediction.grid(True, alpha=0.3)
        
        # Cycle time
        self.ax_cycle_time.clear()
        self.ax_cycle_time.plot(time_array, self.cycle_time_data, 'c-', linewidth=2)
        self.ax_cycle_time.set_title("Cycle Time")
        self.ax_cycle_time.set_ylabel("Time (ms)")
        self.ax_cycle_time.set_ylim(0, max(500, max(self.cycle_time_data) * 1.2) if self.cycle_time_data else 500)
        self.ax_cycle_time.grid(True, alpha=0.3)
        
        # Behavior state scatter
        self.ax_behavior.clear()
        if self.energy_data and self.confidence_data:
            # Show trajectory
            self.ax_behavior.plot(self.energy_data, self.confidence_data, 'b-', alpha=0.3)
            # Show current position
            self.ax_behavior.scatter([self.energy_data[-1]], [self.confidence_data[-1]], 
                                   c='red', s=100, marker='o')
        
        # Redraw behavior regions
        self.ax_behavior.set_title("Behavior State")
        self.ax_behavior.set_xlabel("Energy")
        self.ax_behavior.set_ylabel("Confidence")
        self.ax_behavior.set_xlim(0, 1)
        self.ax_behavior.set_ylim(0, 1)
        self.ax_behavior.axhline(0.6, color='gray', linestyle='--', alpha=0.5)
        self.ax_behavior.axvline(0.3, color='gray', linestyle='--', alpha=0.5)
        self.ax_behavior.axvline(0.7, color='gray', linestyle='--', alpha=0.5)
        self.ax_behavior.grid(True, alpha=0.3)
        
        # Add labels
        self.ax_behavior.text(0.15, 0.8, "Exploring", ha='center', alpha=0.5)
        self.ax_behavior.text(0.5, 0.8, "Learning", ha='center', alpha=0.5)
        self.ax_behavior.text(0.85, 0.8, "Exploiting", ha='center', alpha=0.5)
        self.ax_behavior.text(0.5, 0.3, "Uncertain", ha='center', alpha=0.5)
        
        return []
    
    def run(self):
        """Run the monitor application"""
        try:
            # Initialize animation after mainloop starts
            self.root.after(100, self._start_animation)
            self.root.mainloop()
        finally:
            self.disconnect()
    
    def _start_animation(self):
        """Start the animation after GUI is initialized"""
        self.ani = FuncAnimation(self.fig, self.update_plots, interval=1000, 
                               blit=False, cache_frame_data=False)


def main():
    """Run the real-time brain monitor"""
    print("🧠 Real-time Brain Monitor")
    print("=" * 50)
    print("This tool visualizes brain telemetry in real-time.")
    print("Connect to any running brain server on port 9998.")
    print()
    
    monitor = RealTimeBrainMonitor()
    monitor.run()


if __name__ == "__main__":
    main()