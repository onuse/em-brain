#!/usr/bin/env python3
"""
Simple Brain Monitor

A simplified GUI monitor that avoids threading issues on macOS.
Uses a polling approach instead of animations.
"""

import sys
import socket
import json
import time
from collections import deque
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class SimpleBrainMonitor:
    """Simplified real-time visualization of brain telemetry"""
    
    def __init__(self):
        self.host = 'localhost'
        self.port = 9998
        self.socket = None
        self.connected = False
        
        # Data storage
        self.time_data = deque(maxlen=300)  # 5 minutes at 1Hz
        self.energy_data = deque(maxlen=300)
        self.confidence_data = deque(maxlen=300)
        self.evolution_data = deque(maxlen=300)
        self.cycle_time_data = deque(maxlen=300)
        
        # UI state
        self.selected_session = None
        self.last_update = 0
        
        self.setup_ui()
        
        # Start periodic updates
        self.update_data()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.root = tk.Tk()
        self.root.title("Simple Brain Monitor")
        self.root.geometry("1000x700")
        
        # Control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Connection status
        self.status_label = ttk.Label(control_frame, text="Status: Disconnected", 
                                     font=("Arial", 10, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Connect button
        self.connect_btn = ttk.Button(control_frame, text="Connect", 
                                     command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        # Session selector
        ttk.Label(control_frame, text="Session:").pack(side=tk.LEFT, padx=5)
        self.session_var = tk.StringVar()
        self.session_combo = ttk.Combobox(control_frame, textvariable=self.session_var,
                                         state="readonly", width=30)
        self.session_combo.pack(side=tk.LEFT, padx=5)
        self.session_combo.bind('<<ComboboxSelected>>', self.on_session_selected)
        
        # Refresh sessions button
        self.refresh_btn = ttk.Button(control_frame, text="Refresh",
                                     command=self.refresh_sessions, state="disabled")
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(231)
        self.ax2 = self.fig.add_subplot(232)
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(234)
        self.ax5 = self.fig.add_subplot(235)
        self.ax6 = self.fig.add_subplot(236)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Info panel
        info_frame = ttk.Frame(self.root, padding="10")
        info_frame.pack(fill=tk.X)
        
        self.info_label = ttk.Label(info_frame, text="Waiting for data...", 
                                   font=("Courier", 9))
        self.info_label.pack()
        
    def toggle_connection(self):
        """Connect or disconnect from monitoring server"""
        if self.connected:
            self.disconnect()
        else:
            self.connect()
    
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
                self.status_label.config(text="Status: Connected", foreground="green")
                self.connect_btn.config(text="Disconnect")
                self.refresh_btn.config(state="normal")
                self.refresh_sessions()
        except Exception as e:
            self.status_label.config(text=f"Status: Error - {str(e)[:30]}", foreground="red")
    
    def disconnect(self):
        """Disconnect from monitoring server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        self.connected = False
        self.socket = None
        self.status_label.config(text="Status: Disconnected", foreground="gray")
        self.connect_btn.config(text="Connect")
        self.refresh_btn.config(state="disabled")
        self.session_combo.set("")
        self.session_combo['values'] = []
    
    def _send_json(self, data):
        """Send JSON data to server"""
        if not self.socket:
            return False
        try:
            message = json.dumps(data).encode('utf-8')
            self.socket.sendall(message + b'\n')
            return True
        except:
            self.connected = False
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
            self.connected = False
            return None
    
    def refresh_sessions(self):
        """Refresh list of active sessions"""
        if not self.connected:
            return
        
        self._send_json({'command': 'active_brains'})
        response = self._receive_json()
        
        if response and response.get('status') == 'success':
            sessions = response.get('active_brains', [])
            
            if sessions:
                session_list = []
                for session in sessions:
                    session_id = session.get('session_id', 'unknown')
                    robot_type = session.get('robot_type', 'unknown')
                    session_list.append(f"{session_id} ({robot_type})")
                
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
            self.cycle_time_data.clear()
    
    def get_telemetry(self):
        """Get telemetry data for selected session"""
        if not self.connected or not self.selected_session:
            return None
        
        self._send_json({
            'command': 'brain_stats',
            'session_id': self.selected_session
        })
        
        response = self._receive_json()
        if response and response.get('status') == 'success':
            return response.get('data', {})
        return None
    
    def update_data(self):
        """Update data and plots"""
        # Get new data if connected
        if self.connected and self.selected_session:
            telemetry = self.get_telemetry()
            
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
                self.cycle_time_data.append(telemetry.get('cycle_time_ms', 0))
                
                # Update info
                self.update_info(telemetry)
        
        # Update plots
        self.update_plots()
        
        # Schedule next update
        self.root.after(1000, self.update_data)  # Update every second
    
    def update_info(self, telemetry):
        """Update info panel"""
        evo_state = telemetry.get('evolution_state', {})
        behavior = telemetry.get('behavior_state', 'unknown')
        topology = telemetry.get('topology_regions', {})
        
        info_text = (
            f"Behavior: {behavior} | "
            f"Evolution Cycles: {evo_state.get('evolution_cycles', 0)} | "
            f"Working Memory: {evo_state.get('working_memory', {}).get('n_patterns', 0)} | "
            f"Active Regions: {topology.get('active', 0)}/{topology.get('total', 0)}"
        )
        
        self.info_label.config(text=info_text)
    
    def update_plots(self):
        """Update all plots"""
        if not self.time_data:
            return
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        
        # Convert to numpy arrays
        time_array = np.array(self.time_data)
        
        # Plot 1: Energy
        self.ax1.plot(time_array, self.energy_data, 'b-')
        self.ax1.set_title('Field Energy')
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence
        self.ax2.plot(time_array, self.confidence_data, 'g-')
        self.ax2.set_title('Prediction Confidence')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Evolution
        self.ax3.plot(time_array, self.evolution_data, 'm-')
        self.ax3.set_title('Self-Modification')
        self.ax3.set_ylim(0, max(0.2, max(self.evolution_data) * 1.2) if self.evolution_data else 0.2)
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cycle Time
        self.ax4.plot(time_array, self.cycle_time_data, 'c-')
        self.ax4.set_title('Cycle Time (ms)')
        self.ax4.set_ylim(0, max(500, max(self.cycle_time_data) * 1.2) if self.cycle_time_data else 500)
        self.ax4.grid(True, alpha=0.3)
        
        # Plot 5: Energy vs Confidence scatter
        if self.energy_data and self.confidence_data:
            self.ax5.scatter(self.energy_data, self.confidence_data, alpha=0.5)
            self.ax5.plot(self.energy_data, self.confidence_data, 'b-', alpha=0.3)
            if len(self.energy_data) > 0:
                self.ax5.scatter([self.energy_data[-1]], [self.confidence_data[-1]], 
                               c='red', s=100, marker='o')
        self.ax5.set_title('Behavior State')
        self.ax5.set_xlabel('Energy')
        self.ax5.set_ylabel('Confidence')
        self.ax5.set_xlim(0, 1)
        self.ax5.set_ylim(0, 1)
        self.ax5.grid(True, alpha=0.3)
        
        # Plot 6: Evolution over time (bar chart of recent values)
        if len(self.evolution_data) > 0:
            recent_evo = list(self.evolution_data)[-20:]  # Last 20 values
            self.ax6.bar(range(len(recent_evo)), recent_evo, color='purple', alpha=0.7)
        self.ax6.set_title('Recent Evolution Activity')
        self.ax6.set_ylim(0, 0.2)
        self.ax6.grid(True, alpha=0.3, axis='y')
        
        # Redraw canvas
        self.canvas.draw()
    
    def run(self):
        """Run the monitor application"""
        try:
            self.root.mainloop()
        finally:
            self.disconnect()


def main():
    """Run the simple brain monitor"""
    print("🧠 Simple Brain Monitor")
    print("=" * 50)
    print("Simplified GUI monitor for brain telemetry.")
    print()
    
    monitor = SimpleBrainMonitor()
    monitor.run()


if __name__ == "__main__":
    main()