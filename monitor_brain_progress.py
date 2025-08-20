#!/usr/bin/env python3
"""
Brain Progress Monitor - Real-time visualization of learning
Shows if the brain is actually progressing or just twitching randomly
"""

import socket
import json
import time
import sys
from collections import deque
from datetime import datetime
import numpy as np

class BrainProgressMonitor:
    """Monitor and visualize brain learning progress."""
    
    def __init__(self, telemetry_port=9998):
        self.telemetry_port = telemetry_port
        self.history = {
            'learning_score': deque(maxlen=100),
            'causal_chains': deque(maxlen=100),
            'semantic_meanings': deque(maxlen=100),
            'prediction_accuracy': deque(maxlen=100),
            'exploration_rate': deque(maxlen=100),
            'emergence_score': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
        
        # Milestones to watch for
        self.milestones = {
            'first_concept': False,
            'first_causal_chain': False,
            'first_semantic_meaning': False,
            'prediction_better_than_random': False,
            'stable_behavior': False,
            'problem_solving': False
        }
        
        self.start_time = time.time()
        self.cycle_count = 0
        
    def connect(self):
        """Connect to brain telemetry stream."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(('localhost', self.telemetry_port))
            print(f"✓ Connected to brain telemetry on port {self.telemetry_port}")
            return True
        except Exception as e:
            print(f"✗ Could not connect to telemetry: {e}")
            print("  Make sure brain server is running with telemetry enabled")
            return False
    
    def monitor(self):
        """Main monitoring loop."""
        print("\n" + "="*60)
        print("BRAIN PROGRESS MONITOR")
        print("="*60)
        print("Watching for signs of learning and intelligence emergence...")
        print("-"*60)
        
        buffer = ""
        last_report_time = time.time()
        
        try:
            while True:
                # Receive telemetry data
                data = self.sock.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            telemetry = json.loads(line)
                            self.process_telemetry(telemetry)
                            
                            # Report progress every 5 seconds
                            if time.time() - last_report_time > 5:
                                self.report_progress()
                                last_report_time = time.time()
                                
                        except json.JSONDecodeError:
                            pass  # Ignore malformed messages
                            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.final_report()
    
    def process_telemetry(self, telemetry):
        """Process incoming telemetry data."""
        self.cycle_count = telemetry.get('cycles', self.cycle_count)
        
        # Track key metrics
        if 'learning_score' in telemetry:
            self.history['learning_score'].append(telemetry['learning_score'])
        if 'causal_chains' in telemetry:
            self.history['causal_chains'].append(telemetry['causal_chains'])
        if 'semantic_meanings' in telemetry:
            self.history['semantic_meanings'].append(telemetry['semantic_meanings'])
        if 'prediction_accuracy' in telemetry:
            self.history['prediction_accuracy'].append(telemetry['prediction_accuracy'])
        if 'exploration_rate' in telemetry:
            self.history['exploration_rate'].append(telemetry['exploration_rate'])
        if 'emergence_score' in telemetry:
            self.history['emergence_score'].append(telemetry['emergence_score'])
        
        self.history['timestamps'].append(time.time())
        
        # Check milestones
        self.check_milestones(telemetry)
    
    def check_milestones(self, telemetry):
        """Check for learning milestones."""
        # First concept formed
        if not self.milestones['first_concept'] and telemetry.get('concepts_formed', 0) > 0:
            self.milestones['first_concept'] = True
            self.announce_milestone("🎯 FIRST CONCEPT FORMED! Brain is creating stable patterns")
        
        # First causal chain
        if not self.milestones['first_causal_chain'] and telemetry.get('causal_chains', 0) > 0:
            self.milestones['first_causal_chain'] = True
            self.announce_milestone("🔗 FIRST CAUSAL CHAIN! Brain learned that A leads to B")
        
        # First semantic meaning
        if not self.milestones['first_semantic_meaning'] and telemetry.get('semantic_meanings', 0) > 0:
            self.milestones['first_semantic_meaning'] = True
            self.announce_milestone("💡 SEMANTIC GROUNDING! Patterns now have real-world meaning")
        
        # Prediction better than random
        if not self.milestones['prediction_better_than_random'] and telemetry.get('prediction_accuracy', 0) > 0.6:
            self.milestones['prediction_better_than_random'] = True
            self.announce_milestone("🔮 PREDICTIVE CAPABILITY! Brain predicts better than chance")
        
        # Stable behavior
        if not self.milestones['stable_behavior']:
            if len(self.history['exploration_rate']) > 20:
                recent = list(self.history['exploration_rate'])[-20:]
                if np.std(recent) < 0.1:  # Low variance = stable
                    self.milestones['stable_behavior'] = True
                    self.announce_milestone("⚖️ BEHAVIORAL STABILITY! Consistent exploration/exploitation balance")
        
        # Problem solving (high learning score)
        if not self.milestones['problem_solving'] and telemetry.get('learning_score', 0) > 0.7:
            self.milestones['problem_solving'] = True
            self.announce_milestone("🧩 PROBLEM SOLVING! Brain shows complex adaptive behavior")
    
    def announce_milestone(self, message):
        """Announce a milestone achievement."""
        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"⭐ MILESTONE at {elapsed:.0f}s (cycle {self.cycle_count})")
        print(f"   {message}")
        print(f"{'='*60}\n")
    
    def report_progress(self):
        """Report current progress."""
        if not any(len(h) > 0 for h in self.history.values()):
            return
        
        print(f"\n[Cycle {self.cycle_count}] Progress Report:")
        
        # Learning trajectory
        if len(self.history['learning_score']) > 1:
            current = self.history['learning_score'][-1]
            initial = self.history['learning_score'][0]
            trend = "📈" if current > initial else "📉" if current < initial else "➡️"
            print(f"  Learning Score: {current:.1%} {trend}")
        
        # Causal understanding
        if len(self.history['causal_chains']) > 0:
            chains = self.history['causal_chains'][-1]
            if chains > 0:
                print(f"  Causal Chains: {chains} learned")
        
        # Semantic grounding
        if len(self.history['semantic_meanings']) > 0:
            meanings = self.history['semantic_meanings'][-1]
            if meanings > 0:
                print(f"  Semantic Meanings: {meanings} grounded")
        
        # Prediction accuracy
        if len(self.history['prediction_accuracy']) > 0:
            accuracy = self.history['prediction_accuracy'][-1]
            if accuracy > 0:
                quality = "excellent" if accuracy > 0.8 else "good" if accuracy > 0.6 else "developing"
                print(f"  Prediction: {accuracy:.1%} ({quality})")
        
        # Exploration vs exploitation
        if len(self.history['exploration_rate']) > 0:
            exploration = self.history['exploration_rate'][-1]
            mode = "exploring" if exploration > 0.5 else "exploiting" if exploration < 0.3 else "balanced"
            print(f"  Behavior: {mode} ({exploration:.1%} exploration)")
        
        # Progress indicators
        self.show_progress_bar()
    
    def show_progress_bar(self):
        """Show visual progress indicators."""
        print("\n  Progress Indicators:")
        
        # Concept formation
        concepts = self.cycle_count // 100  # Rough estimate
        bar = self.make_bar(min(concepts, 10), 10)
        print(f"  Concepts:    [{bar}]")
        
        # Learning development
        if len(self.history['learning_score']) > 0:
            learning = int(self.history['learning_score'][-1] * 10)
            bar = self.make_bar(learning, 10)
            print(f"  Learning:    [{bar}]")
        
        # Intelligence emergence
        milestones_achieved = sum(self.milestones.values())
        bar = self.make_bar(milestones_achieved, len(self.milestones))
        print(f"  Milestones:  [{bar}] {milestones_achieved}/{len(self.milestones)}")
    
    def make_bar(self, current, total):
        """Create a progress bar string."""
        filled = int((current / total) * 20)
        return "█" * filled + "░" * (20 - filled)
    
    def final_report(self):
        """Generate final progress report."""
        elapsed = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("FINAL PROGRESS REPORT")
        print("="*60)
        print(f"Monitoring Duration: {elapsed:.0f} seconds")
        print(f"Total Cycles: {self.cycle_count}")
        
        # Milestones achieved
        achieved = sum(self.milestones.values())
        print(f"\nMilestones Achieved: {achieved}/{len(self.milestones)}")
        for milestone, reached in self.milestones.items():
            status = "✓" if reached else "✗"
            print(f"  {status} {milestone.replace('_', ' ').title()}")
        
        # Learning trajectory
        if len(self.history['learning_score']) > 1:
            initial = self.history['learning_score'][0]
            final = self.history['learning_score'][-1]
            improvement = (final - initial) / (initial + 0.001) * 100
            print(f"\nLearning Improvement: {improvement:+.1f}%")
            print(f"  Initial: {initial:.1%}")
            print(f"  Final:   {final:.1%}")
        
        # Assessment
        print("\n" + "-"*60)
        if achieved >= 5:
            print("🌟 EXCELLENT PROGRESS - Genuine learning and intelligence emergence!")
        elif achieved >= 3:
            print("✓ GOOD PROGRESS - Clear signs of learning")
        elif achieved >= 1:
            print("⚡ EARLY PROGRESS - Initial patterns forming")
        else:
            print("🔄 STILL DEVELOPING - More time needed for emergence")
        
        print("="*60)


def main():
    """Run the monitor."""
    monitor = BrainProgressMonitor()
    
    if not monitor.connect():
        print("\nTo monitor brain progress:")
        print("1. Start the brain server: python3 run_server.py --brain enhanced")
        print("2. Run this monitor: python3 monitor_brain_progress.py")
        sys.exit(1)
    
    try:
        monitor.monitor()
    except Exception as e:
        print(f"Error: {e}")
        monitor.final_report()


if __name__ == "__main__":
    main()