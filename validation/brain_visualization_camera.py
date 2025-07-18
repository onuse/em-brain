#!/usr/bin/env python3
"""
Real-Time Brain Visualization Camera

This creates a "brain viewer" that shows what the artificial brain is "thinking"
about in real-time while processing camera feed. It visualizes:

1. Attention patterns - what parts of the image the brain focuses on
2. Pattern recognition - which patterns the brain recognizes
3. Prediction confidence - how confident the brain is about predictions
4. Temporal hierarchy activation - which processing levels are active
5. Memory formation - when the brain forms new memories
6. Sparse activation patterns - which neural patterns are active

This is completely external to the brain and acts as a real-time diagnostic tool.
"""

import sys
import os
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue

# Import brain components
from src.brain import MinimalBrain


class BrainActivityExtractor:
    """Extracts interpretable activity from the brain during processing"""
    
    def __init__(self, brain: MinimalBrain):
        """Initialize extractor with brain instance"""
        self.brain = brain
        self.activity_history = deque(maxlen=100)
        
    def extract_activity(self, sensory_input: List[float], 
                        brain_output: Any, brain_info: Dict) -> Dict:
        """
        Extract meaningful activity from brain processing
        
        Args:
            sensory_input: The input that was processed
            brain_output: The brain's output (prediction)
            brain_info: Additional info from brain processing
            
        Returns:
            Dictionary with interpretable brain activity
        """
        activity = {
            'timestamp': time.time(),
            'sensory_input': sensory_input,
            'brain_output': brain_output,
            'attention_pattern': self._extract_attention_pattern(sensory_input),
            'pattern_recognition': self._extract_pattern_recognition(brain_info),
            'prediction_confidence': self._extract_prediction_confidence(brain_info),
            'temporal_hierarchy': self._extract_temporal_hierarchy(),
            'memory_formation': self._extract_memory_formation(),
            'sparse_activation': self._extract_sparse_activation(sensory_input),
            'novelty_detection': self._extract_novelty_detection(sensory_input)
        }
        
        self.activity_history.append(activity)
        return activity
    
    def _extract_attention_pattern(self, sensory_input: List[float]) -> np.ndarray:
        """Extract what the brain is 'paying attention' to"""
        # Convert sensory input to attention weights
        input_array = np.array(sensory_input)
        
        # Simulate attention as activation magnitude
        attention_raw = np.abs(input_array)
        
        # Normalize to [0, 1]
        if np.max(attention_raw) > 0:
            attention = attention_raw / np.max(attention_raw)
        else:
            attention = attention_raw
        
        # Apply attention focusing (make it more selective)
        attention = np.power(attention, 2)  # Square to make peaks more prominent
        
        return attention
    
    def _extract_pattern_recognition(self, brain_info: Dict) -> Dict:
        """Extract pattern recognition activity"""
        # Try to get pattern info from brain
        pattern_info = {
            'patterns_active': 0,
            'pattern_confidence': 0.0,
            'pattern_novelty': 0.0,
            'pattern_similarity': 0.0
        }
        
        if brain_info:
            # Extract available pattern information
            pattern_info['patterns_active'] = brain_info.get('patterns_active', 0)
            pattern_info['pattern_confidence'] = brain_info.get('confidence', 0.0)
            
            # Simulate pattern metrics based on available info
            if 'similarity_score' in brain_info:
                pattern_info['pattern_similarity'] = brain_info['similarity_score']
            
            # Novelty as inverse of confidence
            pattern_info['pattern_novelty'] = 1.0 - pattern_info['pattern_confidence']
        
        return pattern_info
    
    def _extract_prediction_confidence(self, brain_info: Dict) -> float:
        """Extract how confident the brain is about its predictions"""
        if brain_info and 'confidence' in brain_info:
            return brain_info['confidence']
        
        # Simulate confidence based on available info
        return np.random.uniform(0.3, 0.8)  # Placeholder
    
    def _extract_temporal_hierarchy(self) -> Dict:
        """Extract which temporal processing levels are active"""
        # Simulate temporal hierarchy activation
        # In a real implementation, this would query the brain's temporal system
        
        # Simulate based on recent activity
        if len(self.activity_history) > 0:
            recent_confidences = [a['prediction_confidence'] for a in list(self.activity_history)[-10:]]
            avg_confidence = np.mean(recent_confidences)
            
            # High confidence = reflex level, low confidence = deliberate level
            reflex_activation = avg_confidence
            habit_activation = 1.0 - abs(avg_confidence - 0.5) * 2  # Peak at 0.5 confidence
            deliberate_activation = 1.0 - avg_confidence
        else:
            reflex_activation = 0.5
            habit_activation = 0.3
            deliberate_activation = 0.2
        
        return {
            'reflex': reflex_activation,
            'habit': habit_activation,
            'deliberate': deliberate_activation
        }
    
    def _extract_memory_formation(self) -> Dict:
        """Extract memory formation activity"""
        # Simulate memory formation based on novelty and attention
        if len(self.activity_history) > 1:
            current_input = self.activity_history[-1]['sensory_input']
            prev_input = self.activity_history[-2]['sensory_input']
            
            # Calculate change between frames
            change = np.linalg.norm(np.array(current_input) - np.array(prev_input))
            
            # Memory formation proportional to change
            memory_strength = min(1.0, change * 10)  # Scale factor
            
            return {
                'memory_strength': memory_strength,
                'memory_type': 'episodic' if memory_strength > 0.5 else 'working',
                'consolidation_probability': memory_strength * 0.7
            }
        
        return {
            'memory_strength': 0.0,
            'memory_type': 'working',
            'consolidation_probability': 0.0
        }
    
    def _extract_sparse_activation(self, sensory_input: List[float]) -> Dict:
        """Extract sparse activation patterns"""
        input_array = np.array(sensory_input)
        
        # Calculate sparsity metrics
        threshold = 0.1
        active_neurons = np.sum(input_array > threshold)
        total_neurons = len(input_array)
        sparsity = active_neurons / total_neurons
        
        # Find most active patterns
        active_indices = np.where(input_array > threshold)[0]
        activation_strength = np.sum(input_array[active_indices]) if len(active_indices) > 0 else 0.0
        
        return {
            'sparsity': sparsity,
            'active_neurons': active_neurons,
            'activation_strength': activation_strength,
            'active_indices': active_indices.tolist()
        }
    
    def _extract_novelty_detection(self, sensory_input: List[float]) -> float:
        """Extract novelty detection activity"""
        if len(self.activity_history) < 5:
            return 0.5  # Moderate novelty for new experiences
        
        # Compare current input to recent history
        current = np.array(sensory_input)
        recent_inputs = [np.array(a['sensory_input']) for a in list(self.activity_history)[-5:]]
        
        # Calculate average distance to recent inputs
        distances = [np.linalg.norm(current - prev) for prev in recent_inputs]
        avg_distance = np.mean(distances)
        
        # Normalize to [0, 1] range
        novelty = min(1.0, avg_distance * 5)  # Scale factor
        
        return novelty


class BrainVisualizationOverlay:
    """Creates visual overlays showing brain activity"""
    
    def __init__(self, frame_width: int = 640, frame_height: int = 360):
        """Initialize visualization overlay"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.colors = {
            'attention': (0, 255, 255),      # Yellow for attention
            'recognition': (255, 0, 255),    # Magenta for pattern recognition
            'prediction': (0, 255, 0),       # Green for prediction confidence
            'memory': (255, 255, 0),         # Cyan for memory formation
            'novelty': (0, 0, 255),          # Red for novelty
            'hierarchy': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB for hierarchy levels
        }
        
        # Add attention smoothing
        self.attention_history = deque(maxlen=10)  # Average over 10 frames
        self.attention_threshold = 0.3  # Only show significant attention
        self.last_significant_attention = None
        
        # Motion detection enhancement
        self.motion_boost_factor = 3.0  # Amplify attention during motion
        self.motion_decay_frames = 30   # How long motion boost lasts
    
    def create_overlay(self, frame: np.ndarray, brain_activity: Dict) -> np.ndarray:
        """Create complete brain visualization overlay"""
        overlay = frame.copy()
        
        # Create attention heatmap
        overlay = self._add_attention_heatmap(overlay, brain_activity['attention_pattern'], brain_activity)
        
        # Add pattern recognition indicators
        overlay = self._add_pattern_recognition(overlay, brain_activity['pattern_recognition'])
        
        # Add prediction confidence indicator
        overlay = self._add_prediction_confidence(overlay, brain_activity['prediction_confidence'])
        
        # Add temporal hierarchy visualization
        overlay = self._add_temporal_hierarchy(overlay, brain_activity['temporal_hierarchy'])
        
        # Add memory formation indicator
        overlay = self._add_memory_formation(overlay, brain_activity['memory_formation'])
        
        # Add novelty detection
        overlay = self._add_novelty_detection(overlay, brain_activity['novelty_detection'])
        
        # Add information panel
        overlay = self._add_info_panel(overlay, brain_activity)
        
        return overlay
    
    def _add_attention_heatmap(self, overlay: np.ndarray, attention: np.ndarray, brain_activity: Dict) -> np.ndarray:
        """Add smoothed attention heatmap to overlay"""
        h, w = overlay.shape[:2]
        
        # Add current attention to history
        self.attention_history.append(attention)
        
        # Calculate smoothed attention (average over recent frames)
        if len(self.attention_history) > 1:
            smoothed_attention = np.mean(self.attention_history, axis=0)
        else:
            smoothed_attention = attention
        
        # Boost attention during motion/novelty
        novelty = brain_activity.get('novelty_detection', 0)
        if novelty > 0.5:  # High novelty detected
            smoothed_attention = smoothed_attention * (1 + self.motion_boost_factor * novelty)
        
        # Apply threshold - only show significant attention
        thresholded_attention = np.where(smoothed_attention > self.attention_threshold, 
                                       smoothed_attention, 0)
        
        # Check if there's significant attention
        if np.max(thresholded_attention) > 0:
            self.last_significant_attention = thresholded_attention
            current_attention = thresholded_attention
        else:
            # Use last significant attention with decay
            if self.last_significant_attention is not None:
                current_attention = self.last_significant_attention * 0.9  # Decay
            else:
                current_attention = smoothed_attention
        
        # Reshape attention to approximate grid
        grid_size = int(np.sqrt(len(current_attention)))
        if grid_size * grid_size < len(current_attention):
            grid_size += 1
        
        # Pad attention if needed
        padded_attention = np.zeros(grid_size * grid_size)
        padded_attention[:len(current_attention)] = current_attention
        attention_grid = padded_attention.reshape(grid_size, grid_size)
        
        # Resize to frame dimensions
        attention_resized = cv2.resize(attention_grid, (w, h))
        
        # Create heatmap with more visible colors
        heatmap = cv2.applyColorMap((attention_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame - make heatmap more visible
        overlay = cv2.addWeighted(overlay, 0.5, heatmap, 0.5, 0)
        
        return overlay
    
    def _add_pattern_recognition(self, overlay: np.ndarray, pattern_info: Dict) -> np.ndarray:
        """Add pattern recognition visualization"""
        h, w = overlay.shape[:2]
        
        # Draw pattern recognition strength as circle
        confidence = pattern_info['pattern_confidence']
        radius = int(30 * confidence)
        center = (w - 50, 50)
        
        color = self.colors['recognition']
        cv2.circle(overlay, center, radius, color, -1)
        cv2.circle(overlay, center, 30, color, 2)
        
        # Add text
        cv2.putText(overlay, f"Pattern: {confidence:.2f}", 
                   (w - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def _add_prediction_confidence(self, overlay: np.ndarray, confidence: float) -> np.ndarray:
        """Add prediction confidence bar"""
        h, w = overlay.shape[:2]
        
        # Draw confidence bar
        bar_width = 100
        bar_height = 10
        bar_x = w - bar_width - 10
        bar_y = h - 30
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        color = self.colors['prediction']
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Text
        cv2.putText(overlay, f"Confidence: {confidence:.2f}", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def _add_temporal_hierarchy(self, overlay: np.ndarray, hierarchy: Dict) -> np.ndarray:
        """Add temporal hierarchy activation visualization"""
        h, w = overlay.shape[:2]
        
        # Draw hierarchy levels as bars
        bar_width = 20
        bar_spacing = 25
        start_x = 10
        start_y = h - 100
        
        levels = ['reflex', 'habit', 'deliberate']
        colors = self.colors['hierarchy']
        
        for i, level in enumerate(levels):
            activation = hierarchy[level]
            bar_height = int(60 * activation)
            
            bar_x = start_x + i * bar_spacing
            bar_y = start_y - bar_height
            
            # Draw bar
            cv2.rectangle(overlay, (bar_x, start_y), (bar_x + bar_width, bar_y), 
                         colors[i], -1)
            
            # Draw outline
            cv2.rectangle(overlay, (bar_x, start_y), (bar_x + bar_width, start_y - 60), 
                         colors[i], 1)
            
            # Add label
            cv2.putText(overlay, level[:3], (bar_x - 5, start_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
        
        return overlay
    
    def _add_memory_formation(self, overlay: np.ndarray, memory_info: Dict) -> np.ndarray:
        """Add memory formation visualization"""
        h, w = overlay.shape[:2]
        
        strength = memory_info['memory_strength']
        
        # Draw memory formation as pulsing circle
        radius = int(20 + 10 * strength)
        center = (50, 50)
        
        color = self.colors['memory']
        cv2.circle(overlay, center, radius, color, 2)
        
        # Add inner circle for consolidation
        if memory_info['consolidation_probability'] > 0.5:
            cv2.circle(overlay, center, int(radius * 0.6), color, -1)
        
        # Add text
        cv2.putText(overlay, f"Memory: {strength:.2f}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def _add_novelty_detection(self, overlay: np.ndarray, novelty: float) -> np.ndarray:
        """Add novelty detection visualization"""
        h, w = overlay.shape[:2]
        
        # Draw novelty as border intensity
        if novelty > 0.5:
            border_thickness = int(10 * novelty)
            color = self.colors['novelty']
            
            # Draw border
            cv2.rectangle(overlay, (0, 0), (w, h), color, border_thickness)
        
        # Add novelty text
        cv2.putText(overlay, f"Novelty: {novelty:.2f}", 
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['novelty'], 1)
        
        return overlay
    
    def _add_info_panel(self, overlay: np.ndarray, brain_activity: Dict) -> np.ndarray:
        """Add comprehensive information panel"""
        h, w = overlay.shape[:2]
        
        # Create semi-transparent info panel
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:, :] = (0, 0, 0)  # Black background
        
        # Add information text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        
        y_pos = 15
        line_height = 15
        
        # Add various metrics
        sparse_info = brain_activity['sparse_activation']
        cv2.putText(panel, f"Sparsity: {sparse_info['sparsity']:.2f} "
                          f"({sparse_info['active_neurons']}/{len(brain_activity['sensory_input'])})",
                   (10, y_pos), font, font_scale, color, 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Activation Strength: {sparse_info['activation_strength']:.2f}",
                   (10, y_pos), font, font_scale, color, 1)
        y_pos += line_height
        
        pattern_info = brain_activity['pattern_recognition']
        cv2.putText(panel, f"Pattern Confidence: {pattern_info['pattern_confidence']:.2f}",
                   (10, y_pos), font, font_scale, color, 1)
        y_pos += line_height
        
        memory_info = brain_activity['memory_formation']
        cv2.putText(panel, f"Memory Type: {memory_info['memory_type']} "
                          f"(strength: {memory_info['memory_strength']:.2f})",
                   (10, y_pos), font, font_scale, color, 1)
        y_pos += line_height
        
        hierarchy = brain_activity['temporal_hierarchy']
        cv2.putText(panel, f"Hierarchy: R={hierarchy['reflex']:.2f} "
                          f"H={hierarchy['habit']:.2f} D={hierarchy['deliberate']:.2f}",
                   (10, y_pos), font, font_scale, color, 1)
        y_pos += line_height
        
        cv2.putText(panel, f"Novelty: {brain_activity['novelty_detection']:.2f}",
                   (10, y_pos), font, font_scale, color, 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(panel, f"Time: {timestamp}", 
                   (w - 120, 15), font, font_scale, (150, 150, 150), 1)
        
        # Blend panel with overlay
        overlay_with_panel = np.vstack([overlay, panel])
        
        return overlay_with_panel


class BrainVisualizationCamera:
    """Real-time brain visualization system"""
    
    def __init__(self, brain_type: str = "sparse_goldilocks", camera_id: int = 0):
        """Initialize brain visualization camera"""
        self.brain_type = brain_type
        self.camera_id = camera_id
        
        # Initialize components
        self.brain = MinimalBrain(brain_type=brain_type)
        self.extractor = BrainActivityExtractor(self.brain)
        self.visualizer = BrainVisualizationOverlay()
        
        # Camera setup
        self.camera = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Statistics
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        
        # Visualization controls
        self.show_attention = True
        self.show_info_panel = True
        
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                return False
            
            print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def camera_capture_thread(self):
        """Thread for camera capture"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    self.frame_queue.put(frame, timeout=0.001)
                except queue.Full:
                    pass
            time.sleep(0.01)
    
    def process_frame_to_brain_input(self, frame: np.ndarray) -> List[float]:
        """Convert frame to brain input"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downsample
        small = cv2.resize(gray, (8, 8))  # 64 pixels -> 16D after processing
        
        # Normalize
        normalized = small.astype(np.float32) / 255.0
        
        # Flatten and reduce to brain's sensory dimension
        flattened = normalized.flatten()
        
        # Sample to match brain's sensory dimension
        target_dim = self.brain.sensory_dim
        if len(flattened) > target_dim:
            indices = np.linspace(0, len(flattened) - 1, target_dim, dtype=int)
            sampled = flattened[indices]
        else:
            sampled = np.pad(flattened, (0, target_dim - len(flattened)), mode='constant')
        
        return sampled.tolist()
    
    def run_visualization(self):
        """Run real-time brain visualization"""
        if not self.initialize_camera():
            print("❌ Failed to initialize camera")
            return
        
        print("🧠 Starting Real-Time Brain Visualization")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'a' - Toggle attention heatmap")
        print("  '+' - Increase attention sensitivity")
        print("  '-' - Decrease attention sensitivity")
        print("  'h' - Toggle info panel")
        print("=" * 50)
        
        # Start camera thread
        self.running = True
        capture_thread = threading.Thread(target=self.camera_capture_thread)
        capture_thread.start()
        
        try:
            while self.running:
                try:
                    # Get frame
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Process frame
                    start_time = time.time()
                    
                    # Convert to brain input
                    brain_input = self.process_frame_to_brain_input(frame)
                    
                    # Process through brain
                    brain_output, brain_info = self.brain.process_sensory_input(brain_input)
                    
                    # Extract brain activity
                    brain_activity = self.extractor.extract_activity(
                        brain_input, brain_output, brain_info
                    )
                    
                    # Create visualization
                    if self.show_attention:
                        visualization = self.visualizer.create_overlay(frame, brain_activity)
                    else:
                        # Show minimal visualization without attention heatmap
                        visualization = frame.copy()
                        visualization = self.visualizer._add_pattern_recognition(visualization, brain_activity['pattern_recognition'])
                        visualization = self.visualizer._add_prediction_confidence(visualization, brain_activity['prediction_confidence'])
                        visualization = self.visualizer._add_temporal_hierarchy(visualization, brain_activity['temporal_hierarchy'])
                        if self.show_info_panel:
                            visualization = self.visualizer._add_info_panel(visualization, brain_activity)
                    
                    # Track processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    # Add FPS counter
                    if self.processing_times:
                        fps = 1.0 / np.mean(self.processing_times)
                        cv2.putText(visualization, f"FPS: {fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    
                    # Display
                    cv2.imshow('Brain Visualization', visualization)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Create screenshots directory if it doesn't exist
                        screenshots_dir = "validation/brain_screenshots"
                        os.makedirs(screenshots_dir, exist_ok=True)
                        
                        filename = f"{screenshots_dir}/brain_visualization_{timestamp}.png"
                        cv2.imwrite(filename, visualization)
                        print(f"📸 Screenshot saved: {filename}")
                    elif key == ord('a'):
                        # Toggle attention heatmap
                        self.show_attention = not self.show_attention
                        print(f"🎯 Attention heatmap: {'ON' if self.show_attention else 'OFF'}")
                    elif key == ord('h'):
                        # Toggle info panel
                        self.show_info_panel = not self.show_info_panel
                        print(f"📊 Info panel: {'ON' if self.show_info_panel else 'OFF'}")
                    elif key == ord('+') or key == ord('='):
                        # Increase attention sensitivity
                        self.visualizer.attention_threshold = max(0.1, self.visualizer.attention_threshold - 0.1)
                        print(f"🔍 Attention sensitivity: {self.visualizer.attention_threshold:.1f}")
                    elif key == ord('-'):
                        # Decrease attention sensitivity
                        self.visualizer.attention_threshold = min(0.9, self.visualizer.attention_threshold + 0.1)
                        print(f"🔍 Attention sensitivity: {self.visualizer.attention_threshold:.1f}")
                    
                    self.frame_count += 1
                    
                    # Print periodic stats
                    if self.frame_count % 100 == 0:
                        avg_processing = np.mean(self.processing_times) * 1000
                        print(f"Frame {self.frame_count}: {avg_processing:.1f}ms processing")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Processing error: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n⏹️ Visualization stopped")
        
        finally:
            self.running = False
            capture_thread.join()
            self.camera.release()
            cv2.destroyAllWindows()
            
            print(f"\n📊 Session stats:")
            print(f"Frames processed: {self.frame_count}")
            if self.processing_times:
                avg_time = np.mean(self.processing_times) * 1000
                print(f"Average processing time: {avg_time:.1f}ms")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Brain Visualization")
    parser.add_argument("--brain_type", choices=["minimal", "goldilocks", "sparse_goldilocks"], 
                       default="sparse_goldilocks", help="Type of brain to visualize")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create and run visualization
    visualizer = BrainVisualizationCamera(
        brain_type=args.brain_type,
        camera_id=args.camera_id
    )
    
    visualizer.run_visualization()


if __name__ == "__main__":
    main()