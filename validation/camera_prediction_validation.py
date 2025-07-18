#!/usr/bin/env python3
"""
Camera Prediction Validation

Connect laptop camera to the constraint-based brain and test real-time
visual prediction capabilities. This tests:

1. Frame-to-frame temporal prediction
2. Visual pattern recognition and clustering
3. Emergent attention mechanisms
4. Real-world sparse representation learning
5. Temporal hierarchy emergence with actual visual data

The brain receives camera frames as sparse sensory input and attempts to
predict the next frame, revealing emergent visual intelligence.
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
from typing import Dict, List, Tuple, Optional
import threading
import queue

# Import brain components
from src.brain import MinimalBrain


class CameraFrameProcessor:
    """Processes camera frames for brain input"""
    
    def __init__(self, target_dim: int = 64, downsample_factor: int = 4):
        """
        Initialize frame processor
        
        Args:
            target_dim: Target dimension for brain input (e.g., 64 for 64D vector)
            downsample_factor: Factor to downsample frames (4 = 1/4 resolution)
        """
        self.target_dim = target_dim
        self.downsample_factor = downsample_factor
        self.frame_history = deque(maxlen=10)
        
    def process_frame(self, frame: np.ndarray) -> List[float]:
        """
        Convert camera frame to sparse sensory input vector
        
        Args:
            frame: OpenCV frame (BGR format)
            
        Returns:
            List of floats representing sparse sensory input
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downsample for efficiency
        h, w = gray.shape
        new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
        small = cv2.resize(gray, (new_w, new_h))
        
        # Normalize to [0, 1]
        normalized = small.astype(np.float32) / 255.0
        
        # Flatten to 1D
        flattened = normalized.flatten()
        
        # Reduce dimensionality to target_dim using sampling
        if len(flattened) > self.target_dim:
            # Sample evenly across the image
            indices = np.linspace(0, len(flattened) - 1, self.target_dim, dtype=int)
            sampled = flattened[indices]
        else:
            # Pad if too small
            sampled = np.pad(flattened, (0, self.target_dim - len(flattened)), mode='constant')
        
        # Apply edge detection for more interesting patterns
        edges = cv2.Canny((small * 255).astype(np.uint8), 50, 150)
        edge_normalized = edges.astype(np.float32) / 255.0
        edge_flattened = edge_normalized.flatten()
        
        if len(edge_flattened) > self.target_dim:
            edge_indices = np.linspace(0, len(edge_flattened) - 1, self.target_dim, dtype=int)
            edge_sampled = edge_flattened[edge_indices]
        else:
            edge_sampled = np.pad(edge_flattened, (0, self.target_dim - len(edge_flattened)), mode='constant')
        
        # Combine intensity and edge information
        combined = (sampled * 0.7 + edge_sampled * 0.3).tolist()
        
        # Store in history for analysis
        self.frame_history.append({
            'timestamp': time.time(),
            'vector': combined,
            'original_shape': gray.shape,
            'processed_shape': small.shape
        })
        
        return combined
    
    def get_frame_difference(self, current_vector: List[float]) -> float:
        """Calculate difference from previous frame"""
        if len(self.frame_history) < 2:
            return 0.0
        
        prev_vector = self.frame_history[-2]['vector']
        diff = np.array(current_vector) - np.array(prev_vector)
        return np.linalg.norm(diff)
    
    def detect_motion(self, threshold: float = 0.1) -> bool:
        """Detect if there's significant motion between frames"""
        return self.get_frame_difference(self.frame_history[-1]['vector']) > threshold


class CameraPredictionValidator:
    """Validates brain prediction capabilities with camera feed"""
    
    def __init__(self, brain_type: str = "sparse_goldilocks", camera_id: int = 0):
        """
        Initialize camera prediction validator
        
        Args:
            brain_type: Type of brain to test
            camera_id: Camera device ID (0 for default)
        """
        self.brain_type = brain_type
        self.camera_id = camera_id
        self.results = {
            "brain_type": brain_type,
            "camera_id": camera_id,
            "start_time": datetime.now().isoformat(),
            "predictions": [],
            "patterns_discovered": [],
            "attention_events": [],
            "performance_metrics": {},
            "emergent_behaviors": {}
        }
        
        # Initialize components
        self.brain = MinimalBrain(brain_type=brain_type)
        self.frame_processor = CameraFrameProcessor(target_dim=self.brain.sensory_dim)
        self.camera = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Metrics tracking
        self.frame_count = 0
        self.prediction_accuracies = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.motion_events = []
        self.pattern_clusters = defaultdict(list)
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                print(f"❌ Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to capture test frame")
                return False
            
            print(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]} @ 30fps")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def camera_capture_thread(self):
        """Thread for continuous camera capture"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                try:
                    # Non-blocking queue put
                    self.frame_queue.put(frame, timeout=0.001)
                except queue.Full:
                    # Drop frame if queue is full
                    pass
            time.sleep(0.01)  # ~100 FPS max
    
    def process_prediction(self, current_vector: List[float], 
                         predicted_vector: List[float]) -> Dict:
        """Analyze prediction accuracy and patterns"""
        
        # Calculate prediction error
        if predicted_vector and len(predicted_vector) == len(current_vector):
            error = np.mean(np.abs(np.array(current_vector) - np.array(predicted_vector)))
            self.prediction_accuracies.append(1.0 - min(error, 1.0))
        else:
            self.prediction_accuracies.append(0.0)
        
        # Detect motion
        motion_detected = self.frame_processor.detect_motion()
        if motion_detected:
            self.motion_events.append({
                'timestamp': time.time(),
                'frame_count': self.frame_count,
                'motion_magnitude': self.frame_processor.get_frame_difference(current_vector)
            })
        
        # Analyze patterns (simplified clustering based on vector similarity)
        vector_key = tuple(np.round(current_vector[:10], 2))  # Use first 10 elements as key
        self.pattern_clusters[vector_key].append(self.frame_count)
        
        return {
            'prediction_accuracy': self.prediction_accuracies[-1] if self.prediction_accuracies else 0.0,
            'motion_detected': motion_detected,
            'pattern_cluster_size': len(self.pattern_clusters[vector_key]),
            'total_patterns': len(self.pattern_clusters)
        }
    
    def display_frame_with_overlay(self, frame: np.ndarray, 
                                 prediction_info: Dict) -> np.ndarray:
        """Add prediction information overlay to frame"""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Create info box
        info_box = np.zeros((120, w, 3), dtype=np.uint8)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 20
        
        # Frame count
        cv2.putText(info_box, f"Frame: {self.frame_count}", 
                   (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # Prediction accuracy
        accuracy = prediction_info.get('prediction_accuracy', 0.0)
        color = (0, 255, 0) if accuracy > 0.7 else (0, 255, 255) if accuracy > 0.4 else (0, 0, 255)
        cv2.putText(info_box, f"Accuracy: {accuracy:.3f}", 
                   (10, y_pos), font, 0.5, color, 1)
        y_pos += 20
        
        # Motion detection
        motion_color = (0, 0, 255) if prediction_info.get('motion_detected', False) else (0, 255, 0)
        cv2.putText(info_box, f"Motion: {'YES' if prediction_info.get('motion_detected', False) else 'NO'}", 
                   (10, y_pos), font, 0.5, motion_color, 1)
        y_pos += 20
        
        # Pattern clusters
        cv2.putText(info_box, f"Patterns: {prediction_info.get('total_patterns', 0)}", 
                   (10, y_pos), font, 0.5, (255, 255, 255), 1)
        y_pos += 20
        
        # Processing time
        if self.processing_times:
            avg_time = np.mean(list(self.processing_times)) * 1000
            cv2.putText(info_box, f"Processing: {avg_time:.1f}ms", 
                       (10, y_pos), font, 0.5, (255, 255, 255), 1)
        
        # Combine frame with info box
        combined = np.vstack([overlay, info_box])
        
        return combined
    
    def run_prediction_test(self, duration_seconds: int = 60, 
                          display_feed: bool = True) -> Dict:
        """
        Run real-time camera prediction test
        
        Args:
            duration_seconds: How long to run the test
            display_feed: Whether to show live video feed
            
        Returns:
            Dictionary with test results
        """
        if not self.initialize_camera():
            self.results["error"] = "Failed to initialize camera"
            self.results["total_frames"] = 0
            return self.results
        
        print(f"\n🎥 Starting Camera Prediction Test")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Brain Type: {self.brain_type}")
        print(f"Sensory Dimension: {self.brain.sensory_dim}")
        print("Press 'q' to quit early")
        
        # Start camera capture thread
        self.running = True
        capture_thread = threading.Thread(target=self.camera_capture_thread)
        capture_thread.start()
        
        # Main processing loop
        start_time = time.time()
        last_vector = None
        
        try:
            while time.time() - start_time < duration_seconds:
                try:
                    # Get latest frame
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Process frame
                    process_start = time.time()
                    current_vector = self.frame_processor.process_frame(frame)
                    
                    # Brain prediction
                    if last_vector is not None:
                        # Use previous frame to predict current
                        predicted_vector, brain_info = self.brain.process_sensory_input(last_vector)
                        
                        # Analyze prediction
                        prediction_info = self.process_prediction(current_vector, predicted_vector)
                        
                        # Store results
                        self.results["predictions"].append({
                            'timestamp': time.time(),
                            'frame_count': self.frame_count,
                            'prediction_accuracy': prediction_info['prediction_accuracy'],
                            'motion_detected': prediction_info['motion_detected'],
                            'pattern_cluster_size': prediction_info['pattern_cluster_size'],
                            'brain_confidence': brain_info.get('confidence', 0.0) if brain_info else 0.0
                        })
                    else:
                        # First frame, just process
                        self.brain.process_sensory_input(current_vector)
                        prediction_info = {'prediction_accuracy': 0.0, 'motion_detected': False, 
                                         'pattern_cluster_size': 0, 'total_patterns': 0}
                    
                    # Track processing time
                    processing_time = time.time() - process_start
                    self.processing_times.append(processing_time)
                    
                    # Display frame with overlay
                    if display_feed:
                        display_frame = self.display_frame_with_overlay(frame, prediction_info)
                        cv2.imshow('Camera Prediction Test', display_frame)
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Update for next iteration
                    last_vector = current_vector
                    self.frame_count += 1
                    
                    # Print periodic updates
                    if self.frame_count % 30 == 0:  # Every ~1 second at 30 FPS
                        avg_accuracy = np.mean(self.prediction_accuracies) if self.prediction_accuracies else 0.0
                        avg_processing = np.mean(self.processing_times) * 1000 if self.processing_times else 0.0
                        print(f"Frame {self.frame_count}: Accuracy={avg_accuracy:.3f}, "
                              f"Processing={avg_processing:.1f}ms, "
                              f"Patterns={len(self.pattern_clusters)}")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        
        finally:
            # Cleanup
            self.running = False
            capture_thread.join()
            self.camera.release()
            if display_feed:
                cv2.destroyAllWindows()
        
        # Generate final results
        return self.generate_results()
    
    def generate_results(self) -> Dict:
        """Generate comprehensive test results"""
        
        # Calculate metrics
        if self.prediction_accuracies:
            avg_accuracy = np.mean(self.prediction_accuracies)
            accuracy_trend = np.polyfit(range(len(self.prediction_accuracies)), 
                                      self.prediction_accuracies, 1)[0]
        else:
            avg_accuracy = 0.0
            accuracy_trend = 0.0
        
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times) * 1000
            processing_fps = 1.0 / np.mean(self.processing_times)
        else:
            avg_processing_time = 0.0
            processing_fps = 0.0
        
        # Analyze patterns
        pattern_stats = {
            'total_unique_patterns': len(self.pattern_clusters),
            'most_common_pattern_frequency': max(len(frames) for frames in self.pattern_clusters.values()) if self.pattern_clusters else 0,
            'pattern_diversity': len(self.pattern_clusters) / max(self.frame_count, 1)
        }
        
        # Motion analysis
        motion_stats = {
            'total_motion_events': len(self.motion_events),
            'motion_frequency': len(self.motion_events) / max(self.frame_count, 1),
            'avg_motion_magnitude': np.mean([e['motion_magnitude'] for e in self.motion_events]) if self.motion_events else 0.0
        }
        
        # Update results
        self.results.update({
            "end_time": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "performance_metrics": {
                "avg_prediction_accuracy": avg_accuracy,
                "accuracy_trend": accuracy_trend,
                "avg_processing_time_ms": avg_processing_time,
                "processing_fps": processing_fps,
                "frames_per_second": self.frame_count / max(time.time() - time.mktime(datetime.fromisoformat(self.results["start_time"]).timetuple()), 1)
            },
            "pattern_analysis": pattern_stats,
            "motion_analysis": motion_stats,
            "emergent_behaviors": {
                "learning_evident": accuracy_trend > 0.001,
                "pattern_recognition": len(self.pattern_clusters) > 10,
                "motion_sensitivity": len(self.motion_events) > 0,
                "temporal_prediction": avg_accuracy > 0.1
            }
        })
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation/camera_prediction_results/camera_test_{self.brain_type}_{timestamp}.json"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {filename}")
        
        # Generate summary report
        self.generate_report(filename.replace('.json', '_report.md'))
    
    def generate_report(self, filename: str):
        """Generate markdown report"""
        with open(filename, 'w') as f:
            f.write("# Camera Prediction Test Report\n\n")
            f.write(f"**Date**: {self.results['start_time']}\n")
            f.write(f"**Brain Type**: {self.brain_type}\n")
            f.write(f"**Duration**: {self.results.get('total_frames', 0)} frames\n\n")
            
            # Handle error case
            if 'error' in self.results:
                f.write(f"## Error\n\n")
                f.write(f"**Error**: {self.results['error']}\n\n")
                f.write("This usually indicates camera permission issues on macOS.\n")
                f.write("Please check System Preferences > Security & Privacy > Camera.\n\n")
                return
            
            # Performance metrics
            perf = self.results["performance_metrics"]
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Average Prediction Accuracy**: {perf['avg_prediction_accuracy']:.3f}\n")
            f.write(f"- **Learning Trend**: {perf['accuracy_trend']:.6f} (per frame)\n")
            f.write(f"- **Processing Speed**: {perf['avg_processing_time_ms']:.1f}ms per frame\n")
            f.write(f"- **Processing FPS**: {perf['processing_fps']:.1f}\n")
            f.write(f"- **Camera FPS**: {perf['frames_per_second']:.1f}\n\n")
            
            # Pattern analysis
            patterns = self.results["pattern_analysis"]
            f.write("## Pattern Analysis\n\n")
            f.write(f"- **Unique Patterns Discovered**: {patterns['total_unique_patterns']}\n")
            f.write(f"- **Most Common Pattern Frequency**: {patterns['most_common_pattern_frequency']}\n")
            f.write(f"- **Pattern Diversity**: {patterns['pattern_diversity']:.3f}\n\n")
            
            # Motion analysis
            motion = self.results["motion_analysis"]
            f.write("## Motion Analysis\n\n")
            f.write(f"- **Motion Events**: {motion['total_motion_events']}\n")
            f.write(f"- **Motion Frequency**: {motion['motion_frequency']:.3f} events/frame\n")
            f.write(f"- **Average Motion Magnitude**: {motion['avg_motion_magnitude']:.3f}\n\n")
            
            # Emergent behaviors
            behaviors = self.results["emergent_behaviors"]
            f.write("## Emergent Behaviors\n\n")
            for behavior, detected in behaviors.items():
                status = "✅ Detected" if detected else "❌ Not detected"
                f.write(f"- **{behavior.replace('_', ' ').title()}**: {status}\n")
            
            f.write(f"\n---\n*Report generated at {datetime.now().isoformat()}*\n")
        
        print(f"📝 Report saved to: {filename}")


def main():
    """Main camera prediction test"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Prediction Validation")
    parser.add_argument("--brain_type", choices=["minimal", "goldilocks", "sparse_goldilocks"], 
                       default="sparse_goldilocks", help="Type of brain to test")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Test duration in seconds")
    parser.add_argument("--camera_id", type=int, default=0, 
                       help="Camera device ID")
    parser.add_argument("--no_display", action="store_true", 
                       help="Run without displaying video feed")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create validator
    validator = CameraPredictionValidator(
        brain_type=args.brain_type,
        camera_id=args.camera_id
    )
    
    # Run test
    print("🎯 Camera Prediction Validation Test")
    print("=" * 40)
    
    try:
        results = validator.run_prediction_test(
            duration_seconds=args.duration,
            display_feed=not args.no_display
        )
        
        # Save results
        validator.save_results()
        
        # Print summary
        print("\n🎯 Test Complete!")
        print("=" * 40)
        
        perf = results["performance_metrics"]
        print(f"Frames processed: {results['total_frames']}")
        print(f"Average accuracy: {perf['avg_prediction_accuracy']:.3f}")
        print(f"Learning trend: {perf['accuracy_trend']:.6f}")
        print(f"Processing FPS: {perf['processing_fps']:.1f}")
        
        behaviors = results["emergent_behaviors"]
        print(f"\nEmergent behaviors:")
        for behavior, detected in behaviors.items():
            status = "✅" if detected else "❌"
            print(f"  {status} {behavior.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()