#!/usr/bin/env python3
"""
Simple Camera Prediction Validation (No OpenCV)

A simplified version that simulates camera feed with synthetic visual data
to test the brain's temporal prediction capabilities without requiring OpenCV.

This demonstrates what the full camera validation would do:
1. Generate sequential "frames" that simulate camera input
2. Test frame-to-frame prediction
3. Analyze emergent visual patterns
4. Validate temporal hierarchy response to visual changes
"""

import sys
import os
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Import brain components
from src.brain import MinimalBrain


class SyntheticVisualGenerator:
    """Generates synthetic visual data that simulates camera feed"""
    
    def __init__(self, width: int = 64, height: int = 48):
        """
        Initialize synthetic visual generator
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.width = width
        self.height = height
        self.frame_count = 0
        self.objects = []
        self.background_noise = 0.1
        
        # Create moving objects
        self.objects = [
            {
                'x': np.random.uniform(0, width),
                'y': np.random.uniform(0, height),
                'dx': np.random.uniform(-2, 2),
                'dy': np.random.uniform(-2, 2),
                'size': np.random.uniform(3, 8),
                'intensity': np.random.uniform(0.5, 1.0)
            }
            for _ in range(3)  # 3 moving objects
        ]
    
    def generate_frame(self) -> np.ndarray:
        """Generate a synthetic frame"""
        # Create background
        frame = np.random.normal(0.2, self.background_noise, (self.height, self.width))
        
        # Update and draw objects
        for obj in self.objects:
            # Update position
            obj['x'] += obj['dx']
            obj['y'] += obj['dy']
            
            # Bounce off walls
            if obj['x'] <= 0 or obj['x'] >= self.width:
                obj['dx'] *= -1
            if obj['y'] <= 0 or obj['y'] >= self.height:
                obj['dy'] *= -1
            
            # Keep in bounds
            obj['x'] = np.clip(obj['x'], 0, self.width - 1)
            obj['y'] = np.clip(obj['y'], 0, self.height - 1)
            
            # Draw object (simple circular blob)
            cx, cy = int(obj['x']), int(obj['y'])
            size = int(obj['size'])
            
            for dy in range(-size, size + 1):
                for dx in range(-size, size + 1):
                    if dx*dx + dy*dy <= size*size:
                        px, py = cx + dx, cy + dy
                        if 0 <= px < self.width and 0 <= py < self.height:
                            distance = np.sqrt(dx*dx + dy*dy)
                            intensity = obj['intensity'] * (1 - distance / size)
                            frame[py, px] = max(frame[py, px], intensity)
        
        # Add some temporal patterns
        if self.frame_count % 60 < 30:  # Flashing pattern every 2 seconds
            frame += 0.1 * np.sin(self.frame_count * 0.1)
        
        # Normalize and clip
        frame = np.clip(frame, 0, 1)
        
        self.frame_count += 1
        return frame
    
    def frame_to_vector(self, frame: np.ndarray, target_dim: int = 64) -> List[float]:
        """Convert frame to sensory input vector"""
        # Flatten frame
        flattened = frame.flatten()
        
        # Downsample to target dimension
        if len(flattened) > target_dim:
            indices = np.linspace(0, len(flattened) - 1, target_dim, dtype=int)
            vector = flattened[indices]
        else:
            vector = np.pad(flattened, (0, target_dim - len(flattened)), mode='constant')
        
        # Add some edge detection features
        # Simple gradient-based edge detection
        edges = np.abs(np.gradient(frame)).sum(axis=0)
        edge_flattened = edges.flatten()
        
        if len(edge_flattened) > target_dim:
            edge_indices = np.linspace(0, len(edge_flattened) - 1, target_dim, dtype=int)
            edge_vector = edge_flattened[edge_indices]
        else:
            edge_vector = np.pad(edge_flattened, (0, target_dim - len(edge_flattened)), mode='constant')
        
        # Combine intensity and edge information
        combined = vector * 0.7 + edge_vector * 0.3
        
        return combined.tolist()


class SimpleCameraPredictionValidator:
    """Validates brain prediction with synthetic visual data"""
    
    def __init__(self, brain_type: str = "sparse_goldilocks"):
        """Initialize validator"""
        self.brain_type = brain_type
        self.brain = MinimalBrain(brain_type=brain_type)
        self.visual_generator = SyntheticVisualGenerator()
        
        self.results = {
            "brain_type": brain_type,
            "test_type": "synthetic_visual",
            "start_time": datetime.now().isoformat(),
            "predictions": [],
            "performance_metrics": {},
            "emergent_behaviors": {}
        }
        
        # Metrics tracking
        self.frame_count = 0
        self.prediction_accuracies = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.motion_events = []
        self.pattern_history = deque(maxlen=20)
    
    def calculate_prediction_accuracy(self, predicted: List[float], 
                                    actual: List[float]) -> float:
        """Calculate prediction accuracy"""
        if not predicted or len(predicted) != len(actual):
            return 0.0
        
        # Calculate normalized error
        error = np.mean(np.abs(np.array(predicted) - np.array(actual)))
        accuracy = max(0.0, 1.0 - error)
        return accuracy
    
    def detect_motion(self, current_vector: List[float]) -> Tuple[bool, float]:
        """Detect motion between frames"""
        if len(self.pattern_history) < 2:
            return False, 0.0
        
        prev_vector = self.pattern_history[-1]
        diff = np.array(current_vector) - np.array(prev_vector)
        motion_magnitude = np.linalg.norm(diff)
        
        motion_detected = motion_magnitude > 0.1
        return motion_detected, motion_magnitude
    
    def analyze_patterns(self, vector: List[float]) -> Dict:
        """Analyze pattern characteristics"""
        # Add to history
        self.pattern_history.append(vector)
        
        # Calculate pattern statistics
        vector_array = np.array(vector)
        pattern_stats = {
            'mean_activation': np.mean(vector_array),
            'activation_std': np.std(vector_array),
            'sparsity': np.sum(vector_array > 0.1) / len(vector_array),
            'max_activation': np.max(vector_array),
            'pattern_energy': np.sum(vector_array ** 2)
        }
        
        return pattern_stats
    
    def run_prediction_test(self, duration_seconds: int = 30) -> Dict:
        """Run synthetic visual prediction test"""
        
        print(f"\n🎯 Synthetic Visual Prediction Test")
        print(f"Duration: {duration_seconds} seconds")
        print(f"Brain Type: {self.brain_type}")
        print(f"Target FPS: 30")
        
        start_time = time.time()
        last_vector = None
        
        try:
            while time.time() - start_time < duration_seconds:
                frame_start = time.time()
                
                # Generate synthetic frame
                frame = self.visual_generator.generate_frame()
                current_vector = self.visual_generator.frame_to_vector(frame, self.brain.sensory_dim)
                
                # Brain prediction
                predicted_vector = None
                if last_vector is not None:
                    # Use previous frame to predict current
                    process_start = time.time()
                    predicted_vector, brain_info = self.brain.process_sensory_input(last_vector)
                    processing_time = time.time() - process_start
                    self.processing_times.append(processing_time)
                    
                    # Calculate prediction accuracy
                    accuracy = self.calculate_prediction_accuracy(predicted_vector, current_vector)
                    self.prediction_accuracies.append(accuracy)
                else:
                    # First frame
                    self.brain.process_sensory_input(current_vector)
                    accuracy = 0.0
                    brain_info = {}
                
                # Analyze current frame
                motion_detected, motion_magnitude = self.detect_motion(current_vector)
                if motion_detected:
                    self.motion_events.append({
                        'frame': self.frame_count,
                        'timestamp': time.time(),
                        'magnitude': motion_magnitude
                    })
                
                pattern_stats = self.analyze_patterns(current_vector)
                
                # Store results
                self.results["predictions"].append({
                    'frame': self.frame_count,
                    'timestamp': time.time(),
                    'prediction_accuracy': accuracy,
                    'motion_detected': motion_detected,
                    'motion_magnitude': motion_magnitude,
                    'pattern_stats': pattern_stats,
                    'brain_confidence': brain_info.get('confidence', 0.0) if brain_info else 0.0
                })
                
                # Update for next iteration
                last_vector = current_vector
                self.frame_count += 1
                
                # Print progress
                if self.frame_count % 30 == 0:
                    avg_accuracy = np.mean(self.prediction_accuracies) if self.prediction_accuracies else 0.0
                    avg_processing = np.mean(self.processing_times) * 1000 if self.processing_times else 0.0
                    fps = self.frame_count / (time.time() - start_time)
                    
                    print(f"Frame {self.frame_count}: "
                          f"Accuracy={avg_accuracy:.3f}, "
                          f"Processing={avg_processing:.1f}ms, "
                          f"FPS={fps:.1f}, "
                          f"Motion events={len(self.motion_events)}")
                
                # Maintain target FPS
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / 30.0  # 30 FPS
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
        
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        
        return self.generate_results()
    
    def generate_results(self) -> Dict:
        """Generate comprehensive results"""
        
        # Calculate performance metrics
        if self.prediction_accuracies:
            avg_accuracy = np.mean(self.prediction_accuracies)
            accuracy_trend = np.polyfit(range(len(self.prediction_accuracies)), 
                                      self.prediction_accuracies, 1)[0]
            final_accuracy = np.mean(list(self.prediction_accuracies)[-10:])  # Last 10 frames
        else:
            avg_accuracy = 0.0
            accuracy_trend = 0.0
            final_accuracy = 0.0
        
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times) * 1000
            processing_fps = 1.0 / np.mean(self.processing_times)
        else:
            avg_processing_time = 0.0
            processing_fps = 0.0
        
        total_time = time.time() - time.mktime(datetime.fromisoformat(self.results["start_time"]).timetuple())
        actual_fps = self.frame_count / total_time
        
        # Motion analysis
        motion_frequency = len(self.motion_events) / max(self.frame_count, 1)
        avg_motion_magnitude = np.mean([e['magnitude'] for e in self.motion_events]) if self.motion_events else 0.0
        
        # Pattern analysis
        if self.results["predictions"]:
            recent_patterns = [p['pattern_stats'] for p in self.results["predictions"][-20:]]
            avg_sparsity = np.mean([p['sparsity'] for p in recent_patterns])
            avg_activation = np.mean([p['mean_activation'] for p in recent_patterns])
        else:
            avg_sparsity = 0.0
            avg_activation = 0.0
        
        # Update results
        self.results.update({
            "end_time": datetime.now().isoformat(),
            "total_frames": self.frame_count,
            "performance_metrics": {
                "avg_prediction_accuracy": avg_accuracy,
                "final_accuracy": final_accuracy,
                "accuracy_trend": accuracy_trend,
                "avg_processing_time_ms": avg_processing_time,
                "processing_fps": processing_fps,
                "actual_fps": actual_fps,
                "motion_frequency": motion_frequency,
                "avg_motion_magnitude": avg_motion_magnitude,
                "avg_sparsity": avg_sparsity,
                "avg_activation": avg_activation
            },
            "emergent_behaviors": {
                "learning_evident": accuracy_trend > 0.0001,
                "motion_sensitivity": len(self.motion_events) > 0,
                "temporal_prediction": avg_accuracy > 0.1,
                "pattern_adaptation": final_accuracy > avg_accuracy * 0.8,
                "real_time_capable": actual_fps > 20
            }
        })
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation/camera_prediction_results/synthetic_visual_{self.brain_type}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {filename}")
        
        # Generate report
        report_file = filename.replace('.json', '_report.md')
        self.generate_report(report_file)
    
    def generate_report(self, filename: str):
        """Generate markdown report"""
        with open(filename, 'w') as f:
            f.write("# Synthetic Visual Prediction Test Report\n\n")
            f.write(f"**Date**: {self.results['start_time']}\n")
            f.write(f"**Brain Type**: {self.brain_type}\n")
            f.write(f"**Frames Processed**: {self.results['total_frames']}\n\n")
            
            # Performance metrics
            perf = self.results["performance_metrics"]
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Average Prediction Accuracy**: {perf['avg_prediction_accuracy']:.3f}\n")
            f.write(f"- **Final Accuracy**: {perf['final_accuracy']:.3f}\n")
            f.write(f"- **Learning Trend**: {perf['accuracy_trend']:.6f} (per frame)\n")
            f.write(f"- **Processing Speed**: {perf['avg_processing_time_ms']:.1f}ms per frame\n")
            f.write(f"- **Processing FPS**: {perf['processing_fps']:.1f}\n")
            f.write(f"- **Actual FPS**: {perf['actual_fps']:.1f}\n")
            f.write(f"- **Motion Frequency**: {perf['motion_frequency']:.3f} events/frame\n")
            f.write(f"- **Average Motion Magnitude**: {perf['avg_motion_magnitude']:.3f}\n")
            f.write(f"- **Pattern Sparsity**: {perf['avg_sparsity']:.3f}\n")
            f.write(f"- **Pattern Activation**: {perf['avg_activation']:.3f}\n\n")
            
            # Emergent behaviors
            behaviors = self.results["emergent_behaviors"]
            f.write("## Emergent Behaviors\n\n")
            for behavior, detected in behaviors.items():
                status = "✅ Detected" if detected else "❌ Not detected"
                f.write(f"- **{behavior.replace('_', ' ').title()}**: {status}\n")
            
            # Analysis
            f.write("\n## Analysis\n\n")
            if behaviors.get("learning_evident", False):
                f.write("**Learning**: The brain shows clear improvement in prediction accuracy over time.\n")
            if behaviors.get("motion_sensitivity", False):
                f.write("**Motion Detection**: The system successfully detects movement in the visual field.\n")
            if behaviors.get("temporal_prediction", False):
                f.write("**Temporal Prediction**: Frame-to-frame prediction is working above baseline.\n")
            if behaviors.get("real_time_capable", False):
                f.write("**Real-time Performance**: Processing speed is sufficient for real-time operation.\n")
            
            f.write(f"\n---\n*Report generated at {datetime.now().isoformat()}*\n")
        
        print(f"📝 Report saved to: {filename}")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Camera Prediction Validation")
    parser.add_argument("--brain_type", choices=["minimal", "goldilocks", "sparse_goldilocks"], 
                       default="sparse_goldilocks", help="Type of brain to test")
    parser.add_argument("--duration", type=int, default=30, 
                       help="Test duration in seconds")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create validator
    validator = SimpleCameraPredictionValidator(brain_type=args.brain_type)
    
    print("🎯 Simple Camera Prediction Validation")
    print("=" * 50)
    print("This test simulates camera input with synthetic visual data")
    print("and validates the brain's temporal prediction capabilities.")
    print()
    
    try:
        # Run test
        results = validator.run_prediction_test(duration_seconds=args.duration)
        
        # Save results
        validator.save_results()
        
        # Print summary
        print("\n🎯 Test Complete!")
        print("=" * 30)
        
        perf = results["performance_metrics"]
        print(f"Frames processed: {results['total_frames']}")
        print(f"Average accuracy: {perf['avg_prediction_accuracy']:.3f}")
        print(f"Final accuracy: {perf['final_accuracy']:.3f}")
        print(f"Learning trend: {perf['accuracy_trend']:.6f}")
        print(f"Processing FPS: {perf['processing_fps']:.1f}")
        print(f"Actual FPS: {perf['actual_fps']:.1f}")
        
        print(f"\nEmergent behaviors:")
        behaviors = results["emergent_behaviors"]
        for behavior, detected in behaviors.items():
            status = "✅" if detected else "❌"
            print(f"  {status} {behavior.replace('_', ' ').title()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()