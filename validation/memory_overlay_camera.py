#!/usr/bin/env python3
"""
Memory Overlay Camera - Visualize Brain's Internal Memories on Live Feed

This extends the brain visualization to show what the brain has "learned" and 
"remembers" overlaid directly on the camera feed:

1. Memory Reconstruction - Show what the brain "remembers" about similar scenes
2. Expectation Overlay - Show what the brain expects to happen next
3. Pattern Similarity - Highlight areas similar to learned patterns
4. Feature Detection - Show which learned features are currently active
5. Prediction Confidence - Show where the brain is confident vs uncertain
6. Memory Clusters - Show which memory category the current scene belongs to

This creates a "memory-aware" camera that shows the brain's internal model
of the world overlaid on reality.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Import brain components
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector


class MemoryOverlayGenerator:
    """Generates memory-based overlays for the camera feed"""
    
    def __init__(self, memory_inspector: MemoryInspector, frame_width: int = 640, frame_height: int = 360):
        """Initialize memory overlay generator"""
        self.memory_inspector = memory_inspector
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Overlay settings
        self.show_memory_reconstruction = True
        self.show_expectation_overlay = True
        self.show_pattern_similarity = True
        self.show_feature_detection = True
        self.show_memory_clusters = True
        self.show_edge_detection = True
        
        # Color schemes
        self.colors = {
            'memory': (255, 255, 0),      # Cyan for memory reconstruction
            'expectation': (0, 255, 0),   # Green for expectations
            'similarity': (255, 0, 255),  # Magenta for pattern similarity
            'features': (0, 255, 255),    # Yellow for features
            'clusters': [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Different colors for clusters
        }
        
        # Memory cache for performance
        self.memory_cache = {}
        self.last_cluster_analysis = None
        self.last_feature_analysis = None
        
    def create_memory_overlay(self, frame: np.ndarray, current_pattern: List[float], 
                            brain_activity: Dict, high_res_frame: np.ndarray = None, 
                            edge_frame: np.ndarray = None) -> np.ndarray:
        """Create comprehensive memory overlay with adaptive attention and edge detection"""
        # ALWAYS start with the original camera frame to ensure it's visible
        overlay = frame.copy()
        
        # Convert current pattern to numpy array
        current_array = np.array(current_pattern)
        
        # Always show attention heatmap (even with few memories)
        try:
            # Add adaptive attention heatmap with variable window sizes
            overlay = self._add_focused_attention_heatmap(overlay, current_array, brain_activity, high_res_frame)
        except Exception as e:
            print(f"Attention heatmap error: {e}")
        
        # Add edge detection visualization (if enabled)
        if edge_frame is not None and self.show_edge_detection:
            try:
                overlay = self._add_edge_detection_overlay(overlay, edge_frame)
            except Exception as e:
                print(f"Edge detection overlay error: {e}")
        
        # Show memory count and resolution info
        h, w = overlay.shape[:2]
        memory_count = len(self.memory_inspector.memory_samples)
        cv2.putText(overlay, f"Memories: {memory_count}", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show brain input resolution
        brain_res = f"{int(np.sqrt(len(current_pattern)))}x{int(np.sqrt(len(current_pattern)))}"
        cv2.putText(overlay, f"Brain: {brain_res}", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Only apply other overlays if we have some memories
        if memory_count >= 3:
            # Apply overlays very conservatively to preserve camera feed visibility
            try:
                # Add memory reconstruction overlay (most subtle)
                if self.show_memory_reconstruction:
                    overlay = self._add_memory_reconstruction(overlay, current_array, brain_activity)
                
                # Add expectation overlay (arrows and indicators only)
                if self.show_expectation_overlay:
                    overlay = self._add_expectation_overlay(overlay, current_array, brain_activity)
                
                # Add pattern similarity overlay (very subtle)
                if self.show_pattern_similarity:
                    overlay = self._add_pattern_similarity(overlay, current_array)
                
                # Add feature detection overlay (side bars only)
                if self.show_feature_detection:
                    overlay = self._add_feature_detection(overlay, current_array)
                
                # Add memory cluster overlay (border only)
                if self.show_memory_clusters:
                    overlay = self._add_memory_cluster(overlay, current_array)
                    
            except Exception as e:
                # If any overlay fails, continue with just the attention heatmap
                print(f"Overlay error: {e}")
        else:
            # Show building memories message
            cv2.putText(overlay, "Building memories...", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return overlay
    
    def _add_edge_detection_overlay(self, overlay: np.ndarray, edge_frame: np.ndarray) -> np.ndarray:
        """Add edge detection visualization overlay"""
        h, w = overlay.shape[:2]
        
        # Resize edge frame to match overlay
        edge_resized = cv2.resize(edge_frame, (w, h))
        
        # Create colored edge overlay
        edge_colored = cv2.applyColorMap((edge_resized * 255).astype(np.uint8), cv2.COLORMAP_COOL)
        
        # Apply edge overlay with transparency
        edge_mask = edge_resized > 0.1  # Only show strong edges
        if np.any(edge_mask):
            alpha = 0.3  # 30% opacity for edges
            overlay[edge_mask] = overlay[edge_mask] * (1 - alpha) + edge_colored[edge_mask] * alpha
        
        # Add edge detection info
        edge_count = np.sum(edge_mask)
        edge_density = edge_count / (w * h)
        
        cv2.putText(overlay, f"Edges: {edge_count}", 
                   (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(overlay, f"Density: {edge_density:.3f}", 
                   (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add contour detection and visualization
        try:
            contours, _ = cv2.findContours(
                (edge_resized * 255).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filter contours by size (remove noise)
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
            
            if significant_contours:
                # Draw contours
                cv2.drawContours(overlay, significant_contours, -1, (0, 255, 255), 2)
                
                # Show contour info
                cv2.putText(overlay, f"Contours: {len(significant_contours)}", 
                           (w - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Show largest contour info
                if significant_contours:
                    largest_contour = max(significant_contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    cv2.putText(overlay, f"Max area: {area:.0f}", 
                               (w - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        except:
            pass  # Skip contour detection if it fails
        
        return overlay
    
    def _add_focused_attention_heatmap(self, overlay: np.ndarray, current_pattern: np.ndarray, 
                                     brain_activity: Dict, high_res_frame: np.ndarray = None) -> np.ndarray:
        """Add adaptive attention heatmap with variable window sizes"""
        h, w = overlay.shape[:2]
        
        # Use high-res frame for attention if available, otherwise fall back to overlay
        if high_res_frame is not None:
            attention_source = cv2.resize(high_res_frame, (w, h))
        else:
            attention_source = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) / 255.0
        
        # Calculate adaptive attention map
        attention_map, attention_windows = self._calculate_adaptive_attention_map(
            current_pattern, brain_activity, attention_source
        )
        
        if attention_map is None:
            return overlay
        
        # Adaptive thresholding based on content complexity
        complexity = np.std(attention_source)
        if complexity > 0.2:  # High complexity scene
            attention_threshold = 0.6  # Lower threshold for complex scenes
        else:  # Simple scene
            attention_threshold = 0.8  # Higher threshold for simple scenes
            
        attention_mask = attention_map > attention_threshold
        
        if not np.any(attention_mask):
            # Fallback: show some attention anyway
            attention_threshold = 0.5
            attention_mask = attention_map > attention_threshold
        
        if np.any(attention_mask):
            # Enhance the strong attention areas dramatically
            enhanced_attention = np.zeros_like(attention_map)
            enhanced_attention[attention_mask] = (attention_map[attention_mask] - attention_threshold) / (1.0 - attention_threshold)
            enhanced_attention = np.power(enhanced_attention, 0.4)  # More gradual enhancement
            
            # Create colored heatmap for focus areas only
            heatmap_colored = cv2.applyColorMap((enhanced_attention * 255).astype(np.uint8), cv2.COLORMAP_HOT)
            
            # Apply spotlight effect: blend only where attention is high
            for y in range(h):
                for x in range(w):
                    if attention_mask[y, x]:
                        alpha = enhanced_attention[y, x] * 0.5  # Reduced opacity for better visibility
                        overlay[y, x] = overlay[y, x] * (1 - alpha) + heatmap_colored[y, x] * alpha
            
            # Draw attention windows with variable sizes
            self._draw_attention_windows(overlay, attention_windows)
        
        # Add attention strength indicator
        max_attention = np.max(attention_map)
        hotspot_count = np.sum(attention_mask)
        total_attention_area = np.sum(attention_map > 0.3)  # Count all moderate attention areas
        
        cv2.putText(overlay, f"Focus: {max_attention:.2f}", 
                   (w - 150, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(overlay, f"Hotspots: {hotspot_count}", 
                   (w - 150, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(overlay, f"Attended: {total_attention_area}", 
                   (w - 150, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return overlay
    
    def _calculate_adaptive_attention_map(self, current_pattern: np.ndarray, brain_activity: Dict, 
                                        attention_source: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Calculate adaptive attention map with variable window sizes (optimized for real-time)"""
        try:
            h, w = attention_source.shape[:2]
            
            # Start with broad pre-attentive processing (entire image)
            attention_map = np.ones((h, w)) * 0.3  # Base attention level
            
            # Efficient edge detection with reduced blur
            edges = cv2.Canny((attention_source * 255).astype(np.uint8), 50, 150)
            edge_attention = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (7, 7), 0)  # Reduced blur
            attention_map += edge_attention * 0.4
            
            # Simplified gradient computation (faster)
            grad_x = cv2.Sobel(attention_source, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(attention_source, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_attention = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)  # Reduced blur
            attention_map += gradient_attention * 0.3
            
            # Add brain response if available (simplified)
            brain_output = brain_activity.get('brain_output')
            if brain_output is not None:
                try:
                    brain_array = np.array(brain_output)
                    # Simplified brain spatial mapping
                    brain_resized = cv2.resize(brain_array.reshape(int(np.sqrt(len(brain_array))), -1), (w, h))
                    attention_map += brain_resized * 0.4
                except:
                    pass  # Skip if dimensions don't match
            
            # Add novelty-based attention boost (only if we have enough history)
            if len(self.memory_inspector.pattern_history) > 10:  # Increased threshold
                novelty_score = self.memory_inspector._calculate_novelty_score(current_pattern.tolist())
                attention_map = attention_map * (0.6 + novelty_score * 0.8)  # Higher novelty boost
            
            # Normalize and clip
            attention_map = np.clip(attention_map, 0, 1)
            
            # Simplified attention window identification (faster)
            attention_windows = self._identify_attention_windows_fast(attention_map, attention_source)
            
            return attention_map, attention_windows
            
        except Exception as e:
            print(f"Adaptive attention calculation error: {e}")
            return None, []
    
    def _identify_attention_windows(self, attention_map: np.ndarray, 
                                  source_image: np.ndarray) -> List[Dict]:
        """Identify variable-sized attention windows based on content"""
        h, w = attention_map.shape
        windows = []
        
        # Find peaks in attention map
        threshold = 0.6
        attention_peaks = attention_map > threshold
        
        if not np.any(attention_peaks):
            # If no strong peaks, create broad attention window
            return [{
                'x': w // 4, 'y': h // 4,
                'width': w // 2, 'height': h // 2,
                'type': 'broad', 'strength': np.mean(attention_map)
            }]
        
        # Find connected components (attention regions)
        attention_uint8 = (attention_peaks * 255).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(attention_uint8)
        
        for label_id in range(1, num_labels):  # Skip background (0)
            mask = labels == label_id
            y_coords, x_coords = np.where(mask)
            
            if len(x_coords) < 10:  # Skip tiny regions
                continue
            
            # Calculate bounding box
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # Calculate window size based on content and attention strength
            region_attention = np.mean(attention_map[mask])
            region_complexity = np.std(source_image[y_min:y_max+1, x_min:x_max+1])
            
            # Adaptive window sizing
            base_width = x_max - x_min + 1
            base_height = y_max - y_min + 1
            
            if region_complexity > 0.15:  # Complex region - larger window
                window_width = min(w, int(base_width * 1.8))
                window_height = min(h, int(base_height * 1.8))
                window_type = 'detailed'
            elif region_attention > 0.8:  # High attention - medium window
                window_width = min(w, int(base_width * 1.4))
                window_height = min(h, int(base_height * 1.4))
                window_type = 'focused'
            else:  # Low attention - small window
                window_width = min(w, int(base_width * 1.1))
                window_height = min(h, int(base_height * 1.1))
                window_type = 'minimal'
            
            # Center the window on the attention region
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            window_x = max(0, center_x - window_width // 2)
            window_y = max(0, center_y - window_height // 2)
            
            # Ensure window doesn't exceed image bounds
            window_x = min(window_x, w - window_width)
            window_y = min(window_y, h - window_height)
            
            windows.append({
                'x': window_x, 'y': window_y,
                'width': window_width, 'height': window_height,
                'type': window_type, 'strength': region_attention,
                'complexity': region_complexity
            })
        
        return windows
    
    def _identify_attention_windows_fast(self, attention_map: np.ndarray, 
                                       source_image: np.ndarray) -> List[Dict]:
        """Fast attention window identification for real-time processing"""
        h, w = attention_map.shape
        windows = []
        
        # Simplified approach: find top attention regions without connected components
        threshold = 0.6
        attention_peaks = attention_map > threshold
        
        if not np.any(attention_peaks):
            # If no strong peaks, create single broad attention window
            return [{
                'x': w // 4, 'y': h // 4,
                'width': w // 2, 'height': h // 2,
                'type': 'broad', 'strength': np.mean(attention_map),
                'complexity': 0.0
            }]
        
        # Find peak locations using morphological operations (faster than connected components)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(attention_peaks.astype(np.uint8), kernel, iterations=1)
        
        # Find contours (faster than connected components for simple shapes)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            if w_rect < 5 or h_rect < 5:  # Skip tiny regions
                continue
            
            # Calculate attention strength in this region
            region_attention = np.mean(attention_map[y:y+h_rect, x:x+w_rect])
            
            # Simple complexity measure (standard deviation)
            region_complexity = np.std(source_image[y:y+h_rect, x:x+w_rect])
            
            # Simplified window sizing (faster)
            if region_attention > 0.8:
                window_type = 'focused'
                window_width = min(w, int(w_rect * 1.5))
                window_height = min(h, int(h_rect * 1.5))
            elif region_complexity > 0.15:
                window_type = 'detailed'
                window_width = min(w, int(w_rect * 1.3))
                window_height = min(h, int(h_rect * 1.3))
            else:
                window_type = 'minimal'
                window_width = min(w, int(w_rect * 1.1))
                window_height = min(h, int(h_rect * 1.1))
            
            # Center the window
            center_x = x + w_rect // 2
            center_y = y + h_rect // 2
            
            window_x = max(0, min(w - window_width, center_x - window_width // 2))
            window_y = max(0, min(h - window_height, center_y - window_height // 2))
            
            windows.append({
                'x': window_x, 'y': window_y,
                'width': window_width, 'height': window_height,
                'type': window_type, 'strength': region_attention,
                'complexity': region_complexity
            })
        
        # Limit to top 5 windows to avoid overcrowding
        if len(windows) > 5:
            windows = sorted(windows, key=lambda w: w['strength'], reverse=True)[:5]
        
        return windows
    
    def _draw_attention_windows(self, overlay: np.ndarray, windows: List[Dict]):
        """Draw variable-sized attention windows on overlay"""
        for window in windows:
            x, y = window['x'], window['y']
            w, h = window['width'], window['height']
            window_type = window['type']
            strength = window['strength']
            
            # Color coding by window type
            if window_type == 'detailed':
                color = (0, 255, 255)  # Cyan for detailed attention
                thickness = 3
            elif window_type == 'focused':
                color = (0, 255, 0)    # Green for focused attention
                thickness = 2
            else:  # minimal
                color = (255, 255, 0)  # Yellow for minimal attention
                thickness = 1
            
            # Draw window rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            # Add window info
            cv2.putText(overlay, f"{window_type[:3]}", 
                       (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(overlay, f"{strength:.2f}", 
                       (x + 2, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _calculate_attention_map(self, current_pattern: np.ndarray, brain_activity: Dict, 
                               width: int, height: int) -> np.ndarray:
        """Calculate attention map from brain activity and pattern"""
        try:
            # Get brain output and info
            brain_output = brain_activity.get('brain_output')
            brain_info = brain_activity.get('brain_info', {})
            
            if brain_output is None:
                return None
            
            # Convert brain output to spatial attention map
            brain_array = np.array(brain_output) if isinstance(brain_output, list) else brain_output
            
            # Create base attention from pattern activation
            pattern_spatial = self._pattern_to_spatial(current_pattern, width, height)
            
            # Create attention from brain output (handle different dimensions)
            try:
                brain_spatial = self._pattern_to_spatial(brain_array, width, height)
                # Combine pattern activation and brain response
                attention_map = pattern_spatial * 0.4 + brain_spatial * 0.6
            except:
                # If brain output doesn't match, use pattern activation only
                attention_map = pattern_spatial
            
            # Add temporal prediction attention
            if hasattr(self.memory_inspector, 'prediction_history') and len(self.memory_inspector.prediction_history) > 1:
                recent_predictions = list(self.memory_inspector.prediction_history)[-3:]
                if recent_predictions:
                    try:
                        pred_array = np.mean([np.array(p) for p in recent_predictions if p is not None], axis=0)
                        pred_spatial = self._pattern_to_spatial(pred_array, width, height)
                        attention_map += pred_spatial * 0.3
                    except:
                        # Skip prediction attention if shapes don't match
                        pass
            
            # Add novelty-based attention boost
            if len(self.memory_inspector.pattern_history) > 5:
                novelty_score = self.memory_inspector._calculate_novelty_score(current_pattern.tolist())
                attention_map = attention_map * (0.5 + novelty_score * 0.5)
            
            # Smooth and normalize
            attention_map = cv2.GaussianBlur(attention_map.astype(np.float32), (15, 15), 0)
            attention_map = np.clip(attention_map, 0, 1)
            
            # Add some randomness to prevent static patterns
            noise = np.random.normal(0, 0.05, attention_map.shape)
            attention_map = np.clip(attention_map + noise, 0, 1)
            
            return attention_map
            
        except Exception as e:
            print(f"Attention calculation error: {e}")
            return None
    
    def _add_memory_reconstruction(self, overlay: np.ndarray, current_pattern: np.ndarray, 
                                 brain_activity: Dict) -> np.ndarray:
        """Add memory reconstruction overlay showing what the brain 'remembers'"""
        h, w = overlay.shape[:2]
        
        # Check if we have enough memories for prototypes
        if len(self.memory_inspector.memory_samples) < 3:
            # Not enough memories - show current status instead
            cv2.putText(overlay, "Building memories...", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(overlay, f"Memories: {len(self.memory_inspector.memory_samples)}", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            return overlay
        
        # Get memory prototypes
        prototypes = self.memory_inspector.reconstruct_memory_prototypes(n_prototypes=3)
        
        if 'error' in prototypes:
            return overlay
        
        # Find most similar memory prototype
        best_similarity = -1
        best_prototype = None
        
        for prototype in prototypes['prototypes']:
            prototype_pattern = np.array(prototype['prototype_pattern'])
            similarity = cosine_similarity([current_pattern], [prototype_pattern])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_prototype = prototype
        
        if best_prototype and best_similarity > 0.5:  # Higher threshold for visibility
            # Show memory information as text and subtle indicators only
            memory_color = self.colors['memory']
            
            # Just add text indicators - no full-frame overlays
            cv2.putText(overlay, f"Memory: {best_similarity:.2f}", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, memory_color, 2)
            cv2.putText(overlay, f"Cluster: {best_prototype['cluster_size']} experiences", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, memory_color, 1)
            
            # Add subtle corner indicators instead of full overlay
            corner_size = int(30 * best_similarity)
            cv2.rectangle(overlay, (0, 0), (corner_size, corner_size), memory_color, 3)
            cv2.rectangle(overlay, (w - corner_size, 0), (w, corner_size), memory_color, 3)
            
            # Optional: Add very subtle pattern hints as small dots
            memory_pattern = best_prototype['prototype_pattern']
            memory_visual = self._pattern_to_spatial(memory_pattern, 20, 20)  # Much smaller
            
            # Show pattern as small visualization in corner
            try:
                pattern_preview = cv2.resize(memory_visual, (60, 60))
                pattern_preview = (pattern_preview * 255).astype(np.uint8)
                pattern_preview = cv2.applyColorMap(pattern_preview, cv2.COLORMAP_JET)
                
                # Place in top-right corner
                overlay[10:70, w-70:w-10] = cv2.addWeighted(
                    overlay[10:70, w-70:w-10], 0.7, pattern_preview, 0.3, 0
                )
            except:
                pass  # Skip pattern preview if it fails
        
        return overlay
    
    def _add_expectation_overlay(self, overlay: np.ndarray, current_pattern: np.ndarray, 
                               brain_activity: Dict) -> np.ndarray:
        """Add expectation overlay showing what the brain expects to happen"""
        h, w = overlay.shape[:2]
        
        # Get expectation analysis
        expectations = self.memory_inspector.analyze_expectations()
        
        if 'error' in expectations:
            return overlay
        
        # Get prediction from brain activity
        prediction = brain_activity.get('brain_output')
        if prediction is None:
            return overlay
        
        # Apply expectation strength
        expectation_strength = expectations.get('expectation_strength', 0.5)
        expectation_color = self.colors['expectation']
        
        # Only show if expectation strength is significant
        if expectation_strength > 0.3:
            # Create directional expectation indicators (arrows only)
            prediction_stability = expectations.get('prediction_stability', 0.0)
            if prediction_stability > 0.5:
                # Draw arrows showing expected movement/change
                center_x, center_y = w // 2, h // 2
                arrow_length = int(40 * prediction_stability)
                
                # Calculate arrow direction from prediction trend
                trends = expectations.get('prediction_trends', {})
                if 'overall_trend' in trends:
                    angle = trends['overall_trend'] * 180  # Convert to degrees
                    end_x = center_x + int(arrow_length * np.cos(np.radians(angle)))
                    end_y = center_y + int(arrow_length * np.sin(np.radians(angle)))
                    
                    cv2.arrowedLine(overlay, (center_x, center_y), (end_x, end_y), 
                                   expectation_color, 3, tipLength=0.3)
                    
                    # Add prediction confidence circle
                    radius = int(20 * expectation_strength)
                    cv2.circle(overlay, (center_x, center_y), radius, expectation_color, 2)
            
            # Add expectation info (text only)
            cv2.putText(overlay, f"Expect: {expectation_strength:.2f}", 
                       (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, expectation_color, 2)
            cv2.putText(overlay, f"Stable: {prediction_stability:.2f}", 
                       (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, expectation_color, 1)
        
        return overlay
    
    def _add_pattern_similarity(self, overlay: np.ndarray, current_pattern: np.ndarray) -> np.ndarray:
        """Add pattern similarity overlay showing similar past experiences"""
        h, w = overlay.shape[:2]
        
        if len(self.memory_inspector.pattern_history) < 10:
            return overlay
        
        # Find most similar patterns from history
        pattern_array = np.array(list(self.memory_inspector.pattern_history))
        similarities = cosine_similarity([current_pattern], pattern_array)[0]
        
        # Get top 5 most similar patterns
        top_indices = np.argsort(similarities)[-6:-1]  # Exclude the current pattern
        top_similarities = similarities[top_indices]
        
        # Create similarity visualization
        similarity_strength = np.mean(top_similarities)
        
        if similarity_strength > 0.6:  # Higher threshold, only show strong similarities
            similarity_color = self.colors['similarity']
            
            # Add subtle similarity indicators as corner dots
            dot_size = int(5 + similarity_strength * 10)
            cv2.circle(overlay, (w - 30, h - 30), dot_size, similarity_color, -1)
            cv2.circle(overlay, (30, h - 30), dot_size, similarity_color, 2)
        
        # Add similarity info (text only)
        if similarity_strength > 0.4:  # Show text for moderate similarities
            cv2.putText(overlay, f"Similar: {similarity_strength:.2f}", 
                       (w - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['similarity'], 2)
        
        return overlay
    
    def _add_feature_detection(self, overlay: np.ndarray, current_pattern: np.ndarray) -> np.ndarray:
        """Add feature detection overlay showing active learned features"""
        h, w = overlay.shape[:2]
        
        # Get learned features (cache for performance)
        if self.last_feature_analysis is None or time.time() - self.last_feature_analysis.get('timestamp', 0) > 5:
            features = self.memory_inspector.extract_learned_features()
            features['timestamp'] = time.time()
            self.last_feature_analysis = features
        else:
            features = self.last_feature_analysis
        
        if 'error' in features:
            return overlay
        
        # Project current pattern onto learned features
        feature_components = np.array(features['feature_components'])
        feature_activations = np.dot(current_pattern, feature_components.T)
        
        # Show top 3 most active features
        top_feature_indices = np.argsort(np.abs(feature_activations))[-3:]
        
        feature_color = self.colors['features']
        
        # Draw feature activation indicators
        for i, feature_idx in enumerate(top_feature_indices):
            activation = feature_activations[feature_idx]
            importance = features['feature_importance'][feature_idx]
            
            # Draw feature indicator
            x_pos = 10 + i * 60
            y_pos = 100
            
            # Draw feature strength bar
            bar_height = int(50 * abs(activation))
            bar_color = feature_color if activation > 0 else (0, 0, 255)  # Red for negative
            
            cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + 20, y_pos - bar_height), 
                         bar_color, -1)
            cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + 20, y_pos - 50), 
                         feature_color, 1)
            
            # Add feature label
            cv2.putText(overlay, f"F{feature_idx}", 
                       (x_pos, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, feature_color, 1)
            cv2.putText(overlay, f"{activation:.2f}", 
                       (x_pos - 10, y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, feature_color, 1)
        
        return overlay
    
    def _add_memory_cluster(self, overlay: np.ndarray, current_pattern: np.ndarray) -> np.ndarray:
        """Add memory cluster overlay showing which category the current scene belongs to"""
        h, w = overlay.shape[:2]
        
        # Get cluster analysis (cache for performance)
        if self.last_cluster_analysis is None or time.time() - self.last_cluster_analysis.get('timestamp', 0) > 3:
            clusters = self.memory_inspector.analyze_pattern_clusters()
            clusters['timestamp'] = time.time()
            self.last_cluster_analysis = clusters
        else:
            clusters = self.last_cluster_analysis
        
        if 'error' in clusters:
            return overlay
        
        # Find which cluster the current pattern belongs to
        cluster_centers = clusters['cluster_centers']
        distances = [np.linalg.norm(current_pattern - center) for center in cluster_centers]
        closest_cluster = np.argmin(distances)
        cluster_distance = distances[closest_cluster]
        
        # Get cluster info
        cluster_info = clusters['cluster_statistics'][closest_cluster]
        cluster_color = self.colors['clusters'][closest_cluster % len(self.colors['clusters'])]
        
        # Draw cluster indicator
        confidence = 1.0 / (1.0 + cluster_distance)  # Convert distance to confidence
        
        # Draw cluster border
        border_thickness = int(5 * confidence)
        if border_thickness > 0:
            cv2.rectangle(overlay, (5, 5), (w - 5, h - 5), cluster_color, border_thickness)
        
        # Draw cluster info
        cv2.putText(overlay, f"Cluster {closest_cluster + 1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_color, 2)
        cv2.putText(overlay, f"Size: {cluster_info['size']}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cluster_color, 1)
        cv2.putText(overlay, f"Confidence: {confidence:.2f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cluster_color, 1)
        
        return overlay
    
    def _pattern_to_spatial(self, pattern: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert 1D pattern to 2D spatial representation"""
        # Reshape pattern to approximate spatial layout
        pattern_size = len(pattern)
        grid_size = int(np.sqrt(pattern_size))
        
        if grid_size * grid_size < pattern_size:
            grid_size += 1
        
        # Pad pattern if needed
        padded_pattern = np.zeros(grid_size * grid_size)
        padded_pattern[:pattern_size] = pattern
        
        # Reshape to 2D
        spatial_pattern = padded_pattern.reshape(grid_size, grid_size)
        
        # Normalize to [0, 1]
        if np.max(spatial_pattern) > 0:
            spatial_pattern = spatial_pattern / np.max(spatial_pattern)
        
        return spatial_pattern


class MemoryOverlayCamera:
    """Camera with memory overlay visualization"""
    
    def __init__(self, brain_type: str = "sparse_goldilocks", camera_id: int = 0):
        """Initialize memory overlay camera"""
        self.brain_type = brain_type
        self.camera_id = camera_id
        
        # Initialize components
        self.brain = MinimalBrain(brain_type=brain_type)
        self.memory_inspector = MemoryInspector(self.brain)
        self.overlay_generator = MemoryOverlayGenerator(self.memory_inspector)
        
        # Camera setup
        self.camera = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Statistics
        self.frame_count = 0
        self.processing_times = deque(maxlen=50)
        
        # Control settings
        self.overlay_mode = 'all'  # 'all', 'memory', 'expectation', 'similarity', 'features', 'clusters'
        self.show_edge_detection = True  # Toggle edge detection visualization
        
        # Activity tracking for quiet periods
        self.recent_changes = deque(maxlen=30)  # Track frame-to-frame changes
        self.enable_async_maintenance = False  # Can be toggled
        self.last_input = None
        
    def initialize_camera(self) -> bool:
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
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
    
    def process_frame_to_brain_input(self, frame: np.ndarray) -> Tuple[List[float], np.ndarray, np.ndarray]:
        """Convert frame to brain input with much higher resolution and edge detection"""
        # High-resolution version for attention calculation (128x128)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        high_res = cv2.resize(gray, (128, 128))
        high_res_normalized = high_res.astype(np.float32) / 255.0
        
        # Edge detection for visualization
        edges = cv2.Canny(high_res, 50, 150)
        edge_normalized = edges.astype(np.float32) / 255.0
        
        # Create MUCH higher resolution brain input (64x64 = 4096 pixels vs 16)
        brain_size = 64  # Force 64x64 for meaningful contour detection
        brain_resolution = cv2.resize(gray, (brain_size, brain_size))
        brain_normalized = brain_resolution.astype(np.float32) / 255.0
        brain_flattened = brain_normalized.flatten()
        
        # Handle brain dimension mismatch by expanding brain capacity
        target_dim = self.brain.sensory_dim
        if len(brain_flattened) > target_dim:
            # Use more sophisticated sampling to preserve spatial structure
            # Sample in a grid pattern rather than linear
            grid_size = int(np.sqrt(target_dim))
            if grid_size * grid_size == target_dim:
                # Perfect square - use grid sampling
                step_x = brain_size // grid_size
                step_y = brain_size // grid_size
                sampled = []
                for y in range(0, brain_size, step_y):
                    for x in range(0, brain_size, step_x):
                        if len(sampled) < target_dim:
                            sampled.append(brain_normalized[y, x])
                sampled = np.array(sampled)
            else:
                # Non-square - use linear sampling
                indices = np.linspace(0, len(brain_flattened) - 1, target_dim, dtype=int)
                sampled = brain_flattened[indices]
        else:
            sampled = np.pad(brain_flattened, (0, target_dim - len(brain_flattened)), mode='constant')
        
        return sampled.tolist(), high_res_normalized, edge_normalized
    
    def update_overlay_mode(self, new_mode: str):
        """Update overlay visualization mode"""
        self.overlay_mode = new_mode
        
        # Update overlay generator settings
        self.overlay_generator.show_memory_reconstruction = new_mode in ['all', 'memory']
        self.overlay_generator.show_expectation_overlay = new_mode in ['all', 'expectation']
        self.overlay_generator.show_pattern_similarity = new_mode in ['all', 'similarity']
        self.overlay_generator.show_feature_detection = new_mode in ['all', 'features']
        self.overlay_generator.show_memory_clusters = new_mode in ['all', 'clusters']
    
    def _calculate_activity_level(self, brain_input: List[float]) -> float:
        """Calculate activity level based on frame-to-frame changes"""
        if self.last_input is None:
            self.last_input = brain_input
            return 0.5  # Neutral for first frame
        
        # Calculate change from last frame
        current = np.array(brain_input)
        previous = np.array(self.last_input)
        
        # Normalized change
        change = np.linalg.norm(current - previous) / (np.linalg.norm(current) + np.linalg.norm(previous) + 1e-6)
        
        # Store for next frame
        self.last_input = brain_input
        
        # Track recent changes
        self.recent_changes.append(change)
        
        # Calculate activity level (smoothed over recent frames)
        if len(self.recent_changes) > 5:
            avg_change = np.mean(list(self.recent_changes)[-10:])
            # Map to 0-1 range with reasonable scaling
            activity = np.clip(avg_change * 5.0, 0.0, 1.0)
        else:
            activity = change
        
        return activity
    
    def run_memory_visualization(self):
        """Run memory overlay visualization"""
        if not self.initialize_camera():
            print("❌ Failed to initialize camera")
            return
        
        print("🧠 Starting Memory Overlay Visualization")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '1' - Show all overlays")
        print("  '2' - Show memory reconstruction only")
        print("  '3' - Show expectations only")
        print("  '4' - Show pattern similarity only")
        print("  '5' - Show learned features only")
        print("  '6' - Show memory clusters only")
        print("  'r' - Generate memory report")
        print("  'm' - Toggle async maintenance")
        print("  'e' - Toggle edge detection visualization")
        print("=" * 50)
        
        # Start camera thread
        self.running = True
        capture_thread = threading.Thread(target=self.camera_capture_thread)
        capture_thread.start()
        
        try:
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    start_time = time.time()
                    
                    # Convert to brain input with high-resolution preservation and edge detection
                    brain_input, high_res_frame, edge_frame = self.process_frame_to_brain_input(frame)
                    
                    # Calculate activity level (frame-to-frame change)
                    activity_level = self._calculate_activity_level(brain_input)
                    
                    # Update memory inspector's activity tracking
                    self.memory_inspector.update_activity_level(activity_level)
                    
                    # Process through brain
                    brain_output, brain_info = self.brain.process_sensory_input(brain_input)
                    
                    # Calculate attention map for memory gating using high-res frame
                    current_array = np.array(brain_input)
                    brain_activity = {
                        'brain_output': brain_output,
                        'brain_info': brain_info,
                        'timestamp': time.time()
                    }
                    
                    # Use adaptive attention calculation with high-res frame
                    attention_map, attention_windows = self.overlay_generator._calculate_adaptive_attention_map(
                        current_array, brain_activity, high_res_frame
                    )
                    
                    # Capture memory snapshot with attention gating
                    self.memory_inspector.capture_memory_snapshot(
                        brain_input, brain_output, brain_info, attention_map
                    )
                    
                    # Create memory overlay with high-res attention and edge detection
                    visualization = self.overlay_generator.create_memory_overlay(
                        frame, brain_input, brain_activity, high_res_frame, edge_frame
                    )
                    
                    # Add mode indicator
                    cv2.putText(visualization, f"Mode: {self.overlay_mode}", 
                               (10, visualization.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Track processing time
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                    # Add FPS counter
                    if self.processing_times:
                        fps = 1.0 / np.mean(self.processing_times)
                        cv2.putText(visualization, f"FPS: {fps:.1f}", 
                                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display
                    cv2.imshow('Memory Overlay Visualization', visualization)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshots_dir = "validation/brain_screenshots"
                        os.makedirs(screenshots_dir, exist_ok=True)
                        filename = f"{screenshots_dir}/memory_overlay_{timestamp}.png"
                        cv2.imwrite(filename, visualization)
                        print(f"📸 Screenshot saved: {filename}")
                    elif key == ord('1'):
                        self.update_overlay_mode('all')
                        print("🎯 Mode: All overlays")
                    elif key == ord('2'):
                        self.update_overlay_mode('memory')
                        print("🧠 Mode: Memory reconstruction")
                    elif key == ord('3'):
                        self.update_overlay_mode('expectation')
                        print("🔮 Mode: Expectations")
                    elif key == ord('4'):
                        self.update_overlay_mode('similarity')
                        print("🔍 Mode: Pattern similarity")
                    elif key == ord('5'):
                        self.update_overlay_mode('features')
                        print("⚡ Mode: Learned features")
                    elif key == ord('6'):
                        self.update_overlay_mode('clusters')
                        print("🗂️ Mode: Memory clusters")
                    elif key == ord('r'):
                        print("📊 Generating memory report...")
                        report = self.memory_inspector.generate_memory_report()
                        print("✅ Memory report generated!")
                    elif key == ord('m'):
                        self.enable_async_maintenance = not self.enable_async_maintenance
                        if self.enable_async_maintenance:
                            self.memory_inspector.enable_async_maintenance()
                            print("🔄 Async maintenance: ENABLED")
                        else:
                            self.memory_inspector.disable_async_maintenance()
                            print("⏹️ Async maintenance: DISABLED")
                    elif key == ord('e'):
                        self.show_edge_detection = not self.show_edge_detection
                        if self.show_edge_detection:
                            print("🔍 Edge detection: ENABLED")
                        else:
                            print("⏹️ Edge detection: DISABLED")
                    
                    self.frame_count += 1
                    
                    if self.frame_count % 100 == 0:
                        avg_processing = np.mean(self.processing_times) * 1000
                        gate_stats = self.memory_inspector.memory_gate.get_statistics()
                        print(f"Frame {self.frame_count}: {avg_processing:.1f}ms processing, {len(self.memory_inspector.memory_samples)} memories")
                        
                        # Handle both emergent and explicit statistics
                        storage_rate = gate_stats.get('recent_storage_rate', 0) * 100
                        memory_pressure = gate_stats.get('memory_pressure', gate_stats.get('total_pressure', 0))
                        print(f"   Storage rate: {storage_rate:.1f}%, Memory pressure: {memory_pressure:.2f}")
                        
                        # Show async maintenance stats if enabled
                        if self.enable_async_maintenance and self.memory_inspector.async_maintenance:
                            maint_stats = self.memory_inspector.async_maintenance.get_maintenance_stats()
                            print(f"   Brain state: {maint_stats['brain_state']}, Consolidation debt: {maint_stats['maintenance_pressure']['consolidation_debt']:.2f}")
                
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
            
            print(f"\n📊 Session complete:")
            print(f"Frames processed: {self.frame_count}")
            print(f"Memories captured: {len(self.memory_inspector.memory_samples)}")
            
            # Get memory gating statistics
            gate_stats = self.memory_inspector.memory_gate.get_statistics()
            print(f"\n🧠 Memory Gating Statistics:")
            print(f"Total experiences: {gate_stats['total_experiences']}")
            print(f"Memories formed: {gate_stats['memories_formed']}")
            print(f"Overall storage rate: {gate_stats.get('overall_storage_rate', 0):.1%}")
            
            # Handle both emergent and explicit statistics
            memory_pressure = gate_stats.get('memory_pressure', gate_stats.get('total_pressure', 0))
            print(f"Final memory pressure: {memory_pressure:.2f}")
            
            # Get consolidation statistics (if available)
            if hasattr(self.memory_inspector, 'memory_consolidator') and self.memory_inspector.memory_consolidator:
                consolidation_stats = self.memory_inspector.memory_consolidator.get_statistics()
                if consolidation_stats.get('consolidation_runs', 0) > 0:
                    print(f"\n🧹 Memory Consolidation:")
                    print(f"Consolidation runs: {consolidation_stats['consolidation_runs']}")
                    print(f"Memories pruned: {consolidation_stats['total_memories_pruned']}")
                    print(f"Prototypes created: {consolidation_stats['prototypes_created']}")
            elif 'constraint_enforcer' in gate_stats:
                # Emergent system consolidation stats
                enforcer_stats = gate_stats['constraint_enforcer']
                print(f"\n🌊 Emergent Consolidation:")
                print(f"Consolidation events: {enforcer_stats.get('consolidation_events', 0)}")
                print(f"Emergency interventions: {enforcer_stats.get('emergency_interventions', 0)}")
                print(f"Energy level: {gate_stats.get('energy_level', 100):.0f}%")
            
            if self.processing_times:
                avg_time = np.mean(self.processing_times) * 1000
                print(f"\n⚡ Average processing time: {avg_time:.1f}ms")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Overlay Camera Visualization")
    parser.add_argument("--brain_type", choices=["minimal", "goldilocks", "sparse_goldilocks"], 
                       default="sparse_goldilocks", help="Type of brain to visualize")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create and run visualization
    visualizer = MemoryOverlayCamera(
        brain_type=args.brain_type,
        camera_id=args.camera_id
    )
    
    visualizer.run_memory_visualization()


if __name__ == "__main__":
    main()