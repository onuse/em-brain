#!/usr/bin/env python3
"""
Universal Attention System - Modality-Agnostic Pattern Attention

This system generalizes attention mechanisms to work with any signal type:
- Visual signals (2D images)
- Audio signals (1D time series or 2D spectrograms)
- Tactile signals (sensor arrays)
- Motor signals (joint positions, forces)
- Temporal signals (rhythm, timing patterns)

Core principles:
1. Attention is about signal salience, not specific features
2. Edge detection generalizes to transition detection
3. Spatial attention generalizes to signal attention
4. Window sizing adapts to signal dimensionality
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from enum import Enum


class ModalityType(Enum):
    """Supported modality types"""
    VISUAL = "visual"
    AUDIO = "audio"
    TACTILE = "tactile"
    MOTOR = "motor"
    TEMPORAL = "temporal"


class SignalShape(Enum):
    """Signal dimensionality types"""
    SCALAR = "scalar"      # Single value
    VECTOR = "vector"      # 1D array
    MATRIX = "matrix"      # 2D array
    TENSOR = "tensor"      # 3D+ array


class ModalityFeatureExtractor(ABC):
    """Abstract base class for modality-specific feature extraction"""
    
    @abstractmethod
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract edges/transitions in the signal"""
        pass
    
    @abstractmethod
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract salience/importance map"""
        pass
    
    @abstractmethod
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract gradients/changes in the signal"""
        pass
    
    @abstractmethod
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get modality-specific attention priors"""
        pass


class VisualFeatureExtractor(ModalityFeatureExtractor):
    """Visual-specific feature extraction (specialized for 2D images)"""
    
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract visual edges using Canny"""
        if len(signal.shape) == 2:
            # Grayscale image - ensure proper type conversion
            signal_uint8 = np.clip(signal * 255, 0, 255).astype(np.uint8)
            edges = cv2.Canny(signal_uint8, 50, 150)
            return edges.astype(np.float32) / 255.0
        else:
            raise ValueError(f"Visual edge detection requires 2D signal, got {signal.shape}")
    
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract visual salience using gradients"""
        if len(signal.shape) == 2:
            # Ensure proper type for OpenCV
            signal_f32 = signal.astype(np.float32)
            # Compute gradients
            grad_x = cv2.Sobel(signal_f32, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(signal_f32, cv2.CV_32F, 0, 1, ksize=3)
            salience = np.sqrt(grad_x**2 + grad_y**2)
            return salience
        else:
            raise ValueError(f"Visual salience requires 2D signal, got {signal.shape}")
    
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract visual gradients"""
        return self.extract_salience(signal)
    
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get visual attention priors (center bias, etc.)"""
        h, w = signal.shape
        # Create center bias (humans look at center first)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
        return center_bias


class AudioFeatureExtractor(ModalityFeatureExtractor):
    """Audio-specific feature extraction (specialized for 1D signals)"""
    
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract audio edges (onset detection)"""
        if len(signal.shape) == 1:
            # 1D time series - detect sudden changes
            diff = np.diff(signal, prepend=signal[0])
            edges = np.abs(diff) > np.std(diff) * 2  # Threshold based on std
            return edges.astype(np.float32)
        elif len(signal.shape) == 2:
            # 2D spectrogram - detect spectral edges
            edges = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                diff = np.diff(signal[i], prepend=signal[i, 0])
                edges[i] = np.abs(diff) > np.std(diff) * 2
            return edges.astype(np.float32)
        else:
            raise ValueError(f"Audio edge detection requires 1D or 2D signal, got {signal.shape}")
    
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract audio salience (energy/amplitude)"""
        if len(signal.shape) == 1:
            # 1D - use moving window energy
            window_size = min(32, len(signal) // 4)
            salience = np.convolve(signal**2, np.ones(window_size)/window_size, mode='same')
            return salience
        elif len(signal.shape) == 2:
            # 2D spectrogram - use spectral energy
            salience = np.sum(signal**2, axis=0)  # Sum across frequency bins
            return salience
        else:
            raise ValueError(f"Audio salience requires 1D or 2D signal, got {signal.shape}")
    
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract audio gradients (changes over time)"""
        if len(signal.shape) == 1:
            return np.abs(np.gradient(signal))
        elif len(signal.shape) == 2:
            # For spectrograms, compute time gradients
            grad_t = np.abs(np.gradient(signal, axis=1))
            return np.mean(grad_t, axis=0)  # Average across frequencies
        else:
            raise ValueError(f"Audio gradients require 1D or 2D signal, got {signal.shape}")
    
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get audio attention priors (recency bias, mid-frequency bias)"""
        if len(signal.shape) == 1:
            # Recency bias - recent sounds are more important
            recency = np.linspace(0.5, 1.0, len(signal))
            return recency
        elif len(signal.shape) == 2:
            # Mid-frequency bias - speech range is important
            freq_bins, time_bins = signal.shape
            mid_freq_bias = np.exp(-((np.arange(freq_bins) - freq_bins//2)**2) / (2 * (freq_bins//4)**2))
            return np.outer(mid_freq_bias, np.ones(time_bins))
        else:
            return np.ones_like(signal)


class TactileFeatureExtractor(ModalityFeatureExtractor):
    """Tactile-specific feature extraction (specialized for pressure/touch arrays)"""
    
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract tactile edges (pressure gradients)"""
        if len(signal.shape) == 1:
            # 1D pressure array - detect sudden changes
            diff = np.diff(signal, prepend=signal[0])
            edges = np.abs(diff) > np.std(diff) * 1.5  # Lower threshold for tactile
            return edges.astype(np.float32)
        elif len(signal.shape) == 2:
            # 2D pressure map - detect spatial gradients
            signal_f32 = signal.astype(np.float32)
            grad_x = np.abs(np.gradient(signal_f32, axis=1))
            grad_y = np.abs(np.gradient(signal_f32, axis=0))
            edges = np.sqrt(grad_x**2 + grad_y**2)
            return edges / (np.max(edges) + 1e-8)  # Normalize
        else:
            raise ValueError(f"Tactile edge detection requires 1D or 2D signal, got {signal.shape}")
    
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract tactile salience (pressure intensity)"""
        if len(signal.shape) == 1:
            # 1D - use pressure magnitude with smoothing
            smoothed = np.convolve(signal, np.ones(5)/5, mode='same')
            salience = np.abs(smoothed)
            return salience
        elif len(signal.shape) == 2:
            # 2D pressure map - use local pressure maxima
            from scipy.ndimage import maximum_filter
            local_maxima = maximum_filter(signal, size=3)
            salience = np.where(signal == local_maxima, signal, signal * 0.3)
            return salience
        else:
            raise ValueError(f"Tactile salience requires 1D or 2D signal, got {signal.shape}")
    
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract tactile gradients (pressure changes)"""
        if len(signal.shape) == 1:
            return np.abs(np.gradient(signal))
        elif len(signal.shape) == 2:
            grad_x = np.abs(np.gradient(signal, axis=1))
            grad_y = np.abs(np.gradient(signal, axis=0))
            return np.sqrt(grad_x**2 + grad_y**2)
        else:
            raise ValueError(f"Tactile gradients require 1D or 2D signal, got {signal.shape}")
    
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get tactile attention priors (contact area bias)"""
        if len(signal.shape) == 1:
            # For 1D, bias toward middle (fingertip center)
            center_bias = np.exp(-((np.arange(len(signal)) - len(signal)//2)**2) / (2 * (len(signal)//6)**2))
            return center_bias
        elif len(signal.shape) == 2:
            # For 2D, bias toward center of contact area
            h, w = signal.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            center_bias = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 6)**2))
            return center_bias
        else:
            return np.ones_like(signal)


class MotorFeatureExtractor(ModalityFeatureExtractor):
    """Motor-specific feature extraction (specialized for joint/actuator signals)"""
    
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract motor edges (sudden movements)"""
        if len(signal.shape) == 1:
            # 1D joint/actuator array - detect sudden changes
            velocity = np.diff(signal, prepend=signal[0])
            acceleration = np.diff(velocity, prepend=velocity[0])
            edges = np.abs(acceleration) > np.std(acceleration) * 2  # Acceleration-based edges
            return edges.astype(np.float32)
        elif len(signal.shape) == 2:
            # 2D motor map (multiple joints over time)
            edges = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                velocity = np.diff(signal[i], prepend=signal[i, 0])
                acceleration = np.diff(velocity, prepend=velocity[0])
                edges[i] = np.abs(acceleration) > np.std(acceleration) * 2
            return edges.astype(np.float32)
        else:
            raise ValueError(f"Motor edge detection requires 1D or 2D signal, got {signal.shape}")
    
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract motor salience (movement magnitude)"""
        if len(signal.shape) == 1:
            # 1D - use movement velocity
            velocity = np.abs(np.diff(signal, prepend=signal[0]))
            salience = np.convolve(velocity, np.ones(3)/3, mode='same')  # Smooth
            return salience
        elif len(signal.shape) == 2:
            # 2D - use aggregate movement across joints
            velocities = np.abs(np.diff(signal, axis=1, prepend=signal[:, 0:1]))
            # Return 2D salience map instead of 1D average
            return velocities
        else:
            raise ValueError(f"Motor salience requires 1D or 2D signal, got {signal.shape}")
    
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract motor gradients (movement changes)"""
        if len(signal.shape) == 1:
            return np.abs(np.gradient(signal))
        elif len(signal.shape) == 2:
            # For 2D, compute temporal gradients (changes over time)
            grad_t = np.abs(np.gradient(signal, axis=1))
            # Return 2D gradient map instead of 1D average
            return grad_t
        else:
            raise ValueError(f"Motor gradients require 1D or 2D signal, got {signal.shape}")
    
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get motor attention priors (end-effector bias)"""
        if len(signal.shape) == 1:
            # Bias toward end-effector joints (assume they're at the end)
            end_bias = np.linspace(0.5, 1.0, len(signal))
            return end_bias
        elif len(signal.shape) == 2:
            # For 2D, bias toward recent movements
            recency = np.linspace(0.7, 1.0, signal.shape[1])
            return np.outer(np.ones(signal.shape[0]), recency)
        else:
            return np.ones_like(signal)


class TemporalFeatureExtractor(ModalityFeatureExtractor):
    """Temporal-specific feature extraction (specialized for rhythm/timing patterns)"""
    
    def extract_edges(self, signal: np.ndarray) -> np.ndarray:
        """Extract temporal edges (rhythm changes)"""
        if len(signal.shape) == 1:
            # 1D temporal signal - detect beat/rhythm changes
            # Use second derivative to find rhythm pattern changes
            first_diff = np.diff(signal, prepend=signal[0])
            second_diff = np.diff(first_diff, prepend=first_diff[0])
            edges = np.abs(second_diff) > np.std(second_diff) * 1.5
            return edges.astype(np.float32)
        else:
            raise ValueError(f"Temporal edge detection requires 1D signal, got {signal.shape}")
    
    def extract_salience(self, signal: np.ndarray) -> np.ndarray:
        """Extract temporal salience (rhythm strength)"""
        if len(signal.shape) == 1:
            # Use autocorrelation to find rhythmic patterns
            signal_norm = signal - np.mean(signal)
            autocorr = np.correlate(signal_norm, signal_norm, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peak correlations (rhythmic patterns)
            salience = np.zeros_like(signal)
            for i in range(1, min(len(autocorr), len(signal))):
                if autocorr[i] > 0.3 * np.max(autocorr):  # Significant correlation
                    salience = np.maximum(salience, np.roll(signal_norm, i))
            
            return np.abs(salience)
        else:
            raise ValueError(f"Temporal salience requires 1D signal, got {signal.shape}")
    
    def extract_gradients(self, signal: np.ndarray) -> np.ndarray:
        """Extract temporal gradients (timing changes)"""
        if len(signal.shape) == 1:
            return np.abs(np.gradient(signal))
        else:
            raise ValueError(f"Temporal gradients require 1D signal, got {signal.shape}")
    
    def get_attention_prior(self, signal: np.ndarray) -> np.ndarray:
        """Get temporal attention priors (recency and periodicity bias)"""
        if len(signal.shape) == 1:
            # Combine recency bias with periodicity detection
            recency = np.linspace(0.5, 1.0, len(signal))
            
            # Simple periodicity boost (favor regular patterns)
            periodicity = np.ones_like(signal)
            for period in [4, 8, 16]:  # Common rhythmic periods
                if len(signal) >= period:
                    for i in range(period, len(signal)):
                        if abs(signal[i] - signal[i-period]) < 0.1:
                            periodicity[i] *= 1.2
            
            return recency * periodicity
        else:
            return np.ones_like(signal)


class UniversalAttentionSystem:
    """Universal attention system that works with any signal modality"""
    
    def __init__(self):
        # Initialize GPU device detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_gpu = True
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.use_gpu = True
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
        
        self.extractors = {
            ModalityType.VISUAL: VisualFeatureExtractor(),
            ModalityType.AUDIO: AudioFeatureExtractor(),
            ModalityType.TACTILE: TactileFeatureExtractor(),
            ModalityType.MOTOR: MotorFeatureExtractor(),
            ModalityType.TEMPORAL: TemporalFeatureExtractor(),
        }
    
    def calculate_attention_map(self, signal: np.ndarray, modality: ModalityType, 
                               brain_output: Optional[np.ndarray] = None,
                               novelty_score: float = 0.5) -> Tuple[np.ndarray, List[Dict]]:
        """
        Calculate attention map for any signal modality
        
        Args:
            signal: Input signal (1D, 2D, or 3D array)
            modality: Type of signal (visual, audio, etc.)
            brain_output: Optional brain response to incorporate
            novelty_score: Novelty score for attention boosting
            
        Returns:
            (attention_map, attention_windows)
        """
        try:
            # Get modality-specific extractor
            if modality not in self.extractors:
                raise ValueError(f"Unsupported modality: {modality}")
            
            extractor = self.extractors[modality]
            
            # Base attention from signal priors
            attention_map = extractor.get_attention_prior(signal)
            
            # Add edge-based attention
            edges = extractor.extract_edges(signal)
            attention_map += edges * 0.4
            
            # Add salience-based attention
            salience = extractor.extract_salience(signal)
            attention_map += salience * 0.3
            
            # Add gradient-based attention
            gradients = extractor.extract_gradients(signal)
            attention_map += gradients * 0.2
            
            # Add brain response if available
            if brain_output is not None:
                brain_attention = self._incorporate_brain_response(
                    brain_output, attention_map.shape
                )
                attention_map += brain_attention * 0.4
            
            # Apply GPU acceleration for final processing if available
            if self.use_gpu:
                attention_map = self._process_attention_map_gpu(attention_map, novelty_score)
            else:
                # Apply novelty boost
                attention_map *= (0.6 + novelty_score * 0.8)
                
                # Normalize and clip
                attention_map = np.clip(attention_map, 0, 1)
            
            # Identify attention windows
            attention_windows = self._identify_attention_windows(
                attention_map, signal, modality
            )
            
            return attention_map, attention_windows
            
        except Exception as e:
            print(f"Universal attention calculation error: {e}")
            return np.ones_like(signal) * 0.3, []
    
    def _process_attention_map_gpu(self, attention_map: np.ndarray, novelty_score: float) -> np.ndarray:
        """GPU-accelerated attention map processing"""
        try:
            # Convert to PyTorch tensor and move to GPU
            tensor = torch.from_numpy(attention_map.astype(np.float32)).to(self.device)
            
            # Apply novelty boost
            novelty_factor = 0.6 + novelty_score * 0.8
            tensor *= novelty_factor
            
            # Normalize and clip
            tensor = torch.clamp(tensor, 0, 1)
            
            # Convert back to numpy
            return tensor.cpu().numpy()
            
        except Exception as e:
            # Fallback to CPU processing
            attention_map *= (0.6 + novelty_score * 0.8)
            return np.clip(attention_map, 0, 1)
    
    def _incorporate_brain_response(self, brain_output: np.ndarray, 
                                   target_shape: Tuple[int, ...]) -> np.ndarray:
        """Incorporate brain response into attention map"""
        try:
            # Reshape brain output to match target shape
            if len(target_shape) == 1:
                # 1D target - use direct mapping or interpolation
                if len(brain_output) == target_shape[0]:
                    return brain_output
                else:
                    # Interpolate to match length
                    indices = np.linspace(0, len(brain_output)-1, target_shape[0])
                    return np.interp(indices, np.arange(len(brain_output)), brain_output)
                    
            elif len(target_shape) == 2:
                # 2D target - reshape or resize
                brain_size = int(np.sqrt(len(brain_output)))
                if brain_size * brain_size == len(brain_output):
                    # Perfect square - reshape and resize
                    brain_2d = brain_output.reshape(brain_size, brain_size).astype(np.float32)
                    return cv2.resize(brain_2d, target_shape[::-1])  # cv2 uses (w,h)
                else:
                    # Not square - create 2D representation
                    return np.outer(brain_output[:target_shape[0]], 
                                  np.ones(target_shape[1]))[:target_shape[0], :target_shape[1]]
            else:
                # Higher dimensions - return uniform
                return np.ones(target_shape) * np.mean(brain_output)
                
        except Exception as e:
            print(f"Brain response incorporation error: {e}")
            return np.ones(target_shape) * 0.3
    
    def _identify_attention_windows(self, attention_map: np.ndarray, 
                                   signal: np.ndarray, 
                                   modality: ModalityType) -> List[Dict]:
        """Identify attention windows adapted to signal dimensionality"""
        try:
            if len(attention_map.shape) == 1:
                # 1D signal - identify attention segments
                return self._identify_1d_windows(attention_map, signal, modality)
            elif len(attention_map.shape) == 2:
                # 2D signal - identify attention regions
                return self._identify_2d_windows(attention_map, signal, modality)
            else:
                # Higher dimensions - simplified approach
                return [{'type': 'global', 'strength': np.mean(attention_map), 
                        'shape': attention_map.shape}]
                
        except Exception as e:
            print(f"Attention window identification error: {e}")
            return []
    
    def _identify_1d_windows(self, attention_map: np.ndarray, 
                            signal: np.ndarray, 
                            modality: ModalityType) -> List[Dict]:
        """Identify attention windows for 1D signals"""
        windows = []
        threshold = 0.6
        
        # Find peaks above threshold
        high_attention = attention_map > threshold
        
        if not np.any(high_attention):
            # No peaks - return broad window
            return [{
                'start': 0, 'end': len(attention_map),
                'type': 'broad', 'strength': np.mean(attention_map),
                'modality': modality.value
            }]
        
        # Find continuous segments
        segments = []
        in_segment = False
        start = 0
        
        for i, is_high in enumerate(high_attention):
            if is_high and not in_segment:
                start = i
                in_segment = True
            elif not is_high and in_segment:
                segments.append((start, i))
                in_segment = False
        
        if in_segment:
            segments.append((start, len(attention_map)))
        
        # Convert segments to windows
        for start, end in segments:
            if end - start > 5:  # Minimum window size
                strength = np.mean(attention_map[start:end])
                signal_complexity = np.std(signal[start:end])
                
                # Determine window type
                if strength > 0.8:
                    window_type = 'focused'
                elif signal_complexity > 0.15:
                    window_type = 'detailed'
                else:
                    window_type = 'minimal'
                
                windows.append({
                    'start': start, 'end': end,
                    'type': window_type, 'strength': strength,
                    'complexity': signal_complexity,
                    'modality': modality.value
                })
        
        return windows[:5]  # Limit to top 5 windows
    
    def _identify_2d_windows(self, attention_map: np.ndarray, 
                            signal: np.ndarray, 
                            modality: ModalityType) -> List[Dict]:
        """Identify attention windows for 2D signals (reuse existing logic)"""
        h, w = attention_map.shape
        windows = []
        threshold = 0.6
        
        attention_peaks = attention_map > threshold
        
        if not np.any(attention_peaks):
            return [{
                'x': w // 4, 'y': h // 4,
                'width': w // 2, 'height': h // 2,
                'type': 'broad', 'strength': np.mean(attention_map),
                'modality': modality.value
            }]
        
        # Find contours
        contours, _ = cv2.findContours(
            attention_peaks.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            
            if w_rect < 5 or h_rect < 5:
                continue
            
            region_attention = np.mean(attention_map[y:y+h_rect, x:x+w_rect])
            region_complexity = np.std(signal[y:y+h_rect, x:x+w_rect])
            
            # Adaptive window sizing
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
            
            center_x = x + w_rect // 2
            center_y = y + h_rect // 2
            
            window_x = max(0, min(w - window_width, center_x - window_width // 2))
            window_y = max(0, min(h - window_height, center_y - window_height // 2))
            
            windows.append({
                'x': window_x, 'y': window_y,
                'width': window_width, 'height': window_height,
                'type': window_type, 'strength': region_attention,
                'complexity': region_complexity,
                'modality': modality.value
            })
        
        return windows[:5]  # Limit to top 5 windows
    
    def get_signal_shape(self, signal: np.ndarray) -> SignalShape:
        """Determine signal shape type"""
        if signal.ndim == 0:
            return SignalShape.SCALAR
        elif signal.ndim == 1:
            return SignalShape.VECTOR
        elif signal.ndim == 2:
            return SignalShape.MATRIX
        else:
            return SignalShape.TENSOR
    
    def add_extractor(self, modality: ModalityType, extractor: ModalityFeatureExtractor):
        """Add a new modality-specific extractor"""
        self.extractors[modality] = extractor