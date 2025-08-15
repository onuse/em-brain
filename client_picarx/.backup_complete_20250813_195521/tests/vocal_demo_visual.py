#!/usr/bin/env python3
"""
Visual Vocal Demo - Interactive Robot Emotional Expression

A pygame-based visual demonstration of the robot's digital vocal cords.
Shows emotional states, their meanings, and audio parameters in real-time
while you hear the actual sounds through your Mac speakers.

Usage:
    python3 vocal_demo_visual.py
"""

import sys
import os
import time
import threading
from typing import Dict, Optional

# Add client source to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("❌ pygame not available - install with: pip install pygame")
    sys.exit(1)

from hardware.interfaces.vocal_interface import EmotionalVocalMapper, VocalParameters
from hardware.mock.mac_audio_vocal_driver import MacAudioVocalDriver
from brainstem.brain_client import MockBrainServerClient


class VocalDemoVisual:
    """Visual demonstration of robot vocal expressions."""
    
    def __init__(self):
        """Initialize the visual vocal demo."""
        
        # Initialize vocal system BEFORE pygame to avoid mixer conflicts
        print("🎵 Initializing vocal system...")
        self.vocal_driver = MacAudioVocalDriver()
        self.vocal_driver.initialize_vocal_system()
        self.emotional_mapper = EmotionalVocalMapper()
        
        # Now initialize pygame (but mixer is already set up)
        print("🎮 Initializing pygame display...")
        pygame.init()
        self.screen_width = 1000
        self.screen_height = 700
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🎵 Robot Vocal Expressions - Digital Vocal Cords Demo")
        
        # Colors
        self.colors = {
            'background': (20, 25, 40),
            'title': (255, 255, 255),
            'emotion': (100, 200, 255),
            'description': (200, 200, 200),
            'parameters': (150, 255, 150),
            'current': (255, 200, 100),
            'button': (60, 80, 120),
            'button_hover': (80, 100, 140),
            'button_text': (255, 255, 255),
            'progress': (100, 255, 100),
            'progress_bg': (40, 40, 60)
        }
        
        # Fonts
        self.font_title = pygame.font.Font(None, 48)
        self.font_emotion = pygame.font.Font(None, 36)
        self.font_description = pygame.font.Font(None, 24)
        self.font_parameters = pygame.font.Font(None, 20)
        self.font_button = pygame.font.Font(None, 24)
        
        # Demo state
        self.current_emotion = None
        self.current_params = None
        self.is_playing = False
        self.play_start_time = 0
        self.play_duration = 0
        self.auto_demo_running = False
        self.demo_thread = None
        
        # Emotional states with descriptions
        self.emotions = {
            'curiosity': {
                'name': 'Curiosity',
                'description': 'Rising questioning chirps when learning something new',
                'scenario': 'Robot encounters unknown object',
                'brain_state': {
                    'prediction_confidence': 0.2,
                    'prediction_method': 'bootstrap_random',
                    'total_experiences': 5,
                    'collision_detected': False
                }
            },
            'confidence': {
                'name': 'Confidence', 
                'description': 'Steady rhythmic pulses when certain about decisions',
                'scenario': 'Robot successfully navigating familiar territory',
                'brain_state': {
                    'prediction_confidence': 0.9,
                    'prediction_method': 'consensus',
                    'total_experiences': 100,
                    'collision_detected': False
                }
            },
            'confusion': {
                'name': 'Confusion',
                'description': 'Irregular warbling tones when uncertain',
                'scenario': 'Robot receives conflicting sensor data',
                'brain_state': {
                    'prediction_confidence': 0.1,
                    'prediction_method': 'bootstrap_random',
                    'total_experiences': 50,
                    'collision_detected': True
                }
            },
            'achievement': {
                'name': 'Achievement',
                'description': 'Triumphant crescendos when reaching goals',
                'scenario': 'Robot successfully completes a difficult task',
                'brain_state': {
                    'prediction_confidence': 0.95,
                    'prediction_method': 'consensus',
                    'total_experiences': 200,
                    'collision_detected': False
                }
            },
            'distress': {
                'name': 'Distress',
                'description': 'Sharp urgent calls when encountering problems',
                'scenario': 'Robot detects imminent collision or system failure',
                'brain_state': {
                    'prediction_confidence': 0.8,
                    'prediction_method': 'consensus', 
                    'total_experiences': 75,
                    'collision_detected': True
                }
            },
            'contentment': {
                'name': 'Contentment',
                'description': 'Gentle harmonic tones when operating smoothly',
                'scenario': 'Robot in idle state with all systems nominal',
                'brain_state': {
                    'prediction_confidence': 0.7,
                    'prediction_method': 'consensus',
                    'total_experiences': 150,
                    'collision_detected': False
                }
            }
        }
        
        # Button setup
        self.buttons = self._create_buttons()
        
        print("🎵 Visual Vocal Demo initialized")
        print("🔊 Make sure your Mac volume is up to hear the robot's voice!")
    
    def _create_buttons(self) -> Dict:
        """Create interactive buttons."""
        buttons = {}
        
        # Individual emotion buttons
        y_start = 120
        for i, emotion_key in enumerate(self.emotions.keys()):
            buttons[emotion_key] = pygame.Rect(50, y_start + i * 60, 200, 50)
        
        # Control buttons
        buttons['auto_demo'] = pygame.Rect(300, 500, 150, 50)
        buttons['stop'] = pygame.Rect(470, 500, 100, 50)
        buttons['quit'] = pygame.Rect(590, 500, 100, 50)
        
        return buttons
    
    def run_demo(self):
        """Run the visual vocal demonstration."""
        
        print("🎮 Visual Vocal Demo running - click emotions to hear them!")
        print("   • Click individual emotions to test them")
        print("   • Click 'Auto Demo' for continuous demonstration")
        print("   • Press ESC or click 'Quit' to exit")
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        running = self._handle_click(event.pos, running)
            
            # Update display
            self._draw_interface()
            
            # Update timing for current playback
            if self.is_playing:
                elapsed = time.time() - self.play_start_time
                if elapsed >= self.play_duration:
                    self.is_playing = False
                    self.current_emotion = None
                    self.current_params = None
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        # Cleanup
        self._stop_all()
        pygame.quit()
        print("👋 Visual Vocal Demo ended")
    
    def _handle_click(self, pos, running: bool) -> bool:
        """Handle mouse clicks on buttons."""
        
        # Check emotion buttons
        for emotion_key, button_rect in self.buttons.items():
            if emotion_key in self.emotions and button_rect.collidepoint(pos):
                self._play_emotion(emotion_key)
                return running
        
        # Check control buttons
        if self.buttons['auto_demo'].collidepoint(pos):
            self._start_auto_demo()
        elif self.buttons['stop'].collidepoint(pos):
            self._stop_all()
        elif self.buttons['quit'].collidepoint(pos):
            return False
        
        return running
    
    def _play_emotion(self, emotion_key: str):
        """Play a specific emotional expression."""
        
        if emotion_key not in self.emotions:
            return
        
        print(f"\n🎵 Playing emotion: {emotion_key}")
        
        # Stop any current playback
        self.vocal_driver.stop_vocalization()
        
        # Get brain state and map to vocal parameters
        brain_state = self.emotions[emotion_key]['brain_state']
        vocal_params = self.emotional_mapper.map_brain_state_to_vocal_params(brain_state)
        
        # Start playback
        self.current_emotion = emotion_key
        self.current_params = vocal_params
        self.is_playing = True
        self.play_start_time = time.time()
        self.play_duration = vocal_params.duration
        
        # Synthesize vocalization
        self.vocal_driver.synthesize_vocalization(vocal_params)
    
    def _start_auto_demo(self):
        """Start automatic demonstration of all emotions."""
        
        if self.auto_demo_running:
            return
        
        print("🎭 Starting automatic emotion demonstration...")
        self.auto_demo_running = True
        
        # Start demo thread
        self.demo_thread = threading.Thread(target=self._auto_demo_loop)
        self.demo_thread.daemon = True
        self.demo_thread.start()
    
    def _auto_demo_loop(self):
        """Automatic demonstration loop."""
        
        emotion_list = list(self.emotions.keys())
        
        while self.auto_demo_running:
            for emotion_key in emotion_list:
                if not self.auto_demo_running:
                    break
                
                self._play_emotion(emotion_key)
                
                # Wait for emotion to complete plus pause
                if self.current_params:
                    time.sleep(self.current_params.duration + 1.5)
                else:
                    time.sleep(2.0)
        
        print("🎭 Auto demonstration stopped")
    
    def _stop_all(self):
        """Stop all vocal activity."""
        
        self.auto_demo_running = False
        self.vocal_driver.stop_vocalization()
        self.is_playing = False
        self.current_emotion = None
        self.current_params = None
        
        if self.demo_thread and self.demo_thread.is_alive():
            self.demo_thread.join(timeout=0.5)
    
    def _draw_interface(self):
        """Draw the complete user interface."""
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Title
        title_text = self.font_title.render("🎵 Robot Digital Vocal Cords", True, self.colors['title'])
        title_rect = title_text.get_rect(centerx=self.screen_width//2, y=20)
        self.screen.blit(title_text, title_rect)
        
        subtitle_text = self.font_description.render("Click emotions to hear how the robot expresses its internal states", True, self.colors['description'])
        subtitle_rect = subtitle_text.get_rect(centerx=self.screen_width//2, y=70)
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Emotion buttons and descriptions
        self._draw_emotion_buttons()
        
        # Current playback info
        if self.current_emotion and self.is_playing:
            self._draw_current_playback()
        
        # Control buttons
        self._draw_control_buttons()
        
        # Technical info
        self._draw_technical_info()
    
    def _draw_emotion_buttons(self):
        """Draw emotion selection buttons."""
        
        y_start = 120
        
        for i, (emotion_key, emotion_data) in enumerate(self.emotions.items()):
            y_pos = y_start + i * 60
            button_rect = self.buttons[emotion_key]
            
            # Button background
            is_current = (emotion_key == self.current_emotion and self.is_playing)
            button_color = self.colors['current'] if is_current else self.colors['button']
            pygame.draw.rect(self.screen, button_color, button_rect)
            pygame.draw.rect(self.screen, self.colors['description'], button_rect, 2)
            
            # Button text
            button_text = self.font_button.render(emotion_data['name'], True, self.colors['button_text'])
            button_text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, button_text_rect)
            
            # Description
            desc_text = self.font_description.render(emotion_data['description'], True, self.colors['description'])
            self.screen.blit(desc_text, (270, y_pos + 10))
            
            # Scenario
            scenario_text = self.font_parameters.render(f"Scenario: {emotion_data['scenario']}", True, self.colors['parameters'])
            self.screen.blit(scenario_text, (270, y_pos + 35))
    
    def _draw_current_playback(self):
        """Draw current playback information."""
        
        if not self.current_emotion or not self.current_params:
            return
        
        # Current emotion info box
        info_rect = pygame.Rect(50, 480, 680, 120)
        pygame.draw.rect(self.screen, (30, 35, 50), info_rect)
        pygame.draw.rect(self.screen, self.colors['current'], info_rect, 3)
        
        # Title
        current_text = self.font_emotion.render(f"🎵 Currently Playing: {self.emotions[self.current_emotion]['name']}", True, self.colors['current'])
        self.screen.blit(current_text, (60, 490))
        
        # Progress bar
        elapsed = time.time() - self.play_start_time
        progress = min(elapsed / self.play_duration, 1.0)
        
        progress_rect = pygame.Rect(60, 520, 400, 20)
        pygame.draw.rect(self.screen, self.colors['progress_bg'], progress_rect)
        progress_fill = pygame.Rect(60, 520, int(400 * progress), 20)
        pygame.draw.rect(self.screen, self.colors['progress'], progress_fill)
        
        # Time info
        time_text = self.font_parameters.render(f"{elapsed:.1f}s / {self.play_duration:.1f}s", True, self.colors['description'])
        self.screen.blit(time_text, (480, 525))
        
        # Vocal parameters
        params_y = 550
        params = [
            f"Frequency: {self.current_params.fundamental_frequency:.1f} Hz",
            f"Amplitude: {self.current_params.amplitude:.2f}",
            f"Harmonics: {len(self.current_params.harmonics)}",
            f"Noise: {self.current_params.noise_component:.2f}"
        ]
        
        for i, param in enumerate(params):
            x_pos = 60 + (i * 150)
            param_text = self.font_parameters.render(param, True, self.colors['parameters'])
            self.screen.blit(param_text, (x_pos, params_y))
    
    def _draw_control_buttons(self):
        """Draw control buttons."""
        
        # Auto Demo button
        auto_color = self.colors['current'] if self.auto_demo_running else self.colors['button']
        pygame.draw.rect(self.screen, auto_color, self.buttons['auto_demo'])
        pygame.draw.rect(self.screen, self.colors['description'], self.buttons['auto_demo'], 2)
        
        auto_text = "Auto Demo" if not self.auto_demo_running else "Running..."
        auto_label = self.font_button.render(auto_text, True, self.colors['button_text'])
        auto_rect = auto_label.get_rect(center=self.buttons['auto_demo'].center)
        self.screen.blit(auto_label, auto_rect)
        
        # Stop button
        pygame.draw.rect(self.screen, self.colors['button'], self.buttons['stop'])
        pygame.draw.rect(self.screen, self.colors['description'], self.buttons['stop'], 2)
        
        stop_label = self.font_button.render("Stop", True, self.colors['button_text'])
        stop_rect = stop_label.get_rect(center=self.buttons['stop'].center)
        self.screen.blit(stop_label, stop_rect)
        
        # Quit button
        pygame.draw.rect(self.screen, self.colors['button'], self.buttons['quit'])
        pygame.draw.rect(self.screen, self.colors['description'], self.buttons['quit'], 2)
        
        quit_label = self.font_button.render("Quit", True, self.colors['button_text'])
        quit_rect = quit_label.get_rect(center=self.buttons['quit'].center)
        self.screen.blit(quit_label, quit_rect)
    
    def _draw_technical_info(self):
        """Draw technical information."""
        
        # Technical info box
        tech_rect = pygame.Rect(750, 120, 230, 350)
        pygame.draw.rect(self.screen, (25, 30, 45), tech_rect)
        pygame.draw.rect(self.screen, self.colors['parameters'], tech_rect, 2)
        
        # Title
        tech_title = self.font_description.render("🔧 Technical Info", True, self.colors['parameters'])
        self.screen.blit(tech_title, (760, 130))
        
        # Vocal system info
        info_lines = [
            "",
            "Digital Vocal Cords:",
            "• Brain state → Sound mapping",
            "• Real-time audio synthesis", 
            "• Harmonic generation",
            "• ADSR envelopes",
            "• Frequency/amplitude modulation",
            "",
            "Emotional Mapping:",
            "• Confidence → Frequency",
            "• Uncertainty → Warbling",
            "• Urgency → Sharp attacks",
            "• Calm → Harmonic richness",
            "",
            "Hardware Abstraction:",
            "• Mock implementation",
            "• Mac speakers as surrogate",
            "• Same interface as real robot",
            "",
            "Brain Integration:",
            "• Prediction confidence",
            "• Experience count",
            "• Collision detection",
            "• Learning progress"
        ]
        
        y_offset = 160
        for line in info_lines:
            if line:
                info_text = self.font_parameters.render(line, True, self.colors['description'])
                self.screen.blit(info_text, (760, y_offset))
            y_offset += 18


def main():
    """Run the visual vocal demonstration."""
    
    if not PYGAME_AVAILABLE:
        print("❌ pygame is required for the visual demo")
        print("Install with: pip install pygame")
        return
    
    print("🎵 Starting Visual Vocal Demo...")
    print("🔊 Make sure your Mac volume is turned up!")
    
    try:
        demo = VocalDemoVisual()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()