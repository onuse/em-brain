#!/usr/bin/env python3
"""
Emotional Sound Expression Research for PiCar-X

Explores using the PiCar-X speaker for emotional state expression
and future self-listening feedback loops. This could add a fascinating
new dimension to the artificial life experience.

Research Areas:
1. Brain state to sound mapping
2. Biologically-inspired sound design
3. Real-time audio synthesis
4. Future self-listening implications
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Tuple

# Add brain directory to path
brain_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, brain_dir)


class EmotionalSoundResearcher:
    """Research emotional sound expression for artificial life robots."""
    
    def __init__(self):
        """Initialize the researcher."""
        self.brain_state_mappings = {}
        self.sound_palettes = {}
        
    def analyze_picarx_audio_capabilities(self) -> Dict[str, Any]:
        """Analyze PiCar-X audio hardware capabilities."""
        
        print("🔊 PICAR-X AUDIO HARDWARE ANALYSIS")
        print("=" * 50)
        
        hardware_analysis = {
            'speaker': {
                'availability': 'CONFIRMED - PiCar-X has onboard speaker',
                'typical_specs': {
                    'type': 'Small piezo or dynamic speaker',
                    'power': '1-3W typical',
                    'frequency_range': '200Hz - 8kHz (estimated)',
                    'quality': 'Basic but sufficient for expression'
                },
                'capabilities': [
                    '✅ Tone generation (pure tones, harmonics)',
                    '✅ Simple melody playback', 
                    '✅ Sound effects and alerts',
                    '✅ Volume control via PWM',
                    '✅ Real-time synthesis possible'
                ],
                'limitations': [
                    '⚠️  Limited frequency range vs full-range speakers',
                    '⚠️  Basic fidelity - not hi-fi audio',
                    '⚠️  Power consumption considerations'
                ]
            },
            'microphone': {
                'base_model': 'NOT INCLUDED in standard PiCar-X',
                'upgrade_potential': 'HIGH - USB or I2S microphones easily added',
                'future_capabilities': [
                    '🎯 Self-listening for feedback loops',
                    '🎯 Environmental sound awareness', 
                    '🎯 Voice command potential',
                    '🎯 Echo/reverb analysis for spatial awareness'
                ]
            },
            'processing_power': {
                'raspberry_pi': 'Sufficient for real-time audio synthesis',
                'libraries': 'pygame, pydub, numpy for audio generation',
                'latency': 'Low enough for responsive emotional expression'
            }
        }
        
        print("🔊 Speaker Capabilities:")
        for capability in hardware_analysis['speaker']['capabilities']:
            print(f"   {capability}")
        
        print(f"\n🎤 Microphone Status:")
        print(f"   Base model: {hardware_analysis['microphone']['base_model']}")
        print(f"   Upgrade potential: {hardware_analysis['microphone']['upgrade_potential']}")
        
        return hardware_analysis
    
    def design_brain_state_mappings(self) -> Dict[str, Any]:
        """Design mappings from brain states to emotional sounds."""
        
        print(f"\n🧠 BRAIN STATE TO SOUND MAPPING DESIGN")
        print("=" * 50)
        
        # Map brain states to emotional/behavioral categories
        state_mappings = {
            'exploration_sounds': {
                'brain_conditions': [
                    'prediction_method == "bootstrap_random"',
                    'total_experiences < 50',
                    'prediction_confidence < 0.3'
                ],
                'emotional_state': 'Curious/Uncertain',
                'sound_characteristics': {
                    'frequency': 'Rising tones (200-800Hz)',
                    'pattern': 'Irregular, questioning intervals',
                    'duration': 'Short bursts (0.1-0.3s)',
                    'timbre': 'Pure tones with slight tremolo'
                },
                'biological_inspiration': 'Young animal exploration calls',
                'example_sounds': [
                    'Questioning chirps',
                    'Uncertainty warbles', 
                    'Exploration beeps'
                ]
            },
            
            'confidence_sounds': {
                'brain_conditions': [
                    'prediction_confidence > 0.8',
                    'consensus_rate > 0.7',
                    'prediction_method == "consensus"'
                ],
                'emotional_state': 'Confident/Assured',
                'sound_characteristics': {
                    'frequency': 'Stable mid-tones (300-600Hz)',
                    'pattern': 'Steady, rhythmic pulses',
                    'duration': 'Medium duration (0.3-0.5s)',
                    'timbre': 'Clean tones with harmonics'
                },
                'biological_inspiration': 'Territory establishment calls',
                'example_sounds': [
                    'Confident chirps',
                    'Achievement tones',
                    'Success harmonics'
                ]
            },
            
            'learning_sounds': {
                'brain_conditions': [
                    'working_memory_size > 5',
                    'recent_experience_added == True',
                    'pattern_discovery == True'
                ],
                'emotional_state': 'Learning/Processing',
                'sound_characteristics': {
                    'frequency': 'Ascending sequences (200-1000Hz)',
                    'pattern': 'Complex rhythmic patterns',
                    'duration': 'Longer phrases (0.5-1.0s)',
                    'timbre': 'Layered tones with modulation'
                },
                'biological_inspiration': 'Bird song learning patterns',
                'example_sounds': [
                    'Processing melodies',
                    'Discovery fanfares',
                    'Learning sequences'
                ]
            },
            
            'confusion_sounds': {
                'brain_conditions': [
                    'prediction_error > 0.5',
                    'collision_detected == True',
                    'consensus_rate < 0.3'
                ],
                'emotional_state': 'Confused/Frustrated',
                'sound_characteristics': {
                    'frequency': 'Descending tones (800-200Hz)',
                    'pattern': 'Irregular, broken sequences',
                    'duration': 'Variable, interrupted',
                    'timbre': 'Distorted or modulated tones'
                },
                'biological_inspiration': 'Distress or confusion calls',
                'example_sounds': [
                    'Confusion warbles',
                    'Frustration buzzes',
                    'Error alert tones'
                ]
            },
            
            'social_sounds': {
                'brain_conditions': [
                    'human_interaction_detected == True',
                    'remote_command_received == True',
                    'demonstration_mode == True'
                ],
                'emotional_state': 'Social/Interactive',
                'sound_characteristics': {
                    'frequency': 'Pleasant mid-range (400-800Hz)',
                    'pattern': 'Conversational rhythms',
                    'duration': 'Phrase-like (0.5-2.0s)',
                    'timbre': 'Warm tones with expression'
                },
                'biological_inspiration': 'Social communication calls',
                'example_sounds': [
                    'Greeting chirps',
                    'Acknowledgment tones',
                    'Goodbye melodies'
                ]
            },
            
            'achievement_sounds': {
                'brain_conditions': [
                    'navigation_goal_reached == True',
                    'long_sequence_completed == True',
                    'obstacle_successfully_avoided == True'
                ],
                'emotional_state': 'Satisfied/Accomplished',
                'sound_characteristics': {
                    'frequency': 'Rising to high resolution (300-1200Hz)',
                    'pattern': 'Triumphant progressions',
                    'duration': 'Complete phrases (1.0-2.0s)',
                    'timbre': 'Rich harmonics, bright tones'
                },
                'biological_inspiration': 'Victory or accomplishment calls',
                'example_sounds': [
                    'Success fanfares',
                    'Achievement crescendos',
                    'Completion melodies'
                ]
            }
        }
        
        print("🎵 Emotional Sound Categories:")
        for category, details in state_mappings.items():
            print(f"\n   {category.upper().replace('_', ' ')}:")
            print(f"     Emotion: {details['emotional_state']}")
            print(f"     Inspiration: {details['biological_inspiration']}")
            print(f"     Frequency: {details['sound_characteristics']['frequency']}")
            print(f"     Pattern: {details['sound_characteristics']['pattern']}")
        
        return state_mappings
    
    def design_sound_synthesis_system(self) -> Dict[str, Any]:
        """Design real-time sound synthesis system."""
        
        print(f"\n🎛️  SOUND SYNTHESIS SYSTEM DESIGN")
        print("=" * 50)
        
        synthesis_system = {
            'architecture': {
                'audio_engine': 'pygame.mixer for real-time synthesis',
                'synthesis_method': 'Procedural generation using numpy',
                'buffer_management': 'Double-buffered for smooth playback',
                'latency_target': '<50ms from brain state to sound',
                'memory_usage': 'Minimal - generate sounds on demand'
            },
            
            'sound_generation': {
                'waveform_types': [
                    'Sine waves (pure emotional tones)',
                    'Triangle waves (softer, warmer sounds)',
                    'Square waves (alert, attention-getting)',
                    'Sawtooth waves (complex, rich textures)',
                    'Noise (breathing, natural textures)'
                ],
                'modulation_techniques': [
                    'Amplitude modulation (tremolo, volume expression)',
                    'Frequency modulation (vibrato, pitch bending)',
                    'Filter modulation (tone color changes)',
                    'Envelope shaping (attack, decay, sustain, release)'
                ],
                'effect_processing': [
                    'Reverb (spatial presence)',
                    'Delay (echo, repetition)',
                    'Distortion (intensity, urgency)',
                    'Filtering (tone shaping)'
                ]
            },
            
            'real_time_control': {
                'brain_state_polling': 'Check brain state every 100ms',
                'sound_state_machine': 'Manage transitions between emotional states',
                'priority_system': 'Handle competing emotional expressions',
                'adaptive_timing': 'Adjust sound timing based on robot activity',
                'volume_control': 'Dynamic volume based on environment'
            },
            
            'implementation_approach': [
                '1. Create EmotionalAudioEngine class',
                '2. Implement basic tone generation with numpy',
                '3. Add brain state monitoring and mapping',
                '4. Develop sound transition and blending',
                '5. Integrate with PiCarXBrainstem',
                '6. Add real-time parameter control',
                '7. Optimize performance for continuous operation'
            ]
        }
        
        print("🎼 Synthesis Components:")
        for component, methods in synthesis_system['sound_generation'].items():
            print(f"\n   {component.replace('_', ' ').title()}:")
            for method in methods:
                print(f"     • {method}")
        
        return synthesis_system
    
    def explore_self_listening_implications(self) -> Dict[str, Any]:
        """Explore implications of future self-listening capabilities."""
        
        print(f"\n🎧 SELF-LISTENING FEEDBACK LOOP ANALYSIS")
        print("=" * 50)
        
        self_listening_analysis = {
            'cognitive_implications': {
                'self_awareness': [
                    '🧠 Robot becomes aware of its own emotional expressions',
                    '🧠 Can recognize its own voice vs external sounds',
                    '🧠 Develops internal model of its sound-making capabilities'
                ],
                'feedback_loops': [
                    '🔄 Emotional state → Sound → Hearing → Modified emotional state',
                    '🔄 Can calm itself with soothing sounds',
                    '🔄 Can excite itself with energetic sounds',
                    '🔄 Can develop sound-based self-regulation strategies'
                ],
                'emergent_behaviors': [
                    '🌟 Self-soothing when confused or stuck',
                    '🌟 Sound-based memory triggers and associations',
                    '🌟 Development of personal "voice" or sound signature',
                    '🌟 Sound-guided meditation or focus states'
                ]
            },
            
            'technical_implementation': {
                'microphone_integration': [
                    'USB microphone on Raspberry Pi',
                    'Real-time audio input processing',
                    'Sound classification (own voice vs environment)',
                    'Audio feature extraction for feedback'
                ],
                'self_recognition': [
                    'Acoustic fingerprinting of own sounds',
                    'Delay detection (direct sound vs reflected)',
                    'Frequency analysis to identify own voice',
                    'Volume/distance estimation'
                ],
                'feedback_processing': [
                    'Audio-to-emotional-state mapping',
                    'Integration with existing brain systems',
                    'Temporal delay handling (sound → hearing → response)',
                    'Recursive feedback prevention'
                ]
            },
            
            'biological_inspiration': {
                'animal_examples': [
                    '🐦 Birds learning and refining their own songs',
                    '🐋 Whales using echolocation feedback for navigation',
                    '🐺 Wolves howling and listening to pack responses',
                    '🐵 Primates using vocal feedback for social coordination'
                ],
                'human_parallels': [
                    '👶 Infant babbling and sound experimentation',
                    '🎵 Musicians listening to their own performance',
                    '🗣️ Speech feedback for pronunciation learning',
                    '🧘 Humming/chanting for emotional regulation'
                ]
            },
            
            'research_questions': [
                '❓ How quickly could the robot learn to recognize its own sounds?',
                '❓ Would self-listening create more complex emotional behaviors?',
                '❓ Could the robot develop sound-based habits or preferences?',
                '❓ Would it start to "think out loud" through sound?',
                '❓ How would it react to recordings of its own past sounds?',
                '❓ Could it develop different "voices" for different situations?'
            ],
            
            'experimental_protocols': [
                '🧪 Baseline emotional sound expression (no self-listening)',
                '🧪 Add microphone and self-recognition capability',
                '🧪 Monitor changes in emotional expression patterns',
                '🧪 Test self-soothing and self-regulation behaviors',
                '🧪 Analyze long-term sound pattern evolution',
                '🧪 Compare behavior with/without audio feedback'
            ]
        }
        
        print("🔬 Research Implications:")
        for question in self_listening_analysis['research_questions']:
            print(f"   {question}")
        
        print(f"\n🐾 Biological Parallels:")
        for example in self_listening_analysis['biological_inspiration']['animal_examples']:
            print(f"   {example}")
        
        return self_listening_analysis
    
    def create_implementation_roadmap(self) -> Dict[str, Any]:
        """Create practical implementation roadmap."""
        
        print(f"\n🗺️  IMPLEMENTATION ROADMAP")
        print("=" * 50)
        
        roadmap = {
            'phase_1_basic_expression': {
                'goal': 'Basic emotional sound expression in simulation',
                'duration': '1-2 days',
                'deliverables': [
                    'EmotionalAudioEngine class',
                    'Brain state to sound mapping',
                    'Basic tone synthesis with pygame',
                    'Integration with PiCarXBrainstem',
                    'Demo with audible emotional feedback'
                ],
                'success_criteria': [
                    'Robot makes different sounds based on brain state',
                    'Sounds are clearly distinguishable',
                    'Real-time performance maintained',
                    'Integration doesn\t impact brain performance'
                ]
            },
            
            'phase_2_rich_expression': {
                'goal': 'Rich, nuanced emotional sound palette',
                'duration': '2-3 days',
                'deliverables': [
                    'Advanced sound synthesis (harmonics, modulation)',
                    'Smooth transitions between emotional states',
                    'Contextual sound variations',
                    'Volume and timing adaptation',
                    'Sound recording/playback for analysis'
                ],
                'success_criteria': [
                    'Emotionally expressive and engaging sounds',
                    'Natural transitions, no jarring changes',
                    'Appropriate volume and timing',
                    'Observer can identify robot\s emotional state by sound'
                ]
            },
            
            'phase_3_real_hardware': {
                'goal': 'Deploy emotional sounds on real PiCar-X',
                'duration': '1 day',
                'deliverables': [
                    'Hardware audio output integration',
                    'Performance optimization for Pi',
                    'Real-world testing and tuning',
                    'Volume/timing adjustments for physical environment'
                ],
                'success_criteria': [
                    'Clear audio output on real robot',
                    'No performance impact on navigation',
                    'Appropriate volume for environment',
                    'Stable operation over extended periods'
                ]
            },
            
            'phase_4_self_listening': {
                'goal': 'Add microphone and self-listening capability',
                'duration': '3-5 days',
                'deliverables': [
                    'Microphone integration (USB or I2S)',
                    'Self-voice recognition system',
                    'Audio feedback processing',
                    'Feedback loop safety mechanisms',
                    'Self-regulation behavior experiments'
                ],
                'success_criteria': [
                    'Robot can distinguish its own sounds from environment',
                    'Demonstrates audio-feedback emotional responses',
                    'No harmful feedback loops',
                    'Shows evidence of self-soothing or regulation'
                ]
            }
        }
        
        print("📋 Implementation Phases:")
        for phase, details in roadmap.items():
            print(f"\n   {phase.upper().replace('_', ' ')}:")
            print(f"     Goal: {details['goal']}")
            print(f"     Duration: {details['duration']}")
            print(f"     Key deliverables: {len(details['deliverables'])} items")
        
        total_effort = "7-11 days for complete implementation"
        print(f"\n⏱️  Total estimated effort: {total_effort}")
        
        return roadmap


def main():
    """Run complete emotional sound research."""
    
    print("🎵 Emotional Sound Expression Research for PiCar-X")
    print("=" * 70)
    print("Research Goal: Explore using speaker for emotional expression")
    print("              and future self-listening feedback loops")
    print()
    
    researcher = EmotionalSoundResearcher()
    
    # Conduct research
    hardware_analysis = researcher.analyze_picarx_audio_capabilities()
    brain_mappings = researcher.design_brain_state_mappings()
    synthesis_system = researcher.design_sound_synthesis_system()
    self_listening = researcher.explore_self_listening_implications()
    roadmap = researcher.create_implementation_roadmap()
    
    # Generate recommendations
    print(f"\n🎯 RESEARCH CONCLUSIONS")
    print("=" * 50)
    
    print("✅ FEASIBILITY: HIGH")
    print("   • PiCar-X speaker capable of expressive sound synthesis")
    print("   • Brain states provide rich emotional data for mapping") 
    print("   • Real-time synthesis achievable with minimal performance impact")
    print("   • Clear biological inspiration and precedents")
    
    print(f"\n🚀 IMMEDIATE VALUE:")
    print("   • Adds engaging artificial life dimension to robot")
    print("   • Provides intuitive feedback about robot's internal state")
    print("   • Makes debugging and observation more natural")
    print("   • Creates emotional connection between robot and observer")
    
    print(f"\n🧠 FUTURE POTENTIAL:")
    print("   • Self-listening could enable fascinating self-regulation")
    print("   • Emergent sound-based behaviors and preferences")
    print("   • Potential for sound-based memory and association")
    print("   • Rich research area for artificial life studies")
    
    print(f"\n💡 RECOMMENDATION:")
    print("   START with Phase 1 (basic emotional expression)")
    print("   This is high-value, low-risk, and immediately beneficial")
    print("   Self-listening can be explored later as research progresses")
    
    return {
        'feasibility': 'HIGH',
        'immediate_value': 'HIGH', 
        'future_potential': 'VERY HIGH',
        'recommended_start': 'Phase 1 - Basic Expression',
        'total_phases': 4
    }


if __name__ == "__main__":
    main()