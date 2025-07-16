#!/usr/bin/env python3
"""
Biological Timescale Learning Test

This test respects the brain's biological temporal constants:
- Short-term: 1 minute (pattern recognition)
- Medium-term: 10 minutes (behavior formation) 
- Long-term: 1 hour (memory consolidation)
- Very long-term: 24 hours (major adaptations)

A proper test should run for HOURS to see real learning consolidation,
just like biological brains need sleep cycles to consolidate memories.
"""

import sys
import os
import time
import json
import numpy as np
import datetime
from typing import List, Dict

# Add server directory to path
# From tests/ we need to go up one level to brain/, then into server/
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(brain_root, 'server')
sys.path.insert(0, server_dir)

from src.communication import MinimalBrainClient

class BiologicalTimescaleTester:
    """Test brain learning over biological timescales (hours)."""
    
    def __init__(self, host='localhost', port=9999):
        self.client = MinimalBrainClient(host, port)
        self.start_time = time.time()
        self.session_log = []
        self.checkpoint_interval = 300  # 5 minutes - natural batch processing
        self.consolidation_interval = 3600  # 1 hour - memory consolidation
        
    def connect(self) -> bool:
        """Connect to brain server."""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] 🔗 Connecting to brain server for biological timescale test...")
        return self.client.connect()
    
    def disconnect(self):
        """Disconnect from brain server."""
        self.client.disconnect()
    
    def test_pattern_over_biological_time(self, 
                                        target_duration_hours: float = 2.0,
                                        pattern_session_minutes: int = 5) -> Dict:
        """
        Test pattern learning over biological timescales.
        
        This mimics how biological brains actually learn:
        - Training sessions separated by rest periods
        - Consolidation happens between sessions
        - Long-term improvement measured over hours
        
        Args:
            target_duration_hours: Total test duration (minimum 2 hours for real biology)
            pattern_session_minutes: Length of each training session
        """
        print(f"\n🧬 BIOLOGICAL TIMESCALE LEARNING TEST")
        print(f"   Duration: {target_duration_hours} hours")
        print(f"   Training sessions: {pattern_session_minutes} minutes each")
        print(f"   Consolidation breaks: {self.consolidation_interval//60} minutes")
        print(f"   Expected completion: {(datetime.datetime.now() + datetime.timedelta(hours=target_duration_hours)).strftime('%I:%M %p')}")
        
        pattern_input = [1.0, 0.0, 0.0, 0.0]
        expected_output = [1.0, 0.0, 0.0, 0.0]
        
        total_duration = target_duration_hours * 3600  # Convert to seconds
        session_duration = pattern_session_minutes * 60
        
        session_count = 0
        overall_performance = []
        consolidation_checkpoints = []
        
        test_start = time.time()
        
        while (time.time() - test_start) < total_duration:
            session_count += 1
            session_start = time.time()
            current_time = datetime.datetime.now().strftime('%I:%M:%S %p')
            
            print(f"\n📚 Training Session {session_count} ({current_time})")
            print(f"   Duration: {pattern_session_minutes} minutes")
            
            # Training session
            session_errors = []
            session_predictions = []
            
            training_cycles = 0
            while (time.time() - session_start) < session_duration:
                try:
                    prediction = self.client.get_action(pattern_input, timeout=3.0)
                    if prediction is None:
                        print("   ⚠️ No response from brain")
                        time.sleep(1)
                        continue
                    
                    error = np.mean(np.abs(np.array(prediction) - np.array(expected_output)))
                    session_errors.append(error)
                    session_predictions.append(prediction.copy())
                    training_cycles += 1
                    
                    # Brief pause between predictions (biological realism)
                    time.sleep(0.2)
                    
                except KeyboardInterrupt:
                    print("\n⏹️ Test interrupted by user")
                    return self._generate_results(overall_performance, consolidation_checkpoints)
                except Exception as e:
                    print(f"   ⚠️ Error in session: {e}")
                    time.sleep(5)
                    break
            
            # Analyze session performance
            if session_errors:
                session_avg_error = np.mean(session_errors)
                session_improvement = session_errors[0] - session_errors[-1] if len(session_errors) > 1 else 0
                best_prediction = session_predictions[np.argmin(session_errors)]
                
                session_data = {
                    'session': session_count,
                    'timestamp': time.time(),
                    'duration_minutes': (time.time() - session_start) / 60,
                    'training_cycles': training_cycles,
                    'avg_error': session_avg_error,
                    'improvement_within_session': session_improvement,
                    'best_prediction': best_prediction,
                    'best_error': min(session_errors),
                    'elapsed_hours': (time.time() - test_start) / 3600
                }
                
                overall_performance.append(session_data)
                
                print(f"   📊 Session Results:")
                print(f"      Training cycles: {training_cycles}")
                print(f"      Average error: {session_avg_error:.3f}")
                print(f"      Best error: {min(session_errors):.3f}")
                print(f"      Best prediction: {[f'{x:.2f}' for x in best_prediction]}")
                print(f"      Within-session improvement: {session_improvement:.3f}")
                
                # Check for consolidation checkpoint (every hour)
                if len(overall_performance) > 1:
                    hours_elapsed = session_data['elapsed_hours']
                    if hours_elapsed >= len(consolidation_checkpoints) + 1:
                        checkpoint_data = self._analyze_consolidation_checkpoint(overall_performance)
                        consolidation_checkpoints.append(checkpoint_data)
                        print(f"\n🧠 CONSOLIDATION CHECKPOINT ({hours_elapsed:.1f} hours)")
                        print(f"   Long-term improvement: {checkpoint_data['long_term_improvement']:.3f}")
                        print(f"   Learning stability: {checkpoint_data['stability_score']:.3f}")
                        print(f"   Consolidation strength: {checkpoint_data['consolidation_strength']:.3f}")
            
            # Biological rest period between sessions (consolidation time)
            rest_duration = 180  # 3 minutes (shortened for testing, real biology = 30-60 min)
            if (time.time() - test_start) < total_duration - rest_duration:
                print(f"   😴 Consolidation break: {rest_duration//60} minutes...")
                print(f"      (Brain consolidating memories - biological process)")
                
                # Send keepalive pings during consolidation to maintain connection
                consolidation_start = time.time()
                while (time.time() - consolidation_start) < rest_duration:
                    # Sleep for 30 seconds, then send keepalive
                    time.sleep(30)
                    if (time.time() - consolidation_start) < rest_duration:
                        try:
                            # Send keepalive ping
                            self.client.get_action([0.0, 0.0, 0.0, 0.0], timeout=5.0)
                        except Exception:
                            pass  # Ignore keepalive failures during consolidation
        
        return self._generate_results(overall_performance, consolidation_checkpoints)
    
    def _analyze_consolidation_checkpoint(self, performance_history: List[Dict]) -> Dict:
        """Analyze learning consolidation at biological timescale checkpoints."""
        
        if len(performance_history) < 3:
            return {'long_term_improvement': 0, 'stability_score': 0, 'consolidation_strength': 0}
        
        # Compare first hour vs recent hour
        recent_sessions = [p for p in performance_history if p['elapsed_hours'] >= max(0, performance_history[-1]['elapsed_hours'] - 1)]
        early_sessions = performance_history[:min(3, len(performance_history)//3)]
        
        if not recent_sessions or not early_sessions:
            return {'long_term_improvement': 0, 'stability_score': 0, 'consolidation_strength': 0}
        
        # Long-term improvement
        early_avg_error = np.mean([s['avg_error'] for s in early_sessions])
        recent_avg_error = np.mean([s['avg_error'] for s in recent_sessions])
        long_term_improvement = early_avg_error - recent_avg_error
        
        # Learning stability (consistency of performance)
        recent_errors = [s['avg_error'] for s in recent_sessions]
        stability_score = 1.0 / (1.0 + np.std(recent_errors))  # Higher = more stable
        
        # Consolidation strength (best performance improvement)
        all_best_errors = [s['best_error'] for s in performance_history]
        early_best = min(all_best_errors[:len(all_best_errors)//3]) if len(all_best_errors) >= 3 else 1.0
        recent_best = min(all_best_errors[-len(all_best_errors)//3:]) if len(all_best_errors) >= 3 else 1.0
        consolidation_strength = early_best - recent_best
        
        return {
            'long_term_improvement': long_term_improvement,
            'stability_score': stability_score,
            'consolidation_strength': consolidation_strength,
            'sessions_analyzed': len(recent_sessions),
            'early_avg_error': early_avg_error,
            'recent_avg_error': recent_avg_error
        }
    
    def _generate_results(self, performance_history: List[Dict], checkpoints: List[Dict]) -> Dict:
        """Generate comprehensive biological timescale results."""
        
        if not performance_history:
            return {'error': 'No data collected'}
        
        total_duration_hours = performance_history[-1]['elapsed_hours']
        total_sessions = len(performance_history)
        
        # Overall learning trajectory
        all_avg_errors = [s['avg_error'] for s in performance_history]
        all_best_errors = [s['best_error'] for s in performance_history]
        
        overall_improvement = all_avg_errors[0] - all_avg_errors[-1]
        best_error_achieved = min(all_best_errors)
        best_session = next(s for s in performance_history if s['best_error'] == best_error_achieved)
        
        # Learning curve analysis
        first_third = all_avg_errors[:len(all_avg_errors)//3] if len(all_avg_errors) >= 3 else all_avg_errors
        middle_third = all_avg_errors[len(all_avg_errors)//3:2*len(all_avg_errors)//3] if len(all_avg_errors) >= 3 else []
        final_third = all_avg_errors[2*len(all_avg_errors)//3:] if len(all_avg_errors) >= 3 else []
        
        learning_phases = {
            'initial_learning': np.mean(first_third) if first_third else 0,
            'consolidation_phase': np.mean(middle_third) if middle_third else 0,
            'stable_performance': np.mean(final_third) if final_third else 0
        }
        
        # Biological assessment
        biological_learning_detected = (
            overall_improvement > 0.2 and  # Significant improvement
            total_duration_hours >= 1.0 and  # Sufficient time for consolidation
            best_error_achieved < 0.5  # Achieved good performance
        )
        
        return {
            'test_type': 'biological_timescale',
            'total_duration_hours': total_duration_hours,
            'total_sessions': total_sessions,
            'overall_improvement': overall_improvement,
            'best_error_achieved': best_error_achieved,
            'best_achieved_at_session': best_session['session'],
            'best_achieved_at_hours': best_session['elapsed_hours'],
            'learning_phases': learning_phases,
            'consolidation_checkpoints': checkpoints,
            'biological_learning_detected': biological_learning_detected,
            'performance_history': performance_history,
            'session_summary': {
                'avg_cycles_per_session': np.mean([s['training_cycles'] for s in performance_history]),
                'avg_session_improvement': np.mean([s['improvement_within_session'] for s in performance_history]),
                'most_productive_session': max(performance_history, key=lambda s: s['improvement_within_session'])['session']
            }
        }
    
    def run_biological_test(self, hours: float = 2.0):
        """Run the biological timescale test."""
        
        print("🧬 BIOLOGICAL TIMESCALE BRAIN LEARNING TEST")
        print("=" * 60)
        print("This test respects biological learning timescales:")
        print("- Training sessions with consolidation breaks")
        print("- Long-term improvement measured over hours")
        print("- Memory consolidation between sessions")
        print(f"- Total duration: {hours} hours")
        
        if hours < 0.5:
            print("⚠️ Warning: Less than 30 minutes may not show biological learning")
        
        if not self.connect():
            print("❌ Could not connect to brain server")
            return None
        
        try:
            results = self.test_pattern_over_biological_time(target_duration_hours=hours)
            self.print_biological_assessment(results)
            return results
            
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted - analyzing partial results...")
            return None
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.disconnect()
    
    def print_biological_assessment(self, results: Dict):
        """Print biological learning assessment."""
        
        if 'error' in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"\n🧬 BIOLOGICAL LEARNING ASSESSMENT")
        print("=" * 50)
        
        print(f"📅 Test Duration: {results['total_duration_hours']:.1f} hours")
        print(f"📚 Training Sessions: {results['total_sessions']}")
        print(f"🎯 Overall Improvement: {results['overall_improvement']:.3f}")
        print(f"🏆 Best Error Achieved: {results['best_error_achieved']:.3f}")
        print(f"   (at session {results['best_achieved_at_session']}, {results['best_achieved_at_hours']:.1f} hours)")
        
        # Learning phases
        phases = results['learning_phases']
        print(f"\n📈 Learning Phases:")
        print(f"   Initial Learning: {phases['initial_learning']:.3f}")
        if phases['consolidation_phase']:
            print(f"   Consolidation: {phases['consolidation_phase']:.3f}")
        if phases['stable_performance']:
            print(f"   Stable Performance: {phases['stable_performance']:.3f}")
        
        # Consolidation analysis
        if results['consolidation_checkpoints']:
            print(f"\n🧠 Memory Consolidation:")
            for i, checkpoint in enumerate(results['consolidation_checkpoints']):
                print(f"   Hour {i+1}: {checkpoint['long_term_improvement']:.3f} improvement, "
                      f"{checkpoint['stability_score']:.3f} stability")
        
        # Biological assessment
        biological_detected = results['biological_learning_detected']
        print(f"\n🧬 Biological Learning Detected: {'✅ YES' if biological_detected else '❌ NO'}")
        
        if biological_detected:
            print("   🎉 The brain shows biological learning patterns!")
            print("   📝 Long-term consolidation and improvement detected")
        else:
            print("   💡 Consider longer test duration for biological timescales")
            if results['total_duration_hours'] < 1.0:
                print("   ⚠️ Less than 1 hour - insufficient for consolidation")
        
        # Performance insights
        session_summary = results['session_summary']
        print(f"\n📊 Session Performance:")
        print(f"   Avg cycles per session: {session_summary['avg_cycles_per_session']:.0f}")
        print(f"   Avg within-session improvement: {session_summary['avg_session_improvement']:.3f}")
        print(f"   Most productive session: #{session_summary['most_productive_session']}")


def main():
    """Run biological timescale test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Biological Timescale Brain Learning Test')
    parser.add_argument('--hours', type=float, default=2.0, 
                       help='Test duration in hours (minimum 0.5, recommended 2+)')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick 30-minute test (not biologically realistic)')
    
    args = parser.parse_args()
    
    if args.quick:
        hours = 0.5
        print("⚡ Quick test mode - not biologically realistic but useful for debugging")
    else:
        hours = max(0.5, args.hours)
    
    if hours >= 4:
        print("⚠️ Long test detected - ensure brain server stability")
        response = input(f"Run {hours}-hour test? (y/n): ").lower()
        if response != 'y':
            print("Test cancelled")
            return
    
    tester = BiologicalTimescaleTester()
    tester.run_biological_test(hours=hours)

if __name__ == "__main__":
    main()