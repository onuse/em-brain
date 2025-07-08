#!/usr/bin/env python3
"""
Mental Development Analyzer - Evaluates robot's learning progress from persistent memory

This tool analyzes the saved persistent memory to evaluate:
1. Experience accumulation and quality
2. Memory consolidation patterns  
3. Adaptive parameter evolution
4. Learning trajectory over time
5. Behavioral development indicators
"""

import json
import pickle
import gzip
import os
from datetime import datetime
from typing import Dict, List, Any
import glob


class MentalDevelopmentAnalyzer:
    """Analyzes robot's mental development from persistent memory."""
    
    def __init__(self, memory_path: str = "robot_memory"):
        self.memory_path = memory_path
        self.sessions = []
        self.graphs = []
        self.adaptive_params = []
        
    def load_all_data(self):
        """Load all available persistent memory data."""
        print("📊 Loading persistent memory data...")
        
        # Load session summaries
        session_files = glob.glob(f"{self.memory_path}/sessions/*.json")
        for session_file in sorted(session_files):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                session_data['filename'] = os.path.basename(session_file)
                self.sessions.append(session_data)
        
        # Load graphs (experience data)
        graph_files = glob.glob(f"{self.memory_path}/graphs/*.pkl.gz")
        for graph_file in sorted(graph_files):
            try:
                with gzip.open(graph_file, 'rb') as f:
                    graph_data = pickle.load(f)
                    graph_info = {
                        'filename': os.path.basename(graph_file),
                        'timestamp': os.path.getctime(graph_file),
                        'data': graph_data
                    }
                    self.graphs.append(graph_info)
            except Exception as e:
                print(f"   Warning: Could not load {graph_file}: {e}")
        
        # Load adaptive parameters
        param_files = glob.glob(f"{self.memory_path}/adaptive_params/*.json")
        for param_file in sorted(param_files):
            with open(param_file, 'r') as f:
                param_data = json.load(f)
                param_data['filename'] = os.path.basename(param_file)
                self.adaptive_params.append(param_data)
        
        print(f"   📁 Loaded {len(self.sessions)} sessions")
        print(f"   📁 Loaded {len(self.graphs)} memory graphs")
        print(f"   📁 Loaded {len(self.adaptive_params)} adaptive parameter snapshots")
    
    def analyze_learning_progression(self):
        """Analyze how learning has progressed over time."""
        print("\\n🧠 LEARNING PROGRESSION ANALYSIS")
        print("=" * 50)
        
        if not self.sessions:
            print("❌ No session data available for analysis")
            return
        
        # Analyze session-by-session progression
        total_experiences = 0
        total_adaptations = 0
        
        print("📈 Session-by-Session Development:")
        for i, session in enumerate(self.sessions):
            session_exp = session.get('experiences_count', 0)
            session_adapt = session.get('adaptive_tuning_stats', {}).get('total_adaptations', 0)
            
            total_experiences += session_exp
            total_adaptations += session_adapt
            
            graph_stats = session.get('final_graph_stats', {})
            avg_strength = graph_stats.get('avg_strength', 0)
            max_strength = graph_stats.get('max_strength', 0)
            
            print(f"   Session {i+1}: {session_exp} new experiences, "
                  f"{session_adapt} adaptations")
            print(f"             Avg memory strength: {avg_strength:.1f}, "
                  f"Max: {max_strength:.1f}")
        
        print(f"\\n📊 CUMULATIVE DEVELOPMENT:")
        print(f"   Total experiences across all sessions: {total_experiences}")
        print(f"   Total parameter adaptations: {total_adaptations}")
        print(f"   Learning sessions completed: {len(self.sessions)}")
        
        # Check for learning indicators
        if total_adaptations > 0:
            print("   ✅ System is actively adapting parameters")
        else:
            print("   ⚠️  No parameter adaptations detected")
        
        if total_experiences > 50:
            print("   ✅ Substantial experience accumulation")
        elif total_experiences > 10:
            print("   🔶 Moderate experience accumulation")
        else:
            print("   ⚠️  Limited experience accumulation")
    
    def analyze_memory_quality(self):
        """Analyze the quality and characteristics of accumulated memories."""
        print("\\n🧩 MEMORY QUALITY ANALYSIS")
        print("=" * 50)
        
        if not self.graphs:
            print("❌ No memory graph data available")
            return
        
        # Analyze latest memory graph
        latest_graph = self.graphs[-1]
        graph_data = latest_graph['data']
        
        print(f"📊 Latest Memory Graph Analysis:")
        print(f"   File: {latest_graph['filename']}")
        
        # Try to analyze graph structure
        try:
            if hasattr(graph_data, 'get_graph_statistics'):
                stats = graph_data.get_graph_statistics()
                print(f"   Total memory nodes: {stats.get('total_nodes', 0)}")
                print(f"   Memory consolidations: {stats.get('total_merges', 0)}")
                print(f"   Average memory strength: {stats.get('avg_strength', 0):.1f}")
                print(f"   Memory access count: {stats.get('total_accesses', 0)}")
                
                # Analyze memory quality indicators
                total_nodes = stats.get('total_nodes', 0)
                total_merges = stats.get('total_merges', 0)
                avg_strength = stats.get('avg_strength', 0)
                
                if total_nodes > 20:
                    print("   ✅ Good memory accumulation")
                elif total_nodes > 5:
                    print("   🔶 Moderate memory accumulation")
                else:
                    print("   ⚠️  Limited memory accumulation")
                
                if total_merges > 0:
                    merge_rate = total_merges / total_nodes if total_nodes > 0 else 0
                    print(f"   Memory consolidation rate: {merge_rate:.2f}")
                    if merge_rate > 0.1:
                        print("   ✅ Active memory consolidation")
                    else:
                        print("   🔶 Limited memory consolidation")
                else:
                    print("   ⚠️  No memory consolidation detected")
                
                if avg_strength > 100:
                    print("   ✅ Strong memory formation")
                elif avg_strength > 10:
                    print("   🔶 Moderate memory strength")
                else:
                    print("   ⚠️  Weak memory formation")
            
            elif hasattr(graph_data, 'nodes'):
                # Alternative analysis if graph has nodes attribute
                node_count = len(graph_data.nodes) if hasattr(graph_data.nodes, '__len__') else 0
                print(f"   Memory nodes detected: {node_count}")
            
            else:
                print("   📊 Memory graph structure analysis not available")
                print(f"   Graph type: {type(graph_data)}")
                if hasattr(graph_data, '__dict__'):
                    attrs = [attr for attr in dir(graph_data) if not attr.startswith('_')]
                    print(f"   Available attributes: {attrs[:5]}...")
        
        except Exception as e:
            print(f"   ⚠️  Error analyzing memory graph: {e}")
    
    def analyze_adaptive_parameters(self):
        """Analyze how adaptive parameters have evolved."""
        print("\\n⚙️  ADAPTIVE PARAMETER EVOLUTION")
        print("=" * 50)
        
        if not self.adaptive_params:
            print("❌ No adaptive parameter data available")
            return
        
        # Compare first and last parameter sets
        first_params = self.adaptive_params[0]
        last_params = self.adaptive_params[-1]
        
        print("📈 Parameter Evolution Analysis:")
        print(f"   First snapshot: {first_params['filename']}")
        print(f"   Latest snapshot: {last_params['filename']}")
        
        # Check adaptation statistics
        first_stats = first_params.get('adaptation_statistics', {})
        last_stats = last_params.get('adaptation_statistics', {})
        
        first_adaptations = first_stats.get('total_adaptations', 0)
        last_adaptations = last_stats.get('total_adaptations', 0)
        adaptation_growth = last_adaptations - first_adaptations
        
        print(f"   Adaptation growth: {first_adaptations} → {last_adaptations} (+{adaptation_growth})")
        
        if adaptation_growth > 0:
            print("   ✅ System is actively learning and adapting")
        else:
            print("   ⚠️  No parameter adaptation detected")
        
        # Analyze parameter changes
        first_current = first_params.get('current_parameters', {})
        last_current = last_params.get('current_parameters', {})
        
        print("\\n📊 Key Parameter Changes:")
        key_params = ['similarity_threshold', 'exploration_rate', 'connection_learning_rate']
        
        for param in key_params:
            if param in first_current and param in last_current:
                first_val = first_current[param]
                last_val = last_current[param]
                change = last_val - first_val
                change_pct = (change / first_val * 100) if first_val != 0 else 0
                
                print(f"   {param}: {first_val:.3f} → {last_val:.3f} "
                      f"({change:+.3f}, {change_pct:+.1f}%)")
        
        # Analyze sensory insights
        last_sensory = last_stats.get('sensory_insights', {})
        if last_sensory:
            print("\\n👁️  Sensory Processing Development:")
            bandwidth = last_sensory.get('bandwidth_tier', 'unknown')
            total_dims = last_sensory.get('total_dimensions', 0)
            stable_dims = len(last_sensory.get('stable_dimensions', []))
            
            print(f"   Sensory bandwidth tier: {bandwidth}")
            print(f"   Total sensory dimensions: {total_dims}")
            print(f"   Stable dimensions identified: {stable_dims}")
            
            if stable_dims > 0:
                print("   ✅ Sensory pattern recognition developing")
            else:
                print("   🔶 Early sensory development stage")
    
    def analyze_behavioral_indicators(self):
        """Look for indicators of behavioral development."""
        print("\\n🤖 BEHAVIORAL DEVELOPMENT INDICATORS")
        print("=" * 50)
        
        # Analyze trends across sessions
        if len(self.sessions) < 2:
            print("⚠️  Need multiple sessions to analyze behavioral trends")
            return
        
        print("📈 Cross-Session Behavioral Analysis:")
        
        # Track experience quality over time
        experience_qualities = []
        memory_strengths = []
        
        for session in self.sessions:
            graph_stats = session.get('final_graph_stats', {})
            avg_strength = graph_stats.get('avg_strength', 0)
            total_nodes = graph_stats.get('total_nodes', 0)
            
            if total_nodes > 0:
                experience_qualities.append(total_nodes)
                memory_strengths.append(avg_strength)
        
        if len(experience_qualities) >= 2:
            # Check for improvement trends
            early_avg = sum(experience_qualities[:len(experience_qualities)//2]) / (len(experience_qualities)//2)
            recent_avg = sum(experience_qualities[len(experience_qualities)//2:]) / (len(experience_qualities) - len(experience_qualities)//2)
            
            print(f"   Experience accumulation trend:")
            print(f"     Early sessions avg: {early_avg:.1f} experiences")
            print(f"     Recent sessions avg: {recent_avg:.1f} experiences")
            
            if recent_avg > early_avg * 1.2:
                print("   ✅ Improving learning efficiency")
            elif recent_avg > early_avg * 0.8:
                print("   🔶 Stable learning rate")
            else:
                print("   ⚠️  Declining learning efficiency")
        
        # Check for memory consolidation patterns
        total_merges = sum(s.get('final_graph_stats', {}).get('total_merges', 0) for s in self.sessions)
        total_experiences = sum(s.get('experiences_count', 0) for s in self.sessions)
        
        if total_merges > 0 and total_experiences > 0:
            consolidation_rate = total_merges / total_experiences
            print(f"\\n   Memory consolidation rate: {consolidation_rate:.2f}")
            if consolidation_rate > 0.1:
                print("   ✅ Active pattern recognition and memory optimization")
            else:
                print("   🔶 Basic memory accumulation")
        
        # Overall development assessment
        print("\\n🎯 OVERALL DEVELOPMENT ASSESSMENT:")
        
        development_score = 0
        max_score = 5
        
        if total_experiences > 20:
            development_score += 1
            print("   ✅ Sufficient experience accumulation")
        
        if total_merges > 0:
            development_score += 1
            print("   ✅ Memory consolidation active")
        
        total_adaptations = sum(s.get('adaptive_tuning_stats', {}).get('total_adaptations', 0) for s in self.sessions)
        if total_adaptations > 0:
            development_score += 1
            print("   ✅ Parameter adaptation occurring")
        
        if len(self.sessions) > 3:
            development_score += 1
            print("   ✅ Sustained learning across multiple sessions")
        
        if any(s.get('final_graph_stats', {}).get('avg_strength', 0) > 100 for s in self.sessions):
            development_score += 1
            print("   ✅ Strong memory formation detected")
        
        print(f"\\n📊 Development Score: {development_score}/{max_score}")
        
        if development_score >= 4:
            print("🌟 EXCELLENT: Robot shows strong mental development!")
        elif development_score >= 3:
            print("✅ GOOD: Robot is developing meaningful intelligence")
        elif development_score >= 2:
            print("🔶 MODERATE: Robot shows some learning progress")
        else:
            print("⚠️  LIMITED: Robot needs more learning opportunities")
    
    def generate_report(self):
        """Generate a comprehensive mental development report."""
        print("🧠 ROBOT MENTAL DEVELOPMENT ANALYSIS REPORT")
        print("=" * 60)
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Memory Path: {self.memory_path}")
        print()
        
        self.load_all_data()
        self.analyze_learning_progression()
        self.analyze_memory_quality()
        self.analyze_adaptive_parameters()
        self.analyze_behavioral_indicators()
        
        print("\\n" + "=" * 60)
        print("📋 SUMMARY & RECOMMENDATIONS")
        print("=" * 60)
        
        if not self.sessions:
            print("❌ No learning data found. Robot needs to run some demo sessions.")
            return
        
        total_exp = sum(s.get('experiences_count', 0) for s in self.sessions)
        total_adapt = sum(s.get('adaptive_tuning_stats', {}).get('total_adaptations', 0) for s in self.sessions)
        
        print("📊 Key Metrics:")
        print(f"   Learning sessions: {len(self.sessions)}")
        print(f"   Total experiences: {total_exp}")
        print(f"   Parameter adaptations: {total_adapt}")
        
        print("\\n💡 Recommendations:")
        if total_exp < 50:
            print("   • Run more demo sessions for deeper learning")
        if total_adapt == 0:
            print("   • Check if adaptive systems are properly integrated")
        if len(self.sessions) < 5:
            print("   • Continue running sessions for cross-session learning")
        
        print("\\n✨ The robot's mind is developing through pure experience!")


def main():
    """Run the mental development analysis."""
    analyzer = MentalDevelopmentAnalyzer()
    analyzer.generate_report()


if __name__ == "__main__":
    main()