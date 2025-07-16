"""
Similarity Search Engine

Ultra-fast similarity search through experience memory with adaptive attention.
This is what makes the brain intelligent - finding similar past situations in milliseconds
while naturally suppressing boring memories without losing information.
"""

from .engine import SimilarityEngine
from .adaptive_attention import AdaptiveAttentionScorer, NaturalAttentionSimilarity

__all__ = ['SimilarityEngine', 'AdaptiveAttentionScorer', 'NaturalAttentionSimilarity']