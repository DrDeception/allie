"""
Allie Deception Detection Module

Audiovisual baseline-deviation analysis for psychological and deception research.

Main components:
- deception_features: Extract audiovisual features relevant to deception
- baseline: Create honest behavioral baselines from truthful samples
- deviation: Analyze test samples for deviations from baseline
- deception_detect: Main CLI interface

Example usage:
    # Create baseline
    from deception.baseline import create_baseline
    baseline = create_baseline('S001', './honest_videos/')

    # Test against baseline
    from deception.deviation import analyze_deviation
    report = analyze_deviation('baseline_S001.json', 'test.mp4')
"""

__version__ = '1.0.0'
__author__ = 'Allie Development Team'

# Import main functions for easy access
from .baseline import create_baseline, load_baseline
from .deviation import analyze_deviation

__all__ = ['create_baseline', 'load_baseline', 'analyze_deviation']
