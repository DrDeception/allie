'''
               AAA               lllllll lllllll   iiii
              A:::A              l:::::l l:::::l  i::::i
             A:::::A             l:::::l l:::::l   iiii
            A:::::::A            l:::::l l:::::l
           A:::::::::A            l::::l  l::::l iiiiiii     eeeeeeeeeeee
          A:::::A:::::A           l::::l  l::::l i:::::i   ee::::::::::::ee
         A:::::A A:::::A          l::::l  l::::l  i::::i  e::::::eeeee:::::ee
        A:::::A   A:::::A         l::::l  l::::l  i::::i e::::::e     e:::::e
       A:::::A     A:::::A        l::::l  l::::l  i::::i e:::::::eeeee::::::e
      A:::::AAAAAAAAA:::::A       l::::l  l::::l  i::::i e:::::::::::::::::e
     A:::::::::::::::::::::A      l::::l  l::::l  i::::i e::::::eeeeeeeeeee
    A:::::AAAAAAAAAAAAA:::::A     l::::l  l::::l  i::::i e:::::::e
   A:::::A             A:::::A   l::::::ll::::::li::::::ie::::::::e
  A:::::A               A:::::A  l::::::ll::::::li::::::i e::::::::eeeeeeee
 A:::::A                 A:::::A l::::::ll::::::li::::::i  ee:::::::::::::e
AAAAAAA                   AAAAAAAlllllllllllllllliiiiiiii    eeeeeeeeeeeeee

Deception Detection - Deviation Analysis Module

This module analyzes test videos against an individual's honest baseline to
detect behavioral deviations that may indicate deception. It calculates
statistical deviation scores (z-scores, Mahalanobis distance) and generates
a deception likelihood assessment.

Key Features:
- Loads baseline profile and test video
- Extracts deception features from test sample
- Calculates z-scores for each feature
- Computes overall deviation metrics
- Generates deception likelihood score with confidence
- Identifies most deviant features

Usage: python3 deviation.py --baseline [profile.json] --test_video [video.mp4] --output [report.json]
'''

import os, sys, json, argparse
import numpy as np
from datetime import datetime
from scipy.spatial.distance import mahalanobis
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
from deception_features import deception_featurize

def calculate_z_score(value, mean, std):
    """Calculate z-score for a feature value"""
    if std == 0:
        return 0
    return (value - mean) / std

def calculate_deviation_score(z_scores):
    """
    Calculate overall deviation score from z-scores.
    Uses RMS of absolute z-scores as overall metric.
    """
    abs_z = np.abs(z_scores)
    rms_z = np.sqrt(np.mean(abs_z ** 2))
    return rms_z

def assess_deception_likelihood(deviation_score, significant_deviations_count, total_features):
    """
    Assess likelihood of deception based on deviation metrics.

    Thresholds based on statistical significance:
    - Low: < 1.5 SD from baseline (normal variation)
    - Medium: 1.5-2.5 SD from baseline (noteworthy)
    - High: 2.5-3.5 SD from baseline (significant)
    - Very High: > 3.5 SD from baseline (extreme deviation)
    """

    deviation_ratio = significant_deviations_count / total_features

    if deviation_score < 1.5:
        likelihood = "Very Low"
        confidence = "High"
        interpretation = "Behavior consistent with honest baseline. No significant deviations detected."
    elif deviation_score < 2.5:
        if deviation_ratio < 0.15:
            likelihood = "Low"
            confidence = "Medium"
            interpretation = "Minor deviations from baseline. Likely within normal variation."
        else:
            likelihood = "Medium"
            confidence = "Medium"
            interpretation = "Moderate deviations detected. Further investigation recommended."
    elif deviation_score < 3.5:
        likelihood = "High"
        confidence = "Medium"
        interpretation = "Significant deviations from baseline across multiple features. Possible deception indicators."
    else:
        likelihood = "Very High"
        confidence = "High"
        interpretation = "Extreme deviations from honest baseline. Strong deception indicators present."

    return {
        'likelihood': likelihood,
        'confidence': confidence,
        'interpretation': interpretation,
        'deviation_score': deviation_score,
        'significant_deviations_ratio': deviation_ratio
    }

def analyze_deviation(baseline_path, test_video_path, output_path=None, verbose=True):
    """
    Analyze a test video against baseline to detect deviations.

    Args:
        baseline_path: Path to baseline JSON file
        test_video_path: Path to test video (.mp4)
        output_path: Path to save analysis report (optional)
        verbose: Print detailed output

    Returns:
        analysis_report: Dictionary containing deviation analysis
    """

    if verbose:
        print(f"\n{'='*70}")
        print(f"Deception Deviation Analysis")
        print(f"{'='*70}\n")

    # Load baseline
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        if verbose:
            print(f"✓ Loaded baseline for subject: {baseline['subject_id']}")
            print(f"  Created: {baseline['creation_date']}")
            print(f"  Based on {baseline['n_samples']} samples\n")
    except Exception as e:
        print(f"ERROR loading baseline: {e}")
        return None

    # Extract features from test video
    if verbose:
        print(f"Analyzing test video: {os.path.basename(test_video_path)}\n")

    try:
        test_features, test_labels = deception_featurize(test_video_path)
    except Exception as e:
        print(f"ERROR extracting features from test video: {e}")
        return None

    if len(test_features) == 0:
        print("ERROR: No features extracted from test video")
        return None

    # Calculate deviations for each feature
    feature_deviations = {}
    z_scores = []
    significant_count = 0  # Count features with |z| > 2

    for i, label in enumerate(test_labels):
        if label in baseline['baseline_statistics']:
            test_value = test_features[i]
            baseline_stats = baseline['baseline_statistics'][label]

            # Calculate z-score
            z_score = calculate_z_score(
                test_value,
                baseline_stats['mean'],
                baseline_stats['std']
            )
            z_scores.append(z_score)

            # Check if significantly deviant (|z| > 2 is ~95th percentile)
            is_significant = abs(z_score) > 2.0

            if is_significant:
                significant_count += 1

            feature_deviations[label] = {
                'test_value': float(test_value),
                'baseline_mean': baseline_stats['mean'],
                'baseline_std': baseline_stats['std'],
                'baseline_range': [baseline_stats['min'], baseline_stats['max']],
                'z_score': float(z_score),
                'is_significant': is_significant,
                'deviation_direction': 'higher' if z_score > 0 else 'lower'
            }

    # Calculate overall deviation metrics
    z_scores_array = np.array(z_scores)
    overall_deviation_score = calculate_deviation_score(z_scores_array)

    # Assess deception likelihood
    assessment = assess_deception_likelihood(
        overall_deviation_score,
        significant_count,
        len(test_labels)
    )

    # Create analysis report
    analysis_report = {
        'subject_id': baseline['subject_id'],
        'test_video': os.path.basename(test_video_path),
        'analysis_date': datetime.now().isoformat(),
        'baseline_file': os.path.basename(baseline_path),
        'overall_metrics': {
            'deviation_score': float(overall_deviation_score),
            'mean_absolute_z_score': float(np.mean(np.abs(z_scores_array))),
            'max_absolute_z_score': float(np.max(np.abs(z_scores_array))),
            'significant_deviations': significant_count,
            'total_features': len(test_labels),
            'significant_deviation_percentage': float(significant_count / len(test_labels) * 100)
        },
        'deception_assessment': assessment,
        'feature_deviations': feature_deviations
    }

    # Identify top deviant features
    sorted_features = sorted(
        feature_deviations.items(),
        key=lambda x: abs(x[1]['z_score']),
        reverse=True
    )
    analysis_report['top_deviant_features'] = [
        {
            'feature': feat[0],
            'z_score': feat[1]['z_score'],
            'direction': feat[1]['deviation_direction']
        }
        for feat in sorted_features[:10]
    ]

    # Save report
    if output_path is None:
        output_path = f"deviation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, 'w') as f:
        json.dump(analysis_report, f, indent=2)

    if verbose:
        print(f"{'='*70}")
        print(f"Analysis Results")
        print(f"{'='*70}\n")

        print(f"Overall Deviation Score: {overall_deviation_score:.2f}")
        print(f"Significant Deviations: {significant_count}/{len(test_labels)} ({significant_count/len(test_labels)*100:.1f}%)\n")

        print(f"{'='*70}")
        print(f"Deception Assessment")
        print(f"{'='*70}\n")

        print(f"Likelihood: {assessment['likelihood']}")
        print(f"Confidence: {assessment['confidence']}")
        print(f"Interpretation: {assessment['interpretation']}\n")

        print(f"{'='*70}")
        print(f"Top 10 Most Deviant Features")
        print(f"{'='*70}\n")

        for i, feat in enumerate(analysis_report['top_deviant_features'], 1):
            print(f"{i}. {feat['feature']}")
            print(f"   Z-score: {feat['z_score']:.2f} ({feat['direction']} than baseline)\n")

        print(f"{'='*70}\n")
        print(f"✓ Full analysis report saved to: {output_path}\n")

    return analysis_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze video for deviations from honest baseline')

    parser.add_argument('--baseline', '-b', type=str, required=True,
                        help='Path to baseline profile JSON file')
    parser.add_argument('--test_video', '-t', type=str, required=True,
                        help='Path to test video file (.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for analysis report (default: auto-generated)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.baseline):
        print(f"ERROR: Baseline file not found: {args.baseline}")
        sys.exit(1)

    if not os.path.isfile(args.test_video):
        print(f"ERROR: Test video file not found: {args.test_video}")
        sys.exit(1)

    # Analyze deviation
    report = analyze_deviation(
        baseline_path=args.baseline,
        test_video_path=args.test_video,
        output_path=args.output,
        verbose=not args.quiet
    )

    if report:
        print("✓ Deviation analysis completed successfully!")
        sys.exit(0)
    else:
        print("✗ Deviation analysis failed.")
        sys.exit(1)
