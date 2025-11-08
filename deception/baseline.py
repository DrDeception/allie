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

Deception Detection - Baseline Creation Module

This module creates an individual's honest behavioral baseline from multiple
truthful video samples. The baseline captures the person's normal audiovisual
patterns during honest communication, which can later be compared against
test samples to detect deviations that may indicate deception.

Key Features:
- Processes multiple honest/truthful video samples
- Extracts deception-relevant audiovisual features
- Computes statistical baseline (mean, std, percentiles, ranges)
- Saves baseline profile for later deviation analysis

Usage: python3 baseline.py --subject_id [ID] --baseline_dir [directory] --output [filename]
'''

import os, sys, json, argparse
import numpy as np
from datetime import datetime

# Add parent directory to path to import deception_features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
from deception_features import deception_featurize

def create_baseline(subject_id, baseline_dir, output_path=None):
    """
    Create an honest baseline profile from multiple video samples.

    Args:
        subject_id: Unique identifier for the subject
        baseline_dir: Directory containing honest/truthful video samples (.mp4)
        output_path: Path to save baseline profile (optional)

    Returns:
        baseline_profile: Dictionary containing statistical baseline
    """

    print(f"\n{'='*70}")
    print(f"Creating Honest Baseline for Subject: {subject_id}")
    print(f"{'='*70}\n")

    # Find all video files in baseline directory
    video_files = [f for f in os.listdir(baseline_dir) if f.endswith('.mp4')]

    if len(video_files) < 3:
        print(f"WARNING: Only {len(video_files)} video(s) found. Recommend at least 3-5 samples for reliable baseline.")

    print(f"Found {len(video_files)} baseline video samples\n")

    # Extract features from all baseline videos
    all_features = []
    feature_labels = None

    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(baseline_dir, video_file)
        print(f"[{i}/{len(video_files)}] Processing: {video_file}")

        try:
            features, labels = deception_featurize(video_path)

            if len(features) > 0:
                all_features.append(features)
                if feature_labels is None:
                    feature_labels = labels
            else:
                print(f"  WARNING: No features extracted from {video_file}")

        except Exception as e:
            print(f"  ERROR processing {video_file}: {e}")

    if len(all_features) == 0:
        print("\nERROR: No features extracted from any videos. Baseline creation failed.")
        return None

    # Convert to numpy array for statistical analysis
    features_array = np.array(all_features)  # Shape: (n_samples, n_features)

    print(f"\n{'='*70}")
    print(f"Computing Statistical Baseline")
    print(f"{'='*70}\n")
    print(f"Samples processed: {features_array.shape[0]}")
    print(f"Features per sample: {features_array.shape[1]}\n")

    # Compute baseline statistics for each feature
    baseline_profile = {
        'subject_id': subject_id,
        'creation_date': datetime.now().isoformat(),
        'n_samples': len(all_features),
        'sample_files': video_files,
        'n_features': len(feature_labels),
        'feature_labels': feature_labels,
        'baseline_statistics': {}
    }

    # Calculate statistics for each feature
    for i, label in enumerate(feature_labels):
        feature_values = features_array[:, i]

        baseline_profile['baseline_statistics'][label] = {
            'mean': float(np.mean(feature_values)),
            'std': float(np.std(feature_values)),
            'median': float(np.median(feature_values)),
            'min': float(np.min(feature_values)),
            'max': float(np.max(feature_values)),
            'percentile_25': float(np.percentile(feature_values, 25)),
            'percentile_75': float(np.percentile(feature_values, 75)),
            'iqr': float(np.percentile(feature_values, 75) - np.percentile(feature_values, 25)),
            'range': float(np.max(feature_values) - np.min(feature_values)),
            'cv': float(np.std(feature_values) / np.mean(feature_values)) if np.mean(feature_values) != 0 else 0
        }

    # Save baseline profile
    if output_path is None:
        output_path = f"baseline_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_path, 'w') as f:
        json.dump(baseline_profile, f, indent=2)

    print(f"✓ Baseline profile saved to: {output_path}\n")

    # Print summary statistics
    print(f"{'='*70}")
    print(f"Baseline Summary (Selected Features)")
    print(f"{'='*70}\n")

    # Show some key features
    key_features = [
        'audio_pitch_mean', 'audio_pitch_std', 'audio_pause_rate',
        'audio_energy_mean', 'video_blink_rate', 'video_head_movement_mean'
    ]

    for feat in key_features:
        if feat in baseline_profile['baseline_statistics']:
            stats = baseline_profile['baseline_statistics'][feat]
            print(f"{feat}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")

    print(f"{'='*70}\n")

    return baseline_profile


def load_baseline(baseline_path):
    """
    Load a previously created baseline profile.

    Args:
        baseline_path: Path to baseline JSON file

    Returns:
        baseline_profile: Dictionary containing baseline statistics
    """
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        print(f"✓ Loaded baseline for subject: {baseline['subject_id']}")
        print(f"  Created: {baseline['creation_date']}")
        print(f"  Based on {baseline['n_samples']} samples\n")
        return baseline
    except Exception as e:
        print(f"ERROR loading baseline: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create honest baseline profile for deception detection')

    parser.add_argument('--subject_id', '-s', type=str, required=True,
                        help='Unique identifier for the subject')
    parser.add_argument('--baseline_dir', '-d', type=str, required=True,
                        help='Directory containing honest/truthful video samples (.mp4)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for baseline profile (default: auto-generated)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.baseline_dir):
        print(f"ERROR: Directory not found: {args.baseline_dir}")
        sys.exit(1)

    # Create baseline
    baseline = create_baseline(
        subject_id=args.subject_id,
        baseline_dir=args.baseline_dir,
        output_path=args.output
    )

    if baseline:
        print("✓ Baseline creation completed successfully!")
    else:
        print("✗ Baseline creation failed.")
        sys.exit(1)
