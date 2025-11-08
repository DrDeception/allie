#!/usr/bin/env python3
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

Allie - Deception Detection CLI

Main command-line interface for audiovisual deception detection using
baseline-deviation methodology for psychological research.

Workflow:
1. Create Baseline: Establish honest behavioral baseline from truthful videos
2. Test Sample: Analyze test video against baseline for deviations
3. Batch Analysis: Process multiple test videos against a baseline

Features:
- Individual baseline profiling
- Statistical deviation analysis
- Z-score and distance metrics
- Deception likelihood assessment
- Feature-level analysis

Usage:
  Create baseline:  python3 deception_detect.py create-baseline --subject [ID] --dir [baseline_videos/]
  Test sample:      python3 deception_detect.py test --baseline [profile.json] --video [test.mp4]
  Batch test:       python3 deception_detect.py batch --baseline [profile.json] --dir [test_videos/]
  Extract features: python3 deception_detect.py extract --video [video.mp4]
'''

import os, sys, argparse, json
from datetime import datetime
import baseline
import deviation

def print_header():
    """Print Allie deception detection header"""
    print("\n" + "="*70)
    print(" " * 15 + "ALLIE - DECEPTION DETECTION SYSTEM")
    print(" " * 10 + "Audiovisual Baseline-Deviation Analysis")
    print("="*70 + "\n")

def create_baseline_workflow(args):
    """Workflow for creating an honest baseline"""
    print_header()
    print("MODE: Create Honest Baseline\n")

    baseline_profile = baseline.create_baseline(
        subject_id=args.subject,
        baseline_dir=args.dir,
        output_path=args.output
    )

    if baseline_profile:
        print("\n" + "="*70)
        print("✓ SUCCESS: Baseline created successfully!")
        print("="*70 + "\n")
        return True
    else:
        print("\n" + "="*70)
        print("✗ FAILED: Baseline creation failed")
        print("="*70 + "\n")
        return False

def test_sample_workflow(args):
    """Workflow for testing a single video against baseline"""
    print_header()
    print("MODE: Test Sample Against Baseline\n")

    report = deviation.analyze_deviation(
        baseline_path=args.baseline,
        test_video_path=args.video,
        output_path=args.output,
        verbose=not args.quiet
    )

    if report:
        print("\n" + "="*70)
        print("✓ SUCCESS: Deviation analysis completed!")
        print("="*70 + "\n")
        return True
    else:
        print("\n" + "="*70)
        print("✗ FAILED: Deviation analysis failed")
        print("="*70 + "\n")
        return False

def batch_test_workflow(args):
    """Workflow for batch testing multiple videos against baseline"""
    print_header()
    print("MODE: Batch Test Against Baseline\n")

    # Find all video files
    video_files = [f for f in os.listdir(args.dir) if f.endswith('.mp4')]

    if len(video_files) == 0:
        print(f"ERROR: No video files found in {args.dir}")
        return False

    print(f"Found {len(video_files)} videos to analyze\n")

    results = []
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(args.dir, video_file)
        print(f"\n{'='*70}")
        print(f"[{i}/{len(video_files)}] Analyzing: {video_file}")
        print(f"{'='*70}\n")

        output_name = f"deviation_{video_file.replace('.mp4', '')}.json"
        output_path = os.path.join(args.output_dir if args.output_dir else '.', output_name)

        try:
            report = deviation.analyze_deviation(
                baseline_path=args.baseline,
                test_video_path=video_path,
                output_path=output_path,
                verbose=False
            )

            if report:
                results.append({
                    'video': video_file,
                    'status': 'success',
                    'likelihood': report['deception_assessment']['likelihood'],
                    'deviation_score': report['overall_metrics']['deviation_score'],
                    'report_path': output_path
                })
                print(f"✓ Analysis complete: {report['deception_assessment']['likelihood']} likelihood")
            else:
                results.append({
                    'video': video_file,
                    'status': 'failed',
                    'error': 'Analysis returned None'
                })
                print(f"✗ Analysis failed")

        except Exception as e:
            results.append({
                'video': video_file,
                'status': 'error',
                'error': str(e)
            })
            print(f"✗ Error: {e}")

    # Save batch summary
    summary = {
        'batch_analysis_date': datetime.now().isoformat(),
        'baseline_used': args.baseline,
        'total_videos': len(video_files),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] != 'success'),
        'results': results
    }

    summary_path = f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Batch Analysis Summary")
    print(f"{'='*70}\n")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"\nSummary saved to: {summary_path}\n")

    return True

def extract_features_workflow(args):
    """Workflow for extracting features from a video"""
    print_header()
    print("MODE: Extract Deception Features\n")

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'features'))
    from deception_features import deception_featurize

    print(f"Extracting features from: {args.video}\n")

    try:
        features, labels = deception_featurize(args.video)

        output = {
            'video': os.path.basename(args.video),
            'extraction_date': datetime.now().isoformat(),
            'feature_count': len(features),
            'features': {label: value for label, value in zip(labels, features)}
        }

        output_path = args.output if args.output else args.video.replace('.mp4', '_features.json')

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Extracted {len(features)} features")
        print(f"✓ Saved to: {output_path}\n")

        # Show sample features
        print(f"Sample Features (first 10):")
        for label, value in list(output['features'].items())[:10]:
            print(f"  {label}: {value:.4f}")

        print()
        return True

    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Allie Deception Detection - Audiovisual baseline-deviation analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create baseline from honest videos
  python3 deception_detect.py create-baseline --subject S001 --dir ./honest_videos/

  # Test a video against baseline
  python3 deception_detect.py test --baseline baseline_S001.json --video test_video.mp4

  # Batch test multiple videos
  python3 deception_detect.py batch --baseline baseline_S001.json --dir ./test_videos/

  # Extract features only
  python3 deception_detect.py extract --video sample.mp4
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Create baseline command
    baseline_parser = subparsers.add_parser('create-baseline', help='Create honest baseline profile')
    baseline_parser.add_argument('--subject', '-s', required=True, help='Subject ID')
    baseline_parser.add_argument('--dir', '-d', required=True, help='Directory with honest videos')
    baseline_parser.add_argument('--output', '-o', help='Output baseline file (optional)')

    # Test sample command
    test_parser = subparsers.add_parser('test', help='Test video against baseline')
    test_parser.add_argument('--baseline', '-b', required=True, help='Baseline profile JSON')
    test_parser.add_argument('--video', '-v', required=True, help='Test video file')
    test_parser.add_argument('--output', '-o', help='Output report file (optional)')
    test_parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')

    # Batch test command
    batch_parser = subparsers.add_parser('batch', help='Batch test multiple videos')
    batch_parser.add_argument('--baseline', '-b', required=True, help='Baseline profile JSON')
    batch_parser.add_argument('--dir', '-d', required=True, help='Directory with test videos')
    batch_parser.add_argument('--output_dir', help='Output directory for reports (optional)')

    # Extract features command
    extract_parser = subparsers.add_parser('extract', help='Extract deception features')
    extract_parser.add_argument('--video', '-v', required=True, help='Video file')
    extract_parser.add_argument('--output', '-o', help='Output JSON file (optional)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Execute appropriate workflow
    if args.command == 'create-baseline':
        success = create_baseline_workflow(args)
    elif args.command == 'test':
        success = test_sample_workflow(args)
    elif args.command == 'batch':
        success = batch_test_workflow(args)
    elif args.command == 'extract':
        success = extract_features_workflow(args)
    else:
        parser.print_help()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
