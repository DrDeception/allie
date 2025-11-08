#!/usr/bin/env python3
"""
Example Workflow for Allie Deception Detection

This script demonstrates the complete workflow for baseline-deviation
deception detection research.

Prerequisites:
- Video samples in ./example_data/subject_001/honest/ (3-5 honest videos)
- Video samples in ./example_data/subject_001/test/ (test videos to analyze)

The script will:
1. Create an honest baseline from truthful videos
2. Analyze test videos for deviations
3. Generate reports
"""

import os
import sys

def create_example_structure():
    """Create example directory structure"""
    print("Setting up example directory structure...\n")

    dirs = [
        './example_data/subject_001/honest',
        './example_data/subject_001/test',
        './example_results'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")

    print("\n" + "="*70)
    print("SETUP INSTRUCTIONS")
    print("="*70)
    print("\nPlease add your video files:")
    print("1. Honest/truthful videos → ./example_data/subject_001/honest/")
    print("   - At least 3-5 videos of subject being truthful")
    print("   - 30-120 seconds each")
    print("   - .mp4 format")
    print("\n2. Test videos → ./example_data/subject_001/test/")
    print("   - Videos to analyze for deception")
    print("   - .mp4 format")
    print("\nThen run this script again to proceed with analysis.\n")
    print("="*70 + "\n")

def run_example_workflow():
    """Run the complete deception detection workflow"""

    # Check if data exists
    honest_dir = './example_data/subject_001/honest'
    test_dir = './example_data/subject_001/test'

    if not os.path.exists(honest_dir):
        create_example_structure()
        return

    honest_videos = [f for f in os.listdir(honest_dir) if f.endswith('.mp4')]
    test_videos = [f for f in os.listdir(test_dir) if f.endswith('.mp4')] if os.path.exists(test_dir) else []

    if len(honest_videos) < 3:
        print("\n" + "="*70)
        print("INSUFFICIENT DATA")
        print("="*70)
        print(f"\nFound only {len(honest_videos)} honest video(s).")
        print("Please add at least 3-5 honest/truthful videos to:")
        print(f"  {honest_dir}/\n")
        create_example_structure()
        return

    print("\n" + "="*70)
    print("ALLIE DECEPTION DETECTION - EXAMPLE WORKFLOW")
    print("="*70 + "\n")

    print(f"Found {len(honest_videos)} honest videos for baseline")
    print(f"Found {len(test_videos)} test videos to analyze\n")

    # Step 1: Create baseline
    print("="*70)
    print("STEP 1: Creating Honest Baseline")
    print("="*70 + "\n")

    from baseline import create_baseline

    baseline_path = './example_results/baseline_subject_001.json'

    try:
        baseline = create_baseline(
            subject_id='subject_001',
            baseline_dir=honest_dir,
            output_path=baseline_path
        )

        if not baseline:
            print("✗ Failed to create baseline")
            return

        print(f"\n✓ Baseline created successfully: {baseline_path}\n")

    except Exception as e:
        print(f"✗ Error creating baseline: {e}")
        return

    # Step 2: Analyze test videos (if any)
    if len(test_videos) == 0:
        print("="*70)
        print("No test videos found. Baseline creation complete!")
        print("="*70)
        print("\nTo analyze test videos:")
        print(f"1. Add .mp4 videos to: {test_dir}/")
        print("2. Run this script again")
        print("\nOr use the CLI directly:")
        print(f"   python3 deception_detect.py test -b {baseline_path} -v test_video.mp4")
        print("\n" + "="*70 + "\n")
        return

    print("\n" + "="*70)
    print("STEP 2: Analyzing Test Videos")
    print("="*70 + "\n")

    from deviation import analyze_deviation

    results = []

    for i, video_file in enumerate(test_videos, 1):
        video_path = os.path.join(test_dir, video_file)
        output_path = f'./example_results/analysis_{video_file.replace(".mp4", "")}.json'

        print(f"\n[{i}/{len(test_videos)}] Analyzing: {video_file}")
        print("-" * 70)

        try:
            report = analyze_deviation(
                baseline_path=baseline_path,
                test_video_path=video_path,
                output_path=output_path,
                verbose=True
            )

            if report:
                results.append({
                    'video': video_file,
                    'likelihood': report['deception_assessment']['likelihood'],
                    'deviation_score': report['overall_metrics']['deviation_score']
                })
                print(f"✓ Analysis complete")
            else:
                print(f"✗ Analysis failed")

        except Exception as e:
            print(f"✗ Error: {e}")

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE - SUMMARY")
    print("="*70 + "\n")

    print(f"Baseline: {baseline_path}")
    print(f"Analyzed: {len(results)}/{len(test_videos)} videos\n")

    if results:
        print("Results:")
        for result in results:
            print(f"  {result['video']}: {result['likelihood']} (score: {result['deviation_score']:.2f})")

    print(f"\nAll reports saved to: ./example_results/")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        run_example_workflow()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
