# Allie Deception Detection Module

## Audiovisual Baseline-Deviation Analysis for Psychological Research

This module extends Allie with sophisticated deception detection capabilities using baseline-deviation methodology. It enables researchers to establish an individual's honest behavioral baseline and detect deviations that may indicate deception.

---

## Overview

The deception detection system uses a scientifically-grounded approach:

1. **Baseline Creation**: Establish each individual's unique honest behavioral profile from truthful video samples
2. **Feature Extraction**: Extract 50+ audiovisual features known to correlate with deception
3. **Deviation Analysis**: Calculate statistical deviations from baseline using z-scores and distance metrics
4. **Assessment**: Generate deception likelihood scores with confidence levels

### Key Advantages

- **Individual-specific**: Accounts for personal behavioral differences
- **Multimodal**: Combines audio (prosody, voice quality) and video (facial expressions, movements)
- **Statistical**: Uses robust metrics (z-scores, percentiles, IQR)
- **Research-grade**: Based on peer-reviewed deception detection literature
- **Interpretable**: Provides feature-level explanations

---

## Features Extracted

### Audio Features (26 features)
Based on vocal stress and deception literature:

- **Pitch**: Mean, std, range, variance, coefficient of variation
- **Speech Timing**: Tempo, pause count, pause rate
- **Energy**: Mean, std, variance (loudness control)
- **Voice Quality**: Spectral centroid, spectral rolloff, zero-crossing rate, jitter approximation
- **MFCCs**: 13 coefficients (mean and std) - vocal tract fingerprint

### Video Features (15 features)
Based on nonverbal deception indicators:

- **Blink Behavior**: Blink count, blink rate
- **Head Movement**: Movement mean, std, position variance (x, y)
- **Face Size**: Mean, std, variance (approach/avoidance)
- **Motion**: Overall motion mean, std, variance (fidgeting)
- **Gaze**: Stability, average eyes detected
- **Engagement**: Face presence ratio

---

## Installation & Setup

### Prerequisites

The deception detection module requires additional dependencies:

```bash
cd /path/to/allie
pip install librosa opencv-python moviepy scipy numpy
```

Ensure you have completed the main Allie setup first:
```bash
python3 setup.py
```

---

## Usage

### Quick Start

The deception detection workflow has 4 main modes:

#### 1. Create Baseline (from honest videos)

```bash
cd deception
python3 deception_detect.py create-baseline \
    --subject S001 \
    --dir /path/to/honest_videos/
```

**Best Practices**:
- Use 3-5+ honest video samples (more is better)
- Videos should show natural, truthful responses
- Keep conditions consistent (lighting, distance, questions)
- Each video should be 30-120 seconds

**Output**: `baseline_S001_YYYYMMDD_HHMMSS.json`

---

#### 2. Test Single Video

```bash
python3 deception_detect.py test \
    --baseline baseline_S001.json \
    --video /path/to/test_video.mp4
```

**Output**: Deviation analysis report with:
- Overall deviation score
- Deception likelihood (Very Low → Very High)
- Confidence level
- Top 10 deviant features
- Full feature-level analysis

---

#### 3. Batch Test Multiple Videos

```bash
python3 deception_detect.py batch \
    --baseline baseline_S001.json \
    --dir /path/to/test_videos/ \
    --output_dir ./results/
```

**Output**:
- Individual reports for each video
- Batch summary JSON with all results

---

#### 4. Extract Features Only

```bash
python3 deception_detect.py extract \
    --video /path/to/video.mp4
```

**Output**: JSON file with all extracted features

---

## Understanding the Results

### Deviation Score

The overall deviation score is the RMS (root mean square) of absolute z-scores across all features:

- **< 1.5**: Normal variation (Very Low likelihood)
- **1.5-2.5**: Minor deviations (Low-Medium likelihood)
- **2.5-3.5**: Significant deviations (High likelihood)
- **> 3.5**: Extreme deviations (Very High likelihood)

### Z-Scores

Each feature gets a z-score: `z = (test_value - baseline_mean) / baseline_std`

- **|z| < 2**: Within normal range (95% confidence)
- **|z| > 2**: Statistically significant deviation
- **|z| > 3**: Highly significant deviation (99.7% confidence)

### Deception Assessment

The system provides:

1. **Likelihood**: Very Low, Low, Medium, High, Very High
2. **Confidence**: Based on consistency across features
3. **Interpretation**: Plain language explanation
4. **Top Deviant Features**: Which behaviors changed most

### Example Output

```
Overall Deviation Score: 3.2
Significant Deviations: 12/41 (29.3%)

Deception Assessment
────────────────────────────────────────
Likelihood: High
Confidence: Medium
Interpretation: Significant deviations from baseline across
                multiple features. Possible deception indicators.

Top 10 Most Deviant Features
────────────────────────────────────────
1. audio_pitch_variance
   Z-score: 4.23 (higher than baseline)

2. video_blink_rate
   Z-score: -3.87 (lower than baseline)

3. audio_pause_rate
   Z-score: 3.45 (higher than baseline)
...
```

---

## Research Methodology

### Baseline-Deviation Approach

This system uses the **baseline-deviation** methodology common in psychophysiology and deception research:

1. **Individual Baseline**: Everyone has unique behavioral patterns. Direct comparison between individuals is unreliable.

2. **Within-Subject Design**: By comparing an individual against their own baseline, we control for personality, communication style, and other confounds.

3. **Statistical Deviation**: We use standard deviations (z-scores) to determine if a behavior is "unusual" for that specific individual.

4. **Multimodal Integration**: Deception affects multiple channels simultaneously. Combining audio and video increases detection accuracy.

### Scientific Basis

Features are based on peer-reviewed research in:

- **Voice Stress Analysis**: DePaulo et al. (2003), Vrij (2008)
- **Nonverbal Behavior**: Ekman & Friesen (1969), Zuckerman et al. (1981)
- **Psychophysiology**: Levine & Rill (2017)
- **Computer Vision**: Pérez-Rosas et al. (2015)

---

## Research Applications

This module supports various research scenarios:

### 1. Deception Detection Studies
- Compare truthful vs. deceptive statements
- Analyze behavioral changes under different conditions
- Study individual differences in deception behavior

### 2. Stress & Cognitive Load
- Detect stress-related behavioral changes
- Measure cognitive load via multimodal cues
- Study anxiety indicators

### 3. Interview & Interrogation Research
- Analyze interview subject behavior
- Study interviewer effects
- Compare questioning techniques

### 4. Clinical Psychology
- Assess emotional states
- Study anxiety disorders
- Analyze therapeutic interactions

---

## File Formats

### Baseline Profile JSON

```json
{
  "subject_id": "S001",
  "creation_date": "2025-11-08T10:30:00",
  "n_samples": 5,
  "sample_files": ["honest1.mp4", "honest2.mp4", ...],
  "n_features": 41,
  "feature_labels": ["audio_pitch_mean", ...],
  "baseline_statistics": {
    "audio_pitch_mean": {
      "mean": 156.3,
      "std": 12.4,
      "median": 155.8,
      "min": 142.1,
      "max": 172.5,
      "percentile_25": 148.2,
      "percentile_75": 164.1,
      "iqr": 15.9,
      "range": 30.4,
      "cv": 0.079
    },
    ...
  }
}
```

### Deviation Analysis Report JSON

```json
{
  "subject_id": "S001",
  "test_video": "test_video.mp4",
  "analysis_date": "2025-11-08T14:15:00",
  "overall_metrics": {
    "deviation_score": 3.24,
    "mean_absolute_z_score": 1.87,
    "max_absolute_z_score": 4.23,
    "significant_deviations": 12,
    "total_features": 41,
    "significant_deviation_percentage": 29.3
  },
  "deception_assessment": {
    "likelihood": "High",
    "confidence": "Medium",
    "interpretation": "...",
    "deviation_score": 3.24,
    "significant_deviations_ratio": 0.293
  },
  "feature_deviations": {
    "audio_pitch_mean": {
      "test_value": 182.4,
      "baseline_mean": 156.3,
      "baseline_std": 12.4,
      "baseline_range": [142.1, 172.5],
      "z_score": 2.11,
      "is_significant": true,
      "deviation_direction": "higher"
    },
    ...
  },
  "top_deviant_features": [...]
}
```

---

## Best Practices

### For Baseline Creation

1. **Quantity**: Use 3-5+ videos (more provides better statistical power)
2. **Quality**: Ensure clear audio and video (face visible, minimal noise)
3. **Consistency**: Keep recording conditions similar across samples
4. **Truthfulness**: Use questions/scenarios where subject is being honest
5. **Duration**: 30-120 seconds per video is ideal
6. **Variety**: Include different topics to capture behavioral range

### For Testing

1. **Same Conditions**: Match baseline recording setup (distance, lighting, audio quality)
2. **Same Subject**: Only compare against their own baseline
3. **Time Proximity**: Shorter gap between baseline and test is better
4. **Control Variables**: Account for time of day, fatigue, mood
5. **Multiple Samples**: Test multiple samples when possible
6. **Context**: Document the question/scenario for each test

### Interpretation Caution

⚠️ **Important Research Ethics Notes**:

- This is a **research tool**, not a lie detector
- Deviations indicate behavioral changes, not necessarily deception
- Stress, anxiety, cognitive load can all cause deviations
- Always use in conjunction with other evidence
- Follow ethical guidelines for deception research
- Obtain proper informed consent
- Do not use for high-stakes individual decisions without validation

---

## Advanced Usage

### Custom Thresholds

You can modify thresholds in `deviation.py`:

```python
def assess_deception_likelihood(deviation_score, significant_deviations_count, total_features):
    # Modify these thresholds based on your research needs
    if deviation_score < 1.5:  # Change threshold
        likelihood = "Very Low"
    ...
```

### Feature Selection

To use only specific features, modify the feature extraction in `deception_features.py`:

```python
# Example: Use only audio features
features = audio_features  # Instead of audio_features + video_features
labels = audio_labels
```

### Batch Processing Script

For large-scale studies:

```python
import os
from baseline import create_baseline
from deviation import analyze_deviation

subjects = ['S001', 'S002', 'S003']

for subject in subjects:
    # Create baseline
    baseline_path = create_baseline(
        subject_id=subject,
        baseline_dir=f'./data/{subject}/honest/',
        output_path=f'./baselines/baseline_{subject}.json'
    )

    # Test all deceptive samples
    test_dir = f'./data/{subject}/deceptive/'
    for video in os.listdir(test_dir):
        analyze_deviation(
            baseline_path=baseline_path,
            test_video_path=os.path.join(test_dir, video),
            output_path=f'./results/{subject}_{video}_report.json'
        )
```

---

## Module Components

### 1. `deception_features.py`
- Audiovisual feature extraction
- 41 deception-relevant features
- Standalone feature extractor

### 2. `baseline.py`
- Baseline profile creation
- Statistical analysis (mean, std, percentiles)
- Multi-sample aggregation

### 3. `deviation.py`
- Deviation analysis
- Z-score calculation
- Deception likelihood assessment

### 4. `deception_detect.py`
- Main CLI interface
- Workflow management
- Batch processing

---

## Troubleshooting

### Common Issues

**No features extracted from video**:
- Check video has both audio and video tracks
- Ensure video is .mp4 format
- Verify face is visible in frame
- Check audio is clear and not silent

**Low baseline quality**:
- Use more samples (5+ recommended)
- Ensure consistent recording conditions
- Verify all baseline videos are truly "honest" responses

**Inconsistent results**:
- Check for environmental differences (background noise, lighting)
- Verify subject is in similar emotional state
- Ensure sufficient baseline sample size

**High computation time**:
- Use shorter videos (30-120 seconds optimal)
- Process videos at lower resolution if needed
- Use batch mode for multiple videos

---

## Citation

If you use this module in your research, please cite:

```bibtex
@software{allie_deception_detection,
  title={Allie Deception Detection Module},
  author={Allie Development Team},
  year={2025},
  url={https://github.com/jim-schwoebel/allie}
}
```

And cite relevant deception detection literature used in feature design.

---

## Contributing

To contribute to this module:

1. Follow Allie's contribution guidelines
2. Ensure features are grounded in peer-reviewed research
3. Maintain compatibility with existing Allie structure
4. Add appropriate tests and documentation

---

## License

Same as main Allie framework (Apache 2.0)

---

## Contact & Support

- Report issues: [Allie GitHub Issues](https://github.com/jim-schwoebel/allie/issues)
- Main Allie documentation: [Allie Wiki](https://github.com/jim-schwoebel/allie/wiki)

---

## Acknowledgments

This module builds on decades of deception detection research in psychology, communication studies, and computer science. Features are based on validated indicators from multiple research domains.

**Research Areas**:
- Psychophysiology
- Nonverbal communication
- Voice stress analysis
- Computer vision
- Affective computing
