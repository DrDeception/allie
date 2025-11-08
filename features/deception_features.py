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

Deception Detection - Audiovisual Feature Extractor

This module extracts features specifically designed for deception detection research.
It combines audio prosodic features, voice quality measures, and video behavioral
indicators to create a comprehensive feature set for baseline-deviation analysis.

Key Features Extracted:
- Audio: pitch variation, jitter, shimmer, pause patterns, speech rate, F0, intensity
- Video: facial micro-expressions, blink rate, head movement, gaze patterns

Usage: python3 deception_features.py [videofile_with_audio]
'''

import os, sys, json, numpy as np
import librosa
import cv2
from moviepy.editor import VideoFileClip
from scipy.stats import variation
import warnings
warnings.filterwarnings('ignore')

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        clip = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '_temp_audio.wav')
        clip.audio.write_audiofile(audio_path, logger=None)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def extract_audio_deception_features(audio_path):
    """
    Extract audio features relevant to deception detection.
    Based on literature: pitch variation, jitter, shimmer, pause patterns, speech rate
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        features = {}
        labels = []

        # 1. Pitch (F0) features - deception often shows pitch variation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            features['pitch_variance'] = np.var(pitch_values)
            features['pitch_cv'] = variation(pitch_values) if len(pitch_values) > 1 else 0
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            features['pitch_variance'] = 0
            features['pitch_cv'] = 0

        # 2. Speech rate and pause detection - stress affects timing
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        features['tempo'] = tempo

        # Detect pauses (low energy regions)
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.3
        pauses = rms < threshold
        pause_count = np.sum(np.diff(pauses.astype(int)) == 1)
        features['pause_count'] = pause_count
        features['pause_rate'] = pause_count / (len(y) / sr)  # pauses per second

        # 3. Energy and intensity variation - stress affects loudness control
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_variance'] = np.var(rms)

        # 4. Spectral features - voice quality indicators
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

        # 5. Zero crossing rate - relates to voice quality/stress
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # 6. MFCCs - overall voice quality fingerprint
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])

        # 7. Jitter approximation (pitch period irregularity)
        if len(pitch_values) > 1:
            pitch_diffs = np.diff(pitch_values)
            features['jitter_approx'] = np.mean(np.abs(pitch_diffs))
        else:
            features['jitter_approx'] = 0

        # Create labels list
        labels = list(features.keys())

        return list(features.values()), labels

    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return [], []

def extract_video_deception_features(video_path):
    """
    Extract video features relevant to deception detection.
    Based on literature: facial movements, blink rate, head movement, gaze patterns
    """
    try:
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Load face and eye cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        features = {}

        # Track movement and behavioral patterns
        blink_count = 0
        face_positions = []
        face_sizes = []
        eye_detections = []
        frame_motion = []
        prev_frame = None
        eyes_closed_frames = 0
        total_frames_with_face = 0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                total_frames_with_face += 1
                # Take the largest face (presumably the subject)
                face = max(faces, key=lambda x: x[2] * x[3])
                (x, y, w, h) = face

                # Track face position (head movement)
                face_center_x = x + w/2
                face_center_y = y + h/2
                face_positions.append((face_center_x, face_center_y))
                face_sizes.append(w * h)

                # Detect eyes within face region
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

                # Blink detection (approximation: when eyes not detected in face)
                if len(eyes) < 2:
                    eyes_closed_frames += 1
                    if frame_idx > 0 and len(eye_detections) > 0 and eye_detections[-1] >= 2:
                        blink_count += 1

                eye_detections.append(len(eyes))

            # Frame-to-frame motion estimation
            if prev_frame is not None and len(faces) > 0:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                frame_motion.append(motion_score)

            prev_frame = gray.copy()
            frame_idx += 1

            # Sample every Nth frame for efficiency
            if frame_idx % 5 != 0:
                continue

        cap.release()

        # Calculate video behavioral features

        # 1. Blink rate - deception can increase blink frequency
        features['blink_count'] = blink_count
        features['blink_rate'] = blink_count / duration if duration > 0 else 0

        # 2. Head movement patterns - increased movement under stress
        if len(face_positions) > 1:
            positions_array = np.array(face_positions)
            position_changes = np.diff(positions_array, axis=0)
            features['head_movement_mean'] = np.mean(np.linalg.norm(position_changes, axis=1))
            features['head_movement_std'] = np.std(np.linalg.norm(position_changes, axis=1))
            features['head_position_variance_x'] = np.var(positions_array[:, 0])
            features['head_position_variance_y'] = np.var(positions_array[:, 1])
        else:
            features['head_movement_mean'] = 0
            features['head_movement_std'] = 0
            features['head_position_variance_x'] = 0
            features['head_position_variance_y'] = 0

        # 3. Face size variation - approach/avoidance behavior
        if len(face_sizes) > 1:
            features['face_size_mean'] = np.mean(face_sizes)
            features['face_size_std'] = np.std(face_sizes)
            features['face_size_variance'] = np.var(face_sizes)
        else:
            features['face_size_mean'] = 0
            features['face_size_std'] = 0
            features['face_size_variance'] = 0

        # 4. Overall motion/fidgeting
        if len(frame_motion) > 0:
            features['motion_mean'] = np.mean(frame_motion)
            features['motion_std'] = np.std(frame_motion)
            features['motion_variance'] = np.var(frame_motion)
        else:
            features['motion_mean'] = 0
            features['motion_std'] = 0
            features['motion_variance'] = 0

        # 5. Gaze stability (eye detection consistency)
        if len(eye_detections) > 0:
            features['gaze_stability'] = np.std(eye_detections)
            features['avg_eyes_detected'] = np.mean(eye_detections)
        else:
            features['gaze_stability'] = 0
            features['avg_eyes_detected'] = 0

        # 6. Face presence (engagement level)
        features['face_presence_ratio'] = total_frames_with_face / frame_count if frame_count > 0 else 0

        labels = list(features.keys())

        return list(features.values()), labels

    except Exception as e:
        print(f"Error extracting video features: {e}")
        return [], []

def deception_featurize(videofile):
    """
    Main function to extract combined audiovisual deception features.
    Returns features and labels in Allie's standard format.
    """
    print(f"Extracting deception features from: {videofile}")

    # Extract audio
    audio_path = extract_audio_from_video(videofile)

    # Get audio features
    audio_features, audio_labels = extract_audio_deception_features(audio_path)

    # Get video features
    video_features, video_labels = extract_video_deception_features(videofile)

    # Combine features
    combined_features = audio_features + video_features
    combined_labels = ['audio_' + label for label in audio_labels] + ['video_' + label for label in video_labels]

    # Clean up temporary audio file
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Extracted {len(combined_features)} deception-relevant features")

    return combined_features, combined_labels

if __name__ == '__main__':
    # Test the feature extractor
    if len(sys.argv) > 1:
        videofile = sys.argv[1]
        features, labels = deception_featurize(videofile)

        # Print results
        print("\nFeature Summary:")
        print(f"Total features: {len(features)}")
        print(f"\nSample features:")
        for i, (label, value) in enumerate(zip(labels[:10], features[:10])):
            print(f"  {label}: {value:.4f}")

        # Save to JSON (Allie format)
        output = {
            'features': features,
            'labels': labels,
            'feature_count': len(features)
        }

        output_path = videofile.replace('.mp4', '_deception_features.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nFeatures saved to: {output_path}")
    else:
        print("Usage: python3 deception_features.py [videofile.mp4]")
