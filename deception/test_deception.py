#!/usr/bin/env python3
"""
Unit tests for Allie Deception Detection Module

Tests basic functionality of the deception detection system.
Note: Requires sample video files for full testing.
"""

import os
import sys
import unittest
import json
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDeceptionFeatures(unittest.TestCase):
    """Test deception feature extraction"""

    def test_feature_labels_consistent(self):
        """Test that feature extraction returns consistent labels"""
        # This test verifies the structure without needing actual video
        from features.deception_features import deception_featurize

        # Would need actual video file to run full test
        # For now, test imports work
        self.assertTrue(callable(deception_featurize))

    def test_audio_feature_extraction(self):
        """Test audio feature extraction functions exist"""
        from features.deception_features import extract_audio_deception_features
        self.assertTrue(callable(extract_audio_deception_features))

    def test_video_feature_extraction(self):
        """Test video feature extraction functions exist"""
        from features.deception_features import extract_video_deception_features
        self.assertTrue(callable(extract_video_deception_features))


class TestBaseline(unittest.TestCase):
    """Test baseline creation and loading"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.test_dir, 'test_baseline.json')

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_baseline_module_imports(self):
        """Test baseline module imports correctly"""
        from deception.baseline import create_baseline, load_baseline
        self.assertTrue(callable(create_baseline))
        self.assertTrue(callable(load_baseline))

    def test_baseline_structure(self):
        """Test baseline profile has required structure"""
        # Create a mock baseline
        baseline = {
            'subject_id': 'TEST',
            'creation_date': '2025-11-08',
            'n_samples': 3,
            'sample_files': ['test1.mp4', 'test2.mp4', 'test3.mp4'],
            'n_features': 41,
            'feature_labels': ['audio_pitch_mean', 'audio_pitch_std'],
            'baseline_statistics': {
                'audio_pitch_mean': {
                    'mean': 150.0,
                    'std': 10.0,
                    'median': 149.0,
                    'min': 130.0,
                    'max': 170.0
                }
            }
        }

        # Save and load
        with open(self.baseline_path, 'w') as f:
            json.dump(baseline, f)

        from deception.baseline import load_baseline
        loaded = load_baseline(self.baseline_path)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded['subject_id'], 'TEST')
        self.assertEqual(loaded['n_samples'], 3)


class TestDeviation(unittest.TestCase):
    """Test deviation analysis"""

    def test_deviation_module_imports(self):
        """Test deviation module imports correctly"""
        from deception.deviation import analyze_deviation, calculate_z_score
        self.assertTrue(callable(analyze_deviation))
        self.assertTrue(callable(calculate_z_score))

    def test_z_score_calculation(self):
        """Test z-score calculation"""
        from deception.deviation import calculate_z_score

        # Test normal z-score
        z = calculate_z_score(value=110, mean=100, std=10)
        self.assertAlmostEqual(z, 1.0, places=2)

        # Test negative z-score
        z = calculate_z_score(value=90, mean=100, std=10)
        self.assertAlmostEqual(z, -1.0, places=2)

        # Test zero std (edge case)
        z = calculate_z_score(value=100, mean=100, std=0)
        self.assertEqual(z, 0)

    def test_deviation_score_calculation(self):
        """Test overall deviation score calculation"""
        from deception.deviation import calculate_deviation_score
        import numpy as np

        z_scores = np.array([1.0, -1.0, 2.0, -2.0])
        score = calculate_deviation_score(z_scores)

        # RMS of [1, 1, 2, 2] should be sqrt(10/4) = 1.58
        self.assertAlmostEqual(score, 1.58, places=1)

    def test_deception_assessment(self):
        """Test deception likelihood assessment"""
        from deception.deviation import assess_deception_likelihood

        # Test Very Low
        assessment = assess_deception_likelihood(
            deviation_score=1.0,
            significant_deviations_count=2,
            total_features=41
        )
        self.assertEqual(assessment['likelihood'], 'Very Low')

        # Test High
        assessment = assess_deception_likelihood(
            deviation_score=3.0,
            significant_deviations_count=15,
            total_features=41
        )
        self.assertEqual(assessment['likelihood'], 'High')


class TestCLI(unittest.TestCase):
    """Test CLI interface"""

    def test_cli_imports(self):
        """Test CLI module imports"""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        import deception_detect
        self.assertTrue(hasattr(deception_detect, 'main'))


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_complete_workflow_structure(self):
        """Test that all components can be imported together"""
        try:
            from features.deception_features import deception_featurize
            from deception.baseline import create_baseline
            from deception.deviation import analyze_deviation

            # If we get here, all imports work
            self.assertTrue(True)

        except ImportError as e:
            self.fail(f"Import failed: {e}")


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ALLIE DECEPTION DETECTION - UNIT TESTS")
    print("="*70 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDeceptionFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestBaseline))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviation))
    suite.addTests(loader.loadTestsFromTestCase(TestCLI))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
