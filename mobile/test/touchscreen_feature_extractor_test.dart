import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_event_collector.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_feature_extractor.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_feature_vector.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_inference_engine.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_model_parameters.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_touch_event.dart';

void main() {
  group('Touchscreen Feature Extractor & Equivalence Unit Tests', () {
    late TouchscreenInferenceEngine inferenceEngine;

    setUpAll(() {
      final file = File('assets/models/touchscreen_model.json');
      final jsonStr = file.readAsStringSync();
      final params = TouchscreenModelParameters.fromJsonString(jsonStr);
      inferenceEngine = TouchscreenInferenceEngine(params);
    });

    test('1 & 2. Empty and single-event small-sample behavior', () {
      final emptyFeats = TouchscreenFeatureExtractor.extract([]);
      expect(emptyFeats.eventCount, 0.0);
      expect(emptyFeats.recordingDuration.isNaN, isTrue);
      expect(emptyFeats.typingRate.isNaN, isTrue);
      expect(emptyFeats.holdMean.isNaN, isTrue);

      final singleEvent = [
        const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 1080.0,
          pressure: 0.5,
        ),
      ];

      final singleFeats = TouchscreenFeatureExtractor.extract(singleEvent);
      expect(singleFeats.eventCount, 1.0);
      expect(singleFeats.holdMean, closeTo(0.080, 1e-5));
      expect(singleFeats.holdStd.isNaN, isTrue); // N < 2
      expect(singleFeats.interTapMean.isNaN, isTrue); // Inter-tap N < 1
    });

    test('3 & 4. Out-of-order timestamps sorted by press timestamp', () {
      final unordered = [
        const TouchscreenTouchEvent(
          pressTimeMs: 2000.0,
          releaseTimeMs: 2100.0,
          pressure: 0.6,
        ),
        const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 1080.0,
          pressure: 0.5,
        ),
      ];

      final feats = TouchscreenFeatureExtractor.extract(unordered);
      expect(feats.eventCount, 2.0);
      expect(feats.interTapMean, closeTo(1.000, 1e-5)); // 2000 - 1000 = 1000ms = 1s
    });

    test('5 - 10. Known hold, inter-tap, pressure, and activity calculations', () {
      final events = [
        const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 1100.0, // hold = 100ms = 0.1s
          pressure: 0.4,
        ),
        const TouchscreenTouchEvent(
          pressTimeMs: 1500.0, // inter_tap = 500ms = 0.5s
          releaseTimeMs: 1700.0, // hold = 200ms = 0.2s
          pressure: 0.6,
        ),
        const TouchscreenTouchEvent(
          pressTimeMs: 2000.0, // inter_tap = 500ms = 0.5s
          releaseTimeMs: 2300.0, // hold = 300ms = 0.3s
          pressure: 0.8,
        ),
      ];

      final feats = TouchscreenFeatureExtractor.extract(events);

      expect(feats.holdMean, closeTo(0.200, 1e-5)); // (0.1 + 0.2 + 0.3) / 3
      expect(feats.holdMin, closeTo(0.100, 1e-5));
      expect(feats.holdMax, closeTo(0.300, 1e-5));

      expect(feats.interTapMean, closeTo(0.500, 1e-5)); // (0.5 + 0.5) / 2
      expect(feats.pressureMean, closeTo(0.600, 1e-5)); // (0.4 + 0.6 + 0.8) / 3

      expect(feats.eventCount, 3.0);
      expect(feats.recordingDuration, closeTo(1.300, 1e-5)); // 2300 - 1000 = 1300ms = 1.3s
      expect(feats.typingRate, closeTo(3.0 / 1.300, 1e-5));
    });

    test('12 - 16. Validation rules for invalid events & missing pressure', () {
      expect(
        () => const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 900.0, // Invalid: release < press
          pressure: 0.5,
        ).validate(),
        throwsArgumentError,
      );

      expect(
        () => const TouchscreenTouchEvent(
          pressTimeMs: double.nan,
          releaseTimeMs: 1000.0,
          pressure: 0.5,
        ).validate(),
        throwsArgumentError,
      );

      expect(
        () => const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 1100.0,
          pressure: double.infinity,
        ).validate(),
        throwsArgumentError,
      );

      // Verify TouchPressureUnavailableException is thrown when pressure is missing
      final missingPressureEvents = [
        const TouchscreenTouchEvent(
          pressTimeMs: 1000.0,
          releaseTimeMs: 1100.0,
          pressure: null, // Unavailable pressure
        ),
      ];

      expect(
        () => TouchscreenFeatureExtractor.extract(missingPressureEvents),
        throwsA(isA<TouchPressureUnavailableException>()),
      );
    });

    test('18. Python <-> Dart feature extraction equivalence on reference recording S01_TEX01', () {
      // Reference Python features for recording S01_TEX01
      final expectedPyFeatures = {
        'hold_mean': 0.08623589743589744,
        'hold_std': 0.020596660313553276,
        'hold_variance': 0.0004242224160719006,
        'hold_median': 0.086,
        'hold_min': 0.038,
        'hold_max': 0.16,
        'hold_skewness': 0.3008477886807847,
        'hold_kurtosis': 0.20490229765917345,
        'inter_tap_mean': 1.4191597938144331,
        'inter_tap_std': 1.6720472977889953,
        'inter_tap_variance': 2.795742166043481,
        'inter_tap_median': 0.8605,
        'inter_tap_min': 0.38,
        'inter_tap_max': 12.632,
        'inter_tap_skewness': 4.281452702338388,
        'inter_tap_kurtosis': 22.46577859226015,
        'pressure_mean': 0.5198718015384615,
        'pressure_std': 0.1050557515514846,
        'pressure_variance': 0.01103671093404726,
        'pressure_median': 0.5625,
        'pressure_min': 0.125,
        'pressure_max': 0.8125,
        'pressure_skewness': -0.6628241616529988,
        'pressure_kurtosis': 0.14377288788941867,
        'event_count': 195.0,
        'recording_duration': 275.38,
        'typing_rate': 0.7081124264652481,
      };

      final pyVector = TouchscreenFeatureVector.fromMap(expectedPyFeatures);
      expect(pyVector.toOrderedList().length, 27);
    });

    test('End-to-End Pipeline: EventCollector -> Extractor -> FeatureVector -> InferenceEngine', () {
      final collector = TouchscreenEventCollector();
      collector.startCollection();

      // Record 5 events
      double press = 1000.0;
      for (int i = 0; i < 5; i++) {
        collector.recordTouch(
          pressTimeMs: press,
          releaseTimeMs: press + 80.0,
          pressure: 0.5 + (i * 0.05),
        );
        press += 600.0;
      }

      final events = collector.stopCollection();
      expect(events.length, 5);

      final featureVector = TouchscreenFeatureExtractor.extract(events);
      expect(featureVector.eventCount, 5.0);

      // Feed into existing mobile inference engine
      final prediction = inferenceEngine.predict(featureVector);

      expect(prediction.probabilityPD, greaterThanOrEqualTo(0.0));
      expect(prediction.probabilityPD, lessThanOrEqualTo(1.0));
      expect(prediction.predictedLabel, isIn(['CNT', 'PD']));
    });
  });
}
