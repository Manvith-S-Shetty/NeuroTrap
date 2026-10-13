import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_feature_vector.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_inference_engine.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_model_parameters.dart';

void main() {
  group('Touchscreen ML Mobile Inference Tests', () {
    late TouchscreenModelParameters modelParams;
    late TouchscreenInferenceEngine engine;

    setUpAll(() {
      final file = File('assets/models/touchscreen_model.json');
      expect(file.existsSync(), isTrue,
          reason: 'Asset file assets/models/touchscreen_model.json must exist');
      final jsonStr = file.readAsStringSync();
      modelParams = TouchscreenModelParameters.fromJsonString(jsonStr);
      engine = TouchscreenInferenceEngine(modelParams);
    });

    test('1 & 2. 27-feature vector creation and canonical ordering', () {
      expect(TouchscreenFeatureVector.canonicalFeatureNames.length, 27);
      expect(TouchscreenFeatureVector.canonicalFeatureNames.first, 'hold_mean');
      expect(TouchscreenFeatureVector.canonicalFeatureNames.last, 'typing_rate');
    });

    test('3 - 5. Model parameters parsing and validation', () {
      expect(modelParams.featureCount, 27);
      expect(modelParams.classes, ['CNT', 'PD']);
      expect(modelParams.scalerMean.length, 27);
      expect(modelParams.scalerScale.length, 27);
      expect(modelParams.coefficients.length, 27);
    });

    test('6 & 7. Rejection of NaN and Infinity values', () {
      final validList = List<double>.filled(27, 0.5);

      final nanList = List<double>.from(validList);
      nanList[5] = double.nan;
      expect(() => TouchscreenFeatureVector.validateValues(nanList),
          throwsArgumentError);

      final infList = List<double>.from(validList);
      infList[10] = double.infinity;
      expect(() => TouchscreenFeatureVector.validateValues(infList),
          throwsArgumentError);
    });

    test('8. Rejection of wrong feature count', () {
      final shortList = List<double>.filled(26, 0.5);
      expect(() => TouchscreenFeatureVector.validateValues(shortList),
          throwsArgumentError);
    });

    test('9 - 12. Probability bounds and sum to 1.0', () {
      final sampleVector = TouchscreenFeatureVector.fromMap({
        'hold_mean': 0.086,
        'hold_std': 0.020,
        'hold_variance': 0.0004,
        'hold_median': 0.086,
        'hold_min': 0.038,
        'hold_max': 0.16,
        'hold_skewness': 0.30,
        'hold_kurtosis': 0.20,
        'inter_tap_mean': 1.419,
        'inter_tap_std': 1.672,
        'inter_tap_variance': 2.795,
        'inter_tap_median': 0.860,
        'inter_tap_min': 0.38,
        'inter_tap_max': 12.632,
        'inter_tap_skewness': 4.28,
        'inter_tap_kurtosis': 22.46,
        'pressure_mean': 0.519,
        'pressure_std': 0.105,
        'pressure_variance': 0.011,
        'pressure_median': 0.562,
        'pressure_min': 0.125,
        'pressure_max': 0.812,
        'pressure_skewness': -0.66,
        'pressure_kurtosis': 0.14,
        'event_count': 195.0,
        'recording_duration': 275.38,
        'typing_rate': 0.708,
      });

      final pred = engine.predict(sampleVector);

      expect(pred.probabilityPD, greaterThanOrEqualTo(0.0));
      expect(pred.probabilityPD, lessThanOrEqualTo(1.0));
      expect(pred.probabilityCNT, greaterThanOrEqualTo(0.0));
      expect(pred.probabilityCNT, lessThanOrEqualTo(1.0));

      expect(pred.probabilityPD + pred.probabilityCNT, closeTo(1.0, 1e-6));
      expect(pred.predictedLabel, pred.probabilityPD >= 0.50 ? 'PD' : 'CNT');
    });

    test('13. Python <-> Dart numerical equivalence on development reference samples', () {
      // Reference Sample 1 (Python expected P(PD) = 0.5248076949580375, Label = PD)
      final sample1 = TouchscreenFeatureVector.fromMap({
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
      });

      final pred1 = engine.predict(sample1);
      expect(pred1.probabilityPD, closeTo(0.5248076949580375, 1e-5));
      expect(pred1.predictedLabel, 'PD');

      // Reference Sample 2 (Python expected P(PD) = 0.46763365039512783, Label = CNT)
      final sample2 = TouchscreenFeatureVector.fromMap({
        'hold_mean': 0.0806909090909091,
        'hold_std': 0.01429694700286371,
        'hold_variance': 0.0002044026936026936,
        'hold_median': 0.08,
        'hold_min': 0.054,
        'hold_max': 0.115,
        'hold_skewness': 0.23031606172536506,
        'hold_kurtosis': -0.5481388012683102,
        'inter_tap_mean': 1.028185185185185,
        'inter_tap_std': 0.7398060958109371,
        'inter_tap_variance': 0.5473130593990215,
        'inter_tap_median': 0.746,
        'inter_tap_min': 0.439,
        'inter_tap_max': 4.31,
        'inter_tap_skewness': 3.0103859232615555,
        'inter_tap_kurtosis': 10.79982796478189,
        'pressure_mean': 0.591363645090909,
        'pressure_std': 0.08825313640042524,
        'pressure_variance': 0.007788616084512062,
        'pressure_median': 0.6125,
        'pressure_min': 0.3625,
        'pressure_max': 0.7375,
        'pressure_skewness': -0.9659962978965969,
        'pressure_kurtosis': 0.2070892652508376,
        'event_count': 55.0,
        'recording_duration': 55.611,
        'typing_rate': 0.9890129650608693,
      });

      final pred2 = engine.predict(sample2);
      expect(pred2.probabilityPD, closeTo(0.46763365039512783, 1e-5));
      expect(pred2.predictedLabel, 'CNT');
    });

    test('14 & 15. Deterministic repeated inference', () {
      final vec = List<double>.generate(27, (i) => (i + 1) * 0.1);
      final p1 = engine.predictFromOrderedList(vec);
      final p2 = engine.predictFromOrderedList(vec);

      expect(p1.probabilityPD, equals(p2.probabilityPD));
      expect(p1.decisionScore, equals(p2.decisionScore));
    });
  });
}
