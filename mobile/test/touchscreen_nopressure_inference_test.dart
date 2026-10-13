import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_feature_vector.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_inference_engine.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_model_parameters.dart';

void main() {
  group('Touchscreen 19-Feature No-Pressure ML Mobile Inference Tests', () {
    late TouchscreenNoPressureModelParameters modelParams;
    late TouchscreenNoPressureInferenceEngine engine;

    setUpAll(() {
      final file = File('assets/models/touchscreen_nopressure_model.json');
      expect(file.existsSync(), isTrue,
          reason: 'Asset file assets/models/touchscreen_nopressure_model.json must exist');
      final jsonStr = file.readAsStringSync();
      modelParams = TouchscreenNoPressureModelParameters.fromJsonString(jsonStr);
      engine = TouchscreenNoPressureInferenceEngine(modelParams);
    });

    test('1 & 2. Exactly 19 features produced and canonical feature ordering', () {
      expect(TouchscreenNoPressureFeatureVector.canonicalFeatureNames.length, 19);
      expect(TouchscreenNoPressureFeatureVector.canonicalFeatureNames.first, 'hold_mean');
      expect(TouchscreenNoPressureFeatureVector.canonicalFeatureNames.last, 'typing_rate');
      expect(TouchscreenNoPressureFeatureVector.canonicalFeatureNames, [
        'hold_mean',
        'hold_std',
        'hold_variance',
        'hold_median',
        'hold_min',
        'hold_max',
        'hold_skewness',
        'hold_kurtosis',
        'inter_tap_mean',
        'inter_tap_std',
        'inter_tap_variance',
        'inter_tap_median',
        'inter_tap_min',
        'inter_tap_max',
        'inter_tap_skewness',
        'inter_tap_kurtosis',
        'event_count',
        'recording_duration',
        'typing_rate',
      ]);
    });

    test('3. No pressure feature is present in canonical feature names', () {
      for (final name in TouchscreenNoPressureFeatureVector.canonicalFeatureNames) {
        expect(name.contains('pressure'), isFalse,
            reason: 'Pressure feature found: $name');
      }
    });

    test('4 - 6. Model parameters parsing and length validation', () {
      expect(modelParams.featureCount, 19);
      expect(modelParams.classes, ['CNT', 'PD']);
      expect(modelParams.selectedThreshold, 0.50);
      expect(modelParams.pressureFeaturesExcluded, isTrue);
      expect(modelParams.scalerMean.length, 19);
      expect(modelParams.scalerScale.length, 19);
      expect(modelParams.coefficients.length, 19);
    });

    test('7. Rejection of NaN and Infinity values', () {
      final validList = List<double>.filled(19, 0.5);

      final nanList = List<double>.from(validList);
      nanList[3] = double.nan;
      expect(() => TouchscreenNoPressureFeatureVector.validateValues(nanList),
          throwsArgumentError);

      final infList = List<double>.from(validList);
      infList[8] = double.infinity;
      expect(() => TouchscreenNoPressureFeatureVector.validateValues(infList),
          throwsArgumentError);
    });

    test('8. Rejection of wrong feature vector length', () {
      final shortList = List<double>.filled(18, 0.5);
      expect(() => TouchscreenNoPressureFeatureVector.validateValues(shortList),
          throwsArgumentError);
      final longList = List<double>.filled(20, 0.5);
      expect(() => TouchscreenNoPressureFeatureVector.validateValues(longList),
          throwsArgumentError);
    });

    test('9 - 11. Probability bounds and sum to 1.0', () {
      final sampleVector = TouchscreenNoPressureFeatureVector.fromMap({
        'hold_mean': 0.125,
        'hold_std': 0.022,
        'hold_variance': 0.0005,
        'hold_median': 0.124,
        'hold_min': 0.064,
        'hold_max': 0.198,
        'hold_skewness': 0.079,
        'hold_kurtosis': 0.422,
        'inter_tap_mean': 1.019,
        'inter_tap_std': 1.022,
        'inter_tap_variance': 1.045,
        'inter_tap_median': 0.635,
        'inter_tap_min': 0.24,
        'inter_tap_max': 8.497,
        'inter_tap_skewness': 3.597,
        'inter_tap_kurtosis': 17.973,
        'event_count': 189.0,
        'recording_duration': 191.822,
        'typing_rate': 0.985,
      });

      final pred = engine.predict(sampleVector);

      expect(pred.probabilityPD, greaterThanOrEqualTo(0.0));
      expect(pred.probabilityPD, lessThanOrEqualTo(1.0));
      expect(pred.probabilityCNT, greaterThanOrEqualTo(0.0));
      expect(pred.probabilityCNT, lessThanOrEqualTo(1.0));
      expect(pred.probabilityPD + pred.probabilityCNT, closeTo(1.0, 1e-6));
      expect(pred.predictedLabel, pred.probabilityPD >= 0.50 ? 'PD' : 'CNT');
    });

    test('12. Python <-> Dart numerical equivalence test on reference samples', () {
      // Python Reference Sample 0 (Expected P(PD) = 0.4288027851470557)
      final sample0 = TouchscreenNoPressureFeatureVector.fromMap({
        'hold_mean': 0.12535449735449736,
        'hold_std': 0.022834656457176673,
        'hold_variance': 0.0005214215355172804,
        'hold_median': 0.124,
        'hold_min': 0.064,
        'hold_max': 0.198,
        'hold_skewness': 0.07945494658312441,
        'hold_kurtosis': 0.42289884429307056,
        'inter_tap_mean': 1.0196702127659574,
        'inter_tap_std': 1.022706850677888,
        'inter_tap_variance': 1.045929302423484,
        'inter_tap_median': 0.6355,
        'inter_tap_min': 0.24,
        'inter_tap_max': 8.497,
        'inter_tap_skewness': 3.597694970199808,
        'inter_tap_kurtosis': 17.973653234589754,
        'event_count': 189.0,
        'recording_duration': 191.822,
        'typing_rate': 0.9852884444954176,
      });

      final pred0 = engine.predict(sample0);
      expect(pred0.probabilityPD, closeTo(0.4288027851470557, 1e-5));
      expect(pred0.predictedLabel, 'CNT');

      // Python Reference Sample 10 (Expected P(PD) = 0.4219111744369901)
      final sample10 = TouchscreenNoPressureFeatureVector.fromMap({
        'hold_mean': 0.10451923076923074,
        'hold_std': 0.019862754962260382,
        'hold_variance': 0.00039452903469079946,
        'hold_median': 0.103,
        'hold_min': 0.07,
        'hold_max': 0.146,
        'hold_skewness': 0.1449944320644375,
        'hold_kurtosis': -0.8204150896387361,
        'inter_tap_mean': 0.7925882352941176,
        'inter_tap_std': 0.44893149483949496,
        'inter_tap_variance': 0.2015394870588235,
        'inter_tap_median': 0.658,
        'inter_tap_min': 0.233,
        'inter_tap_max': 1.926,
        'inter_tap_skewness': 1.243673853748716,
        'inter_tap_kurtosis': 0.4822458493752739,
        'event_count': 52.0,
        'recording_duration': 40.546,
        'typing_rate': 1.2824939574803926,
      });

      final pred10 = engine.predict(sample10);
      expect(pred10.probabilityPD, closeTo(0.4219111744369901, 1e-5));
      expect(pred10.predictedLabel, 'CNT');
    });

    test('13. Deterministic repeated inference', () {
      final vec = List<double>.generate(19, (i) => (i + 1) * 0.1);
      final p1 = engine.predictFromOrderedList(vec);
      final p2 = engine.predictFromOrderedList(vec);

      expect(p1.probabilityPD, equals(p2.probabilityPD));
      expect(p1.decisionScore, equals(p2.decisionScore));
    });
  });
}
