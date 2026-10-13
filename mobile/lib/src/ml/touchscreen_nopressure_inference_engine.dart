import 'dart:math' as math;
import 'touchscreen_inference_engine.dart';
import 'touchscreen_nopressure_feature_vector.dart';
import 'touchscreen_nopressure_model_parameters.dart';

/// Pure Dart inference engine for 19-feature pressure-independent touchscreen Logistic Regression ML model.
class TouchscreenNoPressureInferenceEngine {
  final TouchscreenNoPressureModelParameters parameters;

  TouchscreenNoPressureInferenceEngine(this.parameters) {
    parameters.validate();
  }

  /// Evaluates full prediction from an ordered list of 19 feature values.
  TouchscreenPrediction predictFromOrderedList(List<double> rawFeatures,
      {double? threshold}) {
    TouchscreenNoPressureFeatureVector.validateValues(rawFeatures);
    final double activeThreshold = threshold ?? parameters.selectedThreshold;

    double score = parameters.intercept;
    for (int i = 0; i < 19; i++) {
      final double mean = parameters.scalerMean[i];
      final double scale = parameters.scalerScale[i];
      final double coef = parameters.coefficients[i];

      // Standardization: z_i = (x_i - mean_i) / scale_i
      final double z = (rawFeatures[i] - mean) / scale;

      // Decision score: score = intercept + sum(coef_i * z_i)
      score += coef * z;
    }

    // Sigmoid: P(PD) = 1 / (1 + exp(-score))
    final double pPD = 1.0 / (1.0 + math.exp(-score));
    final double pCNT = 1.0 - pPD;

    final String label = pPD >= activeThreshold ? 'PD' : 'CNT';

    return TouchscreenPrediction(
      predictedLabel: label,
      probabilityPD: pPD,
      probabilityCNT: pCNT,
      decisionScore: score,
    );
  }

  /// Evaluates full prediction from a TouchscreenNoPressureFeatureVector instance.
  TouchscreenPrediction predict(TouchscreenNoPressureFeatureVector featureVector,
      {double? threshold}) {
    return predictFromOrderedList(featureVector.toOrderedList(),
        threshold: threshold);
  }

  /// Convenience probability prediction helper.
  double predictProbabilityPD(TouchscreenNoPressureFeatureVector featureVector) {
    return predict(featureVector).probabilityPD;
  }
}
