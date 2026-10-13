import 'dart:math' as math;
import 'touchscreen_feature_vector.dart';
import 'touchscreen_model_parameters.dart';

/// Strongly typed prediction result container.
class TouchscreenPrediction {
  final String predictedLabel;
  final double probabilityPD;
  final double probabilityCNT;
  final double decisionScore;

  const TouchscreenPrediction({
    required this.predictedLabel,
    required this.probabilityPD,
    required this.probabilityCNT,
    required this.decisionScore,
  });

  @override
  String toString() =>
      'TouchscreenPrediction(label: $predictedLabel, P(PD): ${probabilityPD.toStringAsFixed(4)}, P(CNT): ${probabilityCNT.toStringAsFixed(4)}, score: ${decisionScore.toStringAsFixed(4)})';
}

/// Pure Dart inference engine for touchscreen Logistic Regression ML model.
class TouchscreenInferenceEngine {
  final TouchscreenModelParameters parameters;

  TouchscreenInferenceEngine(this.parameters) {
    parameters.validate();
  }

  /// Evaluates full prediction from an ordered list of 27 feature values.
  TouchscreenPrediction predictFromOrderedList(List<double> rawFeatures,
      {double threshold = 0.50}) {
    TouchscreenFeatureVector.validateValues(rawFeatures);

    double score = parameters.intercept;
    for (int i = 0; i < 27; i++) {
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

    final String label = pPD >= threshold ? 'PD' : 'CNT';

    return TouchscreenPrediction(
      predictedLabel: label,
      probabilityPD: pPD,
      probabilityCNT: pCNT,
      decisionScore: score,
    );
  }

  /// Evaluates full prediction from a TouchscreenFeatureVector instance.
  TouchscreenPrediction predict(TouchscreenFeatureVector featureVector,
      {double threshold = 0.50}) {
    return predictFromOrderedList(featureVector.toOrderedList(),
        threshold: threshold);
  }

  /// Convenience probability prediction helper.
  double predictProbabilityPD(TouchscreenFeatureVector featureVector) {
    return predict(featureVector).probabilityPD;
  }
}
