import 'dart:convert';

/// Authoritative parameter container for touchscreen mobile ML inference.
class TouchscreenModelParameters {
  final String modelName;
  final String version;
  final int featureCount;
  final List<String> featureNames;
  final List<String> classes;

  final List<double> scalerMean;
  final List<double> scalerScale;

  final List<double> coefficients;
  final double intercept;

  const TouchscreenModelParameters({
    required this.modelName,
    required this.version,
    required this.featureCount,
    required this.featureNames,
    required this.classes,
    required this.scalerMean,
    required this.scalerScale,
    required this.coefficients,
    required this.intercept,
  });

  /// Parse and validate from exported JSON string or decoded Map.
  factory TouchscreenModelParameters.fromJson(Map<String, dynamic> json) {
    final meta = json['metadata'] as Map<String, dynamic>? ?? {};
    final fCount = (json['feature_count'] as num?)?.toInt() ?? 27;
    final fNames = List<String>.from(json['feature_names'] as List? ?? []);
    final cls = List<String>.from(json['classes'] as List? ?? []);

    final scaler = json['scaler'] as Map<String, dynamic>? ?? {};
    final means = List<double>.from(
        (scaler['mean'] as List? ?? []).map((e) => (e as num).toDouble()));
    final scales = List<double>.from(
        (scaler['scale'] as List? ?? []).map((e) => (e as num).toDouble()));

    final lr = json['logistic_regression'] as Map<String, dynamic>? ?? {};
    final coefs = List<double>.from(
        (lr['coefficients'] as List? ?? []).map((e) => (e as num).toDouble()));
    final inter = (lr['intercept'] as num?)?.toDouble() ?? 0.0;

    final params = TouchscreenModelParameters(
      modelName: meta['model_name'] as String? ?? 'Touchscreen Model',
      version: meta['version'] as String? ?? '1.0.0',
      featureCount: fCount,
      featureNames: fNames,
      classes: cls,
      scalerMean: means,
      scalerScale: scales,
      coefficients: coefs,
      intercept: inter,
    );

    params.validate();
    return params;
  }

  factory TouchscreenModelParameters.fromJsonString(String jsonString) {
    return TouchscreenModelParameters.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>);
  }

  /// Strict validation of model parameter lengths and values.
  void validate() {
    if (featureCount != 27) {
      throw ArgumentError('Expected featureCount == 27, got $featureCount');
    }
    if (featureNames.length != 27) {
      throw ArgumentError(
          'Expected 27 featureNames, got ${featureNames.length}');
    }
    if (classes.length != 2 || classes[0] != 'CNT' || classes[1] != 'PD') {
      throw ArgumentError('Expected classes == ["CNT", "PD"], got $classes');
    }
    if (scalerMean.length != 27) {
      throw ArgumentError('Expected 27 scalerMean, got ${scalerMean.length}');
    }
    if (scalerScale.length != 27) {
      throw ArgumentError('Expected 27 scalerScale, got ${scalerScale.length}');
    }
    if (coefficients.length != 27) {
      throw ArgumentError(
          'Expected 27 coefficients, got ${coefficients.length}');
    }

    for (int i = 0; i < 27; i++) {
      if (scalerMean[i].isNaN || scalerMean[i].isInfinite) {
        throw ArgumentError('scalerMean at $i is invalid');
      }
      if (scalerScale[i].isNaN ||
          scalerScale[i].isInfinite ||
          scalerScale[i] <= 0.0) {
        throw ArgumentError('scalerScale at $i must be positive finite');
      }
      if (coefficients[i].isNaN || coefficients[i].isInfinite) {
        throw ArgumentError('coefficients at $i is invalid');
      }
    }

    if (intercept.isNaN || intercept.isInfinite) {
      throw ArgumentError('intercept is invalid');
    }
  }
}
