import 'dart:convert';

/// Parameter container for 19-feature pressure-independent touchscreen mobile ML inference.
class TouchscreenNoPressureModelParameters {
  final String modelName;
  final String version;
  final int featureCount;
  final List<String> featureNames;
  final List<String> classes;
  final double selectedThreshold;
  final bool pressureFeaturesExcluded;

  final List<double> scalerMean;
  final List<double> scalerScale;

  final List<double> coefficients;
  final double intercept;

  const TouchscreenNoPressureModelParameters({
    required this.modelName,
    required this.version,
    required this.featureCount,
    required this.featureNames,
    required this.classes,
    required this.selectedThreshold,
    required this.pressureFeaturesExcluded,
    required this.scalerMean,
    required this.scalerScale,
    required this.coefficients,
    required this.intercept,
  });

  /// Parse and validate from exported JSON string or decoded Map.
  factory TouchscreenNoPressureModelParameters.fromJson(Map<String, dynamic> json) {
    final mName = json['model_name'] as String? ?? 'Touchscreen No-Pressure Model';
    final ver = json['version'] as String? ?? '1.0.0';
    final fCount = (json['feature_count'] as num?)?.toInt() ?? 19;
    final fNames = List<String>.from(json['feature_names'] as List? ?? []);
    final cls = List<String>.from(json['classes'] as List? ?? []);
    final thresh = (json['selected_threshold'] as num?)?.toDouble() ?? 0.50;
    final noPressure = (json['pressure_features_excluded'] as bool?) ?? true;

    final scaler = json['scaler'] as Map<String, dynamic>? ?? {};
    final meansRaw = scaler['means'] ?? scaler['mean'] ?? [];
    final scalesRaw = scaler['scales'] ?? scaler['scale'] ?? [];

    final means = List<double>.from(
        (meansRaw as List).map((e) => (e as num).toDouble()));
    final scales = List<double>.from(
        (scalesRaw as List).map((e) => (e as num).toDouble()));

    final lr = json['logistic_regression'] as Map<String, dynamic>? ?? {};
    final coefs = List<double>.from(
        (lr['coefficients'] as List? ?? []).map((e) => (e as num).toDouble()));
    final inter = (lr['intercept'] as num?)?.toDouble() ?? 0.0;

    final params = TouchscreenNoPressureModelParameters(
      modelName: mName,
      version: ver,
      featureCount: fCount,
      featureNames: fNames,
      classes: cls,
      selectedThreshold: thresh,
      pressureFeaturesExcluded: noPressure,
      scalerMean: means,
      scalerScale: scales,
      coefficients: coefs,
      intercept: inter,
    );

    params.validate();
    return params;
  }

  factory TouchscreenNoPressureModelParameters.fromJsonString(String jsonString) {
    return TouchscreenNoPressureModelParameters.fromJson(
        jsonDecode(jsonString) as Map<String, dynamic>);
  }

  /// Strict validation of 19-feature model parameters.
  void validate() {
    if (featureCount != 19) {
      throw ArgumentError('Expected featureCount == 19, got $featureCount');
    }
    if (featureNames.length != 19) {
      throw ArgumentError(
          'Expected 19 featureNames, got ${featureNames.length}');
    }
    if (classes.length != 2 || classes[0] != 'CNT' || classes[1] != 'PD') {
      throw ArgumentError('Expected classes == ["CNT", "PD"], got $classes');
    }
    if (scalerMean.length != 19) {
      throw ArgumentError('Expected 19 scalerMean, got ${scalerMean.length}');
    }
    if (scalerScale.length != 19) {
      throw ArgumentError('Expected 19 scalerScale, got ${scalerScale.length}');
    }
    if (coefficients.length != 19) {
      throw ArgumentError(
          'Expected 19 coefficients, got ${coefficients.length}');
    }

    for (int i = 0; i < 19; i++) {
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
