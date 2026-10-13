/// Immutable 19-feature vector contract for pressure-independent touchscreen ML inference.
///
/// Contains exactly 19 named numerical features in canonical order:
/// 1-8: hold_*
/// 9-16: inter_tap_*
/// 17: event_count
/// 18: recording_duration
/// 19: typing_rate
class TouchscreenNoPressureFeatureVector {
  final double holdMean;
  final double holdStd;
  final double holdVariance;
  final double holdMedian;
  final double holdMin;
  final double holdMax;
  final double holdSkewness;
  final double holdKurtosis;

  final double interTapMean;
  final double interTapStd;
  final double interTapVariance;
  final double interTapMedian;
  final double interTapMin;
  final double interTapMax;
  final double interTapSkewness;
  final double interTapKurtosis;

  final double eventCount;
  final double recordingDuration;
  final double typingRate;

  const TouchscreenNoPressureFeatureVector({
    required this.holdMean,
    required this.holdStd,
    required this.holdVariance,
    required this.holdMedian,
    required this.holdMin,
    required this.holdMax,
    required this.holdSkewness,
    required this.holdKurtosis,
    required this.interTapMean,
    required this.interTapStd,
    required this.interTapVariance,
    required this.interTapMedian,
    required this.interTapMin,
    required this.interTapMax,
    required this.interTapSkewness,
    required this.interTapKurtosis,
    required this.eventCount,
    required this.recordingDuration,
    required this.typingRate,
  });

  /// Expected exact 19 feature names in canonical order.
  static const List<String> canonicalFeatureNames = [
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
  ];

  /// Returns the ordered list of 19 feature values.
  List<double> toOrderedList() {
    final list = [
      holdMean,
      holdStd,
      holdVariance,
      holdMedian,
      holdMin,
      holdMax,
      holdSkewness,
      holdKurtosis,
      interTapMean,
      interTapStd,
      interTapVariance,
      interTapMedian,
      interTapMin,
      interTapMax,
      interTapSkewness,
      interTapKurtosis,
      eventCount,
      recordingDuration,
      typingRate,
    ];

    validateValues(list);
    return list;
  }

  /// Construct from a map of feature name to value.
  factory TouchscreenNoPressureFeatureVector.fromMap(Map<String, double> map) {
    for (final name in canonicalFeatureNames) {
      if (!map.containsKey(name)) {
        throw ArgumentError('Missing required feature: $name');
      }
    }

    return TouchscreenNoPressureFeatureVector(
      holdMean: map['hold_mean']!,
      holdStd: map['hold_std']!,
      holdVariance: map['hold_variance']!,
      holdMedian: map['hold_median']!,
      holdMin: map['hold_min']!,
      holdMax: map['hold_max']!,
      holdSkewness: map['hold_skewness']!,
      holdKurtosis: map['hold_kurtosis']!,
      interTapMean: map['inter_tap_mean']!,
      interTapStd: map['inter_tap_std']!,
      interTapVariance: map['inter_tap_variance']!,
      interTapMedian: map['inter_tap_median']!,
      interTapMin: map['inter_tap_min']!,
      interTapMax: map['inter_tap_max']!,
      interTapSkewness: map['inter_tap_skewness']!,
      interTapKurtosis: map['inter_tap_kurtosis']!,
      eventCount: map['event_count']!,
      recordingDuration: map['recording_duration']!,
      typingRate: map['typing_rate']!,
    );
  }

  /// Strict numerical validation rejecting NaN or infinity.
  static void validateValues(List<double> values) {
    if (values.length != 19) {
      throw ArgumentError(
          'Feature vector length mismatch: expected 19, got ${values.length}');
    }
    for (int i = 0; i < values.length; i++) {
      final v = values[i];
      if (v.isNaN) {
        throw ArgumentError(
            'Feature value at index $i (${canonicalFeatureNames[i]}) is NaN');
      }
      if (v.isInfinite) {
        throw ArgumentError(
            'Feature value at index $i (${canonicalFeatureNames[i]}) is Infinite');
      }
    }
  }
}
