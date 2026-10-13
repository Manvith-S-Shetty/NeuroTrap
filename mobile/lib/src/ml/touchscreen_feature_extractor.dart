import 'dart:math' as math;
import 'touchscreen_feature_vector.dart';
import 'touchscreen_touch_event.dart';

/// Exception thrown when real touch pressure measurement is unavailable on the device.
class TouchPressureUnavailableException implements Exception {
  final String message;

  const TouchPressureUnavailableException(
      [this.message =
          'Real touch pressure is unavailable on this device. Cannot perform touchscreen ML inference.']);

  @override
  String toString() => 'TouchPressureUnavailableException: $message';
}

/// Reproduces exact pandas/scipy statistical aggregate functions in Dart.
class TouchscreenFeatureExtractor {
  /// Extract 27-numerical-feature vector from raw touchscreen touch events.
  ///
  /// Throws [TouchPressureUnavailableException] if any event lacks real touch pressure.
  static TouchscreenFeatureVector extract(List<TouchscreenTouchEvent> events) {
    // 1. Filter valid events matching Python validation criteria:
    // pressTimeMs > 0, releaseTimeMs >= pressTimeMs, all values finite
    final List<TouchscreenTouchEvent> validEvents = [];
    for (final event in events) {
      if (event.pressTimeMs > 0 &&
          event.releaseTimeMs >= event.pressTimeMs &&
          !event.pressTimeMs.isNaN &&
          !event.pressTimeMs.isInfinite &&
          !event.releaseTimeMs.isNaN &&
          !event.releaseTimeMs.isInfinite) {
        // Enforce explicit pressure availability check - NEVER substitute artificial values
        if (!event.hasPressure) {
          throw const TouchPressureUnavailableException(
              'Touch event is missing valid real touch pressure measurements.');
        }
        validEvents.add(event);
      }
    }

    // 2. Extract raw numerical series
    final List<double> holdSec =
        validEvents.map((e) => e.holdTimeSec).toList();
    final List<double> pressures =
        validEvents.map((e) => e.pressure!).toList();

    // Enforce pressure variation check for multi-event sessions: constant pressure (single unique value across N>=2 taps)
    // cannot provide meaningful dynamic features for the 8 pressure statistical aggregates.
    final Set<double> uniquePressures = pressures.toSet();
    if (validEvents.length >= 2 && uniquePressures.length <= 1) {
      throw const TouchPressureUnavailableException(
          'Real touch pressure variation is unavailable on this device (constant pressure signal detected). Cannot perform touchscreen ML inference.');
    }

    // 3. Extract positive inter-tap intervals sorted by press timestamp
    final List<double> interTapSec = _calculateInterTapIntervalsSec(validEvents);

    // 4. Compute 8 statistical aggregates for each group
    final holdStats = _computeStats(holdSec);
    final interTapStats = _computeStats(interTapSec);
    final pressureStats = _computeStats(pressures);

    // 5. Activity features
    final double eventCount = validEvents.length.toDouble();
    double durationSec = double.nan;
    double typingRate = double.nan;

    if (validEvents.isNotEmpty) {
      double minPressMs = validEvents.first.pressTimeMs;
      double maxReleaseMs = validEvents.first.releaseTimeMs;
      for (final e in validEvents) {
        if (e.pressTimeMs < minPressMs) minPressMs = e.pressTimeMs;
        if (e.releaseTimeMs > maxReleaseMs) maxReleaseMs = e.releaseTimeMs;
      }
      final double durationMs = maxReleaseMs - minPressMs;

      if (durationMs > 0 && !durationMs.isNaN && !durationMs.isInfinite) {
        durationSec = durationMs / 1000.0;
        typingRate = eventCount / durationSec;
      } else if (durationMs == 0) {
        durationSec = 0.0;
        typingRate = double.nan;
      }
    }

    return TouchscreenFeatureVector(
      holdMean: holdStats['mean']!,
      holdStd: holdStats['std']!,
      holdVariance: holdStats['variance']!,
      holdMedian: holdStats['median']!,
      holdMin: holdStats['min']!,
      holdMax: holdStats['max']!,
      holdSkewness: holdStats['skewness']!,
      holdKurtosis: holdStats['kurtosis']!,
      interTapMean: interTapStats['mean']!,
      interTapStd: interTapStats['std']!,
      interTapVariance: interTapStats['variance']!,
      interTapMedian: interTapStats['median']!,
      interTapMin: interTapStats['min']!,
      interTapMax: interTapStats['max']!,
      interTapSkewness: interTapStats['skewness']!,
      interTapKurtosis: interTapStats['kurtosis']!,
      pressureMean: pressureStats['mean']!,
      pressureStd: pressureStats['std']!,
      pressureVariance: pressureStats['variance']!,
      pressureMedian: pressureStats['median']!,
      pressureMin: pressureStats['min']!,
      pressureMax: pressureStats['max']!,
      pressureSkewness: pressureStats['skewness']!,
      pressureKurtosis: pressureStats['kurtosis']!,
      eventCount: eventCount,
      recordingDuration: durationSec,
      typingRate: typingRate,
    );
  }

  /// Calculates positive inter-tap press_time[i+1] - press_time[i] in seconds.
  /// Uses stable sort on pressTimeMs.
  static List<double> _calculateInterTapIntervalsSec(
      List<TouchscreenTouchEvent> validEvents) {
    if (validEvents.length < 2) return [];

    final sorted = List<TouchscreenTouchEvent>.from(validEvents);
    // Stable merge sort matching pandas sort_values(kind="mergesort")
    sorted.sort((a, b) => a.pressTimeMs.compareTo(b.pressTimeMs));

    final List<double> positiveDiffsSec = [];
    for (int i = 0; i < sorted.length - 1; i++) {
      final diffMs = sorted[i + 1].pressTimeMs - sorted[i].pressTimeMs;
      if (diffMs > 0 && !diffMs.isNaN && !diffMs.isInfinite) {
        positiveDiffsSec.add(diffMs / 1000.0);
      }
    }
    return positiveDiffsSec;
  }

  /// Calculates 8 stats matching pandas Series methods:
  /// - mean(), median(), min(), max()
  /// - std(ddof=1), var(ddof=1) (N >= 2)
  /// - skew() (Fisher-Pearson adjusted, N >= 3)
  /// - kurt() (Fisher excess kurtosis adjusted, N >= 4)
  static Map<String, double> _computeStats(List<double> values) {
    final clean = values.where((v) => !v.isNaN && !v.isInfinite).toList();
    final int n = clean.length;

    if (n == 0) {
      return {
        'mean': double.nan,
        'std': double.nan,
        'variance': double.nan,
        'median': double.nan,
        'min': double.nan,
        'max': double.nan,
        'skewness': double.nan,
        'kurtosis': double.nan,
      };
    }

    // Mean
    final double sum = clean.reduce((a, b) => a + b);
    final double mean = sum / n;

    // Min & Max
    double minVal = clean.first;
    double maxVal = clean.first;
    for (final v in clean) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    // Median
    final sorted = List<double>.from(clean)..sort();
    double median;
    if (n % 2 == 1) {
      median = sorted[n ~/ 2];
    } else {
      median = (sorted[(n ~/ 2) - 1] + sorted[n ~/ 2]) / 2.0;
    }

    // Variance (sample, ddof=1) & Std
    double variance = double.nan;
    double std = double.nan;

    if (n >= 2) {
      double sumSqDiff = 0.0;
      for (final v in clean) {
        final d = v - mean;
        sumSqDiff += d * d;
      }
      variance = sumSqDiff / (n - 1);
      std = math.sqrt(variance);
    }

    // Skewness (pandas Series.skew(): adjusted Fisher-Pearson coefficient)
    // Formula: [n / ((n-1)(n-2))] * [sum((x - mean)^3) / s^3]
    double skewness = double.nan;
    if (n >= 3 && std > 0) {
      double sumCubeDiff = 0.0;
      for (final v in clean) {
        final d = v - mean;
        sumCubeDiff += d * d * d;
      }
      final double factor = n.toDouble() / ((n - 1) * (n - 2));
      skewness = factor * (sumCubeDiff / (std * std * std));
    } else if (n >= 3 && std == 0) {
      skewness = 0.0;
    }

    // Kurtosis (pandas Series.kurt(): adjusted Fisher excess kurtosis)
    // Formula: [n(n+1)/((n-1)(n-2)(n-3))] * [sum((x - mean)^4) / s^4] - [3(n-1)^2 / ((n-2)(n-3))]
    double kurtosis = double.nan;
    if (n >= 4 && std > 0) {
      double sumQuadDiff = 0.0;
      for (final v in clean) {
        final d = v - mean;
        sumQuadDiff += d * d * d * d;
      }
      final double term1Factor =
          (n.toDouble() * (n + 1)) / ((n - 1) * (n - 2) * (n - 3));
      final double term1 = term1Factor * (sumQuadDiff / (variance * variance));
      final double term2 =
          (3.0 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
      kurtosis = term1 - term2;
    } else if (n >= 4 && std == 0) {
      kurtosis = 0.0;
    }

    return {
      'mean': mean,
      'std': std,
      'variance': variance,
      'median': median,
      'min': minVal,
      'max': maxVal,
      'skewness': skewness,
      'kurtosis': kurtosis,
    };
  }
}
