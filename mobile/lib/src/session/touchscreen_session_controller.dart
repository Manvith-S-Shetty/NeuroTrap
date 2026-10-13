import 'dart:async';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import '../ml/touchscreen_event_collector.dart';
import '../ml/touchscreen_feature_extractor.dart';
import '../ml/touchscreen_inference_engine.dart';
import '../ml/touchscreen_nopressure_feature_extractor.dart';
import '../ml/touchscreen_nopressure_inference_engine.dart';
import '../ml/touchscreen_touch_event.dart';

/// Prototype data-quality/usability minimum touch event threshold.
/// Requires at least 30 events based on the controlled passage length.
const int kMinPassageEventCount = 30;

/// Controlled text passage for the touchscreen screening session.
const String kTouchscreenScreeningPassage =
    'The quick brown fox jumps over the lazy dog. Continuous typing dynamics '
    'provide subtle timing and pressure patterns for motor feature extraction.';

/// High-level touchscreen session statuses.
enum TouchscreenSessionStatus {
  idle,
  countdown,
  collecting,
  processing,
  completed,
  error,
}

/// Local development diagnostic summary container for touch pressure signals.
class TouchscreenPressureDiagnostic {
  final int totalEvents;
  final int nullPressureCount;
  final int outOfBoundsCount;
  final List<double> uniquePressureValues;
  final double minPressure;
  final double maxPressure;
  final double meanPressure;
  final double stdPressure;
  final bool hasVariation;

  const TouchscreenPressureDiagnostic({
    required this.totalEvents,
    required this.nullPressureCount,
    required this.outOfBoundsCount,
    required this.uniquePressureValues,
    required this.minPressure,
    required this.maxPressure,
    required this.meanPressure,
    required this.stdPressure,
    required this.hasVariation,
  });

  factory TouchscreenPressureDiagnostic.analyze(
      List<TouchscreenTouchEvent> events) {
    final int total = events.length;
    int nullCount = 0;
    int outOfBoundsCount = 0;
    final Set<double> uniqueSet = {};
    final List<double> validPressures = [];

    for (final e in events) {
      if (e.pressure == null) {
        nullCount++;
      } else {
        final p = e.pressure!;
        if (p < 0.0 || p > 1.0 || p.isNaN || p.isInfinite) {
          outOfBoundsCount++;
        }
        uniqueSet.add(p);
        validPressures.add(p);
      }
    }

    double minVal = double.nan;
    double maxVal = double.nan;
    double meanVal = double.nan;
    double stdVal = double.nan;

    if (validPressures.isNotEmpty) {
      minVal = validPressures.reduce(math.min);
      maxVal = validPressures.reduce(math.max);
      final double sum = validPressures.reduce((a, b) => a + b);
      meanVal = sum / validPressures.length;

      if (validPressures.length >= 2) {
        double sumSqDiff = 0.0;
        for (final p in validPressures) {
          final diff = p - meanVal;
          sumSqDiff += diff * diff;
        }
        final variance = sumSqDiff / (validPressures.length - 1);
        stdVal = math.sqrt(variance);
      } else {
        stdVal = 0.0;
      }
    }

    return TouchscreenPressureDiagnostic(
      totalEvents: total,
      nullPressureCount: nullCount,
      outOfBoundsCount: outOfBoundsCount,
      uniquePressureValues: uniqueSet.toList()..sort(),
      minPressure: minVal,
      maxPressure: maxVal,
      meanPressure: meanVal,
      stdPressure: stdVal,
      hasVariation:
          uniqueSet.length > 1 && (!stdVal.isNaN && stdVal > 0.0),
    );
  }

  @override
  String toString() {
    return 'PressureDiagnostic(totalEvents: $totalEvents, nulls: $nullPressureCount, '
        'outOfBounds: $outOfBoundsCount, uniqueValues: $uniquePressureValues, '
        'min: ${minPressure.toStringAsFixed(4)}, max: ${maxPressure.toStringAsFixed(4)}, '
        'mean: ${meanPressure.toStringAsFixed(4)}, std: ${stdPressure.toStringAsFixed(4)}, '
        'hasVariation: $hasVariation)';
  }
}

/// Internal helper for tracking active pointer presses.
class _ActivePointer {
  final double pressTimeMs;
  final double? pressure;

  const _ActivePointer({
    required this.pressTimeMs,
    this.pressure,
  });
}

/// Immutable state snapshot of a touchscreen screening session.
class TouchscreenSessionState {
  final TouchscreenSessionStatus status;
  final int countdownSeconds;
  final int eventCount;
  final List<TouchscreenTouchEvent> events;
  final String typedText;
  final TouchscreenPrediction? prediction;
  final String? errorMessage;
  final String? userNotice;

  const TouchscreenSessionState({
    required this.status,
    required this.countdownSeconds,
    required this.eventCount,
    required this.events,
    required this.typedText,
    this.prediction,
    this.errorMessage,
    this.userNotice,
  });

  bool get isIdle => status == TouchscreenSessionStatus.idle;
  bool get isCountdown => status == TouchscreenSessionStatus.countdown;
  bool get isCollecting => status == TouchscreenSessionStatus.collecting;
  bool get isProcessing => status == TouchscreenSessionStatus.processing;
  bool get isCompleted => status == TouchscreenSessionStatus.completed;
  bool get isError => status == TouchscreenSessionStatus.error;
}

/// Controller managing the touchscreen screening session lifecycle.
class TouchscreenSessionController {
  TouchscreenSessionController({TouchscreenEventCollector? collector})
      : _collector = collector ?? TouchscreenEventCollector();

  final TouchscreenEventCollector _collector;
  final Map<int, _ActivePointer> _activePointers = {};

  TouchscreenSessionStatus _status = TouchscreenSessionStatus.idle;
  int _countdownSeconds = 3;
  Timer? _countdownTimer;
  TouchscreenPrediction? _prediction;
  String? _errorMessage;
  String? _userNotice;
  String _typedText = '';

  TouchscreenSessionStatus get status => _status;
  int get countdownSeconds => _countdownSeconds;
  TouchscreenPrediction? get prediction => _prediction;
  String? get errorMessage => _errorMessage;
  String? get userNotice => _userNotice;
  String get typedText => _typedText;

  bool get isIdle => _status == TouchscreenSessionStatus.idle;
  bool get isCountdown => _status == TouchscreenSessionStatus.countdown;
  bool get isCollecting => _status == TouchscreenSessionStatus.collecting;
  bool get isProcessing => _status == TouchscreenSessionStatus.processing;
  bool get isCompleted => _status == TouchscreenSessionStatus.completed;
  bool get isError => _status == TouchscreenSessionStatus.error;

  TouchscreenSessionState snapshot() {
    return TouchscreenSessionState(
      status: _status,
      countdownSeconds: _countdownSeconds,
      eventCount: _collector.events.length,
      events: List.unmodifiable(_collector.events),
      typedText: _typedText,
      prediction: _prediction,
      errorMessage: _errorMessage,
      userNotice: _userNotice,
    );
  }

  /// Starts the session by entering the countdown state.
  /// Touch events during countdown are completely ignored.
  void startSession({void Function()? onCountdownComplete}) {
    _countdownTimer?.cancel();
    _activePointers.clear();
    _prediction = null;
    _errorMessage = null;
    _userNotice = null;
    _typedText = '';
    _countdownSeconds = 3;
    _status = TouchscreenSessionStatus.countdown;

    _countdownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (_countdownSeconds > 1) {
        _countdownSeconds--;
      } else {
        timer.cancel();
        _countdownSeconds = 0;
        _status = TouchscreenSessionStatus.collecting;
        _collector.startCollection();
        if (onCountdownComplete != null) {
          onCountdownComplete();
        }
      }
    });
  }

  /// Direct method to start collection immediately (bypassing timer for testing).
  void startCollectionDirectly() {
    _countdownTimer?.cancel();
    _activePointers.clear();
    _prediction = null;
    _errorMessage = null;
    _userNotice = null;
    _typedText = '';
    _countdownSeconds = 0;
    _status = TouchscreenSessionStatus.collecting;
    _collector.startCollection();
  }

  /// Called when a key tap output value is emitted by the custom keyboard.
  void handleKeyTap(String key) {
    if (_status != TouchscreenSessionStatus.collecting) return;
    if (key == '⌫') {
      if (_typedText.isNotEmpty) {
        _typedText = _typedText.substring(0, _typedText.length - 1);
      }
    } else if (key != 'SHIFT') {
      _typedText += key;
    }
  }

  /// Called when a pointer down event is detected.
  void handlePointerDown(int pointerId, double pressTimeMs, double? pressure) {
    if (_status != TouchscreenSessionStatus.collecting) return;
    _activePointers[pointerId] = _ActivePointer(
      pressTimeMs: pressTimeMs,
      pressure: pressure,
    );
  }

  /// Called when a pointer up event is detected.
  void handlePointerUp(int pointerId, double releaseTimeMs, double? pressure) {
    if (_status != TouchscreenSessionStatus.collecting) return;
    final active = _activePointers.remove(pointerId);
    if (active != null) {
      final finalPressure = pressure ?? active.pressure;
      try {
        debugPrint(
            'REAL DEVICE EVENT: press=${active.pressTimeMs}, release=$releaseTimeMs, pressure=$finalPressure');
        _collector.recordTouch(
          pressTimeMs: active.pressTimeMs,
          releaseTimeMs: releaseTimeMs,
          pressure: finalPressure,
        );
      } catch (e) {
        _status = TouchscreenSessionStatus.error;
        _errorMessage = e.toString();
      }
    }
  }

  /// Directly records a pre-formed touch event (for testing or custom input).
  void recordTouchEvent(TouchscreenTouchEvent event) {
    if (_status != TouchscreenSessionStatus.collecting) return;
    try {
      _collector.recordTouch(
        pressTimeMs: event.pressTimeMs,
        releaseTimeMs: event.releaseTimeMs,
        pressure: event.pressure,
      );
    } catch (e) {
      _status = TouchscreenSessionStatus.error;
      _errorMessage = e.toString();
    }
  }

  /// Completes the session, validates minimum data quality, extracts features,
  /// and runs inference.
  bool completeSession(TouchscreenInferenceEngine engine) {
    if (_status != TouchscreenSessionStatus.collecting) {
      return false;
    }

    final rawEvents = _collector.events;

    // Check prototype minimum event count requirement based on passage length
    if (rawEvents.length < kMinPassageEventCount) {
      _userNotice = 'Please type more of the passage before finishing.';
      return false;
    }

    _userNotice = null;
    _status = TouchscreenSessionStatus.processing;
    _collector.stopCollection();

    final diag = TouchscreenPressureDiagnostic.analyze(rawEvents);
    debugPrint('DIAGNOSTIC PRESSURE REPORT: $diag');

    // Check pressure availability across all events
    final hasMissingPressure = rawEvents.any((e) => !e.hasPressure);
    if (hasMissingPressure || rawEvents.isEmpty) {
      _status = TouchscreenSessionStatus.error;
      _errorMessage =
          'Real touch pressure is unavailable on this device. Cannot perform touchscreen ML inference.';
      return false;
    }

    try {
      final featureVector = TouchscreenFeatureExtractor.extract(rawEvents);
      _prediction = engine.predict(featureVector);
      _status = TouchscreenSessionStatus.completed;
      return true;
    } on TouchPressureUnavailableException catch (e) {
      _status = TouchscreenSessionStatus.error;
      _errorMessage = e.message;
      return false;
    } catch (e) {
      _status = TouchscreenSessionStatus.error;
      _errorMessage = 'Session validation error: ${e.toString()}';
      return false;
    }
  }

  /// Completes the session using the 19-feature pressure-independent ML engine.
  ///
  /// Removes pressure requirements, constant pressure blocking, and pressure exceptions.
  bool completeNoPressureSession(TouchscreenNoPressureInferenceEngine engine) {
    if (_status != TouchscreenSessionStatus.collecting) {
      return false;
    }

    final rawEvents = _collector.events;

    // Check prototype minimum event count requirement based on passage length
    if (rawEvents.length < kMinPassageEventCount) {
      _userNotice = 'Please type more of the passage before finishing.';
      return false;
    }

    _userNotice = null;
    _status = TouchscreenSessionStatus.processing;
    _collector.stopCollection();

    try {
      final featureVector = TouchscreenNoPressureFeatureExtractor.extract(rawEvents);
      _prediction = engine.predict(featureVector);
      _status = TouchscreenSessionStatus.completed;
      return true;
    } catch (e) {
      _status = TouchscreenSessionStatus.error;
      _errorMessage = 'Session validation error: ${e.toString()}';
      return false;
    }
  }

  /// Resets the session back to the idle state and clears all raw events.
  void reset() {
    _countdownTimer?.cancel();
    _collector.startCollection();
    _collector.stopCollection();
    _activePointers.clear();
    _prediction = null;
    _errorMessage = null;
    _userNotice = null;
    _typedText = '';
    _countdownSeconds = 3;
    _status = TouchscreenSessionStatus.idle;
  }

  void dispose() {
    _countdownTimer?.cancel();
    _collector.dispose();
  }
}
