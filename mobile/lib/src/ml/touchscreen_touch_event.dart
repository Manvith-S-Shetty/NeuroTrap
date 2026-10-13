/// Immutable raw touchscreen touch event representation.
///
/// Encapsulates press timestamp, release timestamp, and pressure.
/// Note: Touchscreen events contain NO key identity information.
class TouchscreenTouchEvent {
  final double pressTimeMs;
  final double releaseTimeMs;
  final double? pressure;

  const TouchscreenTouchEvent({
    required this.pressTimeMs,
    required this.releaseTimeMs,
    this.pressure,
  });

  /// Hold time duration in milliseconds.
  double get holdTimeMs => releaseTimeMs - pressTimeMs;

  /// Hold time duration in seconds.
  double get holdTimeSec => holdTimeMs / 1000.0;

  /// Returns true if valid real pressure is recorded.
  bool get hasPressure => pressure != null && !pressure!.isNaN && !pressure!.isInfinite;

  /// Strict validation of numeric values and chronologic ordering.
  void validate() {
    if (pressTimeMs.isNaN || pressTimeMs.isInfinite || pressTimeMs <= 0) {
      throw ArgumentError('Invalid pressTimeMs: $pressTimeMs (must be > 0)');
    }
    if (releaseTimeMs.isNaN || releaseTimeMs.isInfinite) {
      throw ArgumentError('Invalid releaseTimeMs: $releaseTimeMs');
    }
    if (releaseTimeMs < pressTimeMs) {
      throw ArgumentError(
          'Invalid event timing: releaseTimeMs ($releaseTimeMs) < pressTimeMs ($pressTimeMs)');
    }
    if (pressure != null && (pressure!.isNaN || pressure!.isInfinite)) {
      throw ArgumentError('Invalid pressure value: $pressure');
    }
  }
}
