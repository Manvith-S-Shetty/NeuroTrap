import 'dart:async';
import 'touchscreen_touch_event.dart';

/// Stream-based and list-based touchscreen event collector abstraction.
///
/// Records raw touch events without key identity.
class TouchscreenEventCollector {
  final List<TouchscreenTouchEvent> _events = [];
  final _controller = StreamController<TouchscreenTouchEvent>.broadcast();

  bool _isCollecting = false;

  bool get isCollecting => _isCollecting;
  List<TouchscreenTouchEvent> get events => List.unmodifiable(_events);
  Stream<TouchscreenTouchEvent> get eventStream => _controller.stream;

  void startCollection() {
    _events.clear();
    _isCollecting = true;
  }

  void recordTouch({
    required double pressTimeMs,
    required double releaseTimeMs,
    double? pressure,
  }) {
    if (!_isCollecting) return;

    final event = TouchscreenTouchEvent(
      pressTimeMs: pressTimeMs,
      releaseTimeMs: releaseTimeMs,
      pressure: pressure,
    );

    event.validate();
    _events.add(event);
    _controller.add(event);
  }

  List<TouchscreenTouchEvent> stopCollection() {
    _isCollecting = false;
    return List.unmodifiable(_events);
  }

  void dispose() {
    _isCollecting = false;
    _controller.close();
  }
}
