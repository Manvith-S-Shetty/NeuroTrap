import 'typing_event.dart';

class PendingKeyPress {
  const PendingKeyPress({required this.key, required this.pressTime});

  final String key;
  final double pressTime;
}

class TypingSessionSnapshot {
  const TypingSessionSnapshot({
    required this.isCollecting,
    required this.events,
    required this.activePressCount,
  });

  final bool isCollecting;
  final List<TypingEvent> events;
  final int activePressCount;

  int get eventCount => events.length;
}

class TypingSessionController {
  bool _isCollecting = false;
  final List<TypingEvent> _events = <TypingEvent>[];
  final Map<String, PendingKeyPress> _activePresses =
      <String, PendingKeyPress>{};

  bool get isCollecting => _isCollecting;
  int get eventCount => _events.length;
  List<TypingEvent> get events => List<TypingEvent>.unmodifiable(_events);

  void startSession() {
    _events.clear();
    _activePresses.clear();
    _isCollecting = true;
  }

  TypingSessionSnapshot stopSession() {
    _isCollecting = false;
    _activePresses.clear();
    return snapshot();
  }

  void recordKeyDown({
    required String keyId,
    required String keyLabel,
    required double timestampSeconds,
    bool isRepeat = false,
  }) {
    if (!_isCollecting || isRepeat) {
      return;
    }
    if (timestampSeconds < 0) {
      throw ArgumentError.value(
        timestampSeconds,
        'timestampSeconds',
        'must be non-negative',
      );
    }
    _activePresses.putIfAbsent(
      keyId,
      () => PendingKeyPress(key: keyLabel, pressTime: timestampSeconds),
    );
  }

  void recordKeyUp({required String keyId, required double timestampSeconds}) {
    if (!_isCollecting) {
      return;
    }
    if (timestampSeconds < 0) {
      throw ArgumentError.value(
        timestampSeconds,
        'timestampSeconds',
        'must be non-negative',
      );
    }

    final pending = _activePresses.remove(keyId);
    if (pending == null) {
      return;
    }
    if (timestampSeconds < pending.pressTime) {
      throw ArgumentError.value(
        timestampSeconds,
        'timestampSeconds',
        'release must be >= press',
      );
    }

    _events.add(
      TypingEvent(
        key: pending.key,
        pressTime: pending.pressTime,
        releaseTime: timestampSeconds,
      ),
    );
  }

  TypingSessionSnapshot snapshot() {
    return TypingSessionSnapshot(
      isCollecting: _isCollecting,
      events: events,
      activePressCount: _activePresses.length,
    );
  }
}
