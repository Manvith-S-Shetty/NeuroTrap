import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/session/key_event_adapter.dart';
import 'package:neurotap_mobile/src/session/typing_event.dart';
import 'package:neurotap_mobile/src/session/typing_session_controller.dart';

void main() {
  test('TypingEvent exposes Python-compatible schema', () {
    const event = TypingEvent(key: 'a', pressTime: 1.0, releaseTime: 1.125);

    expect(event.holdTime, closeTo(0.125, 1e-12));
    expect(event.toJson(), {
      'key': 'a',
      'press_time': 1.0,
      'release_time': 1.125,
      'hold_time': closeTo(0.125, 1e-12),
    });
  });

  test('session lifecycle clears previous events and captures event order', () {
    final controller = TypingSessionController();

    controller.startSession();
    controller.recordKeyDown(
      keyId: 'key-a',
      keyLabel: 'a',
      timestampSeconds: 1.0,
    );
    controller.recordKeyUp(keyId: 'key-a', timestampSeconds: 1.1);
    controller.stopSession();

    expect(controller.eventCount, 1);

    controller.startSession();
    expect(controller.eventCount, 0);
    controller.recordKeyDown(
      keyId: 'key-b',
      keyLabel: 'b',
      timestampSeconds: 2.0,
    );
    controller.recordKeyUp(keyId: 'key-b', timestampSeconds: 2.08);
    controller.recordKeyDown(
      keyId: 'key-c',
      keyLabel: 'c',
      timestampSeconds: 2.2,
    );
    controller.recordKeyUp(keyId: 'key-c', timestampSeconds: 2.31);

    expect(controller.events.map((event) => event.key), ['b', 'c']);
    expect(controller.events.first.pressTime, 2.0);
    expect(controller.events.last.releaseTime, 2.31);
  });

  test('events outside an active session are ignored', () {
    final controller = TypingSessionController();

    controller.recordKeyDown(
      keyId: 'key-a',
      keyLabel: 'a',
      timestampSeconds: 1.0,
    );
    controller.recordKeyUp(keyId: 'key-a', timestampSeconds: 1.1);

    expect(controller.eventCount, 0);
  });

  test('repeat and unmatched key-up events do not fabricate records', () {
    final controller = TypingSessionController()..startSession();

    controller.recordKeyDown(
      keyId: 'key-a',
      keyLabel: 'a',
      timestampSeconds: 1.0,
      isRepeat: true,
    );
    controller.recordKeyUp(keyId: 'key-a', timestampSeconds: 1.1);
    controller.recordKeyUp(keyId: 'key-b', timestampSeconds: 1.2);

    expect(controller.eventCount, 0);
  });

  test('invalid timing is rejected', () {
    final controller = TypingSessionController()..startSession();

    expect(
      () => controller.recordKeyDown(
        keyId: 'key-a',
        keyLabel: 'a',
        timestampSeconds: -1.0,
      ),
      throwsArgumentError,
    );

    controller.recordKeyDown(
      keyId: 'key-a',
      keyLabel: 'a',
      timestampSeconds: 1.0,
    );
    expect(
      () => controller.recordKeyUp(keyId: 'key-a', timestampSeconds: 0.9),
      throwsArgumentError,
    );
  });

  test('stopping a session clears active pending presses', () {
    final controller = TypingSessionController()..startSession();
    controller.recordKeyDown(
      keyId: 'key-a',
      keyLabel: 'a',
      timestampSeconds: 1.0,
    );

    final snapshot = controller.stopSession();

    expect(snapshot.isCollecting, isFalse);
    expect(snapshot.activePressCount, 0);
    expect(controller.eventCount, 0);
  });

  test('Flutter key adapter converts key-down/up into one timing event', () {
    final controller = TypingSessionController()..startSession();
    final adapter = FlutterKeyEventAdapter(controller);

    adapter.handleKeyEvent(
      const KeyDownEvent(
        physicalKey: PhysicalKeyboardKey.keyA,
        logicalKey: LogicalKeyboardKey.keyA,
        character: 'a',
        timeStamp: Duration(milliseconds: 1000),
      ),
    );
    adapter.handleKeyEvent(
      const KeyUpEvent(
        physicalKey: PhysicalKeyboardKey.keyA,
        logicalKey: LogicalKeyboardKey.keyA,
        timeStamp: Duration(milliseconds: 1125),
      ),
    );

    expect(controller.eventCount, 1);
    expect(controller.events.single.key, 'a');
    expect(controller.events.single.pressTime, 1.0);
    expect(controller.events.single.releaseTime, 1.125);
    expect(controller.events.single.holdTime, closeTo(0.125, 1e-12));
  });

  test('Flutter key adapter ignores synthesized events', () {
    final controller = TypingSessionController()..startSession();
    final adapter = FlutterKeyEventAdapter(controller);

    adapter.handleKeyEvent(
      const KeyDownEvent(
        physicalKey: PhysicalKeyboardKey.keyA,
        logicalKey: LogicalKeyboardKey.keyA,
        character: 'a',
        timeStamp: Duration(milliseconds: 1000),
        synthesized: true,
      ),
    );
    adapter.handleKeyEvent(
      const KeyUpEvent(
        physicalKey: PhysicalKeyboardKey.keyA,
        logicalKey: LogicalKeyboardKey.keyA,
        timeStamp: Duration(milliseconds: 1125),
        synthesized: true,
      ),
    );

    expect(controller.eventCount, 0);
  });
}
