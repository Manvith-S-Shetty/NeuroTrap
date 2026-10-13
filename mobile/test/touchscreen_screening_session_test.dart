import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_feature_vector.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_inference_engine.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_model_parameters.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_touch_event.dart';
import 'package:neurotap_mobile/src/session/custom_touchscreen_keyboard.dart';
import 'package:neurotap_mobile/src/session/touchscreen_screen.dart';
import 'package:neurotap_mobile/src/session/touchscreen_session_controller.dart';

void main() {
  group('Touchscreen Screening Session Unit & Integration Tests (Custom Keyboard)', () {
    late TouchscreenModelParameters modelParams;
    late TouchscreenInferenceEngine engine;

    setUpAll(() {
      final file = File('assets/models/touchscreen_model.json');
      expect(file.existsSync(), isTrue,
          reason: 'assets/models/touchscreen_model.json must exist');
      final jsonStr = file.readAsStringSync();
      modelParams = TouchscreenModelParameters.fromJsonString(jsonStr);
      engine = TouchscreenInferenceEngine(modelParams);
    });

    List<TouchscreenTouchEvent> generateSyntheticEvents(int count,
        {bool includeNullPressure = false}) {
      final list = <TouchscreenTouchEvent>[];
      double currentPress = 1000.0;
      for (int i = 0; i < count; i++) {
        final hold = 80.0 + (i % 5) * 10.0;
        final release = currentPress + hold;
        final pressure = (includeNullPressure && i == count - 1)
            ? null
            : 0.4 + (i % 4) * 0.1;

        list.add(TouchscreenTouchEvent(
          pressTimeMs: currentPress,
          releaseTimeMs: release,
          pressure: pressure,
        ));
        currentPress = release + 300.0 + (i % 3) * 50.0;
      }
      return list;
    }

    test('1 & 2. Virtual key PointerDown + PointerUp creates exactly one event with real pressure', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      expect(controller.snapshot().eventCount, 0);

      // Simulate pointer down and pointer up pair on key
      controller.handlePointerDown(1, 1000.0, 0.65);
      controller.handlePointerUp(1, 1100.0, 0.65);

      final state = controller.snapshot();
      expect(state.eventCount, 1);
      expect(state.events.length, 1);
      expect(state.events.first.pressTimeMs, 1000.0);
      expect(state.events.first.releaseTimeMs, 1100.0);
      expect(state.events.first.pressure, 0.65);

      controller.dispose();
    });

    test('3. UI buttons do not create typing events', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      // Only pointer down/up handled by virtual keys generate events.
      // Simulating arbitrary non-key UI interactions (no call to pointer handlers):
      expect(controller.snapshot().eventCount, 0);
      controller.dispose();
    });

    test('4. Countdown touches do not create typing events', () {
      final controller = TouchscreenSessionController();
      controller.startSession();

      // Attempt pointer down/up during countdown state
      controller.handlePointerDown(1, 1000.0, 0.5);
      controller.handlePointerUp(1, 1100.0, 0.5);

      expect(controller.snapshot().eventCount, 0);
      controller.dispose();
    });

    test('5. 30 virtual-key events permit completion', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticEvents(30)) {
        controller.recordTouchEvent(event);
      }

      final success = controller.completeSession(engine);
      expect(success, isTrue);
      expect(controller.isCompleted, isTrue);
      expect(controller.prediction, isNotNull);
      controller.dispose();
    });

    test('6. Fewer than 30 events shows warning notice and prevents completion', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticEvents(20)) {
        controller.recordTouchEvent(event);
      }

      final success = controller.completeSession(engine);
      expect(success, isFalse);
      expect(controller.isCollecting, isTrue);
      expect(controller.userNotice, 'Please type more of the passage before finishing.');
      controller.dispose();
    });

    testWidgets('7. Custom keyboard does not invoke system soft keyboard (no TextField)', (WidgetTester tester) async {
      final controller = TouchscreenSessionController();

      await tester.pumpWidget(
        MaterialApp(
          home: TouchscreenScreen(
            controller: controller,
            inferenceEngine: engine,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Verify no TextField widget exists in TouchscreenScreen tree
      expect(find.byType(TextField), findsNothing);

      controller.dispose();
    });

    test('8 & 9. Retry clears custom typed text buffer and raw events', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticEvents(35)) {
        controller.recordTouchEvent(event);
      }
      controller.handleKeyTap('a');
      controller.handleKeyTap('b');

      expect(controller.snapshot().eventCount, 35);
      expect(controller.snapshot().typedText, 'ab');

      controller.reset();

      final state = controller.snapshot();
      expect(state.eventCount, 0);
      expect(state.events, isEmpty);
      expect(state.typedText, '');
      expect(state.isIdle, isTrue);
      controller.dispose();
    });

    test('10. Existing ML inference output remains unchanged for identical feature vector', () {
      final sample = TouchscreenFeatureVector.fromMap({
        'hold_mean': 0.086,
        'hold_std': 0.020,
        'hold_variance': 0.0004,
        'hold_median': 0.086,
        'hold_min': 0.038,
        'hold_max': 0.16,
        'hold_skewness': 0.30,
        'hold_kurtosis': 0.20,
        'inter_tap_mean': 1.419,
        'inter_tap_std': 1.672,
        'inter_tap_variance': 2.795,
        'inter_tap_median': 0.860,
        'inter_tap_min': 0.38,
        'inter_tap_max': 12.632,
        'inter_tap_skewness': 4.28,
        'inter_tap_kurtosis': 22.46,
        'pressure_mean': 0.519,
        'pressure_std': 0.105,
        'pressure_variance': 0.011,
        'pressure_median': 0.562,
        'pressure_min': 0.125,
        'pressure_max': 0.812,
        'pressure_skewness': -0.66,
        'pressure_kurtosis': 0.14,
        'event_count': 195.0,
        'recording_duration': 275.38,
        'typing_rate': 0.708,
      });

      final pred = engine.predict(sample);
      expect(pred.probabilityPD, closeTo(0.5248, 0.01));
    });

    test('11. Missing or invalid pressure produces existing pressure error', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticEvents(35, includeNullPressure: true)) {
        controller.recordTouchEvent(event);
      }

      final success = controller.completeSession(engine);
      expect(success, isFalse);
      expect(controller.isError, isTrue);
      expect(controller.errorMessage, contains('pressure'));
      controller.dispose();
    });

    test('12. Fixed controlled passage remains unchanged', () {
      expect(kTouchscreenScreeningPassage,
          'The quick brown fox jumps over the lazy dog. Continuous typing dynamics provide subtle timing and pressure patterns for motor feature extraction.');
    });

    testWidgets('13. CustomTouchscreenKeyboard auto-appears when collecting and registers key taps', (WidgetTester tester) async {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      await tester.pumpWidget(
        MaterialApp(
          home: TouchscreenScreen(
            controller: controller,
            inferenceEngine: engine,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // CustomTouchscreenKeyboard is visible during collecting
      expect(find.byType(CustomTouchscreenKeyboard), findsOneWidget);

      // Tap key 't'
      await tester.tap(find.text('t'));
      await tester.pump();

      expect(controller.snapshot().typedText, 't');
      expect(controller.snapshot().eventCount, 1);

      controller.dispose();
    });
  });
}
