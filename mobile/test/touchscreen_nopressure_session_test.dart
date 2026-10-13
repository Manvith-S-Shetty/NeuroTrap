import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_feature_extractor.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_inference_engine.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_nopressure_model_parameters.dart';
import 'package:neurotap_mobile/src/ml/touchscreen_touch_event.dart';
import 'package:neurotap_mobile/src/session/touchscreen_screen.dart';
import 'package:neurotap_mobile/src/session/touchscreen_session_controller.dart';


void main() {
  group('19-Feature No-Pressure Touchscreen Screening Session Tests', () {
    late TouchscreenNoPressureModelParameters modelParams;
    late TouchscreenNoPressureInferenceEngine engine;

    setUpAll(() {
      final file = File('assets/models/touchscreen_nopressure_model.json');
      expect(file.existsSync(), isTrue,
          reason: 'assets/models/touchscreen_nopressure_model.json must exist');
      final jsonStr = file.readAsStringSync();
      modelParams = TouchscreenNoPressureModelParameters.fromJsonString(jsonStr);
      engine = TouchscreenNoPressureInferenceEngine(modelParams);
    });

    List<TouchscreenTouchEvent> generateSyntheticNoPressureEvents(int count) {
      final list = <TouchscreenTouchEvent>[];
      double currentPress = 1000.0;
      for (int i = 0; i < count; i++) {
        final hold = 80.0 + (i % 5) * 10.0;
        final release = currentPress + hold;

        // No pressure (null pressure)
        list.add(TouchscreenTouchEvent(
          pressTimeMs: currentPress,
          releaseTimeMs: release,
          pressure: null,
        ));
        currentPress = release + 300.0 + (i % 3) * 50.0;
      }
      return list;
    }

    test('1. PointerDown + PointerUp creates event when pressure is null', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      expect(controller.snapshot().eventCount, 0);

      // Pointer down/up with null pressure
      controller.handlePointerDown(1, 1000.0, null);
      controller.handlePointerUp(1, 1100.0, null);

      final state = controller.snapshot();
      expect(state.eventCount, 1);
      expect(state.events.first.pressTimeMs, 1000.0);
      expect(state.events.first.releaseTimeMs, 1100.0);
      expect(state.events.first.pressure, isNull);

      controller.dispose();
    });

    test('2. 30 no-pressure events permit completion without pressure errors', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticNoPressureEvents(30)) {
        controller.recordTouchEvent(event);
      }

      final success = controller.completeNoPressureSession(engine);
      expect(success, isTrue);
      expect(controller.isCompleted, isTrue);
      expect(controller.prediction, isNotNull);
      expect(controller.prediction!.probabilityPD, greaterThanOrEqualTo(0.0));
      expect(controller.prediction!.probabilityPD, lessThanOrEqualTo(1.0));

      controller.dispose();
    });

    test('3. Feature extraction produces exactly 19 features from null-pressure events', () {
      final events = generateSyntheticNoPressureEvents(30);
      final vec = TouchscreenNoPressureFeatureExtractor.extract(events);
      final list = vec.toOrderedList();

      expect(list.length, 19);
      expect(vec.eventCount, 30.0);
      expect(vec.holdMean, greaterThan(0.0));
      expect(vec.interTapMean, greaterThan(0.0));
    });

    test('4. Fewer than 30 events shows warning notice and prevents completion', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticNoPressureEvents(20)) {
        controller.recordTouchEvent(event);
      }

      final success = controller.completeNoPressureSession(engine);
      expect(success, isFalse);
      expect(controller.isCollecting, isTrue);
      expect(controller.userNotice, 'Please type more of the passage before finishing.');

      controller.dispose();
    });

    test('5. UI buttons do not create typing events', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      expect(controller.snapshot().eventCount, 0);
      controller.dispose();
    });

    test('6. Retry clears typed text and raw events', () {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticNoPressureEvents(35)) {
        controller.recordTouchEvent(event);
      }
      controller.handleKeyTap('x');
      controller.handleKeyTap('y');

      expect(controller.snapshot().eventCount, 35);
      expect(controller.snapshot().typedText, 'xy');

      controller.reset();

      final state = controller.snapshot();
      expect(state.eventCount, 0);
      expect(state.events, isEmpty);
      expect(state.typedText, '');
      expect(state.isIdle, isTrue);

      controller.dispose();
    });

    testWidgets('7. Custom keyboard active during collection (No TextField / Gboard)', (WidgetTester tester) async {
      final controller = TouchscreenSessionController();

      await tester.pumpWidget(
        MaterialApp(
          home: TouchscreenScreen(
            controller: controller,
            noPressureInferenceEngine: engine,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // No TextField widget (Gboard eliminated)
      expect(find.byType(TextField), findsNothing);

      controller.dispose();
    });

    testWidgets('8. Result view displays P(PD) and P(CNT) with non-diagnostic wording', (WidgetTester tester) async {
      final controller = TouchscreenSessionController();
      controller.startCollectionDirectly();

      for (final event in generateSyntheticNoPressureEvents(35)) {
        controller.recordTouchEvent(event);
      }
      controller.completeNoPressureSession(engine);

      await tester.pumpWidget(
        MaterialApp(
          home: TouchscreenScreen(
            controller: controller,
            noPressureInferenceEngine: engine,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Non-diagnostic headers and descriptions
      expect(find.text('Motor-pattern screening indicator'), findsOneWidget);
      expect(find.textContaining('Model-estimated pattern:'), findsOneWidget);

      // Probabilities displayed
      expect(find.text('PD Probability'), findsOneWidget);
      expect(find.text('CNT Probability'), findsOneWidget);

      // Non-diagnostic research disclaimer
      expect(find.textContaining('This is an experimental research prototype and is not a medical diagnosis'), findsOneWidget);

      // Confirm no diagnostic claims exist
      expect(find.textContaining('Parkinson\'s diagnosis'), findsNothing);
      expect(find.textContaining('clinical detection'), findsNothing);

      controller.dispose();
    });
  });
}
