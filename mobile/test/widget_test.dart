import 'package:flutter_test/flutter_test.dart';
import 'package:neurotap_mobile/main.dart';

void main() {
  testWidgets('NeuroTap typing session screen renders core controls', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const NeuroTapApp());

    expect(find.text('NeuroTap'), findsOneWidget);
    expect(find.text('Typing Dynamics Session'), findsOneWidget);
    expect(find.text('Start Session'), findsOneWidget);
    expect(find.text('Stop Session'), findsOneWidget);
    expect(find.text('Status: Idle'), findsOneWidget);
    expect(find.text('Captured events: 0'), findsOneWidget);
  });

  testWidgets('Start and stop controls update session status', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const NeuroTapApp());

    await tester.tap(find.text('Start Session'));
    await tester.pump();
    expect(
      find.text('Status: Collecting hardware keyboard events'),
      findsOneWidget,
    );

    await tester.tap(find.text('Stop Session'));
    await tester.pump();
    expect(find.text('Status: Stopped'), findsOneWidget);
  });
}
