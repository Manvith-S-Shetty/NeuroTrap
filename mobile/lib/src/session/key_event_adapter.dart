import 'package:flutter/services.dart';

import 'typing_session_controller.dart';

class FlutterKeyEventAdapter {
  const FlutterKeyEventAdapter(this.controller);

  final TypingSessionController controller;

  void handleKeyEvent(KeyEvent event) {
    if (event.synthesized) {
      return;
    }

    final keyId = event.physicalKey.usbHidUsage.toString();
    final keyLabel = event.character ?? event.logicalKey.keyLabel;
    final timestampSeconds =
        event.timeStamp.inMicroseconds / Duration.microsecondsPerSecond;

    if (event is KeyDownEvent) {
      controller.recordKeyDown(
        keyId: keyId,
        keyLabel: keyLabel,
        timestampSeconds: timestampSeconds,
      );
    } else if (event is KeyRepeatEvent) {
      controller.recordKeyDown(
        keyId: keyId,
        keyLabel: keyLabel,
        timestampSeconds: timestampSeconds,
        isRepeat: true,
      );
    } else if (event is KeyUpEvent) {
      controller.recordKeyUp(keyId: keyId, timestampSeconds: timestampSeconds);
    }
  }
}
