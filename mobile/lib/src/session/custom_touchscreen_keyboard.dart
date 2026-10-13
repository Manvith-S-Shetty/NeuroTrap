import 'package:flutter/material.dart';

/// A custom Flutter-owned virtual QWERTY keyboard for touchscreen screening.
///
/// Captures key-level [PointerDownEvent] and [PointerUpEvent] with timestamps
/// and real touch pressure without invoking system soft keyboards (Gboard).
class CustomTouchscreenKeyboard extends StatefulWidget {
  const CustomTouchscreenKeyboard({
    super.key,
    required this.onPointerDown,
    required this.onPointerUp,
    required this.onKeyTap,
  });

  final void Function(int pointerId, double pressTimeMs, double? pressure)
      onPointerDown;
  final void Function(int pointerId, double releaseTimeMs, double? pressure)
      onPointerUp;
  final void Function(String character) onKeyTap;

  @override
  State<CustomTouchscreenKeyboard> createState() =>
      _CustomTouchscreenKeyboardState();
}

class _CustomTouchscreenKeyboardState
    extends State<CustomTouchscreenKeyboard> {
  bool _isShifted = false;

  void _handleKeyRelease(String key) {
    if (key == 'SHIFT') {
      setState(() {
        _isShifted = !_isShifted;
      });
      return;
    }
    widget.onKeyTap(key);
    if (_isShifted && key != 'BACKSPACE') {
      setState(() {
        _isShifted = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Row 1: q w e r t y u i o p
          _buildRow(['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p']),
          const SizedBox(height: 6),
          // Row 2: a s d f g h j k l
          _buildRow(['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
              sidePadding: 12),
          const SizedBox(height: 6),
          // Row 3: Shift z x c v b n m Backspace
          _buildRow3(),
          const SizedBox(height: 6),
          // Row 4: Space Period
          _buildRow4(),
        ],
      ),
    );
  }

  Widget _buildRow(List<String> keys, {double sidePadding = 0}) {
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: sidePadding),
      child: Row(
        children: keys.map((k) {
          final displayChar = _isShifted ? k.toUpperCase() : k;
          return _buildKeyCell(
            label: displayChar,
            outputValue: displayChar,
          );
        }).toList(),
      ),
    );
  }

  Widget _buildRow3() {
    final letters = ['z', 'x', 'c', 'v', 'b', 'n', 'm'];
    return Row(
      children: [
        // Shift key
        _buildKeyCell(
          label: '⇧',
          outputValue: 'SHIFT',
          flex: 1.5,
          color: _isShifted
              ? Theme.of(context).colorScheme.primaryContainer
              : null,
        ),
        ...letters.map((k) {
          final displayChar = _isShifted ? k.toUpperCase() : k;
          return _buildKeyCell(
            label: displayChar,
            outputValue: displayChar,
          );
        }),
        // Backspace key
        _buildKeyCell(
          label: '⌫',
          outputValue: '⌫',
          flex: 1.5,
        ),
      ],
    );
  }

  Widget _buildRow4() {
    return Row(
      children: [
        // Spacebar
        _buildKeyCell(
          label: 'space',
          outputValue: ' ',
          flex: 6.0,
        ),
        // Period
        _buildKeyCell(
          label: '.',
          outputValue: '.',
          flex: 2.0,
        ),
      ],
    );
  }

  Widget _buildKeyCell({
    required String label,
    required String outputValue,
    double flex = 1.0,
    Color? color,
  }) {
    return Expanded(
      flex: (flex * 10).round(),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 2.0),
        child: Listener(
          behavior: HitTestBehavior.opaque,
          onPointerDown: (event) {
            final nowMs = DateTime.now().millisecondsSinceEpoch.toDouble();
            widget.onPointerDown(event.pointer, nowMs, event.pressure);
          },
          onPointerUp: (event) {
            final nowMs = DateTime.now().millisecondsSinceEpoch.toDouble();
            widget.onPointerUp(event.pointer, nowMs, event.pressure);
            _handleKeyRelease(outputValue);
          },
          child: Container(
            height: 48,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              color: color ?? Theme.of(context).colorScheme.surface,
              borderRadius: BorderRadius.circular(6),
              border: Border.all(
                color: Theme.of(context).colorScheme.outlineVariant,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 1,
                  offset: const Offset(0, 1),
                ),
              ],
            ),
            child: Text(
              label,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
        ),
      ),
    );
  }
}
