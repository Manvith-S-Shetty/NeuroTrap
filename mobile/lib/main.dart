import 'package:flutter/material.dart';

import 'src/session/key_event_adapter.dart';
import 'src/session/touchscreen_screen.dart';
import 'src/session/typing_event.dart';
import 'src/session/typing_session_controller.dart';

void main() {
  runApp(const NeuroTapApp());
}

class NeuroTapApp extends StatelessWidget {
  const NeuroTapApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeuroTap',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF006A67)),
        useMaterial3: true,
      ),
      home: const MainNavigationScreen(),
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _tabController.addListener(_handleTabChange);
  }

  void _handleTabChange() {
    if (_tabController.indexIsChanging || mounted) {
      FocusManager.instance.primaryFocus?.unfocus();
      setState(() {});
    }
  }

  @override
  void dispose() {
    _tabController.removeListener(_handleTabChange);
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('NeuroTap'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(
              icon: Icon(Icons.keyboard),
              text: 'Hardware Keyboard',
            ),
            Tab(
              icon: Icon(Icons.touch_app),
              text: 'Touchscreen Screening',
            ),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _tabController.index == 0
              ? const TypingSessionScreen()
              : const SizedBox.shrink(),
          const TouchscreenScreen(),
        ],
      ),
    );
  }
}

class TypingSessionScreen extends StatefulWidget {
  const TypingSessionScreen({super.key});

  @override
  State<TypingSessionScreen> createState() => _TypingSessionScreenState();
}

class _TypingSessionScreenState extends State<TypingSessionScreen> {
  final TypingSessionController _session = TypingSessionController();
  final FocusNode _keyboardFocusNode = FocusNode(
    debugLabel: 'NeuroTap keyboard collector',
  );
  final TextEditingController _textController = TextEditingController();

  late final FlutterKeyEventAdapter _keyEventAdapter = FlutterKeyEventAdapter(
    _session,
  );
  String _status = 'Idle';

  @override
  void dispose() {
    _keyboardFocusNode.dispose();
    _textController.dispose();
    super.dispose();
  }

  void _startSession() {
    setState(() {
      _textController.clear();
      _session.startSession();
      _status = 'Collecting hardware keyboard events';
    });
    _keyboardFocusNode.requestFocus();
  }

  void _stopSession() {
    setState(() {
      _session.stopSession();
      _status = 'Stopped';
    });
  }

  void _handleKeyEvent(KeyEvent event) {
    setState(() {
      _keyEventAdapter.handleKeyEvent(event);
    });
  }

  @override
  Widget build(BuildContext context) {
    final snapshot = _session.snapshot();
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text(
              'Typing Dynamics Session',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            const Text(
              'Research prototype for collecting typing dynamics. This is non-diagnostic and does not provide a medical result.',
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                FilledButton(
                  onPressed: snapshot.isCollecting ? null : _startSession,
                  child: const Text('Start Session'),
                ),
                const SizedBox(width: 12),
                OutlinedButton(
                  onPressed: snapshot.isCollecting ? _stopSession : null,
                  child: const Text('Stop Session'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            _StatusPanel(
              status: _status,
              isCollecting: snapshot.isCollecting,
              eventCount: snapshot.eventCount,
              activePressCount: snapshot.activePressCount,
            ),
            const SizedBox(height: 16),
            KeyboardListener(
              focusNode: _keyboardFocusNode,
              onKeyEvent: _handleKeyEvent,
              child: TextField(
                controller: _textController,
                minLines: 5,
                maxLines: 8,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Typing area',
                  alignLabelWithHint: true,
                ),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'Hardware keyboard key-down/key-up events are captured when available. Software keyboard text changes do not guarantee physical press/release timing.',
            ),
            const SizedBox(height: 16),
            _DebugEventList(events: snapshot.events),
          ],
        ),
      ),
    );
  }
}

class _StatusPanel extends StatelessWidget {
  const _StatusPanel({
    required this.status,
    required this.isCollecting,
    required this.eventCount,
    required this.activePressCount,
  });

  final String status;
  final bool isCollecting;
  final int eventCount;
  final int activePressCount;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Status: $status'),
            Text('Session active: $isCollecting'),
            Text('Captured events: $eventCount'),
            Text('Keys currently down: $activePressCount'),
          ],
        ),
      ),
    );
  }
}

class _DebugEventList extends StatelessWidget {
  const _DebugEventList({required this.events});

  final List<TypingEvent> events;

  @override
  Widget build(BuildContext context) {
    final latest = events.length <= 8
        ? events
        : events.sublist(events.length - 8);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Captured Events', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        if (latest.isEmpty)
          const Text('No captured hardware key events yet.')
        else
          ...latest.map(
            (event) => Text(
              'key=${event.key}, press_time=${event.pressTime.toStringAsFixed(6)}, '
              'release_time=${event.releaseTime.toStringAsFixed(6)}, '
              'hold_time=${event.holdTime.toStringAsFixed(6)}',
            ),
          ),
      ],
    );
  }
}
