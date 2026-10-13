import 'package:flutter/material.dart';
import '../ml/touchscreen_inference_engine.dart';
import '../ml/touchscreen_nopressure_inference_engine.dart';
import '../ml/touchscreen_nopressure_model_parameters.dart';
import 'custom_touchscreen_keyboard.dart';
import 'touchscreen_session_controller.dart';

class TouchscreenScreen extends StatefulWidget {
  const TouchscreenScreen({
    super.key,
    this.controller,
    this.inferenceEngine,
    this.noPressureInferenceEngine,
  });

  final TouchscreenSessionController? controller;
  final TouchscreenInferenceEngine? inferenceEngine;
  final TouchscreenNoPressureInferenceEngine? noPressureInferenceEngine;

  @override
  State<TouchscreenScreen> createState() => _TouchscreenScreenState();
}

class _TouchscreenScreenState extends State<TouchscreenScreen> {
  late final TouchscreenSessionController _controller;
  final ScrollController _scrollController = ScrollController();
  TouchscreenInferenceEngine? _engine;
  TouchscreenNoPressureInferenceEngine? _noPressureEngine;

  bool _isLoadingModel = false;
  String? _modelLoadError;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? TouchscreenSessionController();
    if (widget.noPressureInferenceEngine != null) {
      _noPressureEngine = widget.noPressureInferenceEngine;
    } else if (widget.inferenceEngine != null) {
      _engine = widget.inferenceEngine;
    } else {
      _loadModelParameters();
    }
  }

  Future<void> _loadModelParameters() async {
    setState(() {
      _isLoadingModel = true;
      _modelLoadError = null;
    });

    try {
      final jsonStr = await DefaultAssetBundle.of(context)
          .loadString('assets/models/touchscreen_nopressure_model.json');
      final params = TouchscreenNoPressureModelParameters.fromJsonString(jsonStr);
      if (mounted) {
        setState(() {
          _noPressureEngine = TouchscreenNoPressureInferenceEngine(params);
          _isLoadingModel = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _modelLoadError = 'Failed to load screening model: ${e.toString()}';
          _isLoadingModel = false;
        });
      }
    }
  }


  @override
  void dispose() {
    _scrollController.dispose();
    if (widget.controller == null) {
      _controller.dispose();
    }
    super.dispose();
  }

  void _resetScrollPosition() {
    if (_scrollController.hasClients) {
      _scrollController.jumpTo(0.0);
    }
  }

  void _onStartSession() {
    setState(() {
      _controller.startSession(onCountdownComplete: () {
        if (mounted) setState(() {});
      });
    });
  }

  void _onFinishSession() {
    if (_noPressureEngine == null && _engine == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Screening model is not initialized.')),
      );
      return;
    }

    setState(() {
      if (_noPressureEngine != null) {
        _controller.completeNoPressureSession(_noPressureEngine!);
      } else if (_engine != null) {
        _controller.completeSession(_engine!);
      }
      _resetScrollPosition();
    });
  }


  void _onRetry() {
    setState(() {
      _controller.reset();
      _resetScrollPosition();
    });
  }

  @override
  Widget build(BuildContext context) {
    final state = _controller.snapshot();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Touchscreen Motor Pattern Screening'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          controller: _scrollController,
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_isLoadingModel)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 24),
                  child: Center(child: CircularProgressIndicator()),
                )
              else if (_modelLoadError != null)
                Card(
                  color: Theme.of(context).colorScheme.errorContainer,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(
                      _modelLoadError!,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.onErrorContainer,
                      ),
                    ),
                  ),
                )
              else ...[
                if (state.isIdle) _buildIdleView(context),
                if (state.isCountdown) _buildCountdownView(context, state),
                if (state.isCollecting) _buildCollectingView(context, state),
                if (state.isProcessing) _buildProcessingView(context),
                if (state.isCompleted) _buildCompletedView(context, state),
                if (state.isError) _buildErrorView(context, state),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildIdleView(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Touchscreen Motor Pattern Screening',
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 12),
        const Text(
          'This experimental screening task measures fine touch timing and pressure '
          'characteristics while you type a short controlled text excerpt.',
          style: TextStyle(fontSize: 15),
        ),
        const SizedBox(height: 12),
        const Card(
          child: Padding(
            padding: EdgeInsets.all(12),
            child: Text(
              'Notice: This screening tool produces an experimental research indicator only. '
              'It is NOT a medical diagnosis and does not provide clinical detection.',
              style: TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
        ),
        const SizedBox(height: 16),
        Text(
          'Passage to type:',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Text(
            kTouchscreenScreeningPassage,
            style: TextStyle(fontSize: 15, height: 1.4),
          ),
        ),
        const SizedBox(height: 24),
        FilledButton(
          onPressed: _onStartSession,
          style: FilledButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          child: const Text('START SESSION', style: TextStyle(fontSize: 16)),
        ),
      ],
    );
  }

  Widget _buildCountdownView(
      BuildContext context, TouchscreenSessionState state) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 48),
      alignment: Alignment.center,
      child: Column(
        children: [
          Text(
            'Get Ready to Type...',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 24),
          Text(
            '${state.countdownSeconds}',
            style: Theme.of(context).textTheme.displayLarge?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: Theme.of(context).colorScheme.primary,
                ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Recording will begin after countdown.',
            style: TextStyle(color: Colors.grey),
          ),
        ],
      ),
    );
  }

  Widget _buildCollectingView(
      BuildContext context, TouchscreenSessionState state) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Session Active',
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    color: Theme.of(context).colorScheme.primary,
                  ),
            ),
            Row(
              children: [
                Chip(
                  label: Text('Taps: ${state.eventCount}'),
                ),
                const SizedBox(width: 8),
                FilledButton.tonal(
                  onPressed: _onFinishSession,
                  child: const Text('FINISH'),
                ),
              ],
            ),
          ],
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Text(
            kTouchscreenScreeningPassage,
            style: TextStyle(fontSize: 15, height: 1.4),
          ),
        ),
        const SizedBox(height: 16),
        // Non-editable custom text display container
        Container(
          constraints: const BoxConstraints(minHeight: 80),
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: Theme.of(context).colorScheme.outline,
            ),
          ),
          child: Text(
            state.typedText.isEmpty
                ? 'Typed passage will appear here as you tap keys on the keyboard below...'
                : state.typedText,
            style: TextStyle(
              fontSize: 16,
              height: 1.3,
              color: state.typedText.isEmpty
                  ? Colors.grey
                  : Theme.of(context).colorScheme.onSurface,
            ),
          ),
        ),
        const SizedBox(height: 12),
        // Prominent Finish button above virtual keyboard
        SizedBox(
          width: double.infinity,
          child: FilledButton(
            onPressed: _onFinishSession,
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
            child: const Text('FINISH SCREENING',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          ),
        ),
        const SizedBox(height: 12),
        // Custom Flutter-owned virtual keyboard automatically active
        CustomTouchscreenKeyboard(
          onPointerDown: (pointerId, pressTimeMs, pressure) {
            _controller.handlePointerDown(
                pointerId, pressTimeMs, pressure);
            setState(() {});
          },
          onPointerUp: (pointerId, releaseTimeMs, pressure) {
            _controller.handlePointerUp(
                pointerId, releaseTimeMs, pressure);
            setState(() {});
          },
          onKeyTap: (char) {
            _controller.handleKeyTap(char);
            if ((_noPressureEngine != null || _engine != null) &&
                _controller.snapshot().eventCount >= kMinPassageEventCount &&
                _controller.snapshot().typedText.length >=
                    kTouchscreenScreeningPassage.length) {
              _onFinishSession();
            } else {
              setState(() {});
            }
          },

        ),
        if (state.userNotice != null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.errorContainer,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.info_outline,
                  color: Theme.of(context).colorScheme.onErrorContainer,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    state.userNotice!,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.onErrorContainer,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildProcessingView(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 48),
      alignment: Alignment.center,
      child: const Column(
        children: [
          CircularProgressIndicator(),
          SizedBox(height: 24),
          Text(
            'Analyzing touch timing & pressure features...',
            style: TextStyle(fontSize: 16),
          ),
        ],
      ),
    );
  }

  Widget _buildCompletedView(
      BuildContext context, TouchscreenSessionState state) {
    final pred = state.prediction;
    if (pred == null) return const SizedBox.shrink();

    final isPD = pred.predictedLabel == 'PD';
    final patternText = isPD
        ? 'Model-estimated pattern: PD-associated'
        : 'Model-estimated pattern: control-associated';

    final explanationText = isPD
        ? 'This experimental model produced a PD-associated pattern.'
        : 'This experimental model produced a control-associated pattern.';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Card(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Motor-pattern screening indicator',
                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const Divider(height: 24),
                Text(
                  patternText,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: isPD ? Colors.orange.shade800 : Colors.teal.shade800,
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  explanationText,
                  style: const TextStyle(fontSize: 15),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: _ProbabilityTile(
                        label: 'PD Probability',
                        value:
                            '${(pred.probabilityPD * 100).toStringAsFixed(1)}%',
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: _ProbabilityTile(
                        label: 'CNT Probability',
                        value:
                            '${(pred.probabilityCNT * 100).toStringAsFixed(1)}%',
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        _buildDisclaimerCard(context),
        const SizedBox(height: 24),
        OutlinedButton(
          onPressed: _onRetry,
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          child: const Text('TRY AGAIN', style: TextStyle(fontSize: 16)),
        ),
      ],
    );
  }

  Widget _buildErrorView(BuildContext context, TouchscreenSessionState state) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Card(
          color: Theme.of(context).colorScheme.errorContainer,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.error_outline,
                      color: Theme.of(context).colorScheme.onErrorContainer,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Screening Error',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color:
                                Theme.of(context).colorScheme.onErrorContainer,
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  state.errorMessage ??
                      'An unexpected error occurred during touch screening.',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onErrorContainer,
                    fontSize: 15,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        _buildDisclaimerCard(context),
        const SizedBox(height: 24),
        OutlinedButton(
          onPressed: _onRetry,
          style: OutlinedButton.styleFrom(
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          child: const Text('TRY AGAIN', style: TextStyle(fontSize: 16)),
        ),
      ],
    );
  }

  Widget _buildDisclaimerCard(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: Theme.of(context).colorScheme.outlineVariant,
        ),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Important Disclaimer',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
          ),
          SizedBox(height: 6),
          Text(
            'This is an experimental research prototype and is not a medical diagnosis. '
            'Results can be affected by typing style, device characteristics, and limited training data.',
            style: TextStyle(fontSize: 13, height: 1.3),
          ),
        ],
      ),
    );
  }
}

class _ProbabilityTile extends StatelessWidget {
  const _ProbabilityTile({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(fontSize: 12, color: Colors.grey),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}
