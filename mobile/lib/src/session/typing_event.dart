class TypingEvent {
  const TypingEvent({
    required this.key,
    required this.pressTime,
    required this.releaseTime,
  }) : assert(releaseTime >= pressTime, 'releaseTime must be >= pressTime');

  final String key;
  final double pressTime;
  final double releaseTime;

  double get holdTime => releaseTime - pressTime;

  Map<String, Object> toJson() {
    return {
      'key': key,
      'press_time': pressTime,
      'release_time': releaseTime,
      'hold_time': holdTime,
    };
  }
}
