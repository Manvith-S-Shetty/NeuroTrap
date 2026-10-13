# Data Quality Investigation

This investigation uses the existing read-only loader and validator. It does not modify raw files, delete rows, train a model, or implement feature extraction.

## 1. Summary

Rows investigated: 177,513 canonical raw records.

Validation issue flags:

```text
missing_key: 9
negative_hold_duration: 1
non_monotonic_press_time: 3504
non_positive_timestamp: 819
release_before_press: 1
suspicious_long_hold: 1173
```

The flags are not mutually exclusive. They correspond to 3,559 unique rows, and 1,154 rows have more than one issue category.

The most important finding is that the raw files are mostly stored in `release_time` order, not `press_time` order. This explains many `non_monotonic_press_time` flags: a long-held key can be pressed earlier but released later, so it appears after shorter keys in file row order.

However, some non-monotonic flags also overlap with clearly inconsistent timestamp records, especially records with negative `press_time` and very large `hold_time`. These should not be treated as ordinary out-of-order rows.

## 2. Non-Monotonic Press Times

Current validation rule: within each recording, flag a row when its `press_time` is less than the previous row's `press_time` in file order.

Counts:

- Total `non_monotonic_press_time` flags: 3,504
- MIT-CS1PD: 1,736
- MIT-CS2PD: 1,768
- Recordings affected: 110 of 116

After sorting each recording by `press_time`, the number of press-time decreases becomes 0. This means the flag is caused by file row order rather than by an impossible-to-sort timestamp series.

Release-time ordering is nearly monotonic:

- `press_time` decreases in file order: 3,504
- `release_time` decreases in file order: 2
- recordings with release-time decreases: 2

This supports the interpretation that raw rows are generally ordered by release time.

Representative file-order sequence:

```text
MIT-CS1PD 1401114972.068_001_014.csv, rows 312-318

row  key      hold_time  press_time  release_time
312  o        0.1174     262.1414    262.2588
313  space    0.1251     262.8671    262.9922
314  D        0.1298     265.0319    265.1617
315  Shift_L  1.0453     264.1920    265.2372  <- press_time decreased
316  o        0.1259     266.0699    266.1958
317  n        0.1168     266.5789    266.6957
318  space    0.1159     266.9066    267.0226
```

Here, `Shift_L` was pressed before row 314 but released after row 314. In release-time order this is sensible; in press-time order it appears non-monotonic.

Another representative example:

```text
MIT-CS2PD 1457345994.1056_001_014.csv, rows 1-6

row  key         hold_time  press_time  release_time
1    exclamdown  0.0601     1.4114      1.4715
2    P           0.0819     1.8624      1.9443
3    Shift_L     0.1622     1.7986      1.9608   <- press_time decreased
4    a           0.0837     2.2373      2.3209
5    r           0.0883     2.3860      2.4743
6    e           0.1061     2.4875      2.5935
```

Sorting this local sequence by `press_time` places `Shift_L` before `P`, which is chronologically sensible.

There are also non-monotonic rows that overlap with invalid-looking timestamps:

```text
MIT-CS2PD 1451987802.1042_001_014.csv, rows 1-5

row  key      hold_time       press_time       release_time
1    Shift_R  0.2646          1.1635           1.4281
2    e        1451987800+     -1451987800+     1.4641       <- non-positive press_time and long hold
3    l        0.1494          1.6881           1.8375
4    space    0.1281          1.8919           2.0200
5    p        0.1263          1.9855           2.1118
```

These are not merely release-order effects. They contain negative press timestamps and extremely large hold durations.

Timestamps can be equal only in the problematic negative timestamp records found here:

- Adjacent equal `press_time` within recording: 4
- Positive repeated `press_time` values within recording: 0
- Adjacent equal `release_time`: 0
- `press_time == release_time` in the same row: 0

## 3. Non-Positive Timestamps

Current validation rule: flag rows where `press_time <= 0` or `release_time <= 0`.

Counts:

- Total `non_positive_timestamp` rows: 819
- MIT-CS1PD: 341
- MIT-CS2PD: 478

Breakdown:

```text
dataset    press_time <= 0  release_time <= 0  count
MIT-CS1PD  true             false              341
MIT-CS2PD  true             false              478
```

All 819 cases affect `press_time` only. No non-positive `release_time` values were found.

Representative examples:

```text
dataset    participant  file                           row   key  hold_time       press_time       release_time
MIT-CS1PD  11           1402930351.011_001_014.csv       2   h    1402930000+     -1402930000+     2.1573
MIT-CS1PD  11           1402930351.011_001_014.csv     395   l    1402930000+     -1402930000+     115.4625
MIT-CS2PD  1042         1451987802.1042_001_014.csv      2   e    1451980000+     -1451980000+     1.4641
```

These do not look like missing values or zero placeholders. They are numeric values where:

```text
hold_time ~= release_time - press_time
```

but `press_time` is a large negative value and `hold_time` is correspondingly huge. This suggests an internal timing/reference mismatch in those rows, not a legitimate relative timestamp.

The affected rows appear across many recordings:

- MIT-CS1PD affected recordings: 41
- MIT-CS2PD affected recordings: 37

## 4. Negative/Invalid Hold Times

There is one row with both a negative hold duration and `release_time < press_time`:

```text
dataset    participant  file                            row  key  hold_time  press_time  release_time  label
MIT-CS2PD  1041         1456752658.1041_001_014.csv     542  a    -0.1137    817.9136    817.7999      PD
```

This row is internally inconsistent for any hold-duration calculation because release occurs before press and the duration is negative.

## 5. Long Holds

Current threshold: `hold_time >= 5.0` seconds.

This threshold comes from the official loader's `sanityCheck`, which treats `dataHT >= 5` as a bad label. The threshold was not changed for this investigation.

Counts:

- Total suspicious long holds: 1,173
- MIT-CS1PD: 480
- MIT-CS2PD: 693

Hold-time percentiles:

```text
MIT-CS1PD
min      0.0042
50%      0.1135
90%      0.2018
95%      0.2478
99%      0.6094
99.5%    7.6695
99.9%    1409238000+
max      1419869000+

MIT-CS2PD
min     -0.1137
50%      0.0977
90%      0.1754
95%      0.2186
99%      1.4602
99.5%    1435047000+
99.9%    1457346000+
max      1463173000+

Combined
min     -0.1137
50%      0.1047
90%      0.1914
95%      0.2354
99%      0.8445
99.5%    268.1977
99.9%    1450890000+
max      1463173000+
```

Long holds split into two qualitatively different groups:

- 819 long-hold rows overlap with non-positive `press_time`.
- 354 long-hold rows have positive press/release timestamps and `release_time >= press_time`.

Valid-timestamp long holds by dataset and label:

```text
dataset    label  count
MIT-CS1PD  CNT    45
MIT-CS1PD  PD     94
MIT-CS2PD  CNT    136
MIT-CS2PD  PD     79
```

Representative valid-timestamp long holds:

```text
dataset    participant  file                           row   key    hold_time  press_time  release_time
MIT-CS1PD  11           1402930351.011_001_014.csv     951   L      149.4457   115.3840    264.8297
MIT-CS1PD  11           1402930351.011_001_014.csv    1069   O       83.7851   211.4228    295.2079
MIT-CS1PD  11           1403706430.011_003_014.csv     510   U      133.7496     4.8324    138.5820
MIT-CS2PD  examples include positive-timestamp holds above 5 seconds across 33 recordings
```

Because some long holds have internally consistent positive timestamps, they should not all be assumed to be corrupt. They may represent stuck keys, long modifier holds, acquisition artifacts, or real but unusual typing behavior. The exact interpretation needs feature-specific handling.

## 6. Missing Keys

There are 9 records with missing `key` values:

```text
dataset    participant  file                            row   hold_time  press_time  release_time  label
MIT-CS1PD  72           1403795516.072_001_014.csv       265  0.1344     160.7253    160.8597      PD
MIT-CS1PD  72           1403795516.072_001_014.csv       372  0.1124     228.6414    228.7538      PD
MIT-CS1PD  72           1403795516.072_001_014.csv       478  0.2740     289.3389    289.6130      PD
MIT-CS1PD  72           1403795516.072_001_014.csv       712  0.1406     431.9163    432.0569      PD
MIT-CS1PD  72           1403795516.072_001_014.csv       740  0.0896     452.0041    452.0937      PD
MIT-CS1PD  72           1403795516.072_001_014.csv       967  0.2382     590.4703    590.7085      PD
MIT-CS1PD  72           1403795516.072_001_014.csv      1009  0.1232     618.8482    618.9714      PD
MIT-CS1PD  81           1404313658.081_001_014.csv       821  0.0951     815.0483    815.1434      CNT
MIT-CS2PD  1061         1463172280.1061_001_014.csv      216  0.0348     125.5417    125.5764      CNT
```

The timing values for missing-key rows are otherwise present and look plausible. These rows are missing key identity, not timing.

One missing-key row also has `non_monotonic_press_time`; the other eight are only `missing_key`.

## 7. Overlapping Issues

Issue overlap by unique raw row:

```text
issues                                                                    rows
(non_monotonic_press_time,)                                                2376
(non_monotonic_press_time, non_positive_timestamp, suspicious_long_hold)    794
(non_monotonic_press_time, suspicious_long_hold)                            333
(non_positive_timestamp, suspicious_long_hold)                               25
(suspicious_long_hold,)                                                      21
(missing_key,)                                                               8
(missing_key, non_monotonic_press_time)                                      1
(negative_hold_duration, release_before_press)                               1
```

Long-hold overlap:

```text
issues                                                                    rows
(non_monotonic_press_time, non_positive_timestamp, suspicious_long_hold)    794
(non_monotonic_press_time, suspicious_long_hold)                            333
(non_positive_timestamp, suspicious_long_hold)                               25
(suspicious_long_hold,)                                                      21
```

Non-positive timestamp overlap:

```text
issues                                                                    rows
(non_monotonic_press_time, non_positive_timestamp, suspicious_long_hold)    794
(non_positive_timestamp, suspicious_long_hold)                               25
```

This overlap matters because a feature calculation should not treat all flags equally. For example, a non-monotonic press-time row caused by a held `Shift_L` key is different from a row with a large negative `press_time`.

## 8. Recommended Handling Strategy

These are recommendations for later feature-extraction design only. No records are removed or changed by this investigation.

Records that should definitely be excluded from a particular feature calculation:

- Rows with missing `key` should be excluded from key-specific or key-category-specific calculations because the key identity is unavailable.
- Rows with missing timing values should be excluded from timing calculations requiring the missing field.
- Rows with `hold_time < 0` or `release_time < press_time` should be excluded from hold-duration calculations.
- Rows with `press_time <= 0` should be excluded from calculations that depend on chronological press timing, inter-key timing, or press-time ordering.

Records that should be retained:

- Rows flagged only for `non_monotonic_press_time` should be retained in the raw normalized dataset. Many are explained by file ordering by release time, especially long-held keys released after later-pressed keys.
- For chronological feature calculations, use an explicit sorted view by `press_time` rather than assuming file row order is press order.
- Missing-key rows may still be retained for global timing coverage if a future feature only needs timing and explicitly permits unknown key identity.

Records that need further consideration:

- Long holds with positive timestamps and `release_time >= press_time` should not be automatically removed. They may be genuine prolonged holds, stuck-key behavior, or acquisition artifacts. They should be handled per feature and likely reported separately.
- Long holds that also have non-positive `press_time` are likely timing-reference anomalies and need separate treatment from valid-timestamp long holds.
- The two rows where `release_time` decreases in file order need special attention if a future step assumes release-time order.
- Equal negative press-time records should not be interpreted as legitimate simultaneous key presses. No positive repeated press times were observed.

For NeuroTap, the safest immediate design rule is to preserve raw records, create explicit filtered/sorted views only inside later feature calculations, and document which validation categories each feature excludes.
