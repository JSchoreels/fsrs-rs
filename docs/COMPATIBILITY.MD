# Compatibility Notes

## Default Constructor

- `FSRS::default()` and `FSRS::new(&[])` keep legacy behavior and instantiate
  the FSRS-6 default parameter set.
- This preserves SM-2 bridge behavior through `memory_state_from_sm2()`.

## SM-2 Bridge

- `memory_state_from_sm2()` uses legacy FSRS-6 bridge slots (`w8`, `w9`,
  `w10`, `w20`).
- For compatibility, callers that rely on the default constructor continue to
  get the historical FSRS-6 conversion behavior.
