# Test and coverage report

**Updated:** 2026-10-06 00:44:24 UTC

## Context

This report gives the latest run of the project's tests: how many passed, failed or were skipped, every skipped test with its reason, and how much of the code they run, as line and branch coverage of `src`, `scripts` and `tests`, each apart.
The project requires every test to pass and 100% line and branch coverage of each of the three folders, and every commit checks both.
`scripts/reports.py . tests` reruns the tests with coverage and rewrites the block below; the report is written only when someone runs it.

## Measured

<!-- measured: tests -->

| Passed | Failed | Errors | Skipped | Tests | Time (s) |
| --- | --- | --- | --- | --- | --- |
| 2292 | 0 | 0 | 0 | 2292 | 13.8 |

Skipped tests:

No test was skipped.

| Folder | Lines | Branches | Coverage |
| --- | --- | --- | --- |
| src | 3476 of 3476 | 870 of 870 | 100.00% |
| scripts | 1396 of 1396 | 314 of 314 | 100.00% |
| tests | 8785 of 8785 | 452 of 452 | 100.00% |

<!-- end measured -->

## Analysis

Every test passed and none was skipped, so nothing the tests check is left unchecked on this machine.
A few tests skip themselves only when mutmut runs them, as the mutation report says; they ran here.
Each of `src`, `scripts` and `tests` is covered in full, every line and every branch, which is what the project requires.
Full coverage says every line ran under some test, not that every result was checked; the mutation report measures that.
The whole run takes a few seconds, short enough that every commit runs it.

## Recommendations

- Keep a new feature or fix starting with a test that fails for the reason the work exists, as now.
- When a test has to be skipped, give the reason in its skip, so it shows here.
