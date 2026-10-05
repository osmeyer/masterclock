# Security report

**Updated:** 2026-10-05 21:53:34 UTC

## Context

This report gives what bandit finds over `scripts`, `src` and `tests`, at every severity, and how many comments silence a finding.
The project requires bandit to find nothing of high severity, and every commit checks that, with bandit's check for `assert` left out for the tests, where pytest needs it; this report also counts what the commit check leaves out.
`scripts/reports.py . security` reruns bandit and rewrites the block below; the report is written only when someone runs it.

## Measured

<!-- measured: security -->

| High | Medium | Low |
| --- | --- | --- |
| 0 | 0 | 2035 |

| Test | Severity | Findings |
| --- | --- | --- |
| B101 | low | 2027 |
| B311 | low | 1 |
| B403 | low | 1 |
| B404 | low | 5 |
| B603 | low | 1 |

No finding is above low severity.

Findings silenced by a comment: 17.

<!-- end measured -->

## Analysis

bandit finds nothing of high or medium severity, so the commit check, which fails only on high severity, passes with room to spare.
Nearly every low finding is `B101`, an `assert` in the tests, where pytest needs it; the commit check leaves that test out for the tests folder.
The other low findings are the use of `subprocess` and `pickle`, and a seeded random generator in a test.
Each is the project doing what it must: running its own tools and das_processor, starting worker processes, which pickle what they exchange, and making repeatable test data.
None takes input from outside the project; das_processor's worker processes exchange data only with the process that started them.
Some of those lines silence ruff's check but not bandit's, though they are the same finding: `tests/masterclock/app/test_lock.py` (`S603`, bandit `B603`) and `tests/scripts/test_characterize.py` (`S311`, bandit `B311`).

## Recommendations

- Add the matching `# nosec` to the two test lines that silence only ruff's check of the same finding, as the project's rule asks.
- Leave the other low findings unsilenced: they are known, harmless here, and the commit check ignores them.
