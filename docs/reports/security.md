# Security report

**Updated:** 2026-10-06 00:39:28 UTC

## Context

This report gives what bandit finds over `scripts`, `src` and `tests`, at every severity, and how many comments silence a finding.
The project requires bandit to find nothing of high severity, and every commit checks that, with bandit's check for `assert` left out for the tests, where pytest needs it; this report also counts what the commit check leaves out.
`scripts/reports.py . security` reruns bandit and rewrites the block below; the report is written only when someone runs it.

## Measured

<!-- measured: security -->

| High | Medium | Low |
| --- | --- | --- |
| 0 | 0 | 2032 |

| Test | Severity | Findings |
| --- | --- | --- |
| B101 | low | 2032 |

No finding is above low severity.

Findings silenced by a comment: 25.

<!-- end measured -->

## Analysis

bandit finds nothing of high or medium severity, so the commit check, which fails only on high severity, passes with room to spare.
Every low finding left is `B101`, an `assert` in the tests, where pytest needs it; the commit check leaves that test out for the tests folder.
Every other finding is silenced by a comment that gives its reason: the scripts and tests run the project's own tools, git, Chrome and das_processor, das_processor's worker processes pickle what they exchange with the process that started them, and a test makes repeatable random data.
None of them takes input from outside the project.

## Recommendations

- No change is needed.
- Keep a new silencing comment to the project's form: its rule, both tools' where they overlap, and its reason.
