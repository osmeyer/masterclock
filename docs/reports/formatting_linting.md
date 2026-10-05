# Formatting and linting report

**Updated:** 2026-10-05 21:53:34 UTC

## Context

This report gives what ruff finds in the project: the files `ruff format` would change, the findings of `ruff check` by rule, and every comment that silences a finding, a `noqa` for ruff or a `nosec` for bandit, where it is.
The project requires ruff to find nothing and every silencing comment to name the rule it silences and say why, and every commit checks the first.
`scripts/reports.py . lint` reruns ruff, reads every Python file git tracks, and rewrites the block below; the report is written only when someone runs it.

## Measured

<!-- measured: lint -->

| Files checked | Files to reformat |
| --- | --- |
| 104 | 0 |

Every file is formatted.

ruff found nothing.

| Where | Comment |
| --- | --- |
| `scripts/check_test_modules_alone.py:57` | `# noqa: S603  # nosec B603` |
| `scripts/documents.py:45` | `# nosec B404` |
| `scripts/documents.py:243` | `# noqa: S603  # nosec B603` |
| `scripts/epoch_timing.py:31` | `# nosec B404` |
| `scripts/epoch_timing.py:265` | `# noqa: S603  # nosec B603` |
| `scripts/pdfs.py:26` | `# nosec B404 - runs Chrome, given by its full path` |
| `scripts/pdfs.py:186` | `# noqa: S603  # nosec B603` |
| `scripts/reports.py:45` | `# nosec B404 - runs the project's own tools` |
| `scripts/reports.py:49` | `# nosec B405 - reads only pytest's own file` |
| `scripts/reports.py:131` | `# noqa: S603  # nosec B603 B607 - uv, as checks run` |
| `scripts/reports.py:132` | `# noqa: S607 - uv, as checks run` |
| `scripts/reports.py:168` | `# noqa: S314  # nosec B314 - pytest's own file` |
| `scripts/reports.py:169` | `# noqa: S101 - pytest always writes one suite` |
| `scripts/reports.py:195` | `# noqa: S101 - coverage's JSON holds a mapping` |
| `scripts/reports.py:317` | `# nosec B603 B607 - git, as a user runs it` |
| `scripts/reports.py:318` | `# noqa: S607 - git, as a user runs it` |
| `scripts/reports.py:351` | `# noqa: S101 - bandit's JSON holds a list` |
| `scripts/reports.py:352` | `# noqa: S101 - and a mapping` |
| `scripts/reports.py:381` | `# noqa: S101 - bandit's JSON holds a mapping` |
| `src/masterclock/domain/phase.py:205` | `# noqa: RUF046 - round() of an mpq gives an mpz, not an int` |
| `tests/masterclock/app/test_lock.py:57` | `# noqa: S603 - fixed arguments, this interpreter` |
| `tests/masterclock/das_processor/test_determinism.py:290` | `# noqa: S603  # nosec B603` |
| `tests/scripts/test_characterize.py:85` | `# noqa: S311 - repeatable test data, not a secret` |
| `tests/scripts/test_documents.py:63` | `# noqa: S603  # nosec B603` |
| `tests/scripts/test_documents.py:247` | `# noqa: S603  # nosec B603` |

<!-- end measured -->

## Analysis

ruff leaves every file as it is and finds nothing, with the project's full rule set.
Every suppression names the rule it silences, and where ruff and bandit check the same thing, one line carries both.
The project also requires each to give its reason.
Most give it on the same line or in a comment on the line above; `import subprocess  # nosec B404` in `scripts/documents.py` and `scripts/epoch_timing.py` gives none.
`src` holds a single suppression, in `domain/phase.py`, where ruff's advice would make a value of the wrong type.

## Recommendations

- Give the two `import subprocess` lines in `scripts/documents.py` and `scripts/epoch_timing.py` their reason, as the other scripts do: what the script runs.
- Keep `src` free of suppressions where the code can be changed instead.
