# Mutation report

**Updated:** 2026-10-06 00:44:09 UTC

## Context

This report gives the last mutation run: mutmut changes the code of `src` one small change at a time, each change a *mutant*, and runs the tests that reach it; a mutant the tests fail on is *killed*, and one they all pass with *survived*.
A surviving mutant shows a change to the code that no test notices.
The project requires one of two things for each survivor: a test, when the change would alter behaviour someone could see; otherwise, the reason it survives, recorded below.
mutmut is run by hand, `uv run --frozen mutmut run` and then `uv run --frozen mutmut export-cicd-stats`; `scripts/reports.py . mutation` then reads that run's counts and survivors and rewrites the block below, without running mutmut itself.

## Measured

<!-- measured: mutation -->

| Mutants | Killed | Survived | No tests | Timed out | Suspicious | Skipped |
| --- | --- | --- | --- | --- | --- | --- |
| 6852 | 6785 | 61 | 0 | 6 | 0 | 0 |

| Module | Function | Survived |
| --- | --- | --- |
| `masterclock.app.config` | `as_on_command_line` | 5 |
| `masterclock.app.config` | `read_entries` | 3 |
| `masterclock.app.lock` | `RunLock._holder_pid` | 1 |
| `masterclock.app.log` | `MasterClockLogger.todo` | 4 |
| `masterclock.das_processor.files` | `_truncate` | 2 |
| `masterclock.das_processor.files` | `_width` | 5 |
| `masterclock.das_processor.files` | `read_last_record` | 5 |
| `masterclock.das_processor.workers` | `RecordCapture.emit` | 1 |
| `masterclock.das_processor.workers` | `SeriesShard.__init__` | 4 |
| `masterclock.das_processor.workers` | `SeriesShard._done` | 9 |
| `masterclock.das_processor.workers` | `SeriesShard._keep_newest` | 3 |
| `masterclock.das_processor.workers` | `SeriesShard._started` | 2 |
| `masterclock.das_processor.workers` | `SeriesShard.finish_pairs` | 7 |
| `masterclock.das_processor.workers` | `SeriesShard.start_pairs` | 6 |
| `masterclock.das_processor.workers` | `SeriesShard.work_triples` | 3 |
| `masterclock.das_processor.workers` | `serve` | 1 |

<!-- end measured -->

## Analysis

Nearly every mutant was killed.
The few that timed out made the tests run past mutmut's time limit, so they were noticed too.
Each surviving mutant falls into one of the groups below, and for each group the reason it survives is recorded here, as the project requires; none is a change someone could see that no test checks.
Ten mutants of the run before this one were such changes: a day at the end of the range of DAS days, the counts and wording of the roll-back's log line, and the wording of a logging error.
Each now has a test that fails with the mutant in place, and this run kills them.

| Where | Why it survives |
| --- | --- |
| `das_processor.workers`, every function | This code runs only in worker processes, which import das_processor from the project's own source, not mutmut's changed copy, so the mutant never runs. Applied to the source, such mutants fail the worker tests, which compare every file and log line with a run without workers. |
| `das_processor.files._width`, `app.config.as_on_command_line` | This code runs only when its module is imported, to make constants and settings types, before mutmut switches a mutant on. The column widths and the settings it makes are checked by the format and settings tests. |
| `das_processor.files.read_last_record`, `das_processor.files._truncate` | The mutant drops the error's chained cause, which only a traceback shows; the error and its message are the same. |
| `app.log.MasterClockLogger.todo` | The tests that check which caller a TODO record names skip themselves under mutmut, which adds a call frame to every function it changes. Applied to the source, such a mutant fails those tests. |
| `das_processor.files.read_last_record`, `app.lock.RunLock._holder_pid`, `app.config.read_entries` (`"ascii"` to `"ASCII"`, `"utf-8"` to `"UTF-8"`) | A codec's name is the same in any case, so nothing changes. |
| `app.config.read_entries` (`default_section` set to `None`) | No section header can name `None`, so the parser treats a `[DEFAULT]` section as an ordinary section, as it does now, and the settings refuse it as an unknown section either way. |
| `app.config.read_entries` (no `encoding`) | The file is then read in the locale's encoding, which is UTF-8 wherever das_processor runs; only a machine with another locale would read a non-ASCII settings file differently. |

## Recommendations

- Keep the worker tests comparing every file and log line with a run without workers: they are what checks the code mutmut cannot reach.
- When mutmut is run next, read every new survivor's change with `uv run --frozen mutmut show NAME`, and give it a test or its reason here.
