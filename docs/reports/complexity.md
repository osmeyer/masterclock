# Complexity report

**Updated:** 2026-10-06 00:39:29 UTC

## Context

This report gives how complex the code of `scripts` and `src` is: radon's rank of every function, method and class, by its McCabe complexity, the blocks at the highest rank the project allows, the average, every module's maintainability rank, and whether xenon passes.
The project allows a block at most rank B, a module and the average at most rank A, as xenon checks at every commit, and ruff allows a function a McCabe complexity of at most 12.
`scripts/reports.py . complexity` reruns radon and xenon and rewrites the block below; the report is written only when someone runs it.

## Measured

<!-- measured: complexity -->

| Block rank | Blocks |
| --- | --- |
| A | 536 |
| B | 103 |

Blocks at rank B, the highest a block may have:

| Where | Block | Complexity |
| --- | --- | --- |
| `scripts/characterize.py:259` | `split_at_cold_starts` | 8 |
| `scripts/characterize.py:384` | `drop_unusable_days` | 6 |
| `scripts/characterize.py:486` | `_is_outlier` | 7 |
| `scripts/characterize.py:627` | `_eliminate` | 6 |
| `scripts/characterize.py:698` | `_quadratic` | 7 |
| `scripts/characterize.py:1034` | `_normal_equations` | 6 |
| `scripts/characterize.py:1073` | `_scaled_columns` | 7 |
| `scripts/characterize.py:1126` | `_weighted_fit` | 6 |
| `scripts/characterize.py:1163` | `nonnegative_fit` | 7 |
| `scripts/characterize.py:1197` | `fit_noise_model` | 9 |
| `scripts/characterize.py:1298` | `gap_limit` | 8 |
| `scripts/characterize.py:1486` | `local_triples` | 6 |
| `scripts/characterize.py:1513` | `format_result` | 8 |
| `scripts/check_docstrings.py:71` | `check_files` | 6 |
| `scripts/clean.py:55` | `_cache_folders` | 6 |
| `scripts/clean.py:72` | `generated_paths` | 7 |
| `scripts/document_content.py:329` | `settings_table` | 10 |
| `scripts/document_content.py:417` | `flag_table` | 7 |
| `scripts/document_content.py:484` | `error_table` | 6 |
| `scripts/document_content.py:561` | `attribute_docstring` | 6 |
| `scripts/document_content.py:595` | `written_value` | 10 |
| `scripts/documents.py:336` | `date_problem` | 6 |
| `scripts/documents.py:402` | `check_documents` | 6 |
| `scripts/documents.py:448` | `write_documents` | 7 |
| `scripts/documents.py:500` | `main` | 6 |
| `scripts/epoch_timing.py:99` | `measured_pairs_for` | 6 |
| `scripts/epoch_timing.py:156` | `build_deployment` | 8 |
| `scripts/reports.py:259` | `lint_block` | 8 |
| `scripts/reports.py:334` | `security_block` | 7 |
| `scripts/reports.py:393` | `complexity_block` | 9 |
| `scripts/reports.py:628` | `main` | 6 |
| `src/masterclock/app/config.py:195` | `_checked` | 8 |
| `src/masterclock/app/config.py:246` | `_unknown_names` | 7 |
| `src/masterclock/app/config.py:596` | `merge` | 7 |
| `src/masterclock/app/exceptions.py:115` | `describe_error` | 6 |
| `src/masterclock/app/lock.py:84` | `RunLock.__init__` | 6 |
| `src/masterclock/das_processor/__init__.py:185` | `_run` | 6 |
| `src/masterclock/das_processor/clock_config.py:339` | `RmsLimits` | 9 |
| `src/masterclock/das_processor/clock_config.py:365` | `RmsLimits._check_names` | 8 |
| `src/masterclock/das_processor/clock_config.py:543` | `ClockConfig._type_default` | 8 |
| `src/masterclock/das_processor/clock_config.py:589` | `ClockConfig._default_of` | 7 |
| `src/masterclock/das_processor/clock_config.py:733` | `ClockConfig.locations_at` | 6 |
| `src/masterclock/das_processor/clock_config.py:767` | `ClockConfig.disabled_at` | 6 |
| `src/masterclock/das_processor/files.py:284` | `MeasRecord` | 8 |
| `src/masterclock/das_processor/files.py:308` | `MeasRecord.__post_init__` | 7 |
| `src/masterclock/das_processor/files.py:624` | `_joined` | 6 |
| `src/masterclock/das_processor/files.py:1567` | `check_file` | 8 |
| `src/masterclock/das_processor/files.py:2041` | `write_buffer` | 9 |
| `src/masterclock/das_processor/files.py:2099` | `write_final` | 6 |
| `src/masterclock/das_processor/files.py:2159` | `_prepared` | 8 |
| `src/masterclock/das_processor/files.py:2199` | `_check_existing` | 6 |
| `src/masterclock/das_processor/files.py:2647` | `read_journal` | 6 |
| `src/masterclock/das_processor/read_cd5m5m.py:485` | `parse_line` | 6 |
| `src/masterclock/das_processor/read_cd5m5m.py:749` | `read_measurements` | 6 |
| `src/masterclock/das_processor/read_cd5m5m.py:886` | `find_data_files` | 6 |
| `src/masterclock/das_processor/read_cd5m5m.py:936` | `read_all_blocks` | 6 |
| `src/masterclock/das_processor/read_steering.py:119` | `SteeringFiles` | 6 |
| `src/masterclock/das_processor/read_steering.py:185` | `SteeringFiles._read_new_lines` | 10 |
| `src/masterclock/das_processor/registry.py:115` | `build_registry` | 9 |
| `src/masterclock/das_processor/registry.py:223` | `series_key_of` | 6 |
| `src/masterclock/das_processor/registry.py:259` | `_series_keys` | 7 |
| `src/masterclock/das_processor/run.py:183` | `build_epoch` | 8 |
| `src/masterclock/das_processor/run.py:364` | `_configured_block` | 7 |
| `src/masterclock/das_processor/run.py:400` | `_configured_series` | 7 |
| `src/masterclock/das_processor/run.py:630` | `_innovations` | 7 |
| `src/masterclock/das_processor/run.py:698` | `start_pairs` | 9 |
| `src/masterclock/das_processor/run.py:1067` | `component_of` | 6 |
| `src/masterclock/das_processor/run.py:1389` | `_roll_back_all` | 9 |
| `src/masterclock/das_processor/run.py:1443` | `_kept_epochs` | 6 |
| `src/masterclock/das_processor/run.py:1570` | `process_epoch` | 7 |
| `src/masterclock/das_processor/run.py:1747` | `run` | 10 |
| `src/masterclock/das_processor/run.py:1930` | `log_screening` | 7 |
| `src/masterclock/das_processor/run.py:1971` | `_log_series` | 6 |
| `src/masterclock/das_processor/run.py:2002` | `_log_changes` | 7 |
| `src/masterclock/das_processor/workers.py:489` | `SeriesShard._done` | 7 |
| `src/masterclock/das_processor/workers.py:675` | `WorkerPool.process_epoch` | 9 |
| `src/masterclock/das_processor/workers.py:782` | `WorkerPool._finish_pairs` | 6 |
| `src/masterclock/das_processor/workers.py:877` | `WorkerPool._start_pairs` | 10 |
| `src/masterclock/das_processor/workers.py:957` | `WorkerPool._answers` | 7 |
| `src/masterclock/domain/double_difference.py:218` | `_remote` | 7 |
| `src/masterclock/domain/filter.py:79` | `gains` | 7 |
| `src/masterclock/domain/filter.py:411` | `finish` | 8 |
| `src/masterclock/domain/filter.py:766` | `classify` | 10 |
| `src/masterclock/domain/filter.py:987` | `accept_step` | 8 |
| `src/masterclock/domain/filter.py:1076` | `acquire` | 6 |
| `src/masterclock/domain/filter.py:1167` | `FilterInput` | 6 |
| `src/masterclock/domain/filter.py:1208` | `FilterInput.__post_init__` | 7 |
| `src/masterclock/domain/filter.py:1274` | `filter_step` | 9 |
| `src/masterclock/domain/filter.py:1364` | `_gate` | 10 |
| `src/masterclock/domain/screening.py:235` | `_self_test` | 8 |
| `src/masterclock/domain/screening.py:332` | `_bad_directions` | 7 |
| `src/masterclock/domain/screening.py:376` | `_closure` | 7 |
| `src/masterclock/domain/screening.py:397` | `_triangles` | 8 |
| `src/masterclock/domain/series.py:269` | `_check_numbers` | 6 |
| `src/masterclock/domain/series.py:305` | `_check_rejects` | 6 |
| `src/masterclock/domain/series.py:334` | `_check_flags` | 10 |
| `src/masterclock/domain/series.py:366` | `_check_state` | 7 |
| `src/masterclock/domain/series.py:402` | `_check_model` | 8 |
| `src/masterclock/domain/slips.py:94` | `slip_check` | 6 |
| `src/masterclock/domain/slips.py:184` | `_check_clock` | 8 |
| `src/masterclock/domain/slips.py:283` | `_ds` | 6 |
| `src/masterclock/domain/slips.py:325` | `_attribute_many` | 8 |
| `src/masterclock/domain/slips.py:350` | `_attribute_two` | 6 |

Average complexity of a block: 3.04.

| Module rank | Modules |
| --- | --- |
| A | 40 |
| B | 3 |

xenon passes: no block over rank B, no module or average over rank A.

<!-- end measured -->

## Analysis

xenon passes, so no block, module or average is over the project's limits.
Most blocks are rank A; the rest are rank B, the most a block may have, with a McCabe complexity of 6 to 10, under ruff's limit of 12.
The rank B blocks are where the program makes its decisions: the filter step and the gate, the slip and screening checks, the clock configuration's rules, the run's epoch loop, and the readers of the input files.
Three modules have a maintainability rank of B, below the A that most have; radon's maintainability rank is reported here but no limit applies to it.

## Recommendations

- Keep new code within rank A where it can be, and split a block that would go over rank B along the steps of its algorithm, as the design's pseudocode already does.
- No change is needed for the limits to hold.
