# Performance report

**Updated:** 2026-10-05 21:44:06 UTC

## Context

This report gives how long das_processor takes, against the 600 s between two runs of the scheduler.
`scripts/reports.py . performance --timing-folder FOLDER` runs `scripts/epoch_timing.py` in a new or empty folder, which must be on the disk the real runs write to, since flushing every file is much of a run's time.
The timing script makes up a deployment of the laboratory's size, 5 references and 150 clocks, every reference measuring every clock and steered every hour, writes a day of made-up DAS data for it, and runs das_processor as the scheduler does, one epoch a run, logging at INFO, and then once over the whole day.
It differs from the laboratory in its data: the readings follow made-up clocks with little noise, all clocks are in one building, and nothing is missing or faulty, so no series goes dormant or steps after it starts.
The report is written only when someone runs it.

## Measured

<!-- measured: performance -->

- deployment: 5 references, 150 clocks each measured by every reference, 144 epochs: 775 pair files, 3875 triple files; every reference steered hourly, logging at INFO
- run of one epoch, first, which creates the files: 1.823 s, 0.304% of 600 s
- run of one epoch, median of the next 5: 3.420 s, 0.570% of 600 s
- run of one epoch, longest of the next 5: 8.771 s, 1.462% of 600 s
- batch run of 144 epochs: 24.474 s
- batch run, per epoch: 0.170 s, 0.028% of 600 s

<!-- end measured -->

## Analysis

A run of one epoch, as the scheduler starts it every ten minutes, takes a few seconds, well under 1% of the 600 s it has.
The first run is quicker than the next ones: every series is new, and no triple has a measurement until its pairs have started, so it writes only the pairs' files, a sixth of the files a later run writes.
The later runs vary: the longest takes more than twice the median, on the same data, so the time depends on more than the work done; this report does not break a run's time down to say where it goes.
A whole day in one run takes well under a second per epoch, so catching up after an outage is quick.
The deployment's data are made up, as the context says: on the laboratory's data, dormant series and gaps change how many rows each epoch writes.

## Recommendations

- No change is needed: a scheduled run stays far inside its ten minutes.
- If the laboratory adds many more clocks, measure again: the time per run grows with the number of files.
- Run this report on the machine and disk that run das_processor in operation, which is what its numbers are for.
