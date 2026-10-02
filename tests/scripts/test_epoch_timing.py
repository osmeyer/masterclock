"""Tests for scripts/epoch_timing.py.

The rules covered: the deployment holds every reference against itself and
every other reference, and each clock against one reference in turn; its
DAS files hold one block per epoch with every pair once and no line the
reader refuses, however many pairs there are, and a clock configuration
entry for every clock; the timed runs write the deployment's files, one row
per run of one epoch and one per epoch of the batch; a failed run is
reported with exit status 1; and a folder that is not new or empty, a
number out of range, or more runs than epochs is a usage error.
"""

import logging
import runpy
import sys
from pathlib import Path

import pytest

import epoch_timing
from masterclock.das_processor.clock_config import read_clock_config
from masterclock.das_processor.read_cd5m5m import read_all_blocks


def test_every_reference_meets_every_reference_and_each_clock_one() -> None:
    """Give R * R reference pairs, then each clock against reference n mod R."""
    assert epoch_timing.measured_pairs_for(2, 3) == [
        ("mc0", "mc0"),
        ("mc0", "mc1"),
        ("mc1", "mc0"),
        ("mc1", "mc1"),
        ("mc0", "hm0000"),
        ("mc1", "hm0001"),
        ("mc0", "hm0002"),
    ]


def test_the_deployment_holds_every_pair_once_per_epoch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Write one block per epoch, every pair once, no line refused."""
    das_arguments = epoch_timing.build_deployment(tmp_path, 3, 4, 5)
    measured_pairs = epoch_timing.measured_pairs_for(3, 4)
    with caplog.at_level(logging.WARNING):
        das_blocks = list(read_all_blocks(tmp_path / "das"))
    assert not caplog.records
    assert [das_block.interpolated_datetime for das_block in das_blocks] == [
        epoch_timing.DATA_START + epoch_index * epoch_timing.EPOCH_LENGTH
        for epoch_index in range(5)
    ]
    for das_block in das_blocks:
        assert [
            (das_measurement.reference, das_measurement.clock)
            for das_measurement in das_block.measurements
        ] == measured_pairs
    clock_config = read_clock_config(tmp_path / "clock_config.yaml")
    for _, clock in measured_pairs:
        assert clock_config.entry_for(clock, epoch_timing.DATA_START) is not None
    assert das_arguments[das_arguments.index("--processed-path") + 1] == str(
        tmp_path / "processed"
    )


def test_a_large_deployment_stays_clear_of_the_epoch_s_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep the last of many measurements out of the epoch's refused last seconds."""
    epoch_timing.build_deployment(tmp_path, 10, 200, 1)
    with caplog.at_level(logging.WARNING):
        (das_block,) = read_all_blocks(tmp_path / "das")
    assert not caplog.records
    assert len(das_block.measurements) == 10 * 10 + 200


def data_file_rows(timing_folder: Path) -> dict[str, int]:
    """Give the rows of each data file a run wrote."""
    return {
        data_file.name: sum(
            1 for row_line in data_file.read_text().splitlines() if row_line[0] != "#"
        )
        for data_file in sorted((timing_folder / "processed").rglob("das_a.*.dat"))
    }


def test_the_timed_runs_write_the_deployment_s_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run one epoch per process, and the batch, and print each time."""
    timing_folder = tmp_path / "timing"
    assert epoch_timing.main(
        [str(timing_folder), "--references", "2", "--clocks", "1", "--epochs", "4",
         "--runs", "3"]
    ) == 0  # fmt: skip
    stepped_rows, batch_rows = (
        data_file_rows(timing_folder / "stepped"),
        data_file_rows(timing_folder / "batch"),
    )
    assert len(stepped_rows) == (2 * 2 + 1) + 2 * (2 * 2 + 1)
    assert set(stepped_rows.values()) == {3}
    assert batch_rows.keys() == stepped_rows.keys()
    assert set(batch_rows.values()) == {4}
    printed_output = capsys.readouterr().out
    assert "5 pair files, 10 triple files" in printed_output
    assert "median of the next 2" in printed_output
    assert "batch run of 4 epochs" in printed_output
    assert "% of 600 s" in printed_output


def test_one_run_of_one_epoch_has_no_median(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the first run alone when only one is timed."""
    timing_folder = tmp_path / "timing"
    assert epoch_timing.main(
        [str(timing_folder), "--references", "1", "--clocks", "0", "--epochs", "1",
         "--runs", "1"]
    ) == 0  # fmt: skip
    printed_output = capsys.readouterr().out
    assert "first, which creates the files" in printed_output
    assert "median" not in printed_output


def test_a_failed_run_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 with the run's error output when a run fails."""
    failing_command = (sys.executable, "-c", "import sys; sys.exit('no data here')")
    monkeypatch.setattr(epoch_timing, "DAS_PROCESSOR_COMMAND", failing_command)
    timing_folder = tmp_path / "timing"
    assert epoch_timing.main([str(timing_folder), "--epochs", "1", "--runs", "1"]) == 1
    assert "exit status 1: no data here" in capsys.readouterr().err


def test_an_empty_folder_is_taken(tmp_path: Path) -> None:
    """Build in a folder that is there but empty."""
    assert epoch_timing.main(
        [str(tmp_path), "--references", "1", "--clocks", "0", "--epochs", "1",
         "--runs", "1"]
    ) == 0  # fmt: skip


@pytest.mark.parametrize(
    "cli_arguments",
    [
        ["--references", "0"],
        ["--references", "11"],
        ["--clocks", "-1"],
        ["--clocks", "many"],
        ["--epochs", "145"],
        ["--runs", "0"],
        ["--epochs", "2", "--runs", "3"],
    ],
)
def test_a_number_out_of_range_is_a_usage_error(
    tmp_path: Path, cli_arguments: list[str]
) -> None:
    """Exit 2 for a count out of range, or more runs than epochs."""
    with pytest.raises(SystemExit) as program_exit:
        epoch_timing.main([str(tmp_path / "timing"), *cli_arguments])
    assert program_exit.value.code == 2


@pytest.mark.parametrize("folder_contents", ["file", "full folder"])
def test_a_folder_in_use_is_a_usage_error(tmp_path: Path, folder_contents: str) -> None:
    """Exit 2 for a file, or a folder that holds something."""
    timing_folder = tmp_path / "timing"
    if folder_contents == "file":
        timing_folder.write_text("x")
    else:
        (timing_folder / "kept").mkdir(parents=True)
    with pytest.raises(SystemExit) as program_exit:
        epoch_timing.main([str(timing_folder)])
    assert program_exit.value.code == 2


def test_running_the_file_as_a_script_exits_with_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a program."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["epoch_timing.py", str(tmp_path / "timing"), "--references", "1",
         "--clocks", "0", "--epochs", "1", "--runs", "1"],
    )  # fmt: skip
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(epoch_timing.__file__, run_name="__main__")
    assert program_exit.value.code == 0
