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
    assert epoch_timing.pairs_of(2, 3) == [
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
    arguments = epoch_timing.build(tmp_path, 3, 4, 5)
    pairs = epoch_timing.pairs_of(3, 4)
    with caplog.at_level(logging.WARNING):
        blocks = list(read_all_blocks(tmp_path / "das"))
    assert not caplog.records
    assert [block.interpolated_datetime for block in blocks] == [
        epoch_timing.DAY + i * epoch_timing.EPOCH for i in range(5)
    ]
    for block in blocks:
        assert [(m.reference, m.clock) for m in block.measurements] == pairs
    clocks = read_clock_config(tmp_path / "clock_config.yaml")
    for _, clock in pairs:
        assert clocks.entry_for(clock, epoch_timing.DAY) is not None
    assert arguments[arguments.index("--processed-path") + 1] == str(
        tmp_path / "processed"
    )


def test_a_large_deployment_stays_clear_of_the_epoch_s_end(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Keep the last of many measurements out of the epoch's refused last seconds."""
    epoch_timing.build(tmp_path, 10, 200, 1)
    with caplog.at_level(logging.WARNING):
        (block,) = read_all_blocks(tmp_path / "das")
    assert not caplog.records
    assert len(block.measurements) == 10 * 10 + 200


def rows(folder: Path) -> dict[str, int]:
    """Give the rows of each data file a run wrote."""
    return {
        path.name: sum(1 for line in path.read_text().splitlines() if line[0] != "#")
        for path in sorted((folder / "processed").rglob("das_a.*.dat"))
    }


def test_the_timed_runs_write_the_deployment_s_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run one epoch per process, and the batch, and print each time."""
    folder = tmp_path / "timing"
    assert epoch_timing.main(
        [str(folder), "--references", "2", "--clocks", "1", "--epochs", "4",
         "--runs", "3"]
    ) == 0  # fmt: skip
    stepped, batch = rows(folder / "stepped"), rows(folder / "batch")
    assert len(stepped) == (2 * 2 + 1) + 2 * (2 * 2 + 1)
    assert set(stepped.values()) == {3}
    assert batch.keys() == stepped.keys()
    assert set(batch.values()) == {4}
    out = capsys.readouterr().out
    assert "5 pair files, 10 triple files" in out
    assert "median of the next 2" in out
    assert "batch run of 4 epochs" in out
    assert "% of 600 s" in out


def test_one_run_of_one_epoch_has_no_median(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the first run alone when only one is timed."""
    folder = tmp_path / "timing"
    assert epoch_timing.main(
        [str(folder), "--references", "1", "--clocks", "0", "--epochs", "1",
         "--runs", "1"]
    ) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "first, which creates the files" in out
    assert "median" not in out


def test_a_failed_run_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 1 with the run's error output when a run fails."""
    failing = (sys.executable, "-c", "import sys; sys.exit('no data here')")
    monkeypatch.setattr(epoch_timing, "COMMAND", failing)
    folder = tmp_path / "timing"
    assert epoch_timing.main([str(folder), "--epochs", "1", "--runs", "1"]) == 1
    assert "exit status 1: no data here" in capsys.readouterr().err


def test_an_empty_folder_is_taken(tmp_path: Path) -> None:
    """Build in a folder that is there but empty."""
    assert epoch_timing.main(
        [str(tmp_path), "--references", "1", "--clocks", "0", "--epochs", "1",
         "--runs", "1"]
    ) == 0  # fmt: skip


@pytest.mark.parametrize(
    "arguments",
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
    tmp_path: Path, arguments: list[str]
) -> None:
    """Exit 2 for a count out of range, or more runs than epochs."""
    with pytest.raises(SystemExit) as stopped:
        epoch_timing.main([str(tmp_path / "timing"), *arguments])
    assert stopped.value.code == 2


@pytest.mark.parametrize("what", ["file", "full folder"])
def test_a_folder_in_use_is_a_usage_error(tmp_path: Path, what: str) -> None:
    """Exit 2 for a file, or a folder that holds something."""
    path = tmp_path / "timing"
    if what == "file":
        path.write_text("x")
    else:
        (path / "kept").mkdir(parents=True)
    with pytest.raises(SystemExit) as stopped:
        epoch_timing.main([str(path)])
    assert stopped.value.code == 2


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
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(epoch_timing.__file__, run_name="__main__")
    assert stopped.value.code == 0
