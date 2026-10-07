"""Tests for scripts/reject_stretches.py.

The rules covered: a rejection is read from a log line about a rejected
reading of the channel asked for, its clock the last name of the series
and its references the others, any other line and any other channel's
passed over, and a reference's own rejections left out; an epoch at which
at least the number of clocks asked for are rejected is shared, and its
rejections are left out of every clock's stretches; a clock's rejected
epochs fewer than the shortest stretch apart are joined, and a stretch is
kept when it lasts at least the shortest and enough of its epochs hold a
rejection; the clock's disabled stretches are read from its entries in
date order, an entry that says enabled: false counting as disabled, one
never enabled again open-ended to the data end; a stretch is joined with
each disabled stretch fewer than the shortest apart and counts those it
takes in, and a disabled stretch with no stretch of rejections near it is
not reported; the shared epochs are joined the same way and each stretch
of them names the reference its rejections go through most, a reference
counted once for each rejection; an MJD is written as no_signal.py writes
it; and the script prints one line per stretch of a clock, then one per
shared stretch, each under its header line.

The log lines are invented, in the form das_processor writes them.
"""

import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

import reject_stretches
from masterclock.das_processor.clock_config import read_clock_config
from no_signal import Span

FIRST: Final = 8_775_360
"""An invented first epoch, MJD 60940 at 00:00."""

FIRST_TEXT: Final = "2025-09-22"
"""The day of FIRST, as the log writes it."""

CLOCK_CONFIG: Final = """\
rejects_before_restart: 3
reject_fraction_epochs: 25.0
reject_fraction_limit: 0.5
rms_limit: {default: 50}
types:
  maser:
    filter_states: 2
    time_constant: 30.0
    scale_time_constant: 50.0
    initial_innovation_scale: 5.0
    gap_limit: 432
clocks:
  hm1:
    - {type: maser}
    - {effective_mjd: 60940.5, disabled: true}
    - {effective_mjd: 60940.6, disabled: true}
    - {effective_mjd: 60941.0, enabled: true}
    - {effective_mjd: 60941.2, disabled: false}
    - {effective_mjd: 60943.0, enabled: false}
  hm2: [{type: maser}]
"""
"""An invented clock configuration: hm1 disabled from MJD 60940.5 to 60941 and
from 60943 on, with entries that repeat what is already in force."""


def epoch_text(epoch: int) -> str:
    """Write an epoch's start as the log writes it."""
    minutes = (epoch - FIRST) * 10
    day, minute_of_day = divmod(minutes, 24 * 60)
    hour, minute = divmod(minute_of_day, 60)
    return f"2025-09-{22 + day:02d} {hour:02d}:{minute:02d}:00+00:00"


def rejection_line(series: str, epoch: int) -> str:
    """Write the log line of a rejected reading."""
    return (
        "2026-10-07 20:28:43.185 UTC, MJD 61320.853278 | WARNING |"
        f" masterclock.das_processor.run: {series} rejected at {epoch_text(epoch)}:"
        " innovation 39213.7 ps, scale 559.9 ps, 2 consecutive\n"
    )


def test_an_epoch_is_numbered_from_its_start() -> None:
    """Number an epoch start as an MJD times the epochs in a day."""
    assert reject_stretches.epoch_of(f"{FIRST_TEXT} 00:00:00+00:00") == FIRST
    assert reject_stretches.epoch_of(epoch_text(FIRST + 150)) == FIRST + 150


def test_a_rejection_is_read_from_its_line() -> None:
    """Read the epoch, clock and references; pass over other lines and channels."""
    assert reject_stretches.parse_rejection(
        rejection_line("das_a.mc1.mc3.hm1", FIRST + 7), "a"
    ) == reject_stretches.Rejection(FIRST + 7, "hm1", ("mc1", "mc3"))
    assert (
        reject_stretches.parse_rejection(rejection_line("das_b.mc1.hm1", FIRST), "a")
        is None
    )
    other = "... | INFO | run: das_a.mc1.hm1 cold start at 2025-09-22 00:00:00+00:00"
    assert reject_stretches.parse_rejection(other, "a") is None


@pytest.fixture
def log_file(tmp_path: Path) -> Path:
    """Write an invented log.

    hm1 is rejected at every other epoch from FIRST + 100 for 80 epochs,
    and hm2 at one epoch in six from FIRST + 300 for 120 epochs, too
    thinly to count; at FIRST + 400 to 402 hm1, hm2, hm3 and hm4 are all
    rejected through mc3, and so is mc2, a reference.
    """
    lines = ["2026-10-07 ... | INFO | run: run of channel a starts\n"]
    lines += [
        rejection_line("das_a.mc1.hm1", e) for e in range(FIRST + 100, FIRST + 180, 2)
    ]
    lines += [
        rejection_line("das_a.mc1.mc3.hm1", e)
        for e in range(FIRST + 100, FIRST + 180, 4)
    ]
    lines += [
        rejection_line("das_a.mc2.hm2", e) for e in range(FIRST + 300, FIRST + 420, 6)
    ]
    for epoch in range(FIRST + 400, FIRST + 403):
        lines += [rejection_line(f"das_a.mc3.mc3.hm{n}", epoch) for n in (1, 2, 3, 4)]
        lines += [rejection_line("das_a.mc3.mc2", epoch)]
    lines += [rejection_line("das_b.mc1.hm2", FIRST + 500)]
    written = tmp_path / "das_processor_a.log"
    written.write_text("".join(lines), encoding="utf-8")
    return written


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write the invented clock configuration."""
    written = tmp_path / "clock_config.yaml"
    written.write_text(CLOCK_CONFIG, encoding="utf-8")
    return written


def test_a_references_own_rejections_are_left_out(log_file: Path) -> None:
    """Read every rejection of the channel, a reference's left out."""
    found = reject_stretches.read_rejections(log_file, "a")
    assert len(found) == 40 + 20 + 20 + 12
    assert "mc2" not in {r.clock for r in found}


def test_an_epoch_with_many_clocks_rejected_is_shared(log_file: Path) -> None:
    """Share the epochs with at least the clocks asked for rejected."""
    found = reject_stretches.read_rejections(log_file, "a")
    assert reject_stretches.shared_epochs(found, 4) == {
        FIRST + 400,
        FIRST + 401,
        FIRST + 402,
    }
    assert reject_stretches.shared_epochs(found, 5) == set()


def test_thick_rejections_make_a_burst() -> None:
    """Join rejected epochs close together; keep a stretch long and thick enough."""
    counts = Counter({FIRST + e: 1 for e in range(0, 80, 2)})
    (burst,) = reject_stretches.bursts(counts, 72, 0.3)
    assert burst == reject_stretches.Burst(Span(FIRST, FIRST + 79, False), 40, 40)
    assert reject_stretches.bursts(counts, 80, 0.3) == []
    assert reject_stretches.bursts(counts, 72, 0.6) == []
    apart = reject_stretches.bursts(counts + Counter({FIRST + 200: 3}), 72, 0.3)
    assert [b.span.start for b in apart] == [FIRST]


def test_disabled_stretches_are_read_in_date_order(config_file: Path) -> None:
    """Read each stretch from the entry that disables to the one that enables."""
    clock_config = read_clock_config(config_file)
    assert reject_stretches.disabled_spans(clock_config, "hm1", FIRST + 1000) == [
        Span(FIRST + 72, FIRST + 144, False),
        Span(FIRST + 432, FIRST + 1000, True),
    ]
    assert reject_stretches.disabled_spans(clock_config, "hm1", 0)[-1] == Span(
        FIRST + 432, FIRST + 433, True
    )
    assert reject_stretches.disabled_spans(clock_config, "hm2", FIRST) == []
    assert reject_stretches.disabled_spans(clock_config, "hm9", FIRST) == []


def test_a_burst_takes_in_the_disabled_stretches_near_it() -> None:
    """Join bursts with disabled stretches close by; report none without a burst."""
    burst = reject_stretches.Burst(Span(FIRST, FIRST + 80, False), 50, 40)
    disabled = [
        Span(FIRST + 100, FIRST + 200, False),
        Span(FIRST + 400, FIRST + 500, True),
    ]
    (found,) = reject_stretches.reject_stretches("hm1", [burst], disabled, 72)
    assert found == reject_stretches.RejectStretch(
        "hm1", Span(FIRST, FIRST + 200, False), 50, 40, 1
    )
    (alone,) = reject_stretches.reject_stretches("hm1", [burst], disabled, 10)
    assert (alone.span, alone.replaced) == (burst.span, 0)


def test_a_shared_stretch_names_the_reference_behind_it(log_file: Path) -> None:
    """Join the shared epochs and name the reference most rejections go through."""
    found = reject_stretches.read_rejections(log_file, "a")
    shared = reject_stretches.shared_epochs(found, 4)
    (stretch,) = reject_stretches.shared_stretches(found, shared, 72)
    assert stretch == reject_stretches.SharedStretch(
        Span(FIRST + 400, FIRST + 403, False), 4, "mc3", 12, 13
    )


def test_a_stretch_is_written_as_one_line() -> None:
    """Give the clock, MJDs, hours and counts; '-' for a stretch never enabled."""
    ended = reject_stretches.RejectStretch(
        "hm1", Span(FIRST, FIRST + 18, False), 9, 7, 1
    )
    assert (
        reject_stretches.format_stretch(ended)
        == "hm1 60940.000000 60940.125000 3.0 9 7 1"
    )
    never = ended._replace(span=Span(FIRST, FIRST + 18, True))
    assert reject_stretches.format_stretch(never) == "hm1 60940.000000 - - 9 7 1"
    shared = reject_stretches.SharedStretch(
        Span(FIRST, FIRST + 3, False), 4, "mc3", 12, 13
    )
    assert (
        reject_stretches.format_shared(shared)
        == "60940.000000 60940.020833 0.5 4 mc3 12 13"
    )


def test_the_report_has_a_line_per_stretch(
    log_file: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print each clock's stretches, then the shared ones, each under its header.

    hm1's burst, from FIRST + 100 to FIRST + 178, is joined with its
    disabled stretch from FIRST + 72 to FIRST + 144; its shared rejections
    at FIRST + 400 lie too far away to join it, and are left out. hm2's
    rejections are too thin to make a stretch.
    """
    arguments = [str(log_file), "--rf", "a", "--clock-config", str(config_file)]
    assert reject_stretches.main(arguments) == 0
    assert capsys.readouterr().out.splitlines() == [
        reject_stretches.REPORT_HEADER,
        "hm1 60940.500000 60941.243055 17.8 60 40 1",
        reject_stretches.SHARED_HEADER,
        "60942.777777 60942.798611 0.5 4 mc3 12 13",
    ]


def test_the_options_change_the_stretches(
    log_file: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Find more with a smaller share and fewer shared, less with longer stretches.

    A smaller share finds hm2, more clocks asked for share nothing, and a
    longer shortest stretch leaves every clock's stretch out.

    With nothing shared, hm1's three rejected epochs at FIRST + 400 count,
    but are too few to make a stretch.
    """
    arguments = [str(log_file), "--rf", "a", "--clock-config", str(config_file)]
    reject_stretches.main([*arguments, "--min-share", "0.1", "--shared-clocks", "5"])
    report_lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in report_lines[1:-1]] == ["hm1", "hm2"]
    assert report_lines[-1] == reject_stretches.SHARED_HEADER
    reject_stretches.main([*arguments, "--min-hours", "20"])
    assert capsys.readouterr().out.splitlines()[1] == reject_stretches.SHARED_HEADER


def test_shared_rejections_make_no_stretch(
    log_file: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Leave the shared epochs' rejections out of every clock's stretches.

    With the shortest stretch half an hour, the three shared epochs would
    make a stretch of each of hm1, hm3 and hm4 if they counted.
    """
    arguments = [str(log_file), "--rf", "a", "--clock-config", str(config_file)]
    reject_stretches.main([*arguments, "--min-hours", "0.5"])
    report_lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in report_lines[1:-2]] == ["hm1"]
    assert report_lines[-1].split()[4] == "mc3"


def test_the_script_runs_as_a_program(
    log_file: Path,
    config_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run main from the command line and exit with its status."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reject_stretches.py",
            str(log_file),
            "--rf",
            "a",
            "--clock-config",
            str(config_file),
        ],
    )
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(reject_stretches.__file__, run_name="__main__")
    assert program_exit.value.code == 0
    assert capsys.readouterr().out.startswith(reject_stretches.REPORT_HEADER)
