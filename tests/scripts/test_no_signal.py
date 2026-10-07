"""Tests for scripts/no_signal.py.

The rules covered: the raw measured phase of every row with a reading is
read from its column, rows without one passed over; a one-epoch change is
wrapped into the half period either way; an epoch is judged unlike a
running clock when the changes within HALF_WINDOW epochs of it typically
depart from their median by more than NO_SIGNAL_PS, and not judged with
fewer than MIN_WINDOW_CHANGES of them; a stretch needs every reference that
judged an epoch to agree, and a run ends only at an epoch judged like a
clock; runs fewer than the shortest stretch apart are joined, before and
after their ends are found again, so no two stretches overlap; each end is
found again to the epoch from the changes alone, against the clock's own
changes beside it, the earliest start and latest end of any reference kept,
the run's own ends when no window beside it can judge; a stretch shorter
than the shortest is left out, and one reaching the last epoch judged has
no enable; a stretch whose references disagree by more than DISAGREEING_PS
has no signal, one whose references agree and whose clock typically moves
beyond FAR_OFF_FREQUENCY_PS an epoch is far off, another whose references
agree is quartz, and one they cannot be compared in and that is not far off
has no kind, the references' changes compared relative to the first and
wrapped, so one lurch read either side of the half period is one; a clock
moving beyond FAR_OFF_FREQUENCY_PS an epoch either way is unlike a running
clock however steadily it moves, one moving a little less is not, nor is
one that jumps now and then; the clocks looked at are those measured
against a reference, neither references nor named with a prefix to skip; an
MJD is written rounded down to six decimals, so its first epoch is the
epoch meant; and the script prints one line per stretch, the same report
from several processes as from one.

The readings are invented, from a seeded generator.
"""

import random
import runpy
import sys
from pathlib import Path
from typing import Final

import pytest

import no_signal
from characterize import FAR_OFF_FREQUENCY_PS
from masterclock.das_processor.files import MEAS_COLUMNS, MEAS_HEADER_LINES, SEPARATOR

PERIOD: Final = 200_000
"""The phase period, ps."""

MIN_EPOCHS: Final = 72
"""Twelve hours of epochs: the shortest stretch in these tests."""

FIRST: Final = 8_775_360
"""An invented first epoch, MJD 60940 at 00:00."""

REFERENCES: Final = ("mc1", "mc2", "mc3")
"""The invented deployment's references."""

FAR_OFF_RATE: Final = 30_000
"""How far an invented clock far off frequency moves in an epoch, ps: beyond
FAR_OFF_FREQUENCY_PS."""


def seeded(seed: int) -> random.Random:
    """Give a generator of invented noise, the same on every run."""
    return random.Random(seed)  # noqa: S311  # nosec B311 - repeatable test data, not a secret


def clock_readings(
    epochs: range,
    bad: range,
    kind: str,
    seed: int,
    rate_ps: int = 500,
) -> dict[str, no_signal.Phases]:
    """Give each reference's raw readings of one invented clock.

    The clock runs ``rate_ps`` an epoch fast with a few ps of noise, except
    in ``bad``: there each reference reads the whole period at random for
    ``no_signal``; or all of them read one phase, each with a little noise
    of its own, lurching by tens of nanoseconds an epoch for ``quartz``, or
    moving FAR_OFF_RATE an epoch for ``far_off``.
    """
    generator = seeded(seed)
    true_phase = 0
    readings: dict[str, no_signal.Phases] = {reference: {} for reference in REFERENCES}
    for epoch in epochs:
        if epoch in bad and kind == no_signal.FAR_OFF:
            true_phase += FAR_OFF_RATE + round(generator.gauss(0.0, 3.0))
        elif epoch in bad:
            true_phase += round(generator.gauss(0.0, 40_000.0))
        else:
            true_phase += rate_ps + round(generator.gauss(0.0, 3.0))
        for reference in REFERENCES:
            if epoch in bad and kind == no_signal.NO_SIGNAL:
                reading = generator.randrange(PERIOD)
            else:
                reading = true_phase + round(generator.gauss(0.0, 300.0))
            readings[reference][epoch] = reading % PERIOD
    return readings


def changes_of(readings: dict[str, no_signal.Phases]) -> dict[str, no_signal.Changes]:
    """Give each reference's one-epoch changes."""
    return {
        reference: no_signal.one_epoch_changes(phases)
        for reference, phases in readings.items()
    }


DAY_EPOCHS: Final = range(FIRST, FIRST + 3 * 144)
"""Three invented days."""

BAD: Final = range(FIRST + 150, FIRST + 260)
"""The epochs the invented clock is not running properly in."""


# ------------------------------------------------------------ the readings


def test_a_change_is_wrapped_into_the_half_period() -> None:
    """Bring every change into (-P/2, P/2] by whole periods."""
    assert [
        no_signal.wrapped_change(change)
        for change in (0, 100_000, 100_001, -99_999, -100_000, 350_000)
    ] == [0, 100_000, -99_999, -99_999, 100_000, -50_000]


def test_changes_need_readings_one_epoch_apart() -> None:
    """Give a change only where the epoch before has a reading."""
    assert no_signal.one_epoch_changes({5: 10, 6: 30, 8: 50, 9: 199_990}) == {
        6: 20,
        9: -60,
    }


def meas_row(epoch: int, phase: int | None) -> str:
    """Write a measurement row with only the columns the script reads filled."""
    fields = []
    for column in MEAS_COLUMNS:
        if column.name == "interpolated_mjd":
            text = no_signal.effective_mjd(epoch)
        elif column.name == "measured_phase" and phase is not None:
            text = str(phase)
        else:
            text = "-"
        fields.append(text.rjust(column.width))
    return SEPARATOR.join(fields) + "\n"


def write_pair(
    folder: Path, pair: tuple[str, str], phases: dict[int, int | None]
) -> None:
    """Write an invented measurement file with a header of the right length."""
    meas_folder = folder / "meas"
    meas_folder.mkdir(exist_ok=True)
    lines = ["# header\n"] * MEAS_HEADER_LINES
    lines += [meas_row(epoch, phase) for epoch, phase in sorted(phases.items())]
    (meas_folder / f"das_a.{pair[0]}.{pair[1]}.dat").write_text("".join(lines))


def test_the_raw_phase_of_every_row_with_a_reading_is_read(tmp_path: Path) -> None:
    """Read epoch and phase from their columns, rows without a reading passed over."""
    write_pair(
        tmp_path, ("mc1", "hm1"), {FIRST: 5, FIRST + 1: None, FIRST + 2: 199_999}
    )
    assert no_signal.read_phases(tmp_path / "meas" / "das_a.mc1.hm1.dat") == {
        FIRST: 5,
        FIRST + 2: 199_999,
    }


# ------------------------------------------------------------ judging


def test_a_clock_running_properly_is_never_judged_bad() -> None:
    """Judge every epoch of a steady clock with a signal."""
    changes = changes_of(clock_readings(DAY_EPOCHS, range(0), "", seed=1))["mc1"]
    verdicts = no_signal.judge(changes)
    assert len(verdicts) == len(changes)
    assert not any(verdicts.values())


@pytest.mark.parametrize("kind", [no_signal.NO_SIGNAL, no_signal.QUARTZ])
def test_the_middle_of_a_bad_stretch_is_judged_bad(kind: str) -> None:
    """Judge the epochs well inside a stretch with no signal, or quartz, as bad."""
    changes = changes_of(clock_readings(DAY_EPOCHS, BAD, kind, seed=2))["mc2"]
    verdicts = no_signal.judge(changes)
    inside = range(BAD.start + no_signal.HALF_WINDOW, BAD.stop - no_signal.HALF_WINDOW)
    assert all(verdicts[epoch] for epoch in inside)
    assert not any(verdicts[epoch] for epoch in range(FIRST + 20, BAD.start - 20))


def test_a_window_with_too_few_changes_judges_nothing() -> None:
    """Leave an epoch unjudged when its window holds fewer than the fewest changes."""
    few = {FIRST + 4 * k: 100 for k in range(no_signal.MIN_WINDOW_CHANGES)}
    assert no_signal.judge(few) == {}
    enough = {FIRST + k: 100 for k in range(no_signal.MIN_WINDOW_CHANGES)}
    assert set(no_signal.judge(enough)) == set(enough)


def test_a_clock_that_jumps_now_and_then_is_not_bad() -> None:
    """Judge a clock that runs properly between rare jumps as running properly."""
    changes = changes_of(clock_readings(DAY_EPOCHS, range(0), "", seed=3))["mc1"]
    for epoch in range(BAD.start, BAD.stop, 40):
        changes[epoch] += 90_000
    assert not any(no_signal.judge(changes).values())


def test_a_clock_moving_far_every_epoch_is_bad() -> None:
    """Judge a clock moving more than FAR_OFF_FREQUENCY_PS an epoch bad, however steady.

    One moving a little less than that is running properly.
    """
    limit = round(FAR_OFF_FREQUENCY_PS)
    for rate_ps, bad in ((limit + 1000, True), (limit - 1000, False)):
        readings = clock_readings(DAY_EPOCHS, range(0), "", seed=3, rate_ps=rate_ps)
        verdicts = no_signal.judge(changes_of(readings)["mc1"])
        assert set(verdicts.values()) == {bad}
    readings = clock_readings(DAY_EPOCHS, range(0), "", seed=3, rate_ps=-limit - 1000)
    assert all(no_signal.judge(changes_of(readings)["mc1"]).values())


def test_every_reference_that_judged_must_agree() -> None:
    """Keep an epoch bad only when no reference that judged it found a clock."""
    assert no_signal.agreed([{3: True, 1: True}, {1: False, 2: True}]) == {
        1: False,
        2: True,
        3: True,
    }


def test_a_run_ends_only_at_an_epoch_judged_like_a_clock() -> None:
    """Carry a run over epochs no one judged; leave a run at the end open."""
    verdicts = {1: True, 2: True, 9: True, 10: False, 11: True, 12: False, 13: True}
    assert no_signal.runs(verdicts) == [
        no_signal.Span(1, 10, open_ended=False),
        no_signal.Span(11, 12, open_ended=False),
        no_signal.Span(13, 14, open_ended=True),
    ]
    assert no_signal.runs({1: False}) == []


def test_spans_close_together_are_joined() -> None:
    """Join spans that overlap or lie fewer than the epochs given apart."""
    spans = [
        no_signal.Span(50, 60, False),
        no_signal.Span(0, 10, False),
        no_signal.Span(5, 20, False),
        no_signal.Span(24, 30, True),
    ]
    assert no_signal.joined(spans, 5) == [
        no_signal.Span(0, 30, True),
        no_signal.Span(50, 60, False),
    ]
    assert no_signal.joined(spans[:2], 40) == sorted(spans[:2])
    assert no_signal.joined(spans[:2], 41) == [no_signal.Span(0, 60, False)]


# ------------------------------------------------------------ the ends


def test_the_ends_are_found_to_the_epoch() -> None:
    """Start at the first bad reading and enable at the first proper one."""
    readings = clock_readings(DAY_EPOCHS, BAD, no_signal.NO_SIGNAL, seed=4)
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert (found.disabled_from, found.enabled_at) == (BAD.start, BAD.stop)


def test_the_earliest_start_and_latest_end_of_any_reference_are_kept() -> None:
    """Disable from where any reference first sees the trouble, until the last."""
    readings = clock_readings(DAY_EPOCHS, BAD, no_signal.QUARTZ, seed=5)
    readings["mc1"][BAD.start - 3] = (readings["mc1"][BAD.start - 3] + 70_000) % PERIOD
    readings["mc3"][BAD.stop + 2] = (readings["mc3"][BAD.stop + 2] + 70_000) % PERIOD
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert (found.disabled_from, found.enabled_at) == (BAD.start - 3, BAD.stop + 3)


def test_without_a_window_beside_it_a_run_keeps_its_own_ends() -> None:
    """Keep the run's own start and end when no window before or after can judge."""
    changes = {FIRST + k: 0 for k in range(60)}
    assert no_signal.first_bad(changes, FIRST + 5) == FIRST + 5
    assert no_signal.first_good(changes, FIRST + 55) == FIRST + 55


def test_with_no_spread_any_other_change_strays() -> None:
    """Count any change unlike a clock whose own changes are all the same."""
    assert no_signal._departs(8, [7] * no_signal.MIN_WINDOW_CHANGES)
    assert not no_signal._departs(7, [7] * no_signal.MIN_WINDOW_CHANGES)


def test_with_no_change_straying_a_run_keeps_its_own_ends() -> None:
    """Keep the run's own ends when every change near them is like the clock's."""
    changes = {FIRST + k: 100 + k % 3 for k in range(200)}
    assert no_signal.first_bad(changes, FIRST + 100) == FIRST + 100
    assert no_signal.first_good(changes, FIRST + 100) == FIRST + 100


def test_bad_stretches_close_together_become_one() -> None:
    """Join two bad stretches a few hours apart; never let two overlap."""
    gap = range(FIRST + 200, FIRST + 230)
    bad = range(BAD.start, BAD.stop + 60)
    readings = clock_readings(DAY_EPOCHS, bad, no_signal.NO_SIGNAL, seed=6)
    good_again = clock_readings(DAY_EPOCHS, range(0), "", seed=7)
    for reference in REFERENCES:
        for epoch in gap:
            readings[reference][epoch] = good_again[reference][epoch]
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert (found.disabled_from, found.enabled_at) == (bad.start, bad.stop)
    apart = no_signal.stretches("hm1", changes_of(readings), 10)
    assert len(apart) == 2
    assert apart[0].enabled_at is not None
    assert apart[0].enabled_at <= apart[1].disabled_from


def test_a_stretch_shorter_than_the_shortest_is_left_out() -> None:
    """Report nothing for trouble lasting under the shortest stretch."""
    short = range(BAD.start, BAD.start + 40)
    readings = clock_readings(DAY_EPOCHS, short, no_signal.NO_SIGNAL, seed=8)
    assert no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS) == []
    assert len(no_signal.stretches("hm1", changes_of(readings), 30)) == 1


def test_a_stretch_that_never_ends_has_no_enable() -> None:
    """Give no enable MJD for trouble lasting to the last epoch judged."""
    to_the_end = range(BAD.start, DAY_EPOCHS.stop)
    readings = clock_readings(DAY_EPOCHS, to_the_end, no_signal.NO_SIGNAL, seed=9)
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert found.enabled_at is None
    assert found.disabled_from == BAD.start


# ------------------------------------------------------------ the kind


@pytest.mark.parametrize(
    "kind", [no_signal.NO_SIGNAL, no_signal.QUARTZ, no_signal.FAR_OFF]
)
def test_the_references_tell_no_signal_from_quartz(kind: str) -> None:
    """Call a stretch no signal when its references disagree, quartz when they agree.

    When they agree and the clock moves more than FAR_OFF_FREQUENCY_PS an
    epoch, far off.
    """
    readings = clock_readings(DAY_EPOCHS, BAD, kind, seed=10)
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert found.kind == kind
    assert found.references == REFERENCES


def test_too_few_references_give_no_kind() -> None:
    """Give no kind when fewer than MIN_COMPARED references read the stretch."""
    readings = clock_readings(DAY_EPOCHS, BAD, no_signal.NO_SIGNAL, seed=11)
    del readings["mc3"]
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert found.kind is None


def test_a_far_off_stretch_needs_no_three_references() -> None:
    """Call a stretch far off from two references, which cannot be compared."""
    readings = clock_readings(DAY_EPOCHS, BAD, no_signal.FAR_OFF, seed=11)
    del readings["mc3"]
    (found,) = no_signal.stretches("hm1", changes_of(readings), MIN_EPOCHS)
    assert found.kind == no_signal.FAR_OFF
    assert no_signal.typical_change([{}], no_signal.Span(1, 2, False)) == 0.0


def test_the_reference_spread_is_wrapped() -> None:
    """Measure changes either side of the half period as close together.

    The four changes are one lurch of about 99.5 ns read either side of
    the half period; unwrapped, their median would lie near zero.
    """
    across = [{1: 99_000}, {1: 99_100}, {1: -99_900}, {1: -99_950}]
    assert no_signal.reference_spread(across, no_signal.Span(1, 2, False)) == 500


# ------------------------------------------------------------ the report


def test_an_mjd_is_rounded_down_to_its_epoch() -> None:
    """Write six decimals that never pass the epoch's start."""
    for epoch in range(FIRST, FIRST + 144):
        text = no_signal.effective_mjd(epoch)
        assert len(text.split(".")[1]) == 6
        assert float(text) * 144 <= epoch < float(text) * 144 + 1


def test_a_stretch_is_written_as_one_line() -> None:
    """Give clock, MJDs, hours, kind and references; '-' for what is missing."""
    ended = no_signal.BadStretch("hm1", FIRST, FIRST + 18, no_signal.QUARTZ, REFERENCES)
    assert no_signal.format_stretch(ended) == (
        "hm1 60940.000000 60940.125000 3.0 quartz mc1,mc2,mc3"
    )
    never = no_signal.BadStretch("hm1", FIRST, None, None, ("mc1",))
    assert no_signal.format_stretch(never) == "hm1 60940.000000 - - - mc1"


@pytest.fixture
def processed_path(tmp_path: Path) -> Path:
    """Write invented pair files: one clock with no signal, one quartz, one skipped."""
    for clock, kind, seed in (
        ("hm1", no_signal.NO_SIGNAL, 12),
        ("ox1", no_signal.QUARTZ, 13),
        ("aog1", no_signal.NO_SIGNAL, 14),
    ):
        for reference, phases in clock_readings(DAY_EPOCHS, BAD, kind, seed).items():
            write_pair(tmp_path, (reference, clock), dict(phases))
    for name in ("das_a.mc1.mc2.dat", "das_a.mc1.mc1.hm1.dat", "das_b.mc1.hm1.dat"):
        (tmp_path / "meas" / name).touch()
    (tmp_path / "meas" / "notes.txt").touch()
    write_pair(tmp_path, ("hm1", "mc1"), {FIRST: 0})
    return tmp_path


def test_only_clocks_measured_against_references_are_looked_at(
    processed_path: Path,
) -> None:
    """Pass over references, triples, other channels and names, and skipped prefixes."""
    assert no_signal.measured_clocks(processed_path, "a", ("aog",)) == {
        "hm1": list(REFERENCES),
        "ox1": list(REFERENCES),
    }
    assert "aog1" in no_signal.measured_clocks(processed_path, "a", ())


def test_the_report_has_a_line_per_stretch(
    processed_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the header and each clock's stretches, skipped clocks left out.

    With no signal the clock is enabled at its first proper reading; when
    it lurched, at the reading its last lurch ends on, from which it runs
    properly, one epoch sooner here.
    """
    assert no_signal.main([str(processed_path), "--rf", "a", "--skip", "aog"]) == 0
    report_lines = capsys.readouterr().out.splitlines()
    assert report_lines[0] == no_signal.REPORT_HEADER
    assert [line.split()[0::4] for line in report_lines[1:]] == [
        ["hm1", "no_signal"],
        ["ox1", "quartz"],
    ]
    assert [line.split()[3] for line in report_lines[1:]] == ["18.3", "18.2"]


def test_the_shortest_stretch_can_be_changed(
    processed_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report nothing when every stretch is under the hours asked for."""
    assert no_signal.main([str(processed_path), "--rf", "a", "--min-hours", "24"]) == 0
    assert capsys.readouterr().out.splitlines() == [no_signal.REPORT_HEADER]


def test_clocks_are_looked_at_in_parallel_alike(
    processed_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the same report from two processes as from one."""
    no_signal.main([str(processed_path), "--rf", "a", "--jobs", "1"])
    one_process = capsys.readouterr().out
    no_signal.main([str(processed_path), "--rf", "a", "--jobs", "2"])
    assert capsys.readouterr().out == one_process
    assert len(one_process.splitlines()) == 4


def test_one_clock_is_looked_at_from_one_argument(processed_path: Path) -> None:
    """Give a pool of processes the same result as a direct call."""
    job: no_signal.ClockJob = (processed_path, "a", "hm1", REFERENCES, MIN_EPOCHS)
    assert no_signal._find_one(job) == no_signal.clock_stretches(*job)


def test_the_script_runs_as_a_program(
    processed_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run main from the command line and exit with its status."""
    monkeypatch.setattr(sys, "argv", ["no_signal.py", str(processed_path), "--rf", "a"])
    with pytest.raises(SystemExit) as program_exit:
        runpy.run_path(no_signal.__file__, run_name="__main__")
    assert program_exit.value.code == 0
    assert capsys.readouterr().out.startswith(no_signal.REPORT_HEADER)
