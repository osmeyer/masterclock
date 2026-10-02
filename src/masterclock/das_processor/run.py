"""Running das_processor: building each epoch, processing it, the epoch loop.

Only the run functions take the configuration. :func:`build_epoch` resolves
everything one epoch needs into an :class:`Epoch` of plain values: its
references, every pair and triple, the steering of every reference that
steers a series, and each series' settings. Everything below it receives
that epoch or plain values.
"""

from datetime import datetime, timedelta
from typing import Final, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from masterclock.das_processor.clock_config import ClockConfig
from masterclock.das_processor.config import AppConfig
from masterclock.das_processor.read_cd5m5m import DASData
from masterclock.das_processor.read_steering import read_steering
from masterclock.das_processor.registry import Existing, build_registry, refs_of
from masterclock.domain.phase import EPOCH_SECONDS
from masterclock.domain.series import PairKey, SeriesKey, SeriesParams, TripleKey
from masterclock.domain.steering import SteerEvent, signs

_EPOCH: Final[timedelta] = timedelta(seconds=EPOCH_SECONDS)
"""One epoch, T."""


class Epoch(BaseModel):
    """Everything one epoch needs, with the configuration resolved (design 4.3).

    Parameters
    ----------
    interpolated_datetime : AwareDatetime
        The epoch start E.
    block : DASData or None
        The epoch's DAS block; ``None`` when the DAS measured nothing.
    refs : frozenset of str
        The epoch's references, REFS(e).
    steering : dict of str to tuple of SteerEvent
        The events of every reference that steers a series, in
        (E - T, E + T], in time order.
    pairs : tuple of (str, str)
        Every pair, sorted.
    triples : tuple of (str, str, str)
        Every triple, sorted.
    params : dict of series key to SeriesParams
        Every series' settings at E.

    Raises
    ------
    pydantic.ValidationError
        If the block is of another epoch, or the settings are not for
        exactly the epoch's series.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    interpolated_datetime: AwareDatetime
    block: DASData | None
    refs: frozenset[str]
    steering: dict[str, tuple[SteerEvent, ...]]
    pairs: tuple[PairKey, ...]
    triples: tuple[TripleKey, ...]
    params: dict[SeriesKey, SeriesParams]

    @model_validator(mode="after")
    def _check(self) -> Self:
        """Refuse a block of another epoch, or settings for other series.

        Returns
        -------
        Self
            The epoch, unchanged.

        Raises
        ------
        ValueError
            If the block's mark is not the epoch's, or the settings' keys
            are not the pairs and triples.
        """
        if (
            self.block is not None
            and self.block.interpolated_datetime != self.interpolated_datetime
        ):
            message = (
                f"the block of {self.block.interpolated_datetime} is not of the epoch"
                f" of {self.interpolated_datetime}"
            )
            raise ValueError(message)
        if set(self.params) != {*self.pairs, *self.triples}:
            message = "an epoch holds settings for exactly its pairs and triples"
            raise ValueError(message)
        return self


def build_epoch(
    mark: datetime,
    block: DASData | None,
    existing: Existing,
    config: AppConfig,
    clock_config: ClockConfig,
) -> Epoch:
    """Resolve everything an epoch needs (design 6.3).

    Parameters
    ----------
    mark : datetime
        The epoch start E.
    block : DASData or None
        The epoch's DAS block, or ``None`` when there is none.
    existing : Existing
        The series that existed before the epoch.
    config : AppConfig
        The run's settings, for the steering directory.
    clock_config : ClockConfig
        The clock configuration.

    Returns
    -------
    Epoch
        The epoch: its references, every pair and triple (see
        :func:`~masterclock.das_processor.registry.build_registry`), the
        steering of every reference any series is steered by, read over
        (E - T, E + T] (I4), and each series' settings at E.

    Raises
    ------
    ConfigError
        If a series' clock has no entry in the clock configuration.
    DataFileError
        If a steering file cannot be read.
    """
    refs = refs_of(block)
    pairs, triples = build_registry(block, refs, existing)
    keys: list[SeriesKey] = [*pairs, *triples]
    steered = sorted({mc for key in keys for mc in signs(key)})
    steering = {
        mc: read_steering(config.das.steering_path, mc, mark - _EPOCH, mark + _EPOCH)
        for mc in steered
    }
    params = {key: clock_config.params_for(key, mark) for key in keys}
    return Epoch(
        interpolated_datetime=mark,
        block=block,
        refs=refs,
        steering=steering,
        pairs=pairs,
        triples=triples,
        params=params,
    )
