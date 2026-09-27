"""What can go wrong, and how a failure describes itself.

Every error this project raises descends from :class:`MasterClockError`, so
one ``except`` clause catches all of them while anything unrelated goes past.
That rule is what lets the entry point turn a failure it recognises into an
exit status and say nothing more, on the understanding that the failure was
already reported where it happened.

The root is here, with the errors of being a program. Every package keeps
its own errors in its own ``exceptions`` module, and every exception there
subclasses the root, directly or through an intermediate base. Each package's
tests discover the classes from its module rather than reading a list, so an
exception added later is held to the rule without anything being edited.

An exception is made an intermediate base only when something catches it as
one: :class:`ConfigError` here, because a required setting absent from both
sources is reported differently from any other configuration fault.

:func:`describe_error` is here too. It renders a parsing or validation error,
a Pydantic :class:`~pydantic.ValidationError` included, as one line, since a
record that runs to several lines cannot be read a line at a time.
"""

from pydantic import ValidationError


class MasterClockError(Exception):
    """Base class for all masterclock application errors.

    Behaves exactly like :class:`Exception`; it exists so application errors
    share a common ancestor distinct from built-in and third-party exceptions.

    Examples
    --------
    >>> try:
    ...     raise MasterClockError("something went wrong")
    ... except MasterClockError as exc:
    ...     str(exc)
    'something went wrong'
    """


class ConfigError(MasterClockError):
    """Raised when a configuration a run needs cannot be read or believed.

    Covers every configuration source, the INI file and the deployment's YAML
    files alike: a file that cannot be read or parsed, or that names
    something the program does not read, and a setting, once the sources are
    merged, whose value cannot be used, including one that names a log file
    or directory a run cannot use.

    Examples
    --------
    >>> try:
    ...     raise ConfigError("missing required entries: [paths] output")
    ... except MasterClockError as exc:
    ...     str(exc)
    'missing required entries: [paths] output'
    """


class MissingSettingsError(ConfigError):
    """Raised when a required setting is provided by neither source.

    A :class:`ConfigError` subclass - so an ``except ConfigError`` clause
    still catches it - distinguished so the entry point can treat a
    required setting absent from both the config file and the command line
    as a command-line usage error: it prints the full help and exits with
    status 2, the same as a malformed argument or a bare invocation, rather
    than reporting a generic configuration failure. An *invalid* setting (a
    bad value) remains an ordinary :class:`ConfigError`.

    Examples
    --------
    >>> try:
    ...     raise MissingSettingsError("these settings must be provided ...")
    ... except ConfigError as exc:
    ...     str(exc)
    'these settings must be provided ...'
    """


class RunLockError(MasterClockError):
    """Raised when a run lock cannot be acquired.

    A run holds an exclusive lock on one named file for as long as it runs,
    so a second run that would write the same output - and so write the same
    data twice - aborts with this error instead. Also raised when the lock
    file itself cannot be opened.

    Examples
    --------
    >>> try:
    ...     raise RunLockError("another run already holds the run lock")
    ... except MasterClockError as exc:
    ...     str(exc)
    'another run already holds the run lock'
    """


class LoggingError(MasterClockError):
    """Raised when the application's logging cannot be set up as intended.

    Examples
    --------
    >>> try:
    ...     raise LoggingError("logger 'x' is a Logger, not a MasterClockLogger")
    ... except MasterClockError as exc:
    ...     str(exc)
    "logger 'x' is a Logger, not a MasterClockLogger"
    """


def describe_error(error: Exception) -> str:
    r"""Summarize a parsing or validation error on a single line.

    Parameters
    ----------
    error : Exception
        The error raised while parsing or validating a line or a file; may be
        a Pydantic :class:`~pydantic.ValidationError`, whose per-field
        details are joined into one line.

    Returns
    -------
    str
        A compact description of the problem, on one line and never empty.
        For a validation error, each field's location, dotted, and message,
        the fields joined by ``"; "``; an error that belongs to no field,
        such as one a model validator raises, is its message alone. For any
        other error, its text, or its class name if it has no text. Line
        breaks are written as the escapes ``\n`` and ``\r``.

    Examples
    --------
    >>> describe_error(ValueError("expected 5 fields, found 3"))
    'expected 5 fields, found 3'
    >>> describe_error(ValueError())
    'ValueError'
    """
    if isinstance(error, ValidationError):
        parts = []
        for detail in error.errors():
            location = ".".join(str(part) for part in detail["loc"])
            parts.append(f"{location}: {detail['msg']}" if location else detail["msg"])
        text = "; ".join(parts)
    else:
        text = str(error) or type(error).__name__
    return text.replace("\r", "\\r").replace("\n", "\\n")
