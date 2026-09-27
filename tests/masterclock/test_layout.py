"""Tests for src/masterclock/ as a whole: which part may import which.

The rule covered: app/ holds what any program needs just to be a program and
imports nothing else of the project; domain/ holds the subject matter and
imports only app/ and itself; and every other subpackage is a program, which
imports app/, domain/ and itself, never another program. The standard
library and third-party packages may be imported anywhere.

The subpackages are found by walking the package, so a program added later
is held to the rule without the test being edited. Each module's source is
taken from the imported module itself and read with ast, so an import inside
a function or under TYPE_CHECKING is seen as well as one at the top.
"""

import ast
import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Final

import pytest

import masterclock

PROJECT: Final = "masterclock"
APP: Final = "app"
DOMAIN: Final = "domain"
LOADERS: Final = frozenset({"import_module", "__import__"})


def allowed(part: str) -> frozenset[str]:
    """Return the parts of the project that the part ``part`` may import."""
    if part == APP:
        return frozenset({APP})
    if part == DOMAIN:
        return frozenset({APP, DOMAIN})
    return frozenset({APP, DOMAIN, part})


def part_of(name: str) -> str | None:
    """Return the part of the project ``name`` is in, or None if outside it."""
    if not name.startswith(f"{PROJECT}."):
        return None
    return name.split(".")[1]


def parts() -> list[str]:
    """Return the name of every subpackage of the project."""
    return [
        found.name
        for found in pkgutil.iter_modules(masterclock.__path__)
        if found.ispkg
    ]


def modules_of(part: str) -> dict[str, ModuleType]:
    """Return the subpackage ``part`` and every module below it, by name."""
    package = importlib.import_module(f"{PROJECT}.{part}")
    names = [package.__name__] + [
        found.name
        for found in pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
    ]
    return {name: importlib.import_module(name) for name in names}


def relative_base(module: str, *, is_package: bool, level: int) -> str:
    """Return the package a relative import of ``level`` dots starts from."""
    base = module if is_package else module.rpartition(".")[0]
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    return base


def imported_names(source: str, module: str, *, is_package: bool) -> list[str]:
    """Return every module name the source imports, written out in full.

    An ``import a.b`` gives ``a.b``; a ``from a import b`` gives ``a`` and
    ``a.b``, since ``b`` may be a module; a relative import is resolved from
    ``module``; and a call to ``importlib.import_module`` or ``__import__``
    with a literal name gives that name.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = relative_base(module, is_package=is_package, level=node.level)
                start = f"{base}.{node.module}" if node.module else base
            else:
                start = node.module or ""
            names.append(start)
            names.extend(f"{start}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            called = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", "")
            )
            if (
                called in LOADERS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.append(node.args[0].value)
    return names


def refused(names: list[str], part: str) -> list[str]:
    """Return the names in ``names`` that the part ``part`` may not import."""
    return [
        name
        for name in names
        if (target := part_of(name)) is not None and target not in allowed(part)
    ]


@pytest.mark.parametrize(
    ("source", "module", "is_package", "wrong"),
    [
        ("import masterclock.domain", "masterclock.app.log", False, True),
        ("import masterclock.app.log", "masterclock.app.lock", False, False),
        ("import json", "masterclock.app.log", False, False),
        ("import masterclock", "masterclock.app.log", False, False),
        ("from masterclock import domain", "masterclock.app.log", False, True),
        ("from masterclock import app", "masterclock.app.log", False, False),
        ("from .. import domain", "masterclock.app.log", False, True),
        ("from ..domain import x", "masterclock.app.log", False, True),
        ("from . import log", "masterclock.app.lock", False, False),
        ("from . import domain", "masterclock.app", True, False),
        ("from .. import domain", "masterclock.app", True, True),
        ("def f():\n    import masterclock.domain", "masterclock.app.log", False, True),
        (
            "if TYPE_CHECKING:\n    from masterclock.domain import x",
            "masterclock.app.log",
            False,
            True,
        ),
        (
            "importlib.import_module('masterclock.domain')",
            "masterclock.app.log",
            False,
            True,
        ),
        ("__import__('masterclock.domain')", "masterclock.app.log", False, True),
        ("import_module(name)", "masterclock.app.log", False, False),
        (
            "from masterclock.app.exceptions import x",
            "masterclock.domain.a",
            False,
            False,
        ),
        ("from masterclock.domain import x", "masterclock.domain.a", False, False),
        (
            "from masterclock.das_processor import x",
            "masterclock.domain.a",
            False,
            True,
        ),
        ("from ..das_processor import x", "masterclock.domain.a", False, True),
        ("from masterclock.app import x", "masterclock.das_processor.a", False, False),
        ("from ..domain import x", "masterclock.das_processor.a", False, False),
        ("from . import epochs", "masterclock.das_processor.a", False, False),
        (
            "import masterclock.other_program",
            "masterclock.das_processor.a",
            False,
            True,
        ),
        ("from .. import other_program", "masterclock.das_processor", True, True),
    ],
)
def test_every_way_of_writing_an_import_is_judged(
    source: str, module: str, *, is_package: bool, wrong: bool
) -> None:
    """Refuse an import from a part the module's part may not use, in any form."""
    part = module.split(".")[1]
    names = imported_names(source, module, is_package=is_package)
    assert bool(refused(names, part)) is wrong


def test_every_part_imports_only_what_it_may() -> None:
    """Import, in every module of every part, only the parts that one may use."""
    found = parts()
    assert {APP, DOMAIN, "das_processor"} <= set(found)
    seen: dict[str, int] = {}
    outside: dict[str, list[str]] = {}
    for part in found:
        seen[part] = 0
        for name, module in modules_of(part).items():
            names = imported_names(
                inspect.getsource(module), name, is_package=hasattr(module, "__path__")
            )
            seen[part] += len(names)
            outside[name] = refused(names, part)
    assert all(count > 0 for count in seen.values()), seen
    assert outside == {name: [] for name in outside}
