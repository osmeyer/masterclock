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
IMPORT_LOADERS: Final = frozenset({"import_module", "__import__"})


def allowed_parts(part_name: str) -> frozenset[str]:
    """Return the parts of the project that the part ``part_name`` may import."""
    if part_name == APP:
        return frozenset({APP})
    if part_name == DOMAIN:
        return frozenset({APP, DOMAIN})
    return frozenset({APP, DOMAIN, part_name})


def part_of(module_name: str) -> str | None:
    """Return the part of the project ``module_name`` is in, or None if outside it."""
    if not module_name.startswith(f"{PROJECT}."):
        return None
    return module_name.split(".")[1]


def project_parts_found() -> list[str]:
    """Return the name of every subpackage of the project."""
    return [
        module_info.name
        for module_info in pkgutil.iter_modules(masterclock.__path__)
        if module_info.ispkg
    ]


def modules_of(part_name: str) -> dict[str, ModuleType]:
    """Return the subpackage ``part_name`` and every module below it, by name."""
    package = importlib.import_module(f"{PROJECT}.{part_name}")
    module_names = [package.__name__] + [
        module_info.name
        for module_info in pkgutil.walk_packages(
            package.__path__, f"{package.__name__}."
        )
    ]
    return {
        module_name: importlib.import_module(module_name)
        for module_name in module_names
    }


def relative_base(module_name: str, *, is_package: bool, dot_count: int) -> str:
    """Return the package a relative import of ``dot_count`` dots starts from."""
    base_package = module_name if is_package else module_name.rpartition(".")[0]
    for _ in range(dot_count - 1):
        base_package = base_package.rpartition(".")[0]
    return base_package


def imported_names(source: str, module_name: str, *, is_package: bool) -> list[str]:
    """Return every module name the source imports, written out in full.

    An ``import a.b`` gives ``a.b``; a ``from a import b`` gives ``a`` and
    ``a.b``, since ``b`` may be a module; a relative import is resolved from
    ``module_name``; and a call to ``importlib.import_module`` or ``__import__``
    with a literal name gives that name.
    """
    imported: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = relative_base(
                    module_name, is_package=is_package, dot_count=node.level
                )
                imported_from = f"{base}.{node.module}" if node.module else base
            else:
                imported_from = node.module or ""
            imported.append(imported_from)
            imported.extend(f"{imported_from}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            called = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", "")
            )
            if (
                called in IMPORT_LOADERS
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                imported.append(node.args[0].value)
    return imported


def refused_names(imported: list[str], part_name: str) -> list[str]:
    """Return the names in ``imported`` that the part ``part_name`` may not import."""
    return [
        module_name
        for module_name in imported
        if (target_part := part_of(module_name)) is not None
        and target_part not in allowed_parts(part_name)
    ]


@pytest.mark.parametrize(
    ("source", "module_name", "is_package", "is_refused"),
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
    source: str, module_name: str, *, is_package: bool, is_refused: bool
) -> None:
    """Refuse an import from a part the module's part may not use, in any form."""
    part_name = module_name.split(".")[1]
    imported = imported_names(source, module_name, is_package=is_package)
    assert bool(refused_names(imported, part_name)) is is_refused


def test_every_part_imports_only_what_it_may() -> None:
    """Import, in every module of every part, only the parts that one may use."""
    project_parts = project_parts_found()
    assert {APP, DOMAIN, "das_processor"} <= set(project_parts)
    names_seen: dict[str, int] = {}
    refused_by_module: dict[str, list[str]] = {}
    for part_name in project_parts:
        names_seen[part_name] = 0
        for module_name, module_object in modules_of(part_name).items():
            imported = imported_names(
                inspect.getsource(module_object),
                module_name,
                is_package=hasattr(module_object, "__path__"),
            )
            names_seen[part_name] += len(imported)
            refused_by_module[module_name] = refused_names(imported, part_name)
    assert all(seen_count > 0 for seen_count in names_seen.values()), names_seen
    assert refused_by_module == {module_name: [] for module_name in refused_by_module}
