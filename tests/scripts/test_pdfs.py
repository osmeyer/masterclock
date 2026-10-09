"""Tests for scripts/pdfs.py.

The rules covered: the documents are the README at the top of the project
and every Markdown file under ``docs/``, reports included, each made into a
PDF of the same name under the output folder, in the same folders; a
document's Markdown becomes HTML with its tables, its mermaid blocks left
for mermaid to draw and its display maths, LaTeX between two lines of $$,
set for MathJax with its text as written, while a fenced latex block stays
code, its marker comments not shown, and its links to other documents
pointing at their PDFs; the page loads fixed versions of mermaid and
MathJax, each checked against its hash, and never changes the case of its
text; Chrome is run headless on
each page, and a document Chrome makes no PDF of is reported and the exit
status is 1; a folder without pyproject.toml is refused with status 2; and
run as a program, the script exits with the status main returns.
"""

import runpy
import stat
import sys
from pathlib import Path
from typing import Final

import pytest

import pdfs

FAKE_CHROME: Final = """#!{python}
import sys
from pathlib import Path
printed = [arg for arg in sys.argv if arg.startswith("--print-to-pdf=")]
page = Path(sys.argv[-1].removeprefix("file://"))
if "refuse" in page.read_text(encoding="utf-8"):
    sys.exit(3)
pdf_file = Path(printed[0].removeprefix("--print-to-pdf="))
pdf_file.write_bytes(b"%PDF " + page.name.encode())
"""
"""A stand-in for Chrome: it writes a PDF holding the page's name, or fails
when the page says "refuse"."""


def fake_chrome(tmp_path: Path) -> Path:
    """Write the stand-in for Chrome as an executable file, and give its path."""
    chrome = tmp_path / "chrome"
    chrome.write_text(FAKE_CHROME.format(python=sys.executable), encoding="utf-8")
    chrome.chmod(chrome.stat().st_mode | stat.S_IXUSR)
    return chrome


def make_project(project_root: Path) -> None:
    """Make a project with a README, two documents and a report."""
    (project_root / "docs" / "das_processor").mkdir(parents=True)
    (project_root / "docs" / "reports").mkdir()
    (project_root / "pyproject.toml").write_text("", encoding="utf-8")
    (project_root / "README.md").write_text(
        "# Project\n\nSee the [design](docs/das_processor/design.md#2-notation).\n",
        encoding="utf-8",
    )
    (project_root / "docs" / "das_processor" / "design.md").write_text(
        "# Design\n\nBack to the [README](../../README.md).\n", encoding="utf-8"
    )
    (project_root / "docs" / "das_processor" / "notes.txt").write_text(
        "not a document", encoding="utf-8"
    )
    (project_root / "docs" / "reports" / "security.md").write_text(
        "# Security\n", encoding="utf-8"
    )


def test_the_documents_are_the_readme_and_every_markdown_file_under_docs(
    tmp_path: Path,
) -> None:
    """Give the README, then every .md file under docs/, sorted."""
    make_project(tmp_path)
    assert pdfs.documents(tmp_path) == [
        tmp_path / "README.md",
        tmp_path / "docs" / "das_processor" / "design.md",
        tmp_path / "docs" / "reports" / "security.md",
    ]


def test_a_table_becomes_an_html_table() -> None:
    """Turn a Markdown table into an HTML table."""
    html = pdfs.to_html("| a | b |\n| --- | --- |\n| 1 | 2 |\n")
    assert "<table>" in html
    assert "<td>2</td>" in html


def test_a_mermaid_block_is_left_for_mermaid_to_draw() -> None:
    """Give a mermaid block as a pre of class mermaid, its text escaped."""
    html = pdfs.to_html("```mermaid\nflowchart LR\n    A --> B\n```\n")
    assert '<pre class="mermaid">flowchart LR\n    A --&gt; B\n</pre>' in html


def test_display_maths_is_left_for_mathjax() -> None:
    """Give a $$ block between $$ marks, for MathJax, its text escaped."""
    html = pdfs.to_html("Before.\n\n$$\nx < y\n$$\n\nAfter.\n")
    assert '<div class="math">$$\nx &lt; y\n$$</div>' in html
    assert "<p>Before.</p>" in html
    assert "<p>After.</p>" in html


def test_display_maths_keeps_its_backslashes() -> None:
    """Keep the LaTeX as written; Markdown drops a backslash before a brace."""
    html = pdfs.to_html("$$\nG = \\max\\left\\{ n : x_{n} \\right\\}\n$$\n")
    assert "$$\nG = \\max\\left\\{ n : x_{n} \\right\\}\n$$" in html


def test_a_fenced_latex_block_stays_code() -> None:
    """Give a fenced latex block as code: display maths is written between $$."""
    html = pdfs.to_html("```latex\nx\n```\n")
    assert '<pre><code class="language-latex">x\n</code></pre>' in html


def test_another_code_block_stays_code() -> None:
    """Give any other fenced block as code, as Markdown does."""
    html = pdfs.to_html("```python\nx = 1\n```\n")
    assert '<pre><code class="language-python">x = 1\n</code></pre>' in html


def test_marker_comments_are_not_shown() -> None:
    """Keep the documents' marker comments as comments, never as text."""
    html = pdfs.to_html("A <!-- figure: K -->5<!-- end figure --> ps.\n")
    assert "<!-- figure: K -->5<!-- end figure -->" in html
    assert "&lt;!--" not in html


@pytest.mark.parametrize(
    ("markdown_link", "pdf_link"),
    [
        ("[d](design.md)", 'href="design.pdf"'),
        ("[d](../../README.md#documents)", 'href="../../README.pdf#documents"'),
        ("[w](https://example.com/a.md)", 'href="https://example.com/a.md"'),
        ("[s](#2-notation)", 'href="#2-notation"'),
    ],
)
def test_a_link_to_a_document_points_at_its_pdf(
    markdown_link: str, pdf_link: str
) -> None:
    """Point a link to another document at its PDF, and leave other links alone."""
    assert pdf_link in pdfs.to_html(markdown_link + "\n")


def test_the_page_loads_fixed_versions_of_mermaid_and_mathjax() -> None:
    """Load each library from a URL that names its version, checked by its hash."""
    page = pdfs.page("Design", "<p>x</p>")
    assert f"mermaid@{pdfs.MERMAID_VERSION}/" in page
    assert f"mathjax@{pdfs.MATHJAX_VERSION}/" in page
    assert f'integrity="{pdfs.MERMAID_INTEGRITY}"' in page
    assert f'integrity="{pdfs.MATHJAX_INTEGRITY}"' in page
    assert "<title>Design</title>" in page
    assert "<p>x</p>" in page


def test_the_page_never_changes_the_case_of_its_text() -> None:
    """Never transform text, so a unit symbol keeps its case."""
    assert "text-transform" not in pdfs.page("Design", "<p>1 µs</p>")


def test_every_document_becomes_a_pdf_in_the_same_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print a PDF of each document under the output folder, and exit 0."""
    project_root, output_folder = tmp_path / "project", tmp_path / "pdf"
    project_root.mkdir()
    make_project(project_root)
    exit_status = pdfs.main(
        [
            str(project_root),
            str(output_folder),
            "--chrome",
            str(fake_chrome(tmp_path)),
        ]
    )
    assert exit_status == 0
    made = sorted(
        pdf_file.relative_to(output_folder) for pdf_file in output_folder.rglob("*.pdf")
    )
    assert made == [
        Path("README.pdf"),
        Path("docs") / "das_processor" / "design.pdf",
        Path("docs") / "reports" / "security.pdf",
    ]
    assert (output_folder / "README.pdf").read_bytes() == b"%PDF README.html"
    assert capsys.readouterr().out.endswith("pdfs: documents: 3, PDFs made: 3\n")


def test_a_document_chrome_cannot_print_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report a document Chrome made no PDF of, go on with the rest, and exit 1."""
    project_root, output_folder = tmp_path / "project", tmp_path / "pdf"
    project_root.mkdir()
    make_project(project_root)
    (project_root / "docs" / "reports" / "security.md").write_text(
        "# Security\n\nrefuse\n", encoding="utf-8"
    )
    exit_status = pdfs.main(
        [
            str(project_root),
            str(output_folder),
            "--chrome",
            str(fake_chrome(tmp_path)),
        ]
    )
    assert exit_status == 1
    printed = capsys.readouterr().out
    assert f"no PDF made of {project_root / 'docs' / 'reports' / 'security.md'}" in (
        printed
    )
    assert printed.endswith("pdfs: documents: 3, PDFs made: 2\n")


def test_a_folder_without_pyproject_is_refused(tmp_path: Path) -> None:
    """Exit 2 for a folder that is not a project, making nothing."""
    with pytest.raises(SystemExit) as system_exit:
        pdfs.main([str(tmp_path), str(tmp_path / "pdf"), "--chrome", "/bin/true"])
    assert system_exit.value.code == 2
    assert not (tmp_path / "pdf").exists()


def test_run_as_a_program_it_exits_with_main_s_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit with the status main returns when run as a script."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    make_project(project_root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pdfs.py",
            str(project_root),
            str(tmp_path / "pdf"),
            "--chrome",
            str(fake_chrome(tmp_path)),
        ],
    )
    with pytest.raises(SystemExit) as system_exit:
        runpy.run_path(str(Path(pdfs.__file__)), run_name="__main__")
    assert system_exit.value.code == 0
