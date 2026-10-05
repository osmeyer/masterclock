"""Make a PDF of every document, on demand; the PDFs are never committed.

The documents are the README at the top of the project and every Markdown
file under ``docs/``, the reports included. Each is turned into an HTML
page, its tables as tables, its mermaid blocks drawn by mermaid and its
LaTeX blocks set by MathJax, both loaded at fixed versions from the jsDelivr
CDN as the page is printed, so making the PDFs needs the network; each file
is checked against its hash, and Chrome runs no file that does not match. Headless
Chrome then prints each page to a PDF of the same name, in the same folders,
under the output folder. A link from one document to another points at the
other's PDF.

Give it the project folder, the folder for the PDFs, and Chrome::

    uv run --frozen python scripts/pdfs.py . /tmp/pdf --chrome /usr/bin/google-chrome

It prints each PDF it makes, and each document it could make none of, then
how many documents there were and how many PDFs it made. It exits 0 when it
made every one and 1 otherwise; a folder without ``pyproject.toml`` is
refused with exit status 2.
"""

import argparse
import html
import re
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from markdown_it import MarkdownIt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from markdown_it.renderer import RendererHTML
    from markdown_it.token import Token
    from markdown_it.utils import EnvType, OptionsDict

MERMAID_VERSION: Final = "11.4.1"
"""The version of mermaid the pages load."""

MATHJAX_VERSION: Final = "3.2.2"
"""The version of MathJax the pages load."""

MERMAID_INTEGRITY: Final = (
    "sha384-rbtjAdnIQE/aQJGEgXrVUlMibdfTSa4PQju4HDhN3sR2PmaKFzhEafuePsl9H/9I"
)
"""The hash of that version's mermaid.min.js: Chrome runs no other file."""

MATHJAX_INTEGRITY: Final = (
    "sha384-KKWa9jJ1MZvssLeOoXG6FiOAZfAgmzsIIfw8BXwI9+kYm0lPCbC6yTQPBC00F1/L"
)
"""The hash of that version's tex-svg.js: Chrome runs no other file."""

CHROME_WAIT_MS: Final = 30_000
"""How long Chrome lets a page's scripts run before it prints, ms."""

DOCUMENT_LINK: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>[^:#]*)\.md(?P<anchor>#.*)?$"
)
"""A link to another document: a relative path ending in .md, perhaps an anchor."""

STYLE: Final = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; margin: 0 1.5cm; }
table { border-collapse: collapse; margin: 0.5em 0; }
th, td { border: 1px solid #999; padding: 0.2em 0.4em; vertical-align: top; }
pre { background: #f4f4f4; padding: 0.5em; white-space: pre-wrap; }
pre.mermaid { background: none; text-align: center; }
code { font-size: 9pt; }
"""
"""The pages' style, which never changes the case of any text."""

PAGE: Final = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{style}</style>
<script>window.MathJax = {{tex: {{displayMath: [["$$", "$$"]]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@{mathjax}/es5/tex-svg.js"
 integrity="{mathjax_integrity}" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@{mermaid}/dist/mermaid.min.js"
 integrity="{mermaid_integrity}" crossorigin="anonymous"></script>
<script>mermaid.initialize({{startOnLoad: true}});</script>
</head>
<body>
{body}
</body>
</html>
"""
"""A whole page around a document's HTML."""


def _project_root(cli_argument: str) -> Path:
    """Convert a command-line argument to a folder that holds ``pyproject.toml``."""
    project_folder = Path(cli_argument)
    if not (project_folder / "pyproject.toml").is_file():
        refusal = f"not a project folder (no pyproject.toml): {cli_argument}"
        raise argparse.ArgumentTypeError(refusal)
    return project_folder


def documents(project_folder: Path) -> list[Path]:
    """Give the README, then every Markdown file under ``docs/``, sorted."""
    return [
        project_folder / "README.md",
        *sorted((project_folder / "docs").rglob("*.md")),
    ]


def _fence(
    tokens: Sequence[Token],
    idx: int,
    options: OptionsDict,
    env: EnvType,
    renderer: RendererHTML,
) -> str:
    """Render a fenced block: mermaid for mermaid, display maths for LaTeX."""
    token = tokens[idx]
    language = token.info.strip()
    if language == "mermaid":
        return f'<pre class="mermaid">{html.escape(token.content)}</pre>\n'
    if language == "latex":
        return f'<div class="math">$$\n{html.escape(token.content)}$$</div>\n'
    return renderer.fence(tokens, idx, options, env)


def _pdf_link(href: str) -> str:
    """Point a link to another document at its PDF; leave any other link alone."""
    document_link = DOCUMENT_LINK.match(href)
    if document_link is None:
        return href
    return f"{document_link['path']}.pdf{document_link['anchor'] or ''}"


def to_html(markdown: str) -> str:
    """Turn a document's Markdown into HTML."""
    converter = MarkdownIt("commonmark", {"html": True}).enable("table")

    def fence(
        renderer: RendererHTML,
        tokens: Sequence[Token],
        idx: int,
        options: OptionsDict,
        env: EnvType,
    ) -> str:
        """Render a fenced block (see :func:`_fence`)."""
        return _fence(tokens, idx, options, env, renderer)

    def link_open(
        renderer: RendererHTML,
        tokens: Sequence[Token],
        idx: int,
        options: OptionsDict,
        env: EnvType,
    ) -> str:
        """Render a link, pointed at a document's PDF (see :func:`_pdf_link`)."""
        token = tokens[idx]
        token.attrSet("href", _pdf_link(str(token.attrGet("href"))))
        return renderer.renderToken(tokens, idx, options, env)

    converter.add_render_rule("fence", fence)
    converter.add_render_rule("link_open", link_open)
    return str(converter.render(markdown))


def page(title: str, body: str) -> str:
    """Give the whole page of a document whose HTML is ``body``."""
    return PAGE.format(
        title=html.escape(title),
        style=STYLE,
        mathjax=MATHJAX_VERSION,
        mathjax_integrity=MATHJAX_INTEGRITY,
        mermaid=MERMAID_VERSION,
        mermaid_integrity=MERMAID_INTEGRITY,
        body=body,
    )


def print_pdf(chrome: Path, page_file: Path, pdf_file: Path) -> bool:
    """Print a page to a PDF with headless Chrome; tell whether the PDF was made."""
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    finished = subprocess.run(  # noqa: S603  # nosec B603
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--virtual-time-budget={CHROME_WAIT_MS}",
            f"--print-to-pdf={pdf_file}",
            f"file://{page_file}",
        ],
        capture_output=True,
        check=False,
    )
    return finished.returncode == 0 and pdf_file.is_file()


def main(argv: Sequence[str] | None = None) -> int:
    """Make a PDF of every document and return the exit status."""
    parser = argparse.ArgumentParser(description="Make a PDF of every document.")
    parser.add_argument("root", type=_project_root, help="the project folder")
    parser.add_argument("output", type=Path, help="the folder for the PDFs")
    parser.add_argument("--chrome", type=Path, required=True, help="Chrome's path")
    arguments = parser.parse_args(argv)
    project_folder: Path = arguments.root.resolve()
    output_folder: Path = arguments.output.resolve()
    document_files = documents(project_folder)
    made_count = 0
    with tempfile.TemporaryDirectory() as page_folder:
        for document_file in document_files:
            relative_file = document_file.relative_to(project_folder)
            page_file = Path(page_folder) / relative_file.with_suffix(".html")
            page_file.parent.mkdir(parents=True, exist_ok=True)
            markdown = document_file.read_text(encoding="utf-8")
            page_file.write_text(
                page(document_file.stem, to_html(markdown)), encoding="utf-8"
            )
            pdf_file = output_folder / relative_file.with_suffix(".pdf")
            if print_pdf(arguments.chrome, page_file, pdf_file):
                made_count += 1
                print(f"made {pdf_file}")
            else:
                print(f"no PDF made of {document_file}")
    print(f"pdfs: documents: {len(document_files)}, PDFs made: {made_count}")
    return 0 if made_count == len(document_files) else 1


if __name__ == "__main__":
    sys.exit(main())
