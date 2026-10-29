"""The arXiv archive carries the current paper sources and nothing else.

paper/make_arxiv.py builds the archive and compiles it from an empty directory,
which needs TeX. These checks need none: they read the committed archive and
compare it with the files main.tex uses, so an edit to the paper that is not
followed by a rebuild of the archive fails here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import tarfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

spec = importlib.util.spec_from_file_location("make_arxiv", PAPER / "make_arxiv.py")
make_arxiv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(make_arxiv)


def members() -> dict[str, bytes]:
    with tarfile.open(make_arxiv.ARCHIVE) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_archive_holds_exactly_what_main_tex_uses() -> None:
    expected = {p.relative_to(PAPER).as_posix() for p in make_arxiv.sources()} | {"main.bbl"}
    assert set(members()) == expected


def test_archive_sources_match_the_repository() -> None:
    shipped = members()
    stale = [p.relative_to(PAPER).as_posix() for p in make_arxiv.sources()
             if shipped.get(p.relative_to(PAPER).as_posix()) != make_arxiv.content(p)]
    assert not stale, f"rerun paper/make_arxiv.py; stale in the archive: {stale}"


def test_shipped_bibliography_has_one_entry_per_cited_key() -> None:
    text = "".join(make_arxiv.uncommented(p.read_text(encoding="utf-8"))
                   for p in make_arxiv.sources() if p.suffix == ".tex")
    cited = {key.strip() for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", text)
             for key in group.split(",")}
    bbl = members()["main.bbl"].decode("utf-8")
    items = set(re.findall(r"\\bibitem(?:\[(?:[^\[\]]|\{[^{}]*\})*\])?\s*\{([^}]+)\}", bbl))
    assert cited and items == cited
