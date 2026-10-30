"""Build the arXiv source archive, paper/arxiv-source.tar.gz, and check that it compiles.

    python paper/make_arxiv.py

arXiv compiles a submission with pdfLaTeX and does not run BibTeX, so the archive
carries the generated ``main.bbl`` beside ``refs.bib``. Its contents are traced
from ``main.tex``: every file it inputs, every figure it includes and the
bibliography, and nothing else, so no old log or unused figure is shipped. Text
files are stored with LF line endings and every entry has the same timestamp and
owner, so the archive depends only on the sources.

The check extracts the archive into an empty directory and compiles it there the
way arXiv does: TeX alone, rerun until the labels settle, with no BibTeX, so
every citation has to resolve from the shipped main.bbl. It fails if the log
reports an undefined reference or citation, an overfull box or labels that have
not settled, or if the page count differs from the compile that made the
bibliography. Tectonic's engine is XeTeX, not pdfLaTeX, so the check finds a
missing file or a broken bibliography, not a difference between engines.
"""

from __future__ import annotations

import gzip
import io
import pathlib
import re
import subprocess
import tarfile
import tempfile

PAPER = pathlib.Path(__file__).resolve().parent
ARCHIVE = PAPER / "arxiv-source.tar.gz"
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
BIBLIOGRAPHY = re.compile(r"\\bibliography\{([^}]+)\}")
PROBLEMS = re.compile(r"Overfull \\hbox.*|LaTeX Warning: (?:Reference|Citation).*undefined.*")
PAGES = re.compile(r"Output written on \S+ \((\d+) pages?")
UNSETTLED = re.compile(r".*(?:Rerun to get|may have changed|There were undefined).*")
TEXT = {".tex", ".bib", ".bbl"}


def uncommented(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def resolve(name: str, suffixes: tuple[str, ...]) -> pathlib.Path:
    path = PAPER / name
    for candidate in (path, *(path.with_name(path.name + s) for s in suffixes)):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"main.tex uses {name}, which is not in paper/")


def sources() -> list[pathlib.Path]:
    """main.tex, then every file it inputs, every figure it includes and the .bib, in order."""
    found = {PAPER / "main.tex": None}
    queue = [PAPER / "main.tex"]
    while queue:
        text = uncommented(queue.pop(0).read_text(encoding="utf-8"))
        for name in INPUT.findall(text):
            path = resolve(name, (".tex",))
            if path not in found:
                found[path] = None
                queue.append(path)
        for name in GRAPHICS.findall(text):
            found.setdefault(resolve(name, (".pdf", ".png", ".jpg")), None)
        for names in BIBLIOGRAPHY.findall(text):
            for name in names.split(","):
                found.setdefault(resolve(name.strip(), (".bib",)), None)
    return list(found)


def content(path: pathlib.Path) -> bytes:
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix in TEXT else data


def write_archive(files: dict[str, bytes]) -> None:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    with open(ARCHIVE, "wb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb",
                                                   mtime=0) as gz:
        gz.write(raw.getvalue())


def compile_in(src: pathlib.Path, out: pathlib.Path, exe: str) -> str:
    out.mkdir()
    subprocess.run([exe, "--chatter", "minimal", "--keep-logs", "--keep-intermediates",
                    "--outdir", str(out), "main.tex"], cwd=src, check=True)
    return (out / "main.log").read_text(encoding="utf-8", errors="replace")


def compile_without_bibtex(src: pathlib.Path, exe: str) -> str:
    """TeX passes alone in ``src``, each reading the last one's .aux, until the labels settle."""
    for _ in range(4):
        subprocess.run([exe, "--chatter", "minimal", "--pass", "tex", "--keep-logs",
                        "--keep-intermediates", "main.tex"], cwd=src, check=True)
        log = (src / "main.log").read_text(encoding="utf-8", errors="replace")
        if not UNSETTLED.search(log):
            break
    return log


def page_count(log: str) -> int:
    match = PAGES.search(log)
    if not match:
        raise SystemExit("the log does not report a page count")
    return int(match.group(1))


def main() -> int:
    from build import tectonic  # paper/build.py; the script's own directory is first on sys.path

    exe = tectonic()
    files = {path.relative_to(PAPER).as_posix(): content(path) for path in sources()}
    with tempfile.TemporaryDirectory(prefix="arxiv_") as tmp:
        tmp = pathlib.Path(tmp)
        for name, data in files.items():
            (tmp / "sources" / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp / "sources" / name).write_bytes(data)
        pages = page_count(compile_in(tmp / "sources", tmp / "first", exe))
        files["main.bbl"] = content(tmp / "first" / "main.bbl")
        write_archive(files)

        (tmp / "check").mkdir()
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(tmp / "check", filter="data")
        log = compile_without_bibtex(tmp / "check", exe)
        failures = PROBLEMS.findall(log) + UNSETTLED.findall(log)
        if page_count(log) != pages:
            failures.append(f"{page_count(log)} pages from the archive, {pages} from the sources")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"{len(failures)} problems with paper/{ARCHIVE.name}")
    print(f"wrote paper/{ARCHIVE.name}: {len(files)} files, {pages} pages; it compiles "
          "clean from an empty directory with no BibTeX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
