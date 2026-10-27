"""Check every bibliography DOI against Crossref, and look up candidate references.

    python paper/check_refs.py                 # verify refs.bib, write the report
    python paper/check_refs.py --lookup "title words" --author Surname --year 2018

For each entry of ``paper/refs.bib`` with a DOI, the Crossref record is fetched and
its title, first-author surname and year are compared with the entry. An entry
passes when the DOI resolves, the year matches, the surname matches, and the two
titles share at least 80% of their words. Entries without a DOI are listed for a
manual check against the publisher's page. The result is written to
``paper/refs_check.csv``; nothing in ``refs.bib`` is changed.

The lookup mode searches Crossref's bibliographic index and prints the best
matches with their DOIs, for adding a reference only after its record is seen.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

PAPER = Path(__file__).resolve().parent
BIB = PAPER / "refs.bib"
OUT = PAPER / "refs_check.csv"
AGENT = "hypergamic-matching-markets refcheck (mailto:none)"


def fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - network failure is reported, not raised
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def entries() -> list[dict[str, str]]:
    text = BIB.read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n(?=@)", text):
        head = re.match(r"@(\w+)\{([^,]+),", block.strip())
        if not head:
            continue
        fields = dict(re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", block))
        fields.update(kind=head.group(1), key=head.group(2))
        out.append(fields)
    return out


def words(text: str) -> set[str]:
    text = re.sub(r"\\[a-zA-Z]+|[{}\\'\"`^~]", "", text.lower())
    return {w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 2}


def surname(author_field: str) -> str:
    first = author_field.split(" and ")[0].strip()
    if not first:
        return ""
    name = first.split(",")[0] if "," in first else first.split()[-1]
    return re.sub(r"[^a-z]", "", re.sub(r"\\.|[{}]", "", name.lower()))


def check(entry: dict[str, str]) -> dict[str, str]:
    row = {"key": entry["key"], "doi": entry.get("doi", ""), "status": "", "detail": ""}
    if not row["doi"]:
        row["status"] = "no-doi"
        row["detail"] = entry.get("url", "")
        return row
    try:
        msg = fetch("https://api.crossref.org/works/" + urllib.parse.quote(row["doi"]))["message"]
    except Exception as exc:  # noqa: BLE001
        row["status"], row["detail"] = "unresolved", str(exc)[:120]
        return row
    title = " ".join(msg.get("title", []))
    # A journal article often carries two years: the online-first `issued` date
    # and the print volume's. A citation may use either, so both are accepted.
    years = {str((msg.get(k, {}).get("date-parts") or [[None]])[0][0])
             for k in ("issued", "published-print", "published-online")}
    year = "/".join(sorted(y for y in years if y != "None"))
    authors = msg.get("author", [])
    family = re.sub(r"[^a-z]", "", authors[0].get("family", "").lower()) if authors else ""
    a, b = words(entry.get("title", "")), words(title)
    overlap = len(a & b) / max(1, min(len(a), len(b)))
    problems = []
    if overlap < 0.8:
        problems.append(f"title overlap {overlap:.2f}: {title[:80]}")
    if entry.get("year", "") not in years:
        problems.append(f"year {year} vs {entry.get('year')}")
    want = surname(entry.get("author", ""))
    if family and want and family != want:
        problems.append(f"first author {family} vs {want}")
    row["status"] = "ok" if not problems else "MISMATCH"
    row["detail"] = "; ".join(problems) or title[:90]
    return row


def lookup(query: str, author: str | None, year: str | None) -> None:
    params = {"query.bibliographic": query, "rows": "5"}
    if author:
        params["query.author"] = author
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    for item in fetch(url)["message"]["items"]:
        y = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
        if year and str(y) != str(year):
            continue
        names = ", ".join(a.get("family", "") for a in item.get("author", [])[:4])
        venue = " ".join(item.get("container-title", []))[:60]
        print(f"{item.get('DOI')} | {y} | {names} | {' '.join(item.get('title', []))[:90]} "
              f"| {venue} | {item.get('volume', '')}({item.get('issue', '')}) {item.get('page', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lookup", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--year", default=None)
    args = parser.parse_args()
    if args.lookup:
        lookup(args.lookup, args.author, args.year)
        return 0
    rows = [check(e) for e in entries()]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["key", "doi", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['status']:<10} {r['key']:<20} {r['detail'][:100]}")
    bad = [r for r in rows if r["status"] not in ("ok", "no-doi")]
    print(f"\n{len(rows)} entries, {len(bad)} failing, "
          f"{sum(r['status'] == 'no-doi' for r in rows)} without a DOI")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
