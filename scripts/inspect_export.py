"""Print aggregate statistics about a Jira export without showing issue content.

Usage:
    .venv/bin/python scripts/inspect_export.py /path/to/features.json /path/to/epics.json
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_agent.backlog import _HEADING, normalize_title  # noqa: E402
from feature_agent.sources import FileSource, _load_export  # noqa: E402

MIN_COUNT = 3
_BOLD_LINE = re.compile(r"^\s*\*([^*]{3,40})\*:?\s*$")


def main(features_path: Path, epics_path: Path) -> None:
    source = FileSource(features_path, epics_path)
    items = source.all_items()
    features = [i for i in items if i.issue_type == "Feature"]
    epics = [i for i in items if i.issue_type == "Epic"]
    epic_keys = {e.key for e in epics}

    print(f"Features: {len(features)}   Epics: {len(epics)}")
    for label, group in (("Feature", features), ("Epic", epics)):
        print(f"\n[{label}] status:", dict(Counter(i.status for i in group).most_common()))
        print(f"[{label}] PI:", dict(sorted(Counter(i.pi for i in group).items(), key=str)))
        lengths = sorted(len(i.description) for i in group)
        if lengths:
            print(f"[{label}] description chars: empty={sum(l == 0 for l in lengths)}, "
                  f"median={lengths[len(lengths) // 2]}, max={lengths[-1]}")

    print("\n[Feature] epic link: set=%d, resolvable=%d, missing=%d" % (
        sum(bool(f.epic_key) for f in features),
        sum(f.epic_key in epic_keys for f in features),
        sum(not f.epic_key for f in features)))
    print("[Feature] benefit hypothesis found:", sum(bool(f.benefit_hypothesis) for f in features))
    print("[Feature] acceptance criteria found:", sum(bool(f.acceptance_criteria) for f in features))

    raw = _load_export(features_path) + _load_export(epics_path)
    for label in ("Feature", "Epic"):
        group = [r for r in raw if r["fields"]["issuetype"]["name"] == label]
        headings, bold_lines, filled = Counter(), Counter(), Counter()
        for r in group:
            for line in (r["fields"].get("description") or "").replace("\r\n", "\n").split("\n"):
                if m := _HEADING.match(line):
                    headings[normalize_title(m.group(1))] += 1
                elif m := _BOLD_LINE.match(line):
                    bold_lines[normalize_title(m.group(1))] += 1
            filled.update(k for k, v in r["fields"].items() if v not in (None, "", [], {}))
        print(f"\n[{label}] description headings (>= {MIN_COUNT}x):",
              {h: c for h, c in headings.most_common() if c >= MIN_COUNT})
        print(f"[{label}] bold-only lines (>= {MIN_COUNT}x):",
              {h: c for h, c in bold_lines.most_common() if c >= MIN_COUNT})
        print(f"[{label}] field fill rate (field id: issues):",
              dict(sorted(filled.items())))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
