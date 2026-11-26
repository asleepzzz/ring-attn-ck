"""Build report.html from bench result JSONL files.

python3 bench/make_report.py results/bench.jsonl -o report.html
Fills report_template.html: __DATA__ (bench rows), __VALIDATION__ (correctness table rows).
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--validation", default=None, help="JSON file with correctness results")
    ap.add_argument("--notes", default=None, help="HTML fragment for the observations section")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    rows = []
    for p in args.jsonl:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    # keep the last measurement per (world, seqlen)
    dedup = {}
    for r in rows:
        dedup[(r["world"], r["seqlen"])] = r
    rows = sorted(dedup.values(), key=lambda r: (r["world"], r["seqlen"]))
    validation = [json.loads(l) for l in open(args.validation) if l.strip()] if args.validation else []
    html = open(os.path.join(HERE, "report_template.html"), encoding="utf-8").read()
    html = html.replace("__DATA__", json.dumps(rows)).replace("__VALIDATION__", json.dumps(validation))
    if args.notes:
        html = html.replace("<!--NOTES-->", open(args.notes, encoding="utf-8").read())
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out} ({len(rows)} configs)")


if __name__ == "__main__":
    main()
