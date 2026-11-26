"""Convert check_ring.py text logs into the JSONL that make_report.py --validation reads.

python3 bench/parse_check_logs.py results/logs/*.log -o results/validation.jsonl
A file name containing 'fault<N>' marks a negative-control run that dropped ring step N.
"""
import argparse, json, os, re

HEAD = re.compile(r"^\[W=(\d+)\] (.+) \(S=(\d+), B=(\d+), Hq=(\d+), Hkv=(\d+), D=(\d+)\): (PASS|FAIL)")
ROW = re.compile(r"^\s+(out|dq|dk|dv)\s+ring-vs-fp32 (\S+)\s+ck1-vs-fp32 (\S+)\s+ring-vs-ck1 (\S+)\s+(ok|FAIL)")

ap = argparse.ArgumentParser(); ap.add_argument("logs", nargs="+"); ap.add_argument("-o", required=True)
a = ap.parse_args()
recs = []
for p in a.logs:
    m = re.search(r"fault(\d+)", os.path.basename(p)); fault = int(m.group(1)) if m else None
    cur = None
    for line in open(p):
        if (hm := HEAD.match(line)):
            w, name, s, b, hq, hkv, d, res = hm.groups()
            cur = dict(world=int(w), case=name, seqlen=int(s), batch=int(b), hq=int(hq), hkv=int(hkv), d=int(d),
                       causal="causal" in name, dtype=name.split()[0], passed=res == "PASS", errors={}, fault_step=fault)
            recs.append(cur)
        elif cur is not None and (rm := ROW.match(line)):
            t, r, c, rc, ok = rm.groups()
            cur["errors"][t] = dict(ring=float(r), ck1=float(c), ring_vs_ck1=float(rc), ok=ok == "ok")
with open(a.o, "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
print(len(recs), "records")
