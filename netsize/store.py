"""Where the training runs are kept: one file per task in <out>/runs, one line of JSON per run."""
import json
import os


def file_for(out, task):
    return os.path.join(out, "runs", task.replace("@", "_at_") + ".jsonl")


def load_records(out):
    """Every saved run, grouped by network. A network saved more than once keeps its latest version."""
    latest = {}
    folder = os.path.join(out, "runs")
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".jsonl"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            rev = r.get("rev", 0)
                            if r["key"] not in latest or rev > latest[r["key"]][0]:
                                latest[r["key"]] = (rev, [])
                            if rev == latest[r["key"]][0]:
                                latest[r["key"]][1].append(r)
    return {key: records for key, (rev, records) in latest.items()}


def tidy(out, keep):
    """Rewrite the result files so each holds one version of each network in `keep`, in a steady order."""
    records = load_records(out)
    by_file = {}
    for key in sorted(records):
        if key in keep:
            by_file.setdefault(file_for(out, records[key][0]["task"]), []).extend(records[key])
    folder = os.path.join(out, "runs")
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if name.endswith(".jsonl") and path not in by_file:
            os.remove(path)
    for path, rs in by_file.items():
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
