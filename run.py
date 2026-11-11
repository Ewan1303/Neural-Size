"""Run the experiments.

    python run.py                          # everything
    python run.py --list                   # show what would run and roughly how long it takes
    python run.py --only dial              # one experiment: grid, first_budget, sweeps, shapes, activations, dial
    python run.py --only digits1000        # ... or anything else that appears in a network's name
    python run.py --steps 0.1 --seeds 0 --out trial      # a few-minute miniature

Every finished network is saved to <out>/runs/<task>.jsonl, and anything already
there is skipped, so it is safe to stop and restart.

The work is split into independent pieces, one per network: a learning-rate
search on one seed, then two more seeds at the best rate (five training runs or
a few more). Each piece runs on a single core at below-normal priority.
"""
import argparse
import json
import os
import platform
import sys
import time

# one thread per piece: the pieces themselves are spread over the CPU cores
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from multiprocessing import Pool

import netsize.clock            # noqa: F401  (calibrates the CPU clock once, here, for every worker)
from netsize.experiments import (EXPERIMENTS, SEEDS, all_entries, estimate_seconds, search_is_open, train_network,
                                 unique_networks)
from netsize.store import file_for, load_records, tidy


def _init_worker():
    import torch
    torch.set_num_threads(1)
    try:                                   # let whatever else is running on the machine go first
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(5)
    except Exception:
        pass


def _work(args):
    job, steps_scale, seeds, previous = args
    t0 = time.time()
    try:
        return job, train_network(job, steps_scale, seeds, previous), time.time() - t0, None
    except Exception as e:                 # keep the rest of the batch going
        import traceback
        return job, [], time.time() - t0, f"{e}\n{traceback.format_exc()}"


def machine_info():
    import numpy
    import torch
    cpu = platform.processor() or platform.machine()
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "system": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="an experiment name, or text that appears in a network's name (task|activation|widths)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=float, default=1.0, help="multiply every task's number of training steps by this")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    seeds = tuple(a.seeds) if a.seeds else SEEDS
    entries = EXPERIMENTS[a.only]() if a.only in EXPERIMENTS else [e for e in all_entries() if a.only in e["key"]]
    jobs = unique_networks(entries)
    saved = load_records(a.out)
    todo, previous = [], {}
    for j in jobs:
        records = saved.get(j["key"])
        if records and len({r["seed"] for r in records}) == len(seeds) and not search_is_open(records):
            continue
        if records and len({r["seed"] for r in records}) == len(seeds):
            previous[j["key"]] = records                     # only its learning-rate search needs carrying on
        todo.append(j)
    for j in todo:
        j["estimate"] = estimate_seconds(j, a.steps, seeds) * (0.4 if j["key"] in previous else 1.0)
    todo.sort(key=lambda j: -j["estimate"])                  # longest first, so the cores stay busy to the end
    total = sum(j["estimate"] for j in todo)
    print(f"{len(jobs)} networks, {len(jobs) - len(todo)} already done, {len(todo)} to train "
          f"(about {total / 3600:.1f} CPU-hours, {total / 60 / a.workers:.0f} minutes on {a.workers} workers)")
    if a.list:
        by_exp = {}
        for j in todo:
            for exp in j["exps"]:
                n, s = by_exp.get(exp, (0, 0.0))
                by_exp[exp] = (n + 1, s + j["estimate"])
        for exp, (n, s) in by_exp.items():
            print(f"   {exp:14s} {n:4d} networks  {s / 60:6.1f} CPU-minutes (networks shared between experiments are counted in each)")
        for j in todo[:15]:
            print(f"   {j['key']:46s} {j['params']:>8} parameters  about {j['estimate']:5.0f} s")
        return 0
    if not todo:
        if a.only == "":
            tidy(a.out, {j["key"] for j in jobs})
        return 0
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    with open(os.path.join(a.out, "machine.json"), "w", encoding="utf-8") as f:
        json.dump(dict(machine_info(), seeds=list(seeds), steps_scale=a.steps), f, indent=1)

    t0 = time.time()
    failed, finished = 0, 0.0
    with Pool(a.workers, initializer=_init_worker) as pool:
        work = [(j, a.steps, seeds, previous.get(j["key"])) for j in todo]
        for k, (job, records, wall, err) in enumerate(pool.imap_unordered(_work, work), 1):
            finished += job["estimate"]
            elapsed = time.time() - t0
            left = elapsed * (total - finished) / max(finished, 1e-9)
            if err:
                failed += 1
                print(f"[{k:4d}/{len(todo)}] {job['key']:46s} ERROR: {err}", flush=True)
                continue
            rev = 1 + max((r.get("rev", 0) for r in saved.get(job["key"], [])), default=-1)
            with open(file_for(a.out, job["task"]), "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(dict(r, rev=rev), separators=(",", ":")) + "\n")
            best = [r for r in records if r["chosen"]]
            acc = sum(r["test_acc"] for r in best) / len(best)
            print(f"[{k:4d}/{len(todo)}] {job['key']:46s} {job['params']:>8} par  lr {best[0]['lr']:<6g} test {acc:.3f}"
                  f"  {wall:6.1f} s | {elapsed / 60:5.1f} min gone, about {left / 60:5.1f} left", flush=True)
    if a.only == "" and not failed:
        tidy(a.out, {j["key"] for j in jobs})                # one version of each network, nothing left over
    print(f"done in {(time.time() - t0) / 60:.1f} minutes" + (f", {failed} networks hit an error" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
