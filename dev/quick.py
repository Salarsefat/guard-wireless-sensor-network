"""Development-only quick runner (DEV seeds 3001+). Never used for TEST."""
import json, sys, time, itertools
from multiprocessing import Pool
sys.path.insert(0, "src")
from wsncc.runner import run_one

TUNED = json.load(open("results/tuned_params.json"))
KEYS = ["tdr_1", "gdr", "delay_mean", "delay_p95", "delay_p99", "energy_per_reading_mJ",
        "jain", "frames_per_reading", "readings_per_frame", "hops_mean", "coll_frac_all",
        "sink_coll_frac", "drop_retry", "drop_caf", "drop_buf", "queue_max", "fnd", "dec_us"]

def job(args):
    m, params, scen, seed = args
    t = time.time()
    r = run_one(m, params, scen, seed)
    r["wall"] = time.time() - t
    return r

def params_of(m, override=None):
    p = dict(TUNED[m]["params"]) if m in TUNED else {}
    if override:
        p.update(override)
    return p

def run(methods, scens, seeds, procs=2, overrides=None):
    jobs = []
    for m, sc, sd in itertools.product(methods, scens, seeds):
        name, ov = (m if isinstance(m, str) else m[0]), (None if isinstance(m, str) else m[1])
        jobs.append((name, params_of(name.split(":")[0], ov), sc, sd))
    with Pool(procs) as pool:
        res = pool.map(job, [(j[0].split(":")[0], j[1], j[2], j[3]) for j in jobs])
    for j, r in zip(jobs, res):
        r["label"] = j[0] if j[1] == params_of(j[0].split(":")[0]) else j[0] + json.dumps(j[1])
    return res
