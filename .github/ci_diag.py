# TEMPORARY diagnostic: run identically on Windows and in CI, then diff.
import hashlib
import os
import sys

sys.path.insert(0, "build")
sys.path.insert(0, "build/Release")
sys.path.insert(0, "tests/python")
sys.path.insert(0, ".")
import numpy as np  # noqa: E402
import test_flooding as t  # noqa: E402
from src.python.orchestrator import Simulation, flooding_delta, update_thresholds  # noqa: E402

print("threads", os.environ.get("OMP_NUM_THREADS"))
for run in range(2):
    sim = Simulation(t.small_config(signal_cost=0.0, flooding_weight=0.0), recommender="oracle")
    sim.record_states = True
    frame = sim.run(8)
    print("run", run, "digest", t.trajectory_digest(sim, frame)[:16])
    for e, s in enumerate(sim.states_seen):
        print(" epoch", e, hashlib.sha256(s.tobytes()).hexdigest()[:12],
              "theta0-2", [repr(float(x)) for x in s[:3, 6]])
    print(" n_matched", frame["n_matched"].tolist())

volume, intent, theta, matched = t._fixture_market()
states = np.zeros((theta.size, 14))
states[:, 1] = 1.0
states[:, 6] = theta
states[:, 11] = intent
shipped = states.copy()
update_thresholds(shipped, volume.astype(np.int32), matched, 0.1)
print("yield", [repr(float(x)) for x in t._coefficients(shipped[:, 6] - theta, volume, intent)])
fl = shipped.copy()
fl[:, 6] += flooding_delta(volume, intent, (np.arange(theta.size),), 0.2, 0.5)
print("flood", [repr(float(x)) for x in t._coefficients(fl[:, 6] - theta, volume, intent)])

s0 = sim.states_seen[0]
print("shape", s0.shape, s0.dtype)
for c in range(s0.shape[1]):
    col = np.ascontiguousarray(s0[:, c])
    print(" col", c, hashlib.sha256(col.tobytes()).hexdigest()[:10], repr(float(col[0])), repr(float(col[5])), repr(float(col[100])))
