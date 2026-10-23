# TEMPORARY: print the full trajectory digests for the CI platform.
import sys

sys.path[:0] = ["build", "tests/python", "."]
import test_flooding as t  # noqa: E402
from src.python.orchestrator import Simulation  # noqa: E402

out = []
for cost in (0.0, 1.0):
    sim = Simulation(t.small_config(signal_cost=cost, flooding_weight=0.0), recommender="oracle")
    sim.record_states = True
    out.append(f"PIN {cost} {t.trajectory_digest(sim, sim.run(8))}")
print("::notice::" + "%0A".join(out))
