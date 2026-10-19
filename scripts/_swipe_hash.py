"""Print a stable hash of generate_swipes output on a fixed config/seed.

Used to pin the backward-compatibility guarantee: with taste_sigma = 0 and
shock_scale = 0 the swipe log must be bit-identical to what it was before the
preference-heterogeneity session. Run it before and after the change.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import time

import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "build" / "Release"))

from _hypergamic import generate_swipes, sample_population  # noqa: E402


def main() -> None:
    with open(_ROOT / "experiments" / "base.yaml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Both heterogeneity knobs off: this is the configuration pinned by
    # tests/python/test_heterogeneity.py, and the one whose hash is comparable to
    # any pre-heterogeneity recording. base.yaml ships them on, so they are forced
    # here rather than inherited.
    config["population"]["taste_sigma"] = 0.0
    config["hypergamy"]["shock_scale"] = 0.0

    seed = int(config["population"]["seed"])
    n = int(config["population"]["n"])

    digests = []
    for epoch in (0, 1, 2):
        log = generate_swipes(sample_population(n, config, seed), config, epoch)
        digests.append(hashlib.sha256(log.tobytes()).hexdigest())
        print(f"epoch {epoch}: rows={log.shape[0]:6d} sha256={digests[-1]}")

    combined = hashlib.sha256("".join(digests).encode()).hexdigest()
    print(f"combined sha256={combined}")

    states = sample_population(n, config, seed)
    t0 = time.perf_counter()
    for epoch in range(3):
        generate_swipes(states, config, epoch)
    elapsed = (time.perf_counter() - t0) / 3.0
    print(f"generate_swipes N={n}: {elapsed * 1000:.1f} ms/call")


if __name__ == "__main__":
    main()
