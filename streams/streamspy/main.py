#! /usr/bin/python3
"""Run STREAmS using the gymnasium environment wrapper.

This script replicates the original ``main.py`` functionality but relies on
``StreamsEnvironment.StreamsGymEnv`` to manage solver initialization and time
stepping.  All diagnostics and HDF5 output are handled here using the same
utilities as before.
"""

import json
import os
import runpy

import numpy as np

from StreamsEnvironment import StreamsGymEnv

# ---------------------------------------------------------------------------
# If the configuration requests the adaptive jet, delegate to ``rl_control.py``
# ---------------------------------------------------------------------------
with open("/input/input.json", "r") as f:
    _cfg = json.load(f)
if "Adaptive" in _cfg.get("blowing_bc", {}):
    runpy.run_path(os.path.join(os.path.dirname(__file__), "rl_control.py"), run_name="__main__")
    raise SystemExit

# ---------------------------------------------------------------------------
# Instantiate environment and gather commonly used objects
# ---------------------------------------------------------------------------
env = StreamsGymEnv()
env.initialize_io()

# ---------------------------------------------------------------------------
# Main solver loop
# ---------------------------------------------------------------------------
obs = env.reset()
for _ in range(env.max_episode_steps):
    # ``StreamsGymEnv`` internally handles any actuation logic depending on the
    # chosen jet type.  The action value is ignored for non-adaptive jets.
    obs, reward, done, info = env.step(np.array([0.0], dtype=np.float32))
    env.log_step(info["jet_amplitude"])

# ---------------------------------------------------------------------------
# Finalize solver and MPI stack
# ---------------------------------------------------------------------------
env.close_io()
env.close()
