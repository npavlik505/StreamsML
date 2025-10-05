# Gym and standard imports
import os
import sys
import json
import math
from collections import deque
import numpy as np
import gymnasium
from gymnasium import spaces
from pathlib import Path
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import libstreams as streams # f2py‐wrapped STREAmS library

# ------------------------------------------------------------------
# Transforming STREAmS into a Gym Environment (initialization, restart, step)
# ------------------------------------------------------------------
# Gym Environment: Initialization
class StreamsGymEnv(gymnasium.Env):
    """
    # Observation: the 1D wall‐shear‐stress array τw(x) (length = config.grid.nx)
    Observation: 2D span-averaged U velocity from start from x[0, slot_end], y[0, ny]
    Action:      a single "jet amplitude" scalar ∈ [ -max_amplitude, +max_amplitude ]

    Reward:      Negative squared‐L2 norm of τw(x)  (i.e. agent tries to minimize shear stress)
                 τw(x) is the 1D wall‐shear‐stress array (length = config.grid.nx)
    Done:        When `self.step_count >= self.max_episode_steps`.

    General use (see main.py loops):
        env = StreamsGymEnv(config_path="/input/input.json", max_amplitude=1.0, max_episode_steps=200)
        obs = env.reset()
        for _ in range(max_episode_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
    """
    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        # Start MPI, initialize global variables and import the scripts that rely on the global variables
        streams.wrap_startmpi()
        from mpi4py import rc
        rc.initialize = False
        rc.finalize = False
        from mpi4py import MPI  # solver MPI must be started (wrap_startmpi()) before mpi4py library import
        import globals # contains rank/comm initialization
        #from . import globals  # contains rank/comm initialization        
        globals.init()
        self.rank = globals.rank
        self.comm = globals.comm
        import io_utils # for HDF5
        from config import Config # input.json to Config object
        import utils # helper code; calculate_span_averages, etc.
        import jet_actuator # all actuator code
        #from . import io_utils  # for HDF5
        #from .config import Config  # input.json to Config object
        #from . import utils  # helper code; calculate_span_averages, etc.
        #from . import jet_actuator  # all actuator code
        self.jet_actuator = jet_actuator # bind to self so that it can be used in step method

        # Parse, and later access the entries of, input.json using config
        with open("/input/input.json", "r") as f:
            cfg_json = json.load(f)
            self.config = Config.from_json(cfg_json)

        # Allocate Arrays
        span_average = np.zeros([5, self.config.nx_mpi(), self.config.ny_mpi()], dtype=np.float64)
        temp_field = np.zeros((self.config.nx_mpi(), self.config.ny_mpi(), self.config.nz_mpi()), dtype=np.float64)
        dt_array = np.zeros(1)
        amplitude_array = np.zeros(1)
        time_array = np.zeros(1)
        dissipation_rate_array = np.zeros(1)
        energy_array = np.zeros(1)

        # Execute STREAmS setup routines (wrap_setup and wrap_init_solver)
        self._setup_solver()

        # ``tauw_shape`` is the number of wall points owned by this MPI rank.
        # The global observation is assembled by gathering data from all ranks in `reset` and `step`.
        self.tauw_shape = streams.wrap_get_tauw_x_shape()
        
        # Note the index (first column) is used for Fortran and is therefore 1-based not 0-based
        obs_types = {
            "rho":        (1, 0.0,     10.0),
            "u":          (2, -3e3,    3e3),
            "v":          (3, -3e3,    3e3),
            "w":          (4, -3e3,    3e3),
            "p":          (5, 0.0,     1e7),
            "T":          (6, 50.0,    5e3),
            "rho_sqrd":   (7, 0.0,     1e2),
            "u_sqrd":     (8, 0.0,     9e6),
            "v_sqrd":     (9, 0.0,     9e6),
            "w_sqrd":     (10, 0.0,     9e6),
            "p_sqrd":     (11, 0.0,    1e14),
            "T_sqrd":     (12, 2.5e3,  2.5e7),
            "rhou":       (13, -3e4,   3e4),
            "rhov":       (14, -3e4,   3e4),
            "rhow":       (15, -3e4,   3e4),
            "rhou_sqrd":  (16, 0.0,    9e7),   # ⟨ρ u²⟩
            "rhov_sqrd":  (17, 0.0,    9e7),   # ⟨ρ v²⟩
            "rhow_sqrd":  (18, 0.0,    9e7),   # ⟨ρ w²⟩
            "rhouv":      (19, -9e7,   9e7),
            "dyn_visc":   (20, 1e-6,   1e-3)
        }
        jet_params = self.config.jet.jet_params
        obs_key = jet_params["obs_type"]
        result = obs_types.get(obs_key)
        if result is None:
            valid = ", ".join(obs_types.keys())
            raise ValueError(f"obs_type '{obs_key} not valid. Choose one of: {valid}")
        self._obs_index, min_val, max_val = result
        
        # Store observation bounds so they can be reused when the observation
        # window is recomputed programmatically after initialization.
        self._obs_min_val = min_val
        self._obs_max_val = max_val
        
        if self.rank == 0:
            print(f"Obs consists of {obs_key} data -> index {self._obs_index}, range [{min_val}, {max_val}]")

        self._obs_xstart = int(jet_params["obs_xstart"])
        self._obs_xend = int(jet_params["obs_xend"])
        self._obs_ystart = int(jet_params["obs_ystart"])
        self._obs_yend = int(jet_params["obs_yend"])
        if self.rank == 0:
            print(f"Obs space,  X: {self._obs_xstart} to {self._obs_xend} | Y: {self._obs_ystart} to {self._obs_yend}")

        ## Determine the local slice shape to size the observation space.  The
        ## first dimension is the variable index and has length one, so only the
        ## spatial extents are relevant.
        #sample_slice = streams.wrap_get_w_avzg_slice_gpu(
        #    self._obs_xstart,
        #    self._obs_xend,
        #    self._obs_ystart,
        #    self._obs_yend,
        #    self._obs_index,
        #)
        
        # Determine the portion of the observation window owned by this MPI
        # rank.  Ranks whose local domain does not intersect the requested
        # region return an empty slice.
        nx_local = self.config.nx_mpi()
        global_x_start = self.rank * nx_local + 1
        global_x_end = global_x_start + nx_local - 1
        self._local_xstart = max(self._obs_xstart, global_x_start)
        self._local_xend = min(self._obs_xend, global_x_end)
        self._local_ystart = max(self._obs_ystart, 1)
        self._local_yend = min(self._obs_yend, self.config.grid.ny)
        self._has_values = (
            self._local_xstart <= self._local_xend
            and self._local_ystart <= self._local_yend
        )        
        
        #d1size, d2size = sample_slice.shape[1:3]
        #self._has_values = (d1size > 0 and d2size > 0)
        
        if self._has_values:
            sample_slice = streams.wrap_get_w_avzg_slice_gpu(
                self._local_xstart,
                self._local_xend,
                self._local_ystart,
                self._local_yend,
                self._obs_index,
            )
            d1size, d2size = sample_slice.shape[1:3]
        else:
            d1size = d2size = 0
        
        local_obs_size = d1size * d2size
        global_obs_size = self.comm.allreduce(local_obs_size)
        self._global_obs_size = global_obs_size
        
        w1, w2, w3, w4  = streams.wrap_get_w_shape() # conservative vector (x, y, z, (rho, rho-u, rho-v, rho-w, E))
        self.w_shape   = (w1, w2, w3, w4)
        
        # Define the observation_space as config.grid.nx, the number of grid points in the x (streamwise) direction
        self.nx = self.config.grid.nx

        grid_shape = [self.config.grid.nx, self.config.grid.ny, self.config.grid.nz]
        span_average_shape = [self.config.grid.nx, self.config.grid.ny]

        # 3D flowfield files
        if self.config.temporal.full_flowfield_io_steps not in (None, 0):
            flowfield_writes = int(math.ceil(self.config.temporal.num_iter / self.config.temporal.full_flowfield_io_steps))
        else:
            flowfield_writes = 0

        # span average files
        numwrites = int(math.ceil(self.config.temporal.num_iter / self.config.temporal.span_average_io_steps))

        # Generate Mesh (includes ghost nodes)
        x_mesh = streams.wrap_get_x(self.config.x_start(), self.config.x_end()) 
        y_mesh = streams.wrap_get_y(self.config.y_start(), self.config.y_end())
        z_mesh = streams.wrap_get_z(self.config.z_start(), self.config.z_end())

        # Initialize actuator
        self.actuator = jet_actuator.init_actuator(self.rank, self.config)

        if self.config.jet.jet_method_name == "LearningBased":
            #
            # BEGIN DEVELOPER SECTION: DEFINE YOUR OBSERVATION SPACE
            #
            # Observation Space: Reasonable bounds defined in obs_types dict above. Flattened ROI as shape.
            global_obs_size = self._global_obs_size
            low_obs = np.full((global_obs_size,), min_val, dtype=np.float32)
            high_obs = np.full((global_obs_size,), max_val, dtype=np.float32)
            self.observation_space = spaces.Box(
                                                low=low_obs,
                                                high=high_obs,
                                                shape=(global_obs_size,),
                                                dtype=np.float32)
            
            #
            # END DEVELOPER SECTION: DEFINE YOUR OBSERVATION SPACE
            #

            # Action Space (Bounds determined from input.json, extracted upon actuator initialization above)
            # Action = single continuous amplitude ∈ [ -max_amplitude, +max_amplitude ]
            self.max_amplitude = float(self.actuator.amplitude)
            self.action_space = spaces.Box(low=np.array([-self.max_amplitude], dtype=np.float32),
                                           high=np.array([+self.max_amplitude], dtype=np.float32),
                                           shape=(1,),
                                           dtype=np.float32)

        # Step counting and time
        self.step_count = 0
        self.max_episode_steps = int(self.config.temporal.num_iter)
        self.current_time = 0.0

        # Store actuator geometry for delayed actions
        self.slot_start = int(jet_params["slot_start"])
        self.slot_end = int(jet_params["slot_end"])
        self.sensor_actuator_delay = bool(jet_params.get("sensor_actuator_delay"))

        # Parameters for delayed reward via eligibility traces
        # "lag_steps" controls how many steps elapse before credit is assigned to an action.
        # "lambda_trace" controls exponential decay of eligibility for accumulating rewards.
        # TODO: Calculate lag_steps based on convection jet and let user specify their own lag_steps to overwrite calculation if specified
        self.action_queue = deque()
        self.lag_steps = jet_params["lag_steps"]
        self.lambda_trace = 1.0 - (1.0 / self.lag_steps)

        # State for convection-delayed actuation
        self._delay_actuation_queue = deque()
        self._delay_observation_queue = deque()
        self._delay_skip = False

        if self.rank == 0:
            print(f'[StreamsEnvironment.py] gym environment initialized')
            print(f'Step_count: {self.step_count}')
            print(f'max_episode_steps (--steps in justfile): {self.max_episode_steps}')
            print(f'current_time: {self.current_time}')

    # A helper function for the set_observation_window() method below, which dynamically resets the observation window
    def recompute_obs(self) -> None:
        """Recompute derived observation window values.

        This is useful when ``_obs_xstart``/``_obs_xend``/``_obs_ystart``/
        ``_obs_yend`` are modified programmatically after environment
        creation.  The local slice indices, global observation size and the
        ``observation_space`` are updated in-place so that a subsequent call to
        :meth:`reset` reflects the new window.
        """

        nx_local = self.config.nx_mpi()
        global_x_start = self.rank * nx_local + 1
        global_x_end = global_x_start + nx_local - 1
        self._local_xstart = max(self._obs_xstart, global_x_start)
        self._local_xend = min(self._obs_xend, global_x_end)
        self._local_ystart = max(self._obs_ystart, 1)
        self._local_yend = min(self._obs_yend, self.config.grid.ny)
        self._has_values = (
            self._local_xstart <= self._local_xend
            and self._local_ystart <= self._local_yend
        )

        if self._has_values:
            sample_slice = streams.wrap_get_w_avzg_slice_gpu(
                self._local_xstart,
                self._local_xend,
                self._local_ystart,
                self._local_yend,
                self._obs_index,
            )
            d1size, d2size = sample_slice.shape[1:3]
        else:
            d1size = d2size = 0

        local_obs_size = d1size * d2size
        self._global_obs_size = self.comm.allreduce(local_obs_size)

        if hasattr(self, "observation_space"):
            low_obs = np.full(
                (self._global_obs_size,), self._obs_min_val, dtype=np.float32
            )
            high_obs = np.full(
                (self._global_obs_size,), self._obs_max_val, dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=low_obs,
                high=high_obs,
                shape=(self._global_obs_size,),
                dtype=np.float32,
            )

    # Dynamically reset the observation window (useful for moving and morphing windows)
    def set_observation_window(self, xstart: int, xend: int, ystart: int, yend: int) -> None:
        """Override the observation bounds and recompute derived fields."""

        self._obs_xstart = int(xstart)
        self._obs_xend = int(xend)
        self._obs_ystart = int(ystart)
        self._obs_yend = int(yend)
        self.recompute_obs()

    # MPI helper function
    def _gather_nonempty(self, arr: np.ndarray) -> np.ndarray:
        """Gather 1D data from all ranks while skipping empty contributions.

        Parameters
        ----------
        arr : np.ndarray
            Local array which may have zero length on ranks that do not own a
            portion of the domain for a particular observation.

        Returns
        -------
        np.ndarray
            Concatenated global array containing only the non-empty slices from
            each MPI rank.
        """

        gathered = self.comm.allgather(np.asarray(arr))
        non_empty = [np.ravel(a) for a in gathered if a.size > 0]
        if non_empty:
            return np.concatenate(non_empty)
        return np.empty(0, dtype=arr.dtype)

    # setup_solver definition: initialized solver on first call, closes solver and reinits on subsequent calls. Used in reset method.
    def _setup_solver(self, *, restart_mpi: bool = False) -> None:
        """(Re)initialize the STREAmS solver.

        ``wrap_setup`` and ``wrap_init_solver`` must be called exactly once for a
        running solver.  When resetting the environment we only finalize the
        solver via ``wrap_finalize_solver`` while **keeping MPI alive**.  If a
        full MPI shutdown is required, set ``restart_mpi=True`` to also call
        ``wrap_finalize`` followed by ``wrap_startmpi`` before reinitializing the
        solver.
        """
        
        if self.rank == 0:
            print('[StreamsEnvironment.py] PYTHON _SETUP_SOLVER METHOD CALLED')

        if restart_mpi:
            # Finalize the solver and MPI stack completely.  This is normally
            # only necessary when shutting down the environment or reloading the
            # Python module.
            try:
                streams.wrap_finalize_solver()
                if self.rank == 0:
                    print('[StreamsEnvironment.py] streams.wrap_finalize_solver() called')
            except Exception:
                pass
            try:
                streams.wrap_finalize()
                if self.rank == 0:
                    print('[StreamsEnvironment.py] streams.wrap_finalize() called')
            except Exception:
                pass
            try:
                streams.wrap_deallocate_all()
                if self.rank == 0:
                    print('[StreamsEnvironment.py] streams.wrap_deallocate_all() called')
            except Exception:
                pass
            # When MPI is fully finalized we must start it again before
            # continuing with solver setup.
            streams.wrap_startmpi()
            if self.rank == 0:
                print('[StreamsEnvironment.py] streams.wrap_start_mpi() called')
        else:
            # Standard environment reset: only tear down the solver while MPI
            # remains active.  This avoids the cost and side effects of a full
            # MPI finalize/startup cycle.
            try:
                streams.wrap_finalize_solver()
                if self.rank == 0:
                    print('[StreamsEnvironment.py] streams.wrap_finalize_solver() called')
            except Exception:
                pass
            try:
                streams.wrap_deallocate_all()
                if self.rank == 0:
                    print('[StreamsEnvironment.py] streams.wrap_deallocate_all() called')
            except Exception:
                pass

        # Reinitialize solver data structures.
        streams.wrap_setup()
        streams.wrap_init_solver()
        self.current_time = 0.0
        self.step_count = 0

    # Gym Environment: Restart
    def reset(self, *, seed=None, options=None):
        """
        Re‐initializes the STREAmS solver to a 'cold start' (no previous steps
        taken), then returns the initial span‑averaged streamwise velocity over
        the slot as the observation.
        """
        if self.rank == 0:
            print('[StreamsEnvironment.py] PYTHON RESET() METHOD CALLED')

        super().reset(seed=seed) # seeding not currently used, but kept for future use
        
        self._setup_solver() # End previous solver and rebuild it
        self.actuator = self.jet_actuator.init_actuator(self.rank, self.config) # Re‐build actuator


        #
        # BEGIN DEVELOPER SECTION: RETURN YOUR INITIAL OBSERVATION
        #
        # Immediately compute the span‑averaged conservative variables on the new solver (no time steps taken yet).
        streams.wrap_compute_av() # update the w_avzg
        if self._has_values:
            local_slice = streams.wrap_get_w_avzg_slice_gpu(
                self._local_xstart,
                self._local_xend,
                self._local_ystart,
                self._local_yend,
                self._obs_index,
            )
            local_slice = np.ravel(local_slice)
        else:
            local_slice = np.empty((0,), dtype=np.float32)

        # Gather observation values from all MPI ranks and flatten, ignoring
        # empty slices
        u_global = self._gather_nonempty(local_slice)

        # Reset counters
        self.step_count = 0
        self.current_time = 0.0
        # Rest action_queue
        self.action_queue.clear()
        self._delay_actuation_queue.clear()
        self._delay_observation_queue.clear()
        self._delay_skip = False

        # Gym expects a float32 array
        return u_global.astype(np.float32)
        #
        # END DEVELOPER SECTION: RETURN YOUR INITIAL OBSERVATION
        #

    def delay_action(self, action, observation):
        """Delay a control signal until it convects from sensors to actuator."""

        convection_complete = False

        def _zero_like(obs):
            if torch is not None and isinstance(obs, torch.Tensor):
                return torch.zeros_like(obs)
            if isinstance(obs, np.ndarray):
                return np.zeros_like(obs)
            if isinstance(obs, (list, tuple)):
                zeros = [0 for _ in obs]
                return type(obs)(zeros)
            return 0.0

        def _as_float(value):
            if torch is not None and isinstance(value, torch.Tensor):
                flat = value.detach().reshape(-1)
                if flat.numel() == 0:
                    return 0.0
                return float(flat[0].item())
            arr = np.asarray(value)
            if arr.size == 0:
                return 0.0
            return float(arr.reshape(-1)[0])

        default_prev_obs = _zero_like(observation)
        default_next_obs = _zero_like(observation)

        if self.step_count == 0:
            self._delay_actuation_queue.clear()
            self._delay_observation_queue.clear()
            self._delay_skip = False
            contiguous = self._obs_xend == self.slot_start
            overlaps = self._obs_xend > self.slot_start
            self._delay_skip = contiguous or overlaps
            if self._delay_skip and self.rank == 0:
                print(
                    f"observation window x: {self._obs_xstart}-{self._obs_xend} must be upstream from actuator x: {self.slot_start}-{self.slot_end}"
                )
                print("delay will not be applied")

        if self._delay_skip or not self.sensor_actuator_delay:
            return _as_float(action), default_prev_obs, default_next_obs, False

        self._delay_actuation_queue.append({"actuation": action, "convection": 0.0})
        self._delay_observation_queue.append({"observation": observation})

        rho_slice = streams.wrap_get_w_avzg_slice_gpu(
            self._obs_xstart,
            self._obs_xend,
            self._obs_ystart,
            self._obs_yend,
            1,
        )
        rhou_slice = streams.wrap_get_w_avzg_slice_gpu(
            self._obs_xstart,
            self._obs_xend,
            self._obs_ystart,
            self._obs_yend,
            2,
        )
        u_slice = rhou_slice[0] / rho_slice[0]
        Uc = float(np.mean(u_slice))

        dt = float(streams.wrap_get_dtglobal())
        dx = self.config.length.lx / self.config.grid.nx

        sensor_centroid = 0.5 * (self._obs_xstart + self._obs_xend)
        slot_centroid = 0.5 * (self.slot_start + self.slot_end)
        distance_index = slot_centroid - sensor_centroid

        for entry in self._delay_actuation_queue:
            entry["convection"] += (Uc * dt) / dx

        step_actuation = 0.0
        step_observation = _zero_like(observation)
        next_step_observation = _zero_like(observation)
        if (
            self._delay_actuation_queue
            and self._delay_actuation_queue[0]["convection"] >= distance_index
        ):
            step_actuation = self._delay_actuation_queue.popleft()["actuation"]
            step_observation = self._delay_observation_queue.popleft()["observation"]
            if self._delay_observation_queue:
                next_step_observation = self._delay_observation_queue[0]["observation"]
                convection_complete = True

        return (
            _as_float(step_actuation),
            step_observation,
            next_step_observation,
            convection_complete,
        )


    # Gym Environment: Step
    def step(self, action):
        """
        Given `action` = np.ndarray of shape (1,), pass it to the JetActuator,
        advance the solver one time step, recompute tau for the reward, and
        return (observation, reward, done, info) where the observation is the
        flattened span‑averaged streamwise velocity over the slot.
        """
        # apply action (assign the amplitude calculated by RL to amp)
        amp = float(np.asarray(action).reshape(-1)[0])

        # step actuator
        used_amp = self.actuator.step_actuator(self.current_time, self.step_count, amp)

        # redefine amp for later storage
        amp = float(used_amp)

        # step solver
        streams.wrap_step_solver()

        # Update time
        dt = float(streams.wrap_get_dtglobal())
        self.current_time += dt

        #
        # BEGIN DEVELOPER SECTION: COLLECT YOUR OBSERVATION FROM STREAmS, DEFINE YOUR REWARD
        #
        streams.wrap_compute_av()
        streams.wrap_tauw_calculate()

        # Track actions with an eligibility trace to delay rewards
        self.action_queue.append({"eligibility": 1.0, "accum_reward": 0.0})

        # Immediate reward based on wall shear stress
        if self.tauw_shape > 0:
            tau = streams.wrap_get_tauw_x(self.tauw_shape)
        else:
            tau = np.empty((0,), dtype=np.float64)
        tau_global = self._gather_nonempty(tau)
        r_t = -float(np.sum(tau_global**2))

        for entry in self.action_queue:
            entry["accum_reward"] += entry["eligibility"] * r_t
            entry["eligibility"] *= self.lambda_trace

        if len(self.action_queue) > self.lag_steps:
            reward = self.action_queue.popleft()["accum_reward"]
        else:
            reward = 0.0

        # Observation: span‑averaged conservative variables over the user
        # specified window
        if self._has_values:
            local_slice = streams.wrap_get_w_avzg_slice_gpu(
                self._local_xstart,
                self._local_xend,
                self._local_ystart,
                self._local_yend,
                self._obs_index,
            )
            local_slice = np.ravel(local_slice)
        else:
            local_slice = np.empty((0,), dtype=np.float32)
        u_global = self._gather_nonempty(local_slice)

        # Termination: After `max_episode_steps` steps, done=True
        self.step_count += 1
        done = (self.step_count >= self.max_episode_steps)

        # Required Gym Stats:  (obs, reward, done, info)
        obs = u_global.astype(np.float32)
        
        #
        # END DEVELOPER SECTION: COLLECT YOUR OBSERVATION FROM STREAmS, DEFINE YOUR REWARD
        #
        
        info = {
            "time": self.current_time,
            "step": self.step_count,
            "jet_amplitude": amp,
            "instant_reward": r_t,
        }
        return obs, reward, done, info

    # HDF5 output helpers (called from main.py). Three methods that creat, write to, and close h5 files, respectively.
    def init_h5_io(self, directory) -> None:
        """Create HDF5 files and datasets used for diagnostics."""
        import io_utils  # imported here to avoid modifying global imports
        # from . import io_utils  # imported here to avoid modifying global imports

        # Allocate temporary arrays for diagnostic computations
        self._span_average = np.zeros([5, self.config.nx_mpi(), self.config.ny_mpi()], dtype=np.float64)
        self._temp_field = np.zeros((self.config.nx_mpi(), self.config.ny_mpi(), self.config.nz_mpi()),dtype=np.float64,)
        self._dissipation_rate_array = np.zeros(1)
        self._energy_array = np.zeros(1)

        # Open HDF5 files
        # May have to add conditional if eval loop can't be parallelized
        self.flowfields = io_utils.IoFile(Path(directory) / "flowfields.h5", comm=self.comm)
        self.span_averages = io_utils.IoFile(Path(directory) / "span_averages.h5", comm=self.comm)
        self.trajectories = io_utils.IoFile(Path(directory) / "trajectories.h5", comm=self.comm)
        self.mesh_h5 = io_utils.IoFile(Path(directory) / "mesh.h5", comm=self.comm)  

        grid_shape = [self.config.grid.nx, self.config.grid.ny, self.config.grid.nz,]
        span_average_shape = [self.config.grid.nx, self.config.grid.ny]

        # 3D flowfield files
        if self.config.temporal.full_flowfield_io_steps not in (0, None):
            flowfield_writes = int(math.ceil(self.config.temporal.num_iter / self.config.temporal.full_flowfield_io_steps))
        else:
            flowfield_writes = 0
        self.velocity_dset = io_utils.VectorField3D(self.flowfields, [5, *grid_shape], flowfield_writes, "velocity", self.rank,)
        self.flowfield_time_dset = io_utils.Scalar1D(self.flowfields, [1], flowfield_writes, "time", self.rank)

        # Span average files
        numwrites = int(math.ceil(self.config.temporal.num_iter / self.config.temporal.span_average_io_steps))
        self.span_average_dset = io_utils.VectorFieldXY2D(self.span_averages, [5, *span_average_shape], numwrites, "span_average", self.rank,)
        self.shear_stress_dset = io_utils.ScalarFieldX1D(self.span_averages, [self.config.grid.nx], numwrites, "shear_stress", self.rank,)
        self.span_average_time_dset = io_utils.Scalar0D(self.span_averages, [1], numwrites, "time", self.rank)
        self.dissipation_rate_dset = io_utils.Scalar0D(self.span_averages, [1], numwrites, "dissipation_rate", self.rank)
        self.energy_dset = io_utils.Scalar0D(self.span_averages, [1], numwrites, "energy", self.rank)

        # Trajectory files
        self.dt_dset = io_utils.Scalar0D(self.trajectories, [1], self.config.temporal.num_iter, "dt", self.rank,)
        self.amplitude_dset = io_utils.Scalar0D(self.trajectories, [1], self.config.temporal.num_iter, "jet_amplitude", self.rank,)

        # Mesh datasets
        x_mesh_dset = io_utils.Scalar1DX(self.mesh_h5, [self.config.grid.nx], 1, "x_grid", self.rank)
        y_mesh_dset = io_utils.Scalar1D(self.mesh_h5, [self.config.grid.ny], 1, "y_grid", self.rank)
        z_mesh_dset = io_utils.Scalar1D(self.mesh_h5, [self.config.grid.nz], 1, "z_grid", self.rank)

        # Generate mesh and write to file (includes ghost nodes)
        x_mesh = streams.wrap_get_x(self.config.x_start(), self.config.x_end())
        y_mesh = streams.wrap_get_y(self.config.y_start(), self.config.y_end())
        z_mesh = streams.wrap_get_z(self.config.z_start(), self.config.z_end())
        x_mesh_dset.write_array(x_mesh)
        y_mesh_dset.write_array(y_mesh)
        z_mesh_dset.write_array(z_mesh)

    def log_step_h5(self, jet_amplitude: float) -> None:
        """Write solver data for the current step to the HDF5 datasets."""
        import utils
        #from . import utils

        dt = float(streams.wrap_get_dtglobal())
        self.dt_dset.write_array(np.array([dt], dtype=np.float32))
        self.amplitude_dset.write_array(np.array([float(jet_amplitude)], dtype=np.float32))

        if self.step_count % self.config.temporal.span_average_io_steps == 0:
            utils.hprint("[StreamsGymEnv] writing span average to output")
            streams.wrap_copy_gpu_to_cpu()
            w = self.config.slice_flowfield_array(streams.wrap_get_w(*self.w_shape))
            utils.calculate_span_averages(self.config, self._span_average, self._temp_field, w)
            self.span_average_dset.write_array(self._span_average)
            streams.wrap_tauw_calculate()
            self.shear_stress_dset.write_array(streams.wrap_get_tauw_x(self.tauw_shape))
            self.span_average_time_dset.write_array(np.array([self.current_time], dtype=np.float32))
            streams.wrap_dissipation_calculation()
            self._dissipation_rate_array[0] = streams.wrap_get_dissipation_rate()
            self.dissipation_rate_dset.write_array(self._dissipation_rate_array)
            streams.wrap_energy_calculation()
            self._energy_array[0] = streams.wrap_get_energy()
            self.energy_dset.write_array(self._energy_array)

        if (self.config.temporal.full_flowfield_io_steps not in (None, 0) and self.step_count % self.config.temporal.full_flowfield_io_steps == 0):
            utils.hprint("[StreamsGymEnv] writing flowfield")
            streams.wrap_copy_gpu_to_cpu()
            self.velocity_dset.write_array(self.config.slice_flowfield_array(streams.wrap_get_w(*self.w_shape)))
            self.flowfield_time_dset.write_array(np.array([self.current_time], dtype=np.float32))

    def close_h5_io(self) -> None:
        """Close the HDF5 files opened by :meth:`initialize_io`."""
        for name in [
            "flowfields",
            "span_averages",
            "trajectories",
            "mesh_h5",
        ]:
            if hasattr(self, name):
                try:
                    getattr(self, name).close()
                except Exception:
                    pass

    def close(self):
        """
        Cleanly finalize the solver before shutting down.
        """
        if self.rank == 0:
            print('[StreamsEnvironment.py] PYTHON CLOSE METHOD CALLED')
        try:
            streams.wrap_finalize_solver()
            if self.rank == 0:
                print('[StreamsEnvironment.py] streams.wrap_finalize() called')
        except Exception:
            pass

        try:
            if self.rank == 0:
                print('closing h5 files')

            # Close datasets before calling MPI_Finalize to avoid HDF5 error when MPI is used.
            for ds in [
                getattr(self, name)
                for name in [
                    'velocity_dset',
                    'flowfield_time_dset',
                    'span_average_dset',
                    'shear_stress_dset',
                    'span_average_time_dset',
                    'dissipation_rate_dset',
                    'energy_dset',
                    'dt_dset',
                    'amplitude_dset',
                ]
                if hasattr(self, name)
            ]:
                try:
                    ds.close()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            streams.wrap_finalize()
            if self.rank == 0:
                print('[StreamsEnvironment.py] streams.wrap_finalize() called')
        except Exception:
            pass
            
