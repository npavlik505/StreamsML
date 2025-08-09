# Gym and standard imports
import os
import sys
import json
import math
import numpy as np
import gymnasium
from gymnasium import spaces

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
        from mpi4py import MPI  # solver MPI must be started (wrap_startmpi()) before mpi4py library import
        import globals # contains rank/comm initialization
        globals.init() 
        self.rank = globals.rank
        self.comm = globals.comm
        import io_utils # for HDF5
        from config import Config # input.json to Config object
        import utils # helper code; calculate_span_averages, etc.
        import jet_actuator # all actuator code
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
        # Shape of the span‑averaged streamwise velocity over the slot.
        # ``wrap_get_uoverslot_shape`` returns the local slice owned by this
        # MPI rank.  The global observation is assembled via ``allgather`` in
        # :meth:`reset` and :meth:`step`.
        u1, u2 = streams.wrap_get_uoverslot_shape()
        self.uoverslot_shape = (u1, u2)
        # Some MPI ranks may not intersect the slot and thus own zero cells.
        # Record whether this rank actually holds data for the span-averaged
        # velocity.  Accessing Fortran arrays with a zero extent would trigger
        # runtime errors, so we skip such calls and later gather empty arrays
        # from those ranks.
        self._has_uoverslot = (u1 > 0 and u2 > 0)
        
        # ``wrap_get_uoverslot_shape`` returns the local dimensions of the
        # span‑averaged velocity array owned by this MPI rank.  The global
        # observation is built by gathering the local slices from all ranks in
        # :meth:`reset` and :meth:`step`.  Compute the total size of the global
        # observation by summing the local sizes across all ranks.
        local_obs_size = u1 * u2
        global_obs_size = self.comm.allreduce(local_obs_size)
        self._global_obs_size = global_obs_size
        
        w1, w2, w3, w4  = streams.wrap_get_w_shape() # conservative vector (rho, rho-u, rho-v, rho-w, E)
        self.w_shape   = (w1, w2, w3, w4)
        
        # Define the observation_space as config.grid.nx, the number of grid points in the x (streamwise) direction
        self.nx = self.config.grid.nx

        grid_shape = [self.config.grid.nx, self.config.grid.ny, self.config.grid.nz]
        span_average_shape = [self.config.grid.nx, self.config.grid.ny]

        # 3D flowfield files
        if not (self.config.temporal.full_flowfield_io_steps is None):
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
            # Observation Space (Determine what resonable bounds are)
            
            # # Observation = τw(x) ∈ R^{nx}.  We bound it loosely between [-100, +100] per point.
            # high_obs = np.full((self.nx,), 100.0, dtype=np.float32)
            # self.observation_space = spaces.Box(low=-high_obs,
            #                                     high=+high_obs,
            #                                     shape=(self.nx,),
            #                                     dtype=np.float32)

            # Observation = Flattened span-averaged U velocity over the slot.
            global_obs_size = self._global_obs_size
            high_obs = np.full((global_obs_size,), 2100.0, dtype=np.float32)
            self.observation_space = spaces.Box(
                                                low=-high_obs,
                                                high=+high_obs,
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

        if self.rank == 0:
            print(f'[StreamsEnvironment.py] gym environment initialized')
            print(f'Step_count: {self.step_count}')
            print(f'max_episode_steps (--steps in justfile): {self.max_episode_steps}')
            print(f'current_time: {self.current_time}')

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

    # setup_solver definition: initialized solver on first call, closes solver and reinits on subsequent calls
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
        # Immediately compute tau on the new solver (no time steps taken yet)
        # streams.wrap_copy_gpu_to_cpu() # bring everything from GPU to CPU
        # streams.wrap_tauw_calculate() # ask Fortran to compute τau(x) on the CPU
        # tau = streams.wrap_get_tauw_x(self.tauw_shape)
        # self._tauw_buffer[:] = tau  # temporary tau storage
        
        # # Gather τ_w from all MPI ranks
        # all_tau = self.comm.allgather(self._tauw_buffer)
        # tau_global = np.concatenate(all_tau)

        # # Reset counters
        # self.step_count = 0
        # self.current_time = 0.0

        # # Gym expects a float32 array
        # return tau_global.astype(np.float32)
        
        # Immediately compute the span‑averaged velocity on the new solver
        # (no time steps taken yet)
        streams.wrap_copy_gpu_to_cpu()  # bring everything from GPU to CPU
        streams.wrap_uoverslot_collect()
        if self._has_uoverslot:
            local_u = streams.wrap_get_uoverslot(*self.uoverslot_shape)
        else:
            local_u = np.empty((0,), dtype=np.float32)

        # Gather U values from all MPI ranks and flatten, ignoring empty slices
        u_global = self._gather_nonempty(local_u)

        # Reset counters
        self.step_count = 0
        self.current_time = 0.0

        # Gym expects a float32 array
        # return tau_global.astype(np.float32) 
        return u_global.astype(np.float32)
        #
        # END DEVELOPER SECTION: RETURN YOUR INITIAL OBSERVATION
        #

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
        streams.wrap_copy_gpu_to_cpu()
        streams.wrap_tauw_calculate()
        streams.wrap_uoverslot_collect()

        # Reward based on wall shear stress
        if self.tauw_shape > 0:
            tau = streams.wrap_get_tauw_x(self.tauw_shape)
        else:
            tau = np.empty((0,), dtype=np.float64)
        tau_global = self._gather_nonempty(tau)
        reward = -float(np.sum(tau_global**2))

        # Observation: span‑averaged U over slot
        if self._has_uoverslot:
            local_u = streams.wrap_get_uoverslot(*self.uoverslot_shape)
        else:
            local_u = np.empty((0,), dtype=np.float32)
        u_global = self._gather_nonempty(local_u)

        # Termination: After `max_episode_steps` steps, done=True
        self.step_count += 1
        done = (self.step_count >= self.max_episode_steps)

        # Required Gym Stats:  (obs, reward, done, info)
        # obs = tau_global.astype(np.float32)
        obs = u_global.astype(np.float32)
        
        #
        # END DEVELOPER SECTION: COLLECT YOUR OBSERVATION FROM STREAmS, DEFINE YOUR REWARD
        #
        
        info = {
            "time": self.current_time,
            "step": self.step_count,
            "jet_amplitude": amp
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # HDF5 output helpers (called from main.py)
    # ------------------------------------------------------------------
    def init_h5_io(self, directory) -> None:
        """Create HDF5 files and datasets used for diagnostics."""
        import io_utils  # imported here to avoid modifying global imports

        # Allocate temporary arrays for diagnostic computations
        self._span_average = np.zeros([5, self.config.nx_mpi(), self.config.ny_mpi()], dtype=np.float64)
        self._temp_field = np.zeros((self.config.nx_mpi(), self.config.ny_mpi(), self.config.nz_mpi()),dtype=np.float64,)
        self._dissipation_rate_array = np.zeros(1)
        self._energy_array = np.zeros(1)

        # Open HDF5 files
        self.flowfields = io_utils.IoFile(directory / "flowfields.h5")
        self.span_averages = io_utils.IoFile(directory / "span_averages.h5")
        self.trajectories = io_utils.IoFile(directory / "trajectories.h5")
        self.mesh_h5 = io_utils.IoFile(directory / "mesh.h5")        

        grid_shape = [self.config.grid.nx, self.config.grid.ny, self.config.grid.nz,]
        span_average_shape = [self.config.grid.nx, self.config.grid.ny]

        # 3D flowfield files
        if self.config.temporal.full_flowfield_io_steps is not None:
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

        if (self.config.temporal.full_flowfield_io_steps is not None and self.step_count % self.config.temporal.full_flowfield_io_steps == 0):
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
            
