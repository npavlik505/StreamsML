import numpy as np
from streamspy.base_Classical import BaseController
import libstreams as streams
import math
from collections import deque

class controller(BaseController):
    """Opposition control: act against measured quantity."""

    def __init__(self, env) -> None:
        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        max_action = env.config.jet.jet_params.get("amplitude") # float(env.action_space.high[0])
        min_action = -max_action
 
        # jet parameters
        self.slot_start = env.config.jet.jet_params.get("slot_start")
        self.slot_end = env.config.jet.jet_params.get("slot_end")
        self.org_motion = env.config.jet.jet_params.get("organized_motion")
        self.obs_xstart = env.config.jet.jet_params.get("obs_xstart")
        self.obs_xend = env.config.jet.jet_params.get("obs_xend")
        self.obs_ystart = env.config.jet.jet_params.get("obs_ystart")
        self.obs_yend = env.config.jet.jet_params.get("obs_yend")
        self.gain = env.config.jet.jet_params.get("gain")
        
        # physics parameters
        self.TargetReFriction = env.config.physics.reynolds_friction
        
        # grid parameters
        self.nx = env.config.grid.nx
        self.ny = env.config.grid.ny
        self.nz = env.config.grid.nz
        
        #length parameters
        self.lx = env.config.length.lx
        self.ly = env.config.length.ly

        # store environment and MPI info
        self.env = env
        self.rank = env.rank
        self.comm = env.comm
        self.min_action = min_action
        self.max_action = max_action
        
        self.actuation_queue = deque()
        self.obs_ema = None
        
        self.recompute_obs()

    def reset(self) -> None:
        self.env.close
        pass
        
    def recompute_obs(self) -> None:
        N_buff = max(5, math.ceil(0.1*(self.slot_end - self.slot_start)))
        sensor_x_end = self.slot_start - N_buff
        if self.org_motion == "undefined":
            print("organized-motion flag not defined. Sensor window will not be recomputed")
            return
        elif self.org_motion == "near_wall":
            y_delta = streams.wrap_get_y(1, self.ny + 1)
            sensor_ydelta_loc = 10/self.TargetReFriction # 10+ ideal height for near-wall fluid structure sensing
            sensor_y_loc = int(np.abs(y_delta - sensor_ydelta_loc).argmin())
            
            sensor_x_len = math.ceil(((1000/self.TargetReFriction)/self.lx)*self.nx)
            sensor_x_start = sensor_x_end - sensor_x_len
            
        elif self.org_motion == "LSM":
            y_delta = streams.wrap_get_y(1, self.ny + 1)
            sensor_ydelta_loc = min(max( 3/math.sqrt(self.TargetReFriction), .1 ), 1.5 ) # 3/math.sqrt(self.TargetReFriction) <= log region <= 1.5delta
                                                                                         # Log region height range, ideal for LSM sensing
            sensor_y_loc = int(np.abs(y_delta - sensor_ydelta_loc).argmin()) # Return index of closes y value        
            
            sensor_x_len = math.ceil((2.5/self.lx)*self.nx)
            sensor_x_start = sensor_x_end - sensor_x_len
            
        else:
            raise ValueError("organized_motion must be set to 'near-wall' or 'LSM'")
            
        if sensor_x_start < 0:
            print(f'Sensor outside of simulation domain: Sensor ({sensor_x_start} to {sensor_x_end})')
        else: 
            print(f'Sensor - x: {sensor_x_start} to {sensor_x_end}), y: {sensor_ydelta_loc}, Jet Slot - x: {self.slot_start} to {self.slot_end})')

        if self.rank == 0:
            x_start = sensor_x_start
            x_end = sensor_x_end
            y_loc = sensor_y_loc
        else:
            x_start = x_end = y_loc = None
        x_start = self.comm.bcast(x_start, root=0)
        x_end = self.comm.bcast(x_end, root=0)
        y_loc = self.comm.bcast(y_loc, root=0)
        self.env.set_observation_window(x_start, x_end, y_loc, y_loc)
        

    def compute_action(self, observation):
        """Compute actuator command with convective delay handling."""

        # Exponential moving average of the observation
        obs_avg = float(np.mean(observation))
        if self.obs_ema is None:
            self.obs_ema = obs_avg
        else:
            self.obs_ema = 0.2 * obs_avg + 0.8 * self.obs_ema

        output = -self.gain * self.obs_ema

        # enqueue the control action with zero accumulated convection
        self.actuation_queue.append({"actuation": output, "convection": 0.0})

        # compute local convective velocity at the sensing region
        rho_slice = streams.wrap_get_w_avzg_slice(
            self.env._obs_xstart,
            self.env._obs_xend,
            self.env._obs_ystart,
            self.env._obs_yend,
            1,
        )
        rhou_slice = streams.wrap_get_w_avzg_slice(
            self.env._obs_xstart,
            self.env._obs_xend,
            self.env._obs_ystart,
            self.env._obs_yend,
            2,
        )
        u_slice = rhou_slice[0] / rho_slice[0]
        Uc = float(np.mean(u_slice))

        dt = float(streams.wrap_get_dtglobal())
        dx = self.lx / self.nx

        # distance between sensor and actuator centroids in index units
        sensor_centroid = 0.5 * (self.env._obs_xstart + self.env._obs_xend)
        slot_centroid = 0.5 * (self.slot_start + self.slot_end)
        distance_index = slot_centroid - sensor_centroid

        # update convection progress for all queued actions
        for entry in self.actuation_queue:
            entry["convection"] += (Uc * dt) / dx

        step_actuation = 0.0
        if self.actuation_queue and self.actuation_queue[0]["convection"] >= distance_index:
            step_actuation = self.actuation_queue.popleft()["actuation"]

        if self.min_action is not None or self.max_action is not None:
            low = self.min_action if self.min_action is not None else step_actuation
            high = self.max_action if self.max_action is not None else step_actuation
            step_actuation = np.clip(step_actuation, low, high)

        return float(step_actuation)
