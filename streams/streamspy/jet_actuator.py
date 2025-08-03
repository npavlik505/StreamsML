import numpy as np
from config import Config, Jet
import libstreams as streams
from abc import ABC, abstractmethod
from typing import Optional, Dict
import utils
import math
from mpi4py import rc
rc.initialize = False
from mpi4py import MPI

# the equation of the polynomial for the jet actuation in coordinates local to the jet
# actuator. This means that the jet actuator starts at x = 0 and ends at x = slot_end
# 
# this must be recomputed at every amplitude change
class Polynomial():
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x_idx: int) -> float:
        return self.a * x_idx**2  + \
                self.b * x_idx + \
                self.c

# helper class to recompute the polynomial of the jet actuator
class PolynomialFactory():
    def __init__(self, vertex_x: float, slot_start: float, slot_end: float):
        self.slot_start = slot_start
        self.slot_end = slot_end
        self.vertex_x  = vertex_x

    def poly(self, amplitude: float) -> Polynomial:
        # see streams section of lab documentation for the derivation of this
        #
        # in essence, it is solving for the coefficients a,b,c of the polynomial
        # y = ax^2 + bx +c 
        # using the fact that
        # y(jet start) = 0
        # y(jet end) = 0
        # y((jet start + jet_end) / 2) = amplitude
        a = amplitude/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)
        b = -(amplitude*self.slot_end+ amplitude*self.slot_start)/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)
        c = amplitude*self.slot_end*self.slot_start/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)

        return Polynomial(a, b, c)

class JetActuator():
    def __init__(self, rank: int, config: Config, slot_start: int, slot_end: int):
        self.rank = rank
        self.config = config

        vertex_x = (slot_start + slot_end) / 2
        self.factory = PolynomialFactory(vertex_x, slot_start, slot_end)

        self.local_slot_start_x = int(streams.wrap_get_x_start_slot())
        self.local_slot_nx = int(streams.wrap_get_nx_slot())
        self.local_slot_nz = streams.wrap_get_nz_slot()

        self.has_slot = self.local_slot_start_x != -1

        if self.has_slot:
            sv1, sv2 = streams.wrap_get_blowing_bc_slot_velocity_shape()
            arr = streams.wrap_get_blowing_bc_slot_velocity(sv1, sv2)
            self.bc_velocity = arr.reshape((sv1, sv2))

    def set_amplitude(self, amplitude: float):
        # WARNING: copying to GPU and copying to CPU must happen on ALL mpi procs
        streams.wrap_copy_blowing_bc_to_cpu()

        if not self.has_slot:
            streams.wrap_copy_blowing_bc_to_gpu()
            return None

        # calculate the equation of the polynomial that the velocity of the jet
        # actuator will have with this amplitude
        poly = self.factory.poly(amplitude)

        for idx in range(self.local_slot_nx):
            local_x = self.local_slot_start_x + idx
            global_x = self.config.local_to_global_x(local_x, self.rank)

            velo = poly.evaluate(global_x)
            self.bc_velocity[idx, 0:self.local_slot_nz] = velo

        # copy everything back to the GPU
        streams.wrap_copy_blowing_bc_to_gpu()
        return None


class AbstractActuator(ABC):
    @abstractmethod
    # returns the amplitude of the jet that was used
    def step_actuator(self, time: float, i:int, agent_amplitude: float = None) -> float:
        pass

class NoActuation(AbstractActuator):
    def __init__(self):
        utils.hprint("skipping initialization of jet actuator")
        pass

    # returns the amplitude of the jet that was used
    def step_actuator(self, _:float, i:int, agent_amplitude: float = None) -> float:
        return 0.

class ConstantActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config):
        utils.hprint("initializing a constant velocity actuator")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude

        self.actuator = JetActuator(rank, config, slot_start, slot_end)

    # returns the amplitude of the jet that was used
    def step_actuator(self, _: float, i:int, agent_amplitude: float = None) -> float:
        self.actuator.set_amplitude(self.amplitude)
        return self.amplitude

class SinusoidalActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config, angular_frequency:float ):
        utils.hprint("initializing a sinusoidal velocity actuator")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude

        self.actuator = JetActuator(rank, config, slot_start, slot_end)
        self.angular_frequency = angular_frequency

    # returns the amplitude of the jet that was used
    def step_actuator(self, time: float, i:int, agent_amplitude: float = None) -> float:
        adjusted_amplitude = math.sin(self.angular_frequency * time)
        self.actuator.set_amplitude(adjusted_amplitude)
        return adjusted_amplitude

class DMDcActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config):
        utils.hprint("initializing an actuator for DMDc")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude
        self.config = config
        self.actuator = JetActuator(rank, config, slot_start, slot_end)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.root = 0
        self.rank = rank # should match comm.Get_rank()

    def step_actuator(self, time: float, i: int, agent_amplitude: float = None) -> float:
        n_steps = self.config.temporal.num_iter
        frac    = i / n_steps

        # only root computes the “raw” adjusted_amplitude
        if self.rank == self.root:
            if frac <= .3:
                # print(f'[DEBUG: jet_actuator.py] PRBS running {i}')
                raw = self.amplitude * (2*np.random.rand() - 1)
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
            elif frac <= .6:
                # print(f'[DEBUG: jet_actuator.py] Linear-chirp sine running {i}')
                dt    = (self.config.temporal.fixed_dt
                         if self.config.temporal.fixed_dt is not None
                         else streams.wrap_get_dtglobal())
                T     = n_steps * dt
                phase = 2 * math.pi * (time / T)**2
                raw   = self.amplitude * math.sin(phase)
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
            elif frac <= .7:
                # print(f'[DEBUG: jet_actuator.py] .2 amp running {i}')
                raw = 0.2 * self.amplitude
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
            elif frac <= .8:
                # print(f'[DEBUG: jet_actuator.py] .5 amp running {i}')
                raw = 0.5 * self.amplitude
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
            elif frac <= .9:
                # print(f'[DEBUG: jet_actuator.py] .8 running {i}')
                raw = 0.8 * self.amplitude
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
            else:
                # print(f'[DEBUG: jet_actuator.py] 100% amp running {i}')
                raw = 1.0 * self.amplitude
                # print(f'[DEBUG: jet_actuator.py] amplitude {raw}')
        else:
            raw = None

        # broadcast the single value from root → everyone
        adjusted_amplitude = self.comm.bcast(raw, root=self.root)

        # now every rank sets the same amplitude
        self.actuator.set_amplitude(adjusted_amplitude)
        return adjusted_amplitude

class AdaptiveActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config):
        utils.hprint("initializing an adaptive actuator")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude

        self.actuator = JetActuator(rank, config, slot_start, slot_end)

    # returns the amplitude of the jet that was used
    def step_actuator(self, time: float, i: int, agent_amplitude: float = None) -> float:
        self.actuator.set_amplitude(agent_amplitude)
        return agent_amplitude

def init_actuator(rank: int, config: Config) -> AbstractActuator:
    jet_config = config.jet
    
    if jet_config.jet_method_name == "None":
        return NoActuation()
    
    elif jet_config.jet_method_name == "OpenLoop":
    
        if jet_config.jet_strategy_name  in ("constant", "dmdc"):
            slot_start = jet_config.jet_params["slot_start"]
            slot_end = jet_config.jet_params["slot_end"]
            amplitude = jet_config.jet_params["amplitude"]
            if jet_config.jet_strategy_name  == "constant":
                return ConstantActuator(amplitude, slot_start, slot_end, rank, config);
            elif jet_config.jet_strategy_name  == "dmdc":
                return DMDcActuator(amplitude, slot_start, slot_end, rank, config);
        
        elif jet_config.jet_strategy_name  == "sinusoidal":
            slot_start = jet_config.jet_params["slot_start"]
            slot_end = jet_config.jet_params["slot_end"]
            amplitude = jet_config.jet_params["amplitude"]
            angular_frequency = jet_config.jet_params["angular_frequency"]
            return SinusoidalActuator(amplitude, slot_start, slot_end, rank, config, angular_frequency);

        else:
            pass
    
    elif jet_config.jet_method_name == "Classical":
        print('No classical control algorithms yet')
        exit()
    
    elif jet_config.jet_method_name == "LearningBased":
        if jet_config.jet_strategy_name in ("ddpg", "dqn","ppo"):
            slot_start = jet_config.jet_params["slot_start"]
            slot_end = jet_config.jet_params["slot_end"]
            amplitude = jet_config.jet_params["amplitude"]
            return AdaptiveActuator(amplitude, slot_start, slot_end, rank, config);
    else:
        print('Learning based algorithm does not match available options: ddpg, dqn')
        exit()
