import json
from typing import Any, Dict, Optional
import numpy as np
from enum import Enum, unique

class Length():
    def __init__(self, lx: float, ly: float, lz: float):
        self.lx = lx
        self.ly = ly
        self.lz = lz

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        lx = json_config["x_length"]
        ly = json_config["y_length"]
        lz = json_config["z_length"]

        return Length(lx, ly, lz)

class Grid():
    def __init__(self, nx: int, ny: int, nz: int, ny_wr:int, ly_wr:float, dy_w:float, jb_grid:float, ng: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.ny_wr = ny_wr
        self.ly_wr = ly_wr 

        self.dy_w = dy_w
        self.jb_grid = jb_grid

        self.ng = ng

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        nx = json_config["x_divisions"]
        ny = json_config["y_divisions"]
        nz = json_config["z_divisions"]

        ny_wr = 201
        ly_wr = 2.5
        dy_w = 0.7
        jb_grid = 0

        ng = 3

        return Grid(nx,ny,nz,ny_wr, ly_wr, dy_w, jb_grid, ng)

class Mpi():
    def __init__(self, x_split: int, z_split: int):
        self.x_split = x_split
        self.z_split = z_split

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        x_split = json_config["mpi_x_split"]
        z_split = 1

        return Mpi(x_split, z_split)

class Temporal():
    def __init__(self, num_iter: int, cfl: float, dt_control: int, print_control: int, io_type: int,
            restart_flag: int, disable_restart_io: bool, dtsave_restart: float,
            span_average_io_steps: int, full_flowfield_io_steps: Optional[int], fixed_dt: Optional[float]):
        self.num_iter     = num_iter
        self.cfl          = cfl
        self.dt_control   = dt_control 
        self.print_control= print_control
        self.io_type      = io_type
        self.restart_flag = restart_flag
        self.disable_restart_io = disable_restart_io
        self.dtsave_restart = dtsave_restart

        # number of steps between outputting full 3D vectorfields of the flowfield
        self.full_flowfield_io_steps = full_flowfield_io_steps

        # (optional) parameter of the fixed timestep
        self.fixed_dt = fixed_dt

        # number of steps between writing writing the span average of all the rho / rhou / rhov / rhow/ energy
        self.span_average_io_steps = span_average_io_steps

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        num_iter = json_config["steps"]
        cfl = 0.75
        dt_control = 1
        print_control = 1
        restart_flag = json_config.get("restart_flag", 0)
        disable_restart_io = json_config.get("disable_restart_io", False)
        io_type = 0 if disable_restart_io else 2
        dtsave_restart = json_config.get("dtsave_restart", 50.0)

        fixed_dt = json_config.get("fixed_dt")

        span_average_io_steps = json_config["span_average_io_steps"]
        full_flowfield_io_steps = json_config.get("python_flowfield_steps")

        return Temporal(num_iter, cfl, dt_control, print_control, io_type, restart_flag, disable_restart_io, dtsave_restart, span_average_io_steps, full_flowfield_io_steps, fixed_dt)

class Physics():
    def __init__(self, mach:float, reynolds_friction:float, temp_ratio: float, visc_type:int, Tref: float, turb_inflow: float ):
        self.mach             = mach
        self.reynolds_friction= reynolds_friction
        self.temp_ratio       = temp_ratio
        self.visc_type        = visc_type 
        self.Tref             = Tref
        self.turb_inflow      = turb_inflow

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        mach = json_config["mach_number"]
        reynolds_friction = json_config["reynolds_number"]
        temp_ratio = 1.
        visc_type = 2
        Tref = 160.
        turb_inflow = 0.75

        return Physics(mach, reynolds_friction, temp_ratio, visc_type, Tref, turb_inflow)

from typing import Any, Dict, Optional


class Jet:
    """
    jet_method_name = None, OpenLoop, Classical, LearningBased
    jet_strategy_name = Available actuation strategies within each method (e.g. OpenLoop has Constant, Sinusoidal, DMDc)
    jet_params = All other user-provided, jet-specific parameters
    """
    def __init__(self, jet_method: str, jet_strategy: Optional[str], jet_params: Optional[Dict[str, Any]]):
        self.jet_method_name = jet_method
        self.jet_strategy_name = jet_strategy
        self.jet_params = jet_params

    @staticmethod
    def from_json(cfg_json: Dict[str, Any]) -> "Jet":
        jet_json = cfg_json["blowing_bc"]
        jet_method = jet_json["method"]
        jet_strategy = jet_json["strategy"]
        jet_params = {k: v for k, v in jet_json.items() if k not in {"method", "strategy"}}

        if jet_method == "None":
            return Jet(jet_method, None, {})
            
        else:
            return Jet(jet_method, jet_strategy, jet_params)

class Config():
    def __init__(self, length: Length, grid: Grid, mpi: Mpi, temporal: Temporal, physics: Physics, jet: Jet):
        self.length = length
        self.grid = grid
        self.mpi = mpi
        self.temporal = temporal
        self.physics = physics
        self.jet = jet

    @staticmethod
    def from_json(json_config: Dict[str, Any]):
        length = Length.from_json(json_config)
        grid = Grid.from_json(json_config)
        mpi = Mpi.from_json(json_config)
        temporal = Temporal.from_json(json_config)
        physics = Physics.from_json(json_config)
        jet = Jet.from_json(json_config)

        return Config(length, grid, mpi, temporal, physics, jet)

    def x_start(self) -> int:
        return self.grid.ng

    def x_end(self) -> int:
        return self.x_start() + int(self.grid.nx / self.mpi.x_split)

    def nx_mpi(self) -> int:
        return self.x_end() - self.x_start()

    def y_start(self) -> int:
        return self.grid.ng

    def y_end(self) -> int:
        return self.y_start() + self.grid.ny

    def ny_mpi(self) -> int:
        return self.y_end() - self.y_start()

    def z_start(self) -> int:
        return self.grid.ng

    def z_end(self) -> int:
        return self.z_start() + self.grid.nz

    def nz_mpi(self) -> int:
        return self.z_end() - self.z_start()

    def slice_flowfield_array(self, array: np.ndarray) -> np.ndarray:
        return array[:, \
                self.x_start():self.x_end(), \
                self.y_start():self.y_end(), \
                self.z_start():self.z_end()\
        ]

    def local_to_global_x(self, x: int, rank: int):
        previous_mpi_x = rank * self.nx_mpi();
        return previous_mpi_x + x
