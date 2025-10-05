use crate::prelude::*;
use cli::ConfigGenerator;
use cli::FlowType;
use cli::JetMethod;

#[derive(thiserror::Error, Debug, From)]
pub(crate) enum ConfigError {
    #[error("There was an error with the z-divisions / mpi-x-split values chosen. The domain cannot be evenly split: {0}")]
    MpiSplitX(MpiSplitX),
    #[error("Domain requires too much memory: {0}")]
    Memory(Memory),
    #[error("{0}")]
    Custom(String),
}

#[derive(Debug, Display, Constructor)]
#[display(
    fmt = "x divisions: {} mpi divisions: {} remainder: {}",
    x_div,
    split,
    remainder
)]
pub(crate) struct MpiSplitX {
    x_div: usize,
    split: usize,
    remainder: usize,
}

#[derive(Debug, Display, Constructor)]
#[display(fmt = "invalid number of mpi splits: {}", split)]
pub(crate) struct MpiSplitZero {
    split: usize,
}

#[derive(Debug, Display, Constructor)]
#[display(
    fmt = "gpu memory capacity: {}, memory required for simulation: {}",
    gpu_memory_required,
    required_memory
)]
pub(crate) struct Memory {
    gpu_memory_required: Megabytes,
    required_memory: Megabytes,
}

#[derive(Debug, Display, PartialEq, PartialOrd, Clone, Copy)]
#[display(fmt = "{} Mb", _0)]
pub(crate) struct Megabytes(pub(crate) usize);

pub(crate) fn config_generator(args: ConfigGenerator) -> anyhow::Result<()> {
    let output_path = args.output_path.clone();
    let dry = args.dry;
    let json = args.json;

    let config = args.into_serializable();

    // 11 gb to megabytes
    // 11 gb is what is available on the 2080 TI available in the lab
    let gpu_memory = Some(Megabytes(11 * 10usize.pow(3)));

    // validate that the parameters can be run on the gpu
    config.validate(gpu_memory)?;

    if !dry {
        if json {
            let file = std::fs::File::create(&output_path).with_context(|| {
                format!(
                    "failed to create json output file at {}",
                    output_path.display()
                )
            })?;
            serde_json::to_writer_pretty(file, &config).with_context(|| {
                format!("failed to serialize data to json file. This should not happen")
            })?;
            Ok(())
        } else {
            _config_generator(&config, output_path)
        }
    } else {
        Ok(())
    }
}

/// create a streams config file to be used in the solver
pub(crate) fn _config_generator(config: &Config, output_path: PathBuf) -> anyhow::Result<()> {
    const CFL: f64 = 0.75;

    let cfl = if let Some(fixed_dt) = config.fixed_dt {
        -1. * fixed_dt
    } else {
        CFL
    };

    let output = format!(
        r#"!=============================================================
!
! ███████╗████████╗██████╗ ███████╗ █████╗ ███╗   ███╗███████╗
! ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔════╝
! ███████╗   ██║   ██████╔╝█████╗  ███████║██╔████╔██║███████╗
! ╚════██║   ██║   ██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║╚════██║
! ███████║   ██║   ██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████║
! ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝
!
! Supersonic TuRbulEnt Accelerated navier stokes Solver
!
! input file
!
!=============================================================

 flow_type (0==>channel, 1==>BL, 2==>SBLI)
 {flow_type}   

  Lx(rlx)             Ly(rly)         Lz(rlz)
  {lx}          {ly}         {lz}
 
  Nx(nxmax)     Ny(nymax)     Nz(nzmax)
  {nx}          {ny}        {nz}
 
 Ny_wr(nymax_wr)     Ly_wr(rly_wr)      dy+_w  jbgrid
 {nymax_wr}                   {rly_wr}             .7       0

 ng  visc_ord  ep_ord  weno_par (1==>ord_1,2==>ord_3, 3==>ord_5, 4==>ord_7)
  3     6      6       3
 
 MPI_x_split     MPI_z_split
 {mpi_x_split}               1 

 sensor_threshold   xshock_imp   deflec_shock    pgrad (0==>constant bulk)
  {shock_sensitivity}               {shock_imp}             {angle}              0.
      
 restart   num_iter   cfl   dt_control  print_control  io_type
   0        {steps}      {cfl}      1       1              2
      
 Mach      Reynolds (friction)  temp_ratio   visc_type   Tref (dimensional)   turb_inflow
 {mach}      {re}                   1.            2         160.                0.75
  
 stat_control  xstat_num
  500           10

 xstat_list
   10. 20. 30. 35. 40. 45. 50. 55. 60. 65.
 
 dtsave dtsave_restart  enable_plot3d   enable_vtk
  5.       50.                0          {snapshots_3d}

 rand_type
   -1

 save_probe_steps save_span_average_steps
    {probe_steps}         {span_average_steps}

sbli_blowing_bc        slot_start_x_global     slot_end_x_global
 {sbli_blowing_bc}              {slot_start}                {slot_end}
   "#,
        flow_type = config.flow_type.as_streams_int(),
        lx = config.x_length,
        nx = config.x_divisions,
        ly = config.y_length,
        ny = config.y_divisions,
        lz = config.z_length,
        nz = config.z_divisions,
        mach = config.mach_number,
        re = config.reynolds_number,
        angle = config.shock_angle,
        mpi_x_split = config.mpi_x_split,
        steps = config.steps,
        probe_steps = config.probe_io_steps,
        span_average_steps = config.span_average_io_steps,
        sbli_blowing_bc = config.blowing_bc.blowing_bc_as_streams_int(),
        snapshots_3d = config.snapshots_3d as usize,
        cfl = cfl,
        nymax_wr = config.nymax_wr,
        rly_wr = config.rly_wr,
        slot_start = config.blowing_bc.slot_start_as_streams_int(),
        slot_end = config.blowing_bc.slot_end_as_streams_int(),
        shock_sensitivity = config.sensor_threshold,
        shock_imp = config.shock_impingement
    );

    std::fs::write(&output_path, output.as_bytes())
        .with_context(|| format!("failed to write to file {} ", output_path.display()))?;

    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    /// (friction) Reynolds number (Reynolds in input file)
    pub(crate) reynolds_number: f64,

    /// type of flow to generate
    pub(crate) flow_type: FlowType,

    /// Mach number (Mach in input file, rm in code)
    pub(crate) mach_number: f64,

    /// Shock angle (degrees) (deflec_shock in input file)
    pub(crate) shock_angle: f64,

    /// total length in the x direction
    ///
    /// in streams, this parameter is rlx
    pub(crate) x_length: f64,

    /// total length in the x direction
    ///
    /// in streams, this parameter is nxmax
    pub(crate) x_divisions: usize,

    /// total length in the y direction.
    ///
    /// in streams, this parameter is rly
    pub(crate) y_length: f64,

    /// total length in the y direction
    ///
    /// in streams, this parameter is nymax
    pub(crate) y_divisions: usize,

    /// total length in the z direction
    ///
    /// in streams, this parameter is rlz
    pub(crate) z_length: f64,

    /// total length in the z direction
    ///
    /// in streams, this parameter is nzmax
    pub(crate) z_divisions: usize,

    /// number of MPI divisions along the x axis. The config generated
    /// will have 1 mpi division along the z axis as some extensions
    /// to the code assume there are no z divisions.
    ///
    /// The value supplied to this argument MUST be used for the -np
    /// argument in `mpirun`
    pub(crate) mpi_x_split: usize,

    /// number of steps for the solver to take
    pub(crate) steps: usize,
    
    /// number of steps between writing probe information.
    /// (0 => never)
    /// (n >0 => every n steps)
    pub(crate) probe_io_steps: usize,

    /// number of steps between span average flowfields
    /// (0 => never)
    /// (n >0 => every n steps)
    pub(crate) span_average_io_steps: usize,

    /// information on how to setup the blowing boundary condition on the
    /// bottom surface.
    pub(crate) blowing_bc: cli::JetActuator,

    /// enable exporting 3D flowfields to VTK files
    ///
    /// If not present, no 3D flowfields will be written
    pub(crate) snapshots_3d: bool,

    /// run the python solver with bindings, not the fortran solver
    pub(crate) use_python: bool,

    /// specify a fixed timestep to use
    pub(crate) fixed_dt: Option<f64>,

    /// how often to export full flowfields to hdf5 files (PYTHON ONLY!)
    pub(crate) python_flowfield_steps: Option<usize>,

    /// (currently not well understood): it is required that nymax-wr > y-divisions
    pub(crate) nymax_wr: usize,

    /// (currently not well understood): it is required that rly-wr > y-length
    pub(crate) rly_wr: f64,

    /// X locations for vertical probes (along different values of y) at a (X, _, Z) location.
    /// You must provide the same number of x locations here as you do z locations in `--probe-locations-z`
    pub(crate) probe_locations_x: Vec<usize>,

    /// Z locations for vertical probes (along different values of y) at a (X, _, Z) location.
    /// You must provide the same number of z locations here as you do x locations in `--probe-locations-x`
    pub(crate) probe_locations_z: Vec<usize>,

    /// shock capturing sensor threshold. x < 1 enables it (lower is more sensitive), x >= 1
    /// disables it
    pub(crate) sensor_threshold: f64,

    /// location where the shock strikes the bottom surface
    pub(crate) shock_impingement: f64,
}

impl Config {
    /// load the config data at a given path with `serde_json`
    pub(crate) fn from_path(path: &Path) -> Result<Self, Error> {
        // load the config file specified
        let config_bytes = fs::read(&path).map_err(|e| FileError::new(path.to_owned(), e))?;
        let config: Config = serde_json::from_slice(&config_bytes)?;
        Ok(config)
    }

    /// check all the parameters of the input file to guarantee that the given input
    /// file will (likely) work in the solver without runtime error
    ///
    /// `max_gpu_mem` must only be specified if you are running the config on a gpu system
    pub(crate) fn validate(&self, max_gpu_mem: Option<Megabytes>) -> Result<(), ConfigError> {
        // make sure that the number of divisions with mpi is acceptable
        let split_remainder = self.x_divisions % self.mpi_x_split;
        if split_remainder != 0 {
            return Err(MpiSplitX::new(self.x_divisions, self.mpi_x_split, split_remainder).into());
        }

        if let Some(gpu_mem) = max_gpu_mem {
            self.check_gpu_mem(gpu_mem)?;
        }

        // from config file
        if self.y_divisions < self.nymax_wr {
            return Err(ConfigError::Custom(format!(
                "y-divisions ({}) must be greater than {}",
                self.y_divisions, self.nymax_wr
            )));
        }

        if self.y_length <= self.rly_wr {
            return Err(ConfigError::Custom(format!(
                "y-length ({}) must be greater than rly-wr {}",
                self.y_length, self.rly_wr
            )));
        }

        if self.y_divisions <= self.nymax_wr {
            return Err(ConfigError::Custom(format!(
                "y-divisions ({}) must be greater than nymax-wr {}",
                self.y_divisions, self.nymax_wr
            )));
        }

        if self.y_divisions % self.mpi_x_split != 0 {
            return Err(ConfigError::Custom(format!(
                "nymax (y-divisions) @ {} must be divisible by mpi-x-split @ {} (remainder: {})",
                self.y_divisions,
                self.mpi_x_split,
                self.y_divisions % self.mpi_x_split
            )));
        }

        // make sure that the coordinates of the x probe and z probe make sense. There should
        // be equal numbers of points for x probes as there are for z probes.
        if self.probe_locations_x.len() != self.probe_locations_z.len() {
            return Err(ConfigError::Custom(format!(
                "There were not equal numbers of probe  locations ( (x,z) coordinates from --probe-locations-x and --probe_locations_z) there were {} x locations and {} locations.",
                self.probe_locations_x.len(),
                self.probe_locations_z.len(),
            )));
        }
        
        // ensure observation window is within the domain
        if matches!(self.blowing_bc.method, JetMethod::LearningBased) {
            if let (Some(xs), Some(xe), Some(ys), Some(ye)) = (
                self.blowing_bc.params.get("obs_xstart").and_then(|v| v.as_u64()),
                self.blowing_bc.params.get("obs_xend").and_then(|v| v.as_u64()),
                self.blowing_bc.params.get("obs_ystart").and_then(|v| v.as_u64()),
                self.blowing_bc.params.get("obs_yend").and_then(|v| v.as_u64()),
            ) {
                let xs = xs as usize;
                let xe = xe as usize;
                let ys = ys as usize;
                let ye = ye as usize;

                if xs < 1 || xe < 1 || xs > self.x_divisions || xe > self.x_divisions {
                    return Err(ConfigError::Custom(format!(
                        "obs-xstart ({xs}) and obs-xend ({xe}) must be between 1 and x_divisions ({})",
                        self.x_divisions
                    )));
                }

                if ys < 1 || ye < 1 || ys > self.y_divisions || ye > self.y_divisions {
                    return Err(ConfigError::Custom(format!(
                        "obs-ystart ({ys}) and obs-yend ({ye}) must be between 1 and y_divisions ({})",
                        self.y_divisions
                    )));
                }
            }
        }
        
        if let Some(lag_steps_value) = self.blowing_bc.params.get("lag_steps") {
            let Some(lag_steps) = lag_steps_value.as_i64() else {
                return Err(ConfigError::Custom(
                    "lag-steps must be provided as an integer".to_string(),
                ));
            };

            if lag_steps < 1 {
                return Err(ConfigError::Custom(format!(
                    "lag-steps ({lag_steps}) must be greater than or equal to 1"
                )));
            }
        }

        Ok(())
    }

    /// check that there is enough memory available on the gpu to run the simulation
    ///
    /// this code is a replication of the memory checking code in fortran
    fn check_gpu_mem(&self, max_gpu_mem: Megabytes) -> Result<(), ConfigError> {
        // fortran memory checking code :
        // gpu_used_mem = 43._mykind      ! Number of 3D arrays on GPU
        // correction_factor = 1.5_mykind ! Safety margin
        // gpu_used_mem = gpu_used_mem+correction_factor
        // gpu_used_mem = gpu_used_mem*real((nx+2*ng),mykind)*real((ny+2*ng),mykind)*real((nz+2*ng),mykind)
        // gpu_used_mem = gpu_used_mem*storage_size(1._mykind)/8._mykind/(1024._mykind**2)
        let n_ghost = 3;
        let mut gpu_used_mem = 43.;
        // number of bytes for floating point
        let n_bytes = 8usize;
        gpu_used_mem += 1.5;
        // gpu_used_mem *= ((self.x_divisions + (2 * n_ghost))
        let local_x_divisions = self.x_divisions / self.mpi_x_split;

        gpu_used_mem *= ((local_x_divisions + (2 * n_ghost))
            * (self.y_divisions + (2 * n_ghost))
            * (self.z_divisions + (2 * n_ghost))) as f64;
        gpu_used_mem *= (n_bytes as f64) / (1024. * 1024.);
        let gpu_mem_required = Megabytes(gpu_used_mem as usize);

        if gpu_mem_required > max_gpu_mem {
            return Err(Memory::new(gpu_mem_required, max_gpu_mem).into());
        }

        Ok(())
    }

    pub(crate) fn to_writer<W: Write>(&self, writer: &mut W) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub(crate) fn to_file<T: AsRef<Path>>(&self, path: T) -> anyhow::Result<()> {
        let path = path.as_ref().to_owned();
        let mut file = std::fs::File::create(&path).with_context(|| {
            format!(
                "failed to serialize create file at {} for config",
                path.display()
            )
        })?;

        self.to_writer(&mut file).with_context(|| {
            format!(
                "failed to serialize config to file at path {}",
                path.display()
            )
        })?;

        Ok(())
    }
}
