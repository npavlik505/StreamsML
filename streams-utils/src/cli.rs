use crate::prelude::*;
use clap::Parser;
use clap::Subcommand;
use clap::ValueEnum;
use std::path::PathBuf;

/// utilities for working with the streams solver
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub(crate) struct Args {
    #[clap(subcommand)]
    pub(crate) mode: Command,
}

#[derive(Subcommand, Debug, Clone)]
pub(crate) enum Command {
    /// generate a config file (input.dat) for use in the solver
    ConfigGenerator(ConfigGenerator),
    /// run the solver once inside the apptainer container
    RunContainer(RunContainer),
    /// run an the apptainer solver locally
    RunLocal(RunLocal),
    /// parse probe data to .mat files
    Probe(ParseProbe),
    /// convert a span average VTK file to a .mat file for analysis
    VtkToMat(VtkToMat),
    /// convert a partial solver folder with span binaries to VTK files.
    /// usually this is performed automatically by the `run-solver` subcommand
    SpansToVtk(SpansToVtk),
    /// convert a flowfields.h5 file into a series of vtk files
    HDF5ToVtk(HDF5ToVtk),
    Animate(Animate),
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct Animate {
    /// path to the output folder for which the data should be animated
    pub(crate) data_folder: PathBuf,

    #[clap(long, default_value_t = 1)]
    /// number to adjust the range by when iterating over the slices. If indexing
    /// from 1:100, and decimate = 5 then it will iterate by 1:5:100
    pub(crate) decimate: usize,
}

#[derive(Parser, Debug, Clone)]
/// Fields that are configurable for generating input.dat files for the solver
pub(crate) struct ConfigGenerator {
    /// path to write the resulting config file to
    pub(crate) output_path: PathBuf,

    /// type of flow to generate
    pub(crate) flow_type: FlowType,

    /// (friction) Reynolds number (Reynolds in input file)
    #[clap(long, default_value_t = 250.0)]
    pub(crate) reynolds_number: f64,

    /// Mach number (Mach in input file, rm in code)
    #[clap(long, default_value_t = 2.28)]
    pub(crate) mach_number: f64,

    /// Shock angle (degrees) (deflec_shock in input file)
    #[clap(long, default_value_t = 8.0)]
    pub(crate) shock_angle: f64,

    /// total length in the x direction
    #[clap(long, default_value_t = 27.)]
    pub(crate) x_length: f64,

    /// total length in the x direction
    #[clap(long, default_value_t = 800)]
    pub(crate) x_divisions: usize,

    /// total length in the y direction
    #[clap(long, default_value_t = 6.)]
    pub(crate) y_length: f64,

    /// total length in the y direction
    #[clap(long, default_value_t = 208)]
    pub(crate) y_divisions: usize,

    /// total length in the z direction
    #[clap(long, default_value_t = 3.8)]
    pub(crate) z_length: f64,

    /// total length in the z direction
    #[clap(long, default_value_t = 150)]
    pub(crate) z_divisions: usize,

    /// number of MPI divisions along the x axis. The config generated
    /// will have 1 mpi division along the z axis as some extensions
    /// to the code assume there are no z divisions.
    ///
    /// The value supplied to this argument MUST be used for the -np
    /// argument in `mpirun`
    #[clap(long, default_value_t = 4)]
    pub(crate) mpi_x_split: usize,

    #[clap(long)]
    /// skip writing the actual config file
    pub(crate) dry: bool,

    #[clap(long, default_value_t = 50_000)]
    /// number of steps for the solver to take
    pub(crate) steps: usize,

    #[clap(long, default_value_t = 0)]
    /// number of steps between writing probe information.
    /// (0 => never)
    /// (n >0 => every n steps)
    pub(crate) probe_io_steps: usize,

    #[clap(long, default_value_t = 100)]
    /// number of steps between span average flowfields
    /// (0 => never)
    /// (n >0 => every n steps)
    pub(crate) span_average_io_steps: usize,

    #[command(subcommand)]
    /// whether or not to use blowing boundary condition on the bottom surface
    /// in the sbli case
    pub(crate) blowing_bc: JetActuator,

    #[clap(long)]
    /// enable exporting 3D flowfields to VTK files
    ///
    /// If not present, no 3D flowfields will be written
    pub(crate) snapshots_3d: bool,

    #[clap(long)]
    /// save output to json format
    pub(crate) json: bool,

    #[clap(long)]
    /// run the solver with python bindings instead of fortran mode
    pub(crate) use_python: bool,

    /// specify a fixed timestep to use
    #[clap(long)]
    pub(crate) fixed_dt: Option<f64>,

    /// how often to export full flowfields to hdf5 files (PYTHON ONLY!)
    #[clap(long)]
    pub(crate) python_flowfield_steps: Option<usize>,

    /// (currently not well understood): it is required that nymax-wr > y-divisions
    #[clap(long, default_value_t = 201)]
    pub(crate) nymax_wr: usize,

    /// (currently not well understood): it is required that rly-wr > y-length
    #[clap(long, default_value_t = 2.5)]
    pub(crate) rly_wr: f64,

    #[clap(long)]
    #[arg(requires = "probe_locations_z")]
    /// X locations for vertical probes (along different values of y) at a (X, _, Z) location.
    /// You must provide the same number of x locations here as you do z locations in `--probe-locations-z`
    pub(crate) probe_locations_x: Vec<usize>,

    #[clap(long)]
    #[arg(requires = "probe_locations_x")]
    /// Z locations for vertical probes (along different values of y) at a (X, _, Z) location.
    /// You must provide the same number of z locations here as you do x locations in `--probe-locations-x`
    pub(crate) probe_locations_z: Vec<usize>,

    /// shock capturing sensor threshold. x < 1 enables it (lower is more sensitive), x >= 1
    /// disables it
    #[clap(long, default_value_t = 0.1)]
    pub(crate) sensor_threshold: f64,

    /// location where the shock strikes the bottom surface
    #[clap(long, default_value_t = 15.)]
    pub(crate) shock_impingement: f64,
}

impl ConfigGenerator {
    /// create a default config to be written to a given path
    pub(crate) fn with_path(output_path: PathBuf) -> Self {
        // commented values in here are the default values from the solver file
        // that we are overwriting
        Self {
            output_path,
            reynolds_number: 250.0,
            mach_number: 2.28,
            shock_angle: 8.0,
            //x_length: 70.0,
            x_length: 27.0,
            //x_divisions: 2048,
            x_divisions: 800,
            //y_length: 12.,
            y_length: 6.,
            //y_divisions: 400,
            y_divisions: 208,
            //z_length: 6.5,
            z_length: 3.8,
            //z_divisions: 256,
            z_divisions: 150,
            mpi_x_split: 4,
            dry: false,
            steps: 50_000,
            probe_io_steps: 0,
            span_average_io_steps: 100,
            blowing_bc: JetActuator::None,
            snapshots_3d: true,
            json: false,
            use_python: false,
            fixed_dt: None,
            python_flowfield_steps: None,
            rly_wr: 2.5,
            nymax_wr: 201,
            probe_locations_x: Vec::new(),
            probe_locations_z: Vec::new(),
            flow_type: FlowType::ShockBoundaryLayer,
            sensor_threshold: 0.1,
            shock_impingement: 15.,
        }
    }

    pub(crate) fn into_serializable(self) -> crate::config_generator::Config {
        let Self {
            reynolds_number,
            mach_number,
            shock_angle,
            x_length,
            x_divisions,
            y_length,
            y_divisions,
            z_length,
            z_divisions,
            mpi_x_split,
            steps,
            probe_io_steps,
            span_average_io_steps,
            blowing_bc,
            snapshots_3d,
            use_python,
            fixed_dt,
            python_flowfield_steps,
            rly_wr,
            nymax_wr,
            probe_locations_x,
            probe_locations_z,
            flow_type,
            sensor_threshold,
            shock_impingement,
            ..
        } = self;

        crate::config_generator::Config {
            reynolds_number,
            mach_number,
            shock_angle,
            x_length,
            x_divisions,
            y_length,
            y_divisions,
            z_length,
            z_divisions,
            mpi_x_split,
            steps,
            probe_io_steps,
            span_average_io_steps,
            blowing_bc,
            snapshots_3d,
            use_python,
            fixed_dt,
            python_flowfield_steps,
            rly_wr,
            nymax_wr,
            probe_locations_x,
            probe_locations_z,
            flow_type,
            sensor_threshold,
            shock_impingement,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum FlowType {
    ChannelFlow,
    BoundaryLayer,
    ShockBoundaryLayer,
}

impl FlowType {
    pub(crate) fn as_streams_int(&self) -> u8 {
        match &self {
            Self::ChannelFlow => 0,
            Self::BoundaryLayer => 1,
            Self::ShockBoundaryLayer => 2,
        }
    }
}

#[derive(Subcommand, Debug, Clone, Serialize, Deserialize)]
pub(crate) enum JetActuator {
    None,
    /// jet actuator with constant amplitude
    Constant {
        #[clap(long)]
        /// amplitude of the parabolic jet velocity
        amplitude: f64,
        #[clap(long)]
        /// the x location (index) at which the slot starts blowing
        ///
        /// required if slot-end or sbli-blowing-bc is set
        slot_start: usize,
        /// the x location (index) at which the slot stop blowing
        ///
        /// required if slot-start or sbli-blowing-bc is set
        #[clap(long)]
        slot_end: usize,
    },
    Sinusoidal {
        #[clap(long)]
        /// amplitude of the parabolic jet velocity
        amplitude: f64,

        #[clap(long)]
        /// angular frqeuency \omega of the sin wave. An angular frequency of pi
        /// has a period of 2 seconds (1 second is positive velocity, 1 second is negative
        /// velocity)
        angular_frequency: f64,

        #[clap(long)]
        /// the x location (index) at which the slot starts blowing
        ///
        /// required if slot-end or sbli-blowing-bc is set
        slot_start: usize,
        /// the x location (index) at which the slot stop blowing
        ///
        /// required if slot-start or sbli-blowing-bc is set
        #[clap(long)]
        slot_end: usize,
    },
    #[clap(name = "DMDc")]
    DMDc {
        #[clap(long)]
        /// amplitude of the parabolic jet velocity
        amplitude: f64,
        #[clap(long)]
        /// the x location (index) at which the slot starts blowing
        ///
        /// required if slot-end or sbli-blowing-bc is set
        slot_start: usize,
        /// the x location (index) at which the slot stop blowing
        ///
        /// required if slot-start or sbli-blowing-bc is set
        #[clap(long)]
        slot_end: usize,
    },
    /// use RL controller for the jet amplitude
    Adaptive {
        #[clap(long, default_value_t = 1.0)] amplitude: f64,
        #[clap(long)]
        /// the x location (index) at which the slot starts blowing
        ///
        /// required if slot-end or sbli-blowing-bc is set
        slot_start: usize,
        #[clap(long)]
        /// the x location (index) at which the slot stop blowing
        ///
        /// required if slot-start or sbli-blowing-bc is set
        slot_end: usize,
        // RL hyper-parameters --------------------------
        #[clap(long, default_value_t = 10)]   train_episodes:     usize,
        #[clap(long, default_value_t = 10)]   eval_episodes:     usize,
        #[clap(long, default_value_t = 1000)] eval_max_steps:    usize,
        #[clap(long, default_value = "/RL_metrics/checkpoint")] checkpoint_dir:    String,
        #[clap(long, default_value_t = 5)]    checkpoint_interval: usize,
        #[clap(long, default_value_t = 42)]   seed:              u64,
        #[clap(long, default_value_t = 3e-4)] learning_rate:     f64,
        #[clap(long, default_value_t = 0.99)] gamma:             f64,
        #[clap(long, default_value_t = 0.005)] tau:              f64,
        #[clap(long, default_value_t = 1_000_000)] buffer_size:  usize,
        #[clap(long, default_value = "/RL_metrics/eval")] eval_output:  String,
        #[clap(long, default_value = "false")] verbose:  bool,
        #[clap(long, default_value = "/RL_metrics/training")] training_output: String,
    },
}

impl JetActuator {
    pub(crate) fn blowing_bc_as_streams_int(&self) -> u8 {
        match &self {
            Self::None => 0,
            _ => 1,
        }
    }

    pub(crate) fn slot_start_as_streams_int(&self) -> i32 {
        match &self {
            Self::None => -1,
            Self::Constant { slot_start, .. } => *slot_start as i32,
            Self::Sinusoidal { slot_start, .. } => *slot_start as i32,
            Self::DMDc { slot_start, .. } => *slot_start as i32,
            Self::Adaptive { slot_start, .. } => *slot_start as i32,
        }
    }

    pub(crate) fn slot_end_as_streams_int(&self) -> i32 {
        match &self {
            Self::None => -1,
            Self::Constant { slot_end, .. } => *slot_end as i32,
            Self::Sinusoidal { slot_end, .. } => *slot_end as i32,
            Self::DMDc { slot_end, .. } => *slot_end as i32,
            Self::Adaptive { slot_end, .. } => *slot_end as i32,
        }
    }
}

#[derive(Debug, Clone, Parser)]
pub(crate) struct BlowingSlot {}

#[derive(Parser, Debug, Clone)]
pub(crate) struct SbliCases {
    #[clap(value_enum)]
    /// mode to run the case generation with
    pub(crate) mode: SbliMode,

    /// the location where all `distribute` files will
    /// be written
    pub(crate) output_directory: PathBuf,

    /// a matrix_id that you want to ping after the jobs are
    /// finished. Should look like: `@user_id:matrix.org`
    #[clap(long)]
    pub(crate) matrix: Option<distribute::OwnedUserId>,

    #[clap(long)]
    /// input databse file to use
    pub(crate) database_bl: PathBuf,

    /// path to the streams .sif file you wish to use
    /// to run this batch
    #[clap(long)]
    pub(crate) solver_sif: PathBuf,

    #[clap(long)]
    /// copy the .sif file to the output directory so
    /// that the run can be replicated later. if not
    /// passed the distribute-jobs.yaml file will reference
    /// the solver .sif file that may change at a later time
    pub(crate) copy_sif: bool,
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct JetValidation {
    /// the location where all `distribute` files will
    /// be written
    pub(crate) output_directory: PathBuf,

    /// name of the batch to use in `distribute`
    #[clap(long)]
    pub(crate) batch_name: String,

    /// a matrix_id that you want to ping after the jobs are
    /// finished. Should look like: `@user_id:matrix.org`
    #[clap(long)]
    pub(crate) matrix: Option<distribute::OwnedUserId>,

    /// path to the streams .sif file you wish to use
    /// to run this batch
    #[clap(long)]
    pub(crate) solver_sif: PathBuf,

    #[clap(long)]
    /// copy the .sif file to the output directory so
    /// that the run can be replicated later. if not
    /// passed the distribute-jobs.yaml file will reference
    /// the solver .sif file that may change at a later time
    pub(crate) copy_sif: bool,

    #[clap(long)]
    /// number of steps to include in the simulation
    pub(crate) steps: usize,

    #[clap(long)]
    /// input databse file to use
    pub(crate) database_bl: PathBuf,
}

#[derive(Debug, Clone, Parser, ValueEnum)]
pub(crate) enum SbliMode {
    /// generate sweeps for reynolds number, shock angle, and mach number
    Sweep,
    /// validate the blowing boundary condition case
    CheckBlowingCondition,
    /// ensure that the probes are working properly
    CheckProbes,
    /// run a single case
    OneCase,
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct RunContainer {
    // no arguments required, the number of MPI processes
    // allowed is set based on the number required by the input file
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct RunLocal {
    /// the number of processes that this program is allowed to use
    pub(crate) nproc: usize,

    #[clap(long)]
    /// working dir to run the solver in
    pub(crate) workdir: PathBuf,

    #[clap(long)]
    /// input.json file to load into the solver
    pub(crate) config: PathBuf,

    #[clap(long)]
    /// path to database.bl file required to run streams
    pub(crate) database: PathBuf,

    #[clap(long)]
    /// mount some python code into the container to run instead of the
    /// code contained in the solver image
    pub(crate) python_mount: Option<PathBuf>,
}

#[derive(Parser, Debug, Clone, Constructor)]
pub(crate) struct ParseProbe {
    /// mode to run the case generation with
    pub(crate) probe_directory: PathBuf,

    /// location where .mat files will be written
    pub(crate) output_directory: PathBuf,

    /// config json file that was used to generate probe data
    #[clap(long)]
    pub(crate) config: PathBuf,
}

#[derive(Parser, Debug, Clone, Constructor)]
pub(crate) struct VtkToMat {
    /// all the input files to write to the output directory
    pub(crate) input_files: Vec<PathBuf>,

    #[clap(long)]
    pub(crate) config: PathBuf,

    /// .mat file that is exported
    #[clap(long)]
    pub(crate) output_file: PathBuf,
}

#[derive(Parser, Debug, Clone, Constructor)]
pub(crate) struct SpansToVtk {
    /// the path to the solver results. Should contain the input.json file, x.dat, y.dat, z.dat
    /// as well as a ./spans/ folder containing .binary files to convert
    pub(crate) solver_results: PathBuf,

    #[clap(long)]
    /// remove the old binary files after converting to
    pub(crate) clean_binary: bool,
}

#[derive(Parser, Debug, Clone, Constructor)]
pub(crate) struct HDF5ToVtk {
    /// the path to the solver results.
    ///
    /// if run with streams-utils, this will be `./distribute_save` locally.
    ///
    /// This folder should contain a flowfields.h5 file. Results are written to a `vtks` folder
    /// within solver-results
    pub(crate) solver_results: PathBuf,
}
