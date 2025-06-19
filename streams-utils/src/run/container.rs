use crate::prelude::*;

use super::create_dirs;
use super::postprocess;

/// running routine for the solver once activated within the container
pub(crate) fn run_container(_args: cli::RunContainer) -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    let path = PathBuf::from("/input/input.json");
    let dist_save = PathBuf::from("/distribute_save");
    let input_dat = PathBuf::from("/distribute_save/input.dat");

    // initialize some base directories within the folder we will work in
    create_dirs(&dist_save)?;

    // copy input.json to the output
    fs::copy(&path, "/distribute_save/input.json").unwrap();
    fs::copy("/input/database_bl.dat", "/distribute_save/database_bl.dat").unwrap();

    // read in the config json file
    let file = fs::File::open(&path)
        .with_context(|| format!("failed to open input.json file at {}", path.display()))?;

    let config: Config = serde_json::from_reader(file)?;

    // change the current working directory to the distribute_save directory. That way, all the
    // file that we need to run and work with will be output here
    let target_dir = PathBuf::from("/distribute_save");
    std::env::set_current_dir(&target_dir).with_context(|| {
        format!(
            "could not change current working directory to {}",
            target_dir.display()
        )
    })?;

    // then, generate the actual config for an output to the solver
    crate::config_generator::_config_generator(&config, input_dat)?;

    //
    // setup shell environment and run the solver
    //

    let sh = xshell::Shell::new()?;

    // choose the nproc
    let nproc = (config.mpi_x_split * 1).to_string();

    if config.use_python {
        let runtime_py = PathBuf::from("/runtimesolver/");
        let static_py = PathBuf::from("/streamspy/");

        let solver_py = if runtime_py.exists() {
            println!("running python bindings with runtime solver");
            runtime_py
        } else {
            println!("running static python bindings");
            static_py
        };

        let exec = xshell::cmd!(sh, "mpirun -np {nproc} {solver_py}/main.py");

        println!("Now running solver, STDOUT will be hidden until it finishes");
        exec.run()?;
    } else {
        // TODO: make this command share the current stdout
        let exec = xshell::cmd!(sh, "mpirun -np {nproc} /streams.exe");

        println!("Now running solver, STDOUT will be hidden until it finishes");
        exec.run()?;
    }

    postprocess(&config)?;

    let end = start.elapsed();
    let hours = end.as_secs() / 3600;
    let minutes = (end.as_secs() / 60) - (hours * 60);
    let seconds = end.as_secs() - (hours * 3600) - (minutes * 60);
    println!(
        "runtime information (hhhh:mm:ss): {:04}:{:02}:{02}",
        hours, minutes, seconds
    );

    Ok(())
}
