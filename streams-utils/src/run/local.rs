use crate::prelude::*;
pub(crate) use anyhow::Result;

use xshell::{cmd, Shell};

struct Solver {
    #[allow(dead_code)]
    working_dir: PathBuf,
    input: PathBuf,
    dist_save: PathBuf,
}

impl Solver {
    fn new(working_dir: PathBuf) -> Result<Self> {
        //
        // create folders for input
        //
        let input = working_dir.join("input");
        if input.exists() {
            std::fs::remove_dir_all(&input).with_context(|| {
                format!(
                    "failed to remove full directory at {} for input",
                    input.display()
                )
            })?;
        }

        std::fs::create_dir(&input).with_context(|| {
            format!(
                "failed to create directory at {} for input",
                input.display()
            )
        })?;

        //
        // create folders for distribute save
        //
        let dist_save = working_dir.join("distribute_save");
        if dist_save.exists() {
            std::fs::remove_dir_all(&dist_save).with_context(|| {
                format!(
                    "failed to remove full directory at {} for distribute save",
                    dist_save.display()
                )
            })?;
        }
        std::fs::create_dir(&dist_save).with_context(|| {
            format!(
                "failed to create directory at {} for distribute save",
                dist_save.display()
            )
        })?;

        let s = Solver {
            working_dir,
            input,
            dist_save,
        };

        Ok(s)
    }

    fn load_input_file(&self, host_path: &Path, container_name: &str) -> Result<()> {
        let container_path = self.input.join(container_name);
        std::fs::copy(host_path, &container_path).with_context(|| {
            format!(
                "failed to copy {} to {}",
                host_path.display(),
                container_path.display()
            )
        })?;

        Ok(())
    }

    fn run(&self, nproc: usize, python_mount: String) -> Result<()> {
        let sh = Shell::new()?;

        let results_path = &self.dist_save;
        let input_path = &self.input;
        let nproc = nproc.to_string();

        let exec = cmd!(sh, "apptainer run --nv --bind {results_path}:/distribute_save,{input_path}:/input{python_mount} --app distribute ./streams.sif {nproc}")
            // ignore the output status so we get more STDOUT information?
            .ignore_status();

        exec.run()?;

        Ok(())
    }
}

pub(crate) fn run_local(args: cli::RunLocal) -> Result<()> {
    let sif_file = PathBuf::from("./streams.sif");

    if !sif_file.exists() {
        anyhow::bail!("streams.sif does not exist in the current directory. Are you sure you are running from the ./streams-utils folder");
    }

    let solver = Solver::new(args.workdir)?;

    solver.load_input_file(&args.config, "input.json")?;
    solver.load_input_file(&args.database, "database_bl.dat")?;

    // if a directory was specified to run the solver then we format it to a binding for the
    // `apptainer run` comamnd, otherwise an empty string will not change the output
    let python_mount = if let Some(mount_path) = args.python_mount {
        format!(
            ",{}:/runtimesolver",
            mount_path.to_string_lossy().into_owned()
        )
    } else {
        "".to_string()
    };

    solver.run(args.nproc, python_mount)?;

    Ok(())
}
