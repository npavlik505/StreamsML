use crate::prelude::*;
use anyhow::Result;

/// convert a general solver folder full of span binaries to vtk files
pub(crate) fn spans_to_vtk(args: cli::SpansToVtk) -> Result<()> {
    // load the config file
    let config = Config::from_path(&args.solver_results.join("input.json"))?;

    // load the mesh information
    let mesh = run::MeshInfo::from_base_path(&args.solver_results, &config)?;

    // then convert all the binar files to vtk files
    run::convert_spans(&args.solver_results, &config, &mesh, args.clean_binary)?;

    Ok(())
}
