use crate::prelude::*;
use anyhow::Result;

use ndarray::s;
use ndarray::Array3;
use ndarray::Array4;

use rayon::prelude::*;

#[derive(vtk::DataArray, Clone)]
#[vtk_write(encoding = "binary")]
struct FlowfieldData {
    rho: vtk::Scalar3D<f32>,
    velocity: vtk::Vector3D<f32>,
    energy: vtk::Scalar3D<f32>,
}

pub(crate) fn hdf5_to_vtk(args: cli::HDF5ToVtk) -> Result<()> {
    let flowfields_file = args.solver_results.join("flowfields.h5");
    let vtk_output_folder = args.solver_results.join("vtk");

    //
    // create vtk output folder
    //

    if vtk_output_folder.exists() {
        std::fs::remove_dir_all(&vtk_output_folder).with_context(|| {
            format!(
                "failed to clear vtk output folder at {}",
                vtk_output_folder.display()
            )
        })?;
    }

    std::fs::create_dir(&vtk_output_folder).with_context(|| {
        format!(
            "failed to create vtk output folder at {}",
            vtk_output_folder.display()
        )
    })?;

    //
    // check hdf5 files exist where we expect them w/ correct datasets
    // and dimensions
    //

    let file = hdf5::file::File::open(&flowfields_file)
        .with_context(|| format!("failed to open flowfields file `{}`. Are you sure the directory you passed in was correct?", flowfields_file.display()))?;

    let dset = file.dataset("velocity").with_context(|| {
        format!(
            "dataset `velocity` was missing from h5 file {}",
            flowfields_file.display()
        )
    })?;

    // shape of the data is
    // <numwrites, 5, NX, NY, NZ>
    let shape = dset.shape();
    if shape.len() != 5 {
        anyhow::bail!("velocity flowfields file was not 5 dimensional, this should not happen")
    }

    let nwrite = shape[0];

    //
    // load config files for the run
    //

    let config_path = args.solver_results.join("input.json");
    let config = Config::from_path(&config_path)
        .with_context(|| format!("failed to read config at path {}", config_path.display()))?;

    let mesh = run::MeshInfo::from_base_path(&args.solver_results, &config)?;
    let vtk_mesh: vtk::Mesh3D<_, vtk::Binary> =
        vtk::Mesh3D::new(mesh.x_data, mesh.y_data, mesh.z_data);

    let nx = config.x_divisions;
    let ny = config.y_divisions;
    let nz = config.z_divisions;

    //
    // setup vtk containers and types
    //

    let spans = vtk::Spans3D::new(nx, ny, nz);
    let domain = vtk::Rectilinear3D::new(vtk_mesh, spans);

    let rho: Array3<f32> = Array3::zeros((nx, ny, nz));
    let velocity: Array4<f32> = Array4::zeros((3, nx, ny, nz));
    let energy: Array3<f32> = Array3::zeros((nx, ny, nz));

    let data = FlowfieldData {
        rho: vtk::Scalar3D::new(rho),
        velocity: vtk::Vector3D::new(velocity),
        energy: vtk::Scalar3D::new(energy),
    };

    let vtk_container = vtk::VtkData::new(domain, data);

    let _: Result<()> = (0..nwrite)
        .into_par_iter()
        .map(|write| -> Result<(), anyhow::Error> {
            let mut vtk_container = vtk_container.clone();
            let slice = ndarray::s![write, .., .., .., ..];

            let curr_data: Array4<f32> = dset
                .read_slice(slice)
                .with_context(|| "failed to read hdf5 dataset")?;

            let rho_slice = s![0, .., .., ..];
            let velocity_slice = s![1..=3, .., .., ..];
            let energy_slice = s![4, .., .., ..];

            let rho_ref = curr_data.slice(rho_slice);
            let rho_velocity_ref = curr_data.slice(velocity_slice);
            let rho_energy_ref = curr_data.slice(energy_slice);

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        vtk_container.data.rho[[i, j, k]] = rho_ref[[i, j, k]];
                        vtk_container.data.energy[[i, j, k]] =
                            rho_energy_ref[[i, j, k]] / rho_ref[[i, j, k]];

                        for v in 0..3 {
                            vtk_container.data.velocity[[v, i, j, k]] =
                                rho_velocity_ref[[v, i, j, k]] / rho_ref[[i, j, k]];
                        }
                    }
                }
            }

            println!("writing flowfield file {write}/{nwrite}");

            // after we have updated all the arrays, write the vtk out to a file
            let write_path = vtk_output_folder.join(format!("flowfield_{write:05}.vtr"));
            let file = std::fs::File::create(&write_path).with_context(|| {
                format!("unable to create vtr output at {}", write_path.display())
            })?;
            let writer = std::io::BufWriter::new(file);

            vtk::write_vtk(writer, vtk_container)?;

            Ok(())
        })
        .collect();

    Ok(())
}
