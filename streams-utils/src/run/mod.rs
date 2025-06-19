use crate::prelude::*;
mod container;
mod local;

pub(crate) use container::run_container;
pub(crate) use local::run_local;

use anyhow::Result;

/// create all the folders that data is written to in the solver
fn create_dirs(base: &Path) -> Result<()> {
    let csv = base.join("csv_data");
    fs::create_dir(&csv).with_context(|| format!("failed to create dir {}", csv.display()))?;

    let spans = base.join("spans");
    fs::create_dir(&spans).with_context(|| format!("failed to create dir {}", csv.display()))?;

    Ok(())
}

fn read_mesh_info(path: &Path, ghost_nodes: usize, values: usize) -> Result<Vec<f64>> {
    let file_data = fs::read_to_string(path)
        .with_context(|| format!("could not read contents of file {}", path.display()))?;

    let data = file_data
        .split('\n')
        .into_iter()
        .map(|row| row.trim().parse())
        .take_while(|x| x.is_ok())
        .map(|x| x.unwrap())
        .skip(ghost_nodes)
        .take(values)
        .collect();

    Ok(data)
}

/// information on the meshing `dx` `dy` `dz` from the streams output files
#[derive(mat5::MatFile, Debug)]
pub(crate) struct MeshInfo {
    pub(crate) x_data: Vec<f64>,
    pub(crate) y_data: Vec<f64>,
    pub(crate) z_data: Vec<f64>,
}

impl MeshInfo {
    pub(crate) fn from_base_path(base: &Path, config: &Config) -> Result<Self> {
        let xg = base.join("x.dat");
        let yg = base.join("y.dat");
        let zg = base.join("z.dat");

        // TODO: update this in the config generation - but i doubt it will ever change
        let ghost_nodes = 3;

        let x_data = read_mesh_info(&xg, ghost_nodes, config.x_divisions)?;
        let y_data = read_mesh_info(&yg, ghost_nodes, config.y_divisions)?;
        let z_data = read_mesh_info(&zg, ghost_nodes, config.z_divisions)?;

        Ok(Self {
            x_data,
            y_data,
            z_data,
        })
    }
}

/// general parent postprocessing routine to be called after the solver has finished
fn postprocess(config: &Config) -> Result<()> {
    let data_location = PathBuf::from("/distribute_save");
    let mesh_info = MeshInfo::from_base_path(&data_location, config)?;

    // convert all the binary spans to vtk files
    convert_spans(&data_location, config, &mesh_info, true)?;

    // write the probes to a folder + create matfiles folder
    write_probes(&data_location)?;

    // write the mesh information to the matfiles folder
    let mesh_path = data_location.join("matfiles/mesh.mat");
    let writer = fs::File::create(&mesh_path).with_context(|| {
        format!(
            "failed to create directory for mesh information: {}",
            mesh_path.display()
        )
    })?;
    mat5::MatFile::write_contents(&mesh_info, writer)?;

    Ok(())
}

/// helper function for assembling all the elements to write all binary data to .mat files
fn write_probes(location: &Path) -> Result<(), Error> {
    let probe_folder = location.join("csv_data");
    let output_folder = location.join("matfiles");
    let config = location.join("/input/input.json");
    let args = cli::ParseProbe::new(probe_folder, output_folder, config);
    crate::probe::probe(args)?;
    Ok(())
}

/// Convert all .binary files in the ./spans directory to Vtk files using mesh information
pub(crate) fn convert_spans(
    data_location: &Path,
    config: &Config,
    mesh_info: &MeshInfo,
    remove_binary: bool,
) -> Result<(), Error> {
    let spans_folder = data_location.join("spans");

    // currently not possible to write arrays in binary for 2D files
    let mesh =
        vtk::Mesh2D::<_, vtk::Ascii>::new(mesh_info.x_data.clone(), mesh_info.y_data.clone());

    let spans = vtk::Spans2D::new(config.x_divisions, config.y_divisions);
    let domain = vtk::Rectilinear2D::new(mesh, spans);

    for file in walkdir::WalkDir::new(&spans_folder)
        .into_iter()
        .filter_map(|e| e.ok())
        // the first item will be the root folder we created
        // this makes sure we skip any item that is a directory
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext != "vtr")
                .unwrap_or(false)
        })
    {
        let path = file.path();

        let file_name = path.file_stem().unwrap().to_string_lossy();
        let output_name = format!("{}.vtr", file_name);
        let output_path = spans_folder.join(output_name);

        // read the data to something we can write a vtk with

        let mut file = fs::File::open(path).map_err(|e| FileError::new(path.to_owned(), e))?;

        // five arrays, each taking nx * ny points, and each point uses 8 bytes
        let mut buffer = Vec::with_capacity(8 * config.x_divisions * config.y_divisions * 5);
        file.read_to_end(&mut buffer).unwrap();
        let float_bytes = utils::bytes_to_float(&buffer);

        let data = binary_to_vtk::convert_binary_to_vtk_information(&float_bytes, config)?;

        let vtk = vtk::VtkData::new(domain.clone(), data);

        let writer = io::BufWriter::new(
            fs::File::create(&output_path)
                .map_err(|e| FileError::new(output_path.to_owned(), e))?,
        );

        vtk::write_vtk(writer, vtk)?;

        if remove_binary {
            fs::remove_file(path).unwrap()
        }
    }

    Ok(())
}

#[test]
fn read_mesh_file() {
    let file = PathBuf::from("./static/x.dat");
    assert_eq!(file.exists(), true);

    let ghost = 3;
    let nx = 840;

    let x = read_mesh_info(&file, ghost, nx).unwrap();
    dbg!(&x);
    assert_eq!(x.len(), 840);
}
