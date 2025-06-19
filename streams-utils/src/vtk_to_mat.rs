use crate::prelude::*;
use ndarray::s;

pub(crate) fn vtk_to_mat(mut args: cli::VtkToMat) -> Result<(), Error> {
    let config = Config::from_path(&args.config)?;

    // sort all the input files
    args.input_files.sort_unstable();

    let num_files = args.input_files.len();

    let mut span_averages = Array4::zeros((num_files, 4, config.x_divisions, config.y_divisions));

    for (idx, file) in args.input_files.into_iter().enumerate() {
        println!("reading {}", file.display());
        let vtk_data: vtk::VtkData<
            vtk::Rectilinear2D<f64, vtk::Binary>,
            crate::binary_to_vtk::SpanVtkInformation,
        > = vtk::read_vtk(&file)?;

        // unpack data from the arrays
        let rho: ndarray::Array2<f64> = vtk_data.data.rho.into();
        let velocity: ndarray::Array3<f64> = vtk_data.data.velocity.into();
        let u = velocity.slice(s!(0usize, .., ..));
        let v = velocity.slice(s!(1usize, .., ..));
        let w = velocity.slice(s!(2usize, .., ..));

        // store that new data in the array at the appropriate time step
        span_averages
            .slice_mut(s![idx, 0usize, .., ..])
            .assign(&rho);
        span_averages.slice_mut(s![idx, 1usize, .., ..]).assign(&u);
        span_averages.slice_mut(s![idx, 2usize, .., ..]).assign(&v);
        span_averages.slice_mut(s![idx, 3usize, .., ..]).assign(&w);
    }

    let spans = SpanAverages::new(span_averages);

    let writer = std::io::BufWriter::new(
        std::fs::File::create(&args.output_file)
            .map_err(|e| FileError::new(args.output_file, e))?,
    );
    mat5::MatFile::write_contents(&spans, writer)?;

    Ok(())
}

#[derive(Debug, Constructor, mat5::MatFile)]
struct SpanAverages {
    span_averages: Array4,
}
