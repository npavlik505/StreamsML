//! parse probe binary and export it .mat files

use crate::prelude::*;
use rayon::prelude::*;

#[derive(Debug, thiserror::Error, From)]
/// An error that arises from parsing binary files of probe data
pub(crate) enum ProbeBinaryError {
    #[error("{0}")]
    InsufficientLength(Length),
    #[error("{0}")]
    File(FileError),
    #[error("Could not write the file using mat5: {0}")]
    Mat5(mat5::Error),
}

#[derive(Display, Debug, Constructor)]
#[display(
    fmt = "Data was unexpected length for nz = {}. Length was {} but expected a length of {} ({} difference)",
    nz,
    length,
    expected_length,
    difference
)]
/// Represents an error caused by data being an unexpected length
pub(crate) struct Length {
    nz: usize,
    length: usize,
    expected_length: usize,
    difference: usize,
}

/// Array data as it is stored in a single file written by fortran
struct ProbeFile {
    viscous: Array2,
    log_law: Array2,
    freestream: Array2,
}

/// Probe data as it is stored in a matfile
///
/// the matfile will contain arrays named after each of the fields. Each array
/// is 3 dimensional with
///
/// | Dimension | Data chosen |
/// |-----------|-------------|
/// | 1st       | Timestep, in the order that they are created |
/// | 2nd       | Data type. This dimension should have 4 indicies for : rho, u, v, w |
/// | 3rd       | Probe data at the given timestep and data type |
#[derive(mat5::MatFile)]
struct ProbeFileAllTimesteps {
    viscous: Array3,
    log_law: Array3,
    freestream: Array3,
}

/// multithreaded parser for probe data.
///
/// The files send to this function must be sorted to the
/// chronological order in which they were created as no attempt to
/// sort them is made in this function.
pub(crate) fn parse_file_group<W: Write>(
    files: &[PathBuf],
    nz: usize,
    writer: W,
) -> Result<(), ProbeBinaryError> {
    let parse_results: Result<Vec<(usize, ProbeFile)>, _> = files
        .into_par_iter()
        .enumerate()
        .map(|(idx, path)| {
            // load and parse the binary file in parallel
            parse_binary_file(path.as_ref(), nz).map(|x| (idx, x))
        })
        .collect();

    let shape = (files.len(), 4, nz);

    let mut viscous = Array3::zeros(shape);
    let mut log_law = Array3::zeros(shape);
    let mut freestream = Array3::zeros(shape);

    // copy all of the slices into big matricies
    for (idx, slice) in parse_results? {
        viscous
            .slice_mut(ndarray::s![idx, .., ..])
            .assign(&slice.viscous);
        log_law
            .slice_mut(ndarray::s![idx, .., ..])
            .assign(&slice.log_law);
        freestream
            .slice_mut(ndarray::s![idx, .., ..])
            .assign(&slice.freestream);
    }

    // assemble the data and write it out to a file
    let probes = ProbeFileAllTimesteps {
        viscous,
        log_law,
        freestream,
    };
    mat5::MatFile::write_contents(&probes, writer)?;

    Ok(())
}

/// load a single binary file to its constituent arrays
fn parse_binary_file(path: &Path, nz: usize) -> Result<ProbeFile, ProbeBinaryError> {
    let bytes = std::fs::read(path).map_err(|e| FileError::new(path.to_owned(), e))?;
    let floats = utils::bytes_to_float(bytes.as_slice());

    // 4 pieces of information are written per data point, there are nz data points
    let slice_len = nz * 4;

    if slice_len * 3 != floats.len() {
        let diff = ((floats.len() as isize) - (slice_len * 3) as isize).abs() as usize;
        return Err(Length::new(nz, floats.len(), slice_len * 3, diff).into());
    }

    let viscous_floats = floats.get(0..slice_len).unwrap();
    let log_law_floats = floats.get(slice_len..2 * slice_len).unwrap();
    let freestream_floats = floats.get(2 * slice_len..3 * slice_len).unwrap();

    let viscous = read_array(viscous_floats, nz)?;
    let log_law = read_array(log_law_floats, nz)?;
    let freestream = read_array(freestream_floats, nz)?;

    Ok(ProbeFile {
        viscous,
        log_law,
        freestream,
    })
}

/// from a buffer of bytes that is
#[inline(always)]
fn read_array(buffer: &[f64], nz: usize) -> Result<Array2, ProbeBinaryError> {
    // check that the buffer is the length that we expect it to be
    if buffer.len() != nz * 4 {
        let diff = ((buffer.len() as isize) - (nz * 4) as isize).abs() as usize;
        return Err(Length::new(nz, buffer.len(), nz * 4, diff).into());
    }

    let mut arr = Array2::zeros((4, nz));

    buffer
        .chunks(4)
        .into_iter()
        .enumerate()
        .for_each(|(idx, chunk)| {
            // iterate through each of the probe points of information
            for information_idx in 0..4 {
                *arr.get_mut((information_idx, idx)).unwrap() = chunk[information_idx]
            }
        });

    Ok(arr)
}

#[test]
// ensure a .binary file written by the solver parses as we expect it to
fn parse_probe_binary() {
    let file = PathBuf::from("./static/span_probe_example.binary");
    let nz = 150;
    parse_binary_file(&file, nz).unwrap();
}
