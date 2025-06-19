//! handles cli commands for exporting probe data to .mat files

use crate::prelude::*;

/// for a given configuration file and probe directory, parse the probe binary information and
/// transform the data to .mat files
pub(crate) fn probe(args: cli::ParseProbe) -> Result<(), Error> {
    let config = Config::from_path(&args.config)?;

    // group all of the probes data together
    let paths = fs::read_dir(&args.probe_directory)
        .map_err(|e| FileError::new(args.probe_directory.clone(), e))?
        .into_iter()
        .filter_map(|entry_res| entry_res.ok())
        .map(|entry: fs::DirEntry| entry.path())
        .filter(|entry_path| {
            entry_path
                .file_name()
                .map(|name| name.to_str().unwrap().contains("span_probe"))
                .unwrap_or(false)
        });

    let grouping = group_probes_by_number(paths);

    // create the output directory if it does not exist
    if !args.output_directory.exists() {
        std::fs::create_dir(&args.output_directory)
            .map_err(|e| FileError::new(args.output_directory.clone(), e))?;
    }

    // helper function to create a writer for the given .mat file destination path and parse the data to
    // that file
    fn parse_probe_helper(
        probe_info: Vec<ProbeInfo>,
        config: &Config,
        path: &Path,
    ) -> Result<(), Error> {
        let writer = io::BufWriter::new(
            std::fs::File::create(path).map_err(|e| FileError::new(path.to_owned(), e))?,
        );
        let paths = probe_info
            .into_iter()
            .map(|info| info.path)
            .collect::<Vec<_>>();

        crate::probe_binary::parse_file_group(paths.as_slice(), config.z_divisions, writer)?;
        Ok(())
    }

    parse_probe_helper(
        grouping.one,
        &config,
        &args.output_directory.join("probe_1.mat"),
    )?;
    parse_probe_helper(
        grouping.two,
        &config,
        &args.output_directory.join("probe_2.mat"),
    )?;
    parse_probe_helper(
        grouping.three,
        &config,
        &args.output_directory.join("probe_3.mat"),
    )?;

    Ok(())
}

#[derive(PartialEq, Eq, Debug)]
/// metadata parsed from probe filesystem path on where it is
/// and the timestep at which the data was collected
struct ProbeInfo {
    path: PathBuf,
    step_number: usize,
    probe_number: usize,
}

/// all probe paths for a given run, grouped by the fortran probe_number and ordered
/// by the step at which they were written
struct ProbeGrouping {
    one: Vec<ProbeInfo>,
    two: Vec<ProbeInfo>,
    three: Vec<ProbeInfo>,
}

fn group_probes_by_number(probe_paths: impl Iterator<Item = PathBuf>) -> ProbeGrouping {
    let mut one = Vec::new();
    let mut two = Vec::new();
    let mut three = Vec::new();

    probe_paths
        //.into_iter()
        .map(probe_metadata)
        .for_each(|probe_meta: ProbeInfo| {
            // match the probe number to the container that it belongs to
            //
            // this could be made more robust with hashmaps but I elect for it to
            // be more explicit for clairty
            if probe_meta.probe_number == 1 {
                one.push(probe_meta);
            } else if probe_meta.probe_number == 2 {
                two.push(probe_meta);
            } else if probe_meta.probe_number == 3 {
                three.push(probe_meta);
            } else {
                panic!(
                    "unknown probe number parsed ({}) for path {}",
                    probe_meta.probe_number,
                    probe_meta.path.display()
                );
            }
        });

    // make sure they are ordered by the step number
    one.sort_unstable_by_key(|x| x.step_number);
    two.sort_unstable_by_key(|x| x.step_number);
    three.sort_unstable_by_key(|x| x.step_number);

    ProbeGrouping { one, two, three }
}

/// parse the probe number and step at which a binary file of probe data was written at
// TODO: better error handling in this function
fn probe_metadata(probe: PathBuf) -> ProbeInfo {
    let filename = probe
        .file_name()
        .expect(&format!(
            "probe file {} was missing a file name",
            probe.display()
        ))
        .to_string_lossy();

    // based on the fortran code the file name follows this format
    // span_probe_[1 char]_[5 char step number].binary
    let header = "span_probe";
    let sep = "_";

    let probe_number_length = 1;
    let cycle_number_length = 5;

    let probe_num_start = header.len() + sep.len();
    let cycle_num_start = probe_num_start + sep.len() + probe_number_length;

    let probe_number_str = filename
        .get(probe_num_start..probe_num_start + probe_number_length)
        .unwrap();
    let probe_step_str = filename
        .get(cycle_num_start..cycle_num_start + cycle_number_length)
        .unwrap();

    let probe_number = probe_number_str.parse().expect(&format!(
        "failed to parse probe number for file name {} - probe number string to parse was {}",
        filename, probe_number_str
    ));
    let step_number = probe_step_str.parse().expect(&format!(
        "failed to parse probe step for file name {} - probe number string to parse was {}",
        filename, probe_step_str
    ));

    ProbeInfo {
        path: probe,
        step_number,
        probe_number,
    }
}

#[test]
fn check_probe_parse() {
    let path = PathBuf::from("./some/probe/path/span_probe_1_99999.binary");
    let expected = ProbeInfo {
        path: path.clone(),
        step_number: 99999,
        probe_number: 1,
    };
    let parsed_info = probe_metadata(path);
    assert_eq!(expected, parsed_info);
}
