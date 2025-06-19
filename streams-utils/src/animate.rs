use crate::prelude::*;
use anyhow::Result;
use rayon::prelude::*;

use xshell::{cmd, Shell};

const NUM_THREADS: usize = 4;
const ANIMATE_SCRIPT: &str = "/home/brooks/github/streams-utils/analysis/scripts/animate.jl";

pub(crate) fn animate(args: cli::Animate) -> Result<()> {
    let animation_output_folder = args.data_folder.join("animation");

    if animation_output_folder.exists() {
        std::fs::remove_dir_all(&animation_output_folder).ok();
    }

    std::fs::create_dir(&animation_output_folder).with_context(|| {
        format!(
            "failed to create animation output folder at {}",
            animation_output_folder.display()
        )
    })?;

    let span_averages_path = args.data_folder.join("span_averages.h5");

    let span_averages_file = hdf5::File::open(&span_averages_path)
        .with_context(|| format!("failed to open file {}", span_averages_path.display()))?;

    let averages = span_averages_file
        .dataset("span_average")
        .with_context(|| {
            format!(
                "failed to find span_average dataset in {}",
                span_averages_path.display()
            )
        })?;

    let num_writes = averages.shape()[0];

    let partitions = partition_animation_indicies(NUM_THREADS, num_writes);

    let folder_list = args.data_folder.display().to_string();
    let decimate = args.decimate.to_string();

    partitions.into_par_iter().for_each(|partition| {
        let sh = Shell::new().unwrap();
        let AnimationSpan {
            start_idx,
            end_idx,
            cpu_number,
        } = partition;
        let start_idx = start_idx.to_string();
        let end_idx = end_idx.to_string();

        // julia needs about 30 seconds to compile each thread, and each thread takes quite a bit
        // of memory to compile so we stagger them here
        std::thread::sleep(std::time::Duration::from_secs(30 * cpu_number as u64));

        let cmd = cmd!(
            sh,
            "julia {ANIMATE_SCRIPT} {start_idx} {end_idx} {decimate} {folder_list}"
        );
        cmd.run().unwrap();
    });

    if args.decimate != 1 {
        reorganize_folder(&animation_output_folder)?;
    }

    Ok(())
}

fn reorganize_folder(animation_folder: &Path) -> Result<()> {
    let mut entries = walkdir::WalkDir::new(&animation_folder)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|f| f.file_type().is_file())
        .map(|f| f.into_path())
        .collect::<Vec<PathBuf>>();

    entries.sort_unstable();

    for (idx, file) in entries.into_iter().enumerate() {
        let new_path = animation_folder.join(format!("anim_{idx:05}.png"));

        std::fs::rename(&file, &new_path).with_context(|| {
            format!(
                "failed to rename {} to {}",
                file.display(),
                new_path.display()
            )
        })?;
    }

    Ok(())
}

fn join_folders_to_list(folder_list: &[PathBuf]) -> String {
    let mut out = String::new();

    for folder in folder_list {
        out.push_str(&folder.display().to_string());
        out.push(' ');
    }

    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AnimationSpan {
    cpu_number: usize,
    start_idx: usize,
    end_idx: usize,
}

fn partition_animation_indicies(num_cpus: usize, total_indicies: usize) -> Vec<AnimationSpan> {
    let estimation = total_indicies / num_cpus;

    let mut last_end_idx = 0;

    let mut spans = Vec::new();

    for cpu_number in 0..(num_cpus - 1) {
        let start_idx = last_end_idx + 1;
        let end_idx = start_idx + estimation - 1;

        last_end_idx = end_idx;

        spans.push(AnimationSpan {
            start_idx,
            end_idx,
            cpu_number,
        });
    }

    let start_idx = last_end_idx + 1;
    let end_idx = total_indicies;

    spans.push(AnimationSpan {
        start_idx,
        end_idx,
        cpu_number: num_cpus - 1,
    });

    spans
}

#[test]
fn even_cpu_split() {
    let cpus = 4;
    let total_indicies = 16;

    let partition = partition_animation_indicies(cpus, total_indicies);

    let expected = vec![
        AnimationSpan {
            cpu_number: 0,
            start_idx: 1,
            end_idx: 4,
        },
        AnimationSpan {
            cpu_number: 1,
            start_idx: 5,
            end_idx: 8,
        },
        AnimationSpan {
            cpu_number: 2,
            start_idx: 9,
            end_idx: 12,
        },
        AnimationSpan {
            cpu_number: 3,
            start_idx: 13,
            end_idx: 16,
        },
    ];

    assert_eq!(partition, expected);
}

#[test]
fn noneven_cpu_split() {
    let cpus = 4;
    let total_indicies = 18;

    let partition = partition_animation_indicies(cpus, total_indicies);

    let expected = vec![
        AnimationSpan {
            cpu_number: 0,
            start_idx: 1,
            end_idx: 4,
        },
        AnimationSpan {
            cpu_number: 1,
            start_idx: 5,
            end_idx: 8,
        },
        AnimationSpan {
            cpu_number: 2,
            start_idx: 9,
            end_idx: 12,
        },
        AnimationSpan {
            cpu_number: 3,
            start_idx: 13,
            end_idx: 18,
        },
    ];

    assert_eq!(partition, expected);
}
