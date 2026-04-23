# StreamsML

StreamsML is a control, modeling, and analysis platform for compressible flow simulations built around the [STREAmS solver](https://github.com/matteobernardini/STREAmS) and a Python/Gymnasium interface. It extends STREAmS with a customizable jet actuator and a workflow for testing control, reduced-order modeling, and analysis methods on high-speed boundary-layer and shock/boundary-layer-interaction simulations.

## What StreamsML is for

StreamsML was built to make high-fidelity flow-control experiments more reproducible and easier to modify. It provides a structured environment for:

- running boundary-layer (BL) and shock/boundary-layer-interaction (SBLI) simulations,
- testing open-loop, classical, and learning-based control methods,
- applying reduced-order modeling methods such as POD, DMD, and DMDc, and
- generating analysis and visualization outputs from simulation results.

## Repository structure

- `streams/` — solver-side code, Python interface, control/modeling/analysis modules, and simulation assets
- `streams-utils/` — build, configuration, and execution utilities
- `install_dependencies.sh` — environment bootstrap script
- `README.md` — overview and quick-start instructions
- `rust-toolchain.toml` — Rust toolchain pinning for the utilities

## Personal contributions

The original STREAmS solver was written in Fortran. Prior work in the lab had introduced a jet actuator and open-loop control capability, but the software was no longer reproducibly buildable when I began working on it. My work on StreamsML focused on making the platform usable as a repeatable experimentation environment for reinforcement learning and other complex forms of control:

- updated and pinned the build environment,
- added reproducible setup and execution utilities,
- re-wrapped and extended Fortran routines for Python-side use,
- designed and implemented a Python-based control framework for both classical and learning-based methods, and
- created a Python-based modeling and analysis workflow.

## Quick start

For detailed usage beyond a standard run, see the [detailed user instructions (PDF)](streams/svgs/StreamsMLmanual.pdf).

All commands below are run from the `streams-utils/` directory unless noted otherwise.

### 1. Clone the repository

```bash
git clone https://github.com/npavlik505/StreamsML.git --depth 1
cd StreamsML
```

### 2. Set the required paths

Add the following paths to your shell configuration and reload it:

```bash
export STREAMS_DIR="/path/to/your/StreamsML/streams"
export STREAMS_UTILS_DIR="/path/to/your/StreamsML/streams-utils"
```

For example:

```bash
export STREAMS_DIR="/home/username/Desktop/StreamsML/streams"
export STREAMS_UTILS_DIR="/home/username/Desktop/StreamsML/streams-utils"
```

### 3. Install dependencies

```bash
chmod +x install_dependencies.sh
bash install_dependencies.sh
```

If Apptainer is not installed successfully by the script, install it separately using the [official Apptainer documentation](https://apptainer.org/docs/admin/main/installation.html).

### 4. Build the containerized software stack

From `streams-utils/`, run:

```bash
just nv
just base
just build
```

This produces the final `streams.sif` container used to run the software.

## Build overview

The build process uses three main layers:

- **NVIDIA HPC SDK container** — provides the base compiler and CUDA-capable HPC environment
- **`base.apptainer`** — installs required software such as Python packages and system dependencies
- **`build.apptainer`** — compiles the Fortran and Rust components and assembles the runnable StreamsML environment

Before building, make sure your system satisfies the CUDA/driver requirements for the NVIDIA HPC SDK tag used in this repository, included below:

![Nvidia HPC SDK system requirements](streams/svgs/NvidiaHPCsysreq.png)

## Configure and run a standard simulation

### 1. Choose the flow type

In the `Justfile`, select one of the two flow configurations:

```make
# streams_flow_type := "shock-boundary-layer"
# streams_flow_type := "boundary-layer"
```

### 2. Edit the `config` recipe

The main user interface is the `config` recipe in the `Justfile`. This is where you choose:

- simulation parameters,
- control strategy and algorithm,
- actuator settings,
- training and evaluation settings, and
- output/checkpoint locations.

Control strategies are grouped into categories such as open-loop, classical, and learning-based. The README example uses a learning-based DDPG configuration, but the utilities support multiple choices depending on the selected mode.

A shortened example looks like this:

```make
config:
	echo {{config_output}}

	cargo r -- \
		config-generator {{config_output}} {{streams_flow_type}} \
		--steps 6 \
		--reynolds-number 250 \
		--mach-number 2.28 \
		--x-divisions 600 \
		--y-divisions 208 \
		... \
		--use-python \
		learning-based ddpg \
		    --slot-start 100 \
		    --slot-end 149 \
		    --train-episodes 2 \
		    --training-output {{training}} \
		    --eval-episodes 2 \
		    --eval-output {{eval}} \
		    --checkpoint-dir {{checkpoint}}
```

Use the templates in `JustfileExamples/` to quickly assemble valid configurations.

You can also inspect the available flags with:

```bash
just help tree
```

### 3. Generate the run configuration

```bash
just config
```

### 4. Launch the simulation

```bash
just run
```

## Notes for users

- The README is intentionally a quick-start guide.
- More detailed instructions for modeling, analysis, visualization, and adding new methods are provided in the linked PDF documentation, [Detailed user instructions (PDF)](streams/svgs/StreamsMLmanual.pdf)
- Most user-facing configuration happens through the `Justfile` and the `streams-utils/` tooling rather than through a polished standalone CLI.

## Why this README is structured this way

StreamsML serves two audiences: someone evaluating the project at a glance, and someone trying to actually build and run it. This README keeps the top of the page focused on what the project is and why it exists, then moves quickly into the minimum steps needed to build and run a standard case.
