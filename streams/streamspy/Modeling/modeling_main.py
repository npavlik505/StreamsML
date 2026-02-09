"""Entry point for modeling utilities."""
import argparse
import json
import sys
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Script imports
from config import Config
from dmdc import run_dmdc, validate_rom_on_case, plot_dmd_modes, plot_pod_energy_spectrum, plot_pod_modes

# Function to collect u and/or v and/or w from components flag 
def _parse_components(value: str | None) -> list[int] | list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None
    if all(item.lstrip("-").isdigit() for item in items):
        return [int(item) for item in items]
    return items

def main() -> None:
    # modeling flags for dmdc
    parser = argparse.ArgumentParser(description="Run modeling on StreamsML output")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--episode-tag", default="", help="Episode tag used for learning based runs")
    parser.add_argument("--model-type", default="dmdc", help="Modeling method to apply")
    parser.add_argument("--workflow", default="run_dmdc", choices=("run_dmdc", "plot_modes", "validate_rom"), help="Modeling workflow to run")
    parser.add_argument("--energy", type=float, default=100.0, help="Percent of snapshot energy to retain in the ROM (0-100]")
    parser.add_argument("--velocity-components", default="1,2,3", help="Velocity component indices (comma-separated string)")
    parser.add_argument("--validation-sa", type=Path, help="Path to span_averages.h5 file you are evaluating ROM against")
    parser.add_argument("--validation-traj", type=Path, help="Path to trajectories.h5 file containing forcing values producing validation-sa, req'd for ROM eval")
    parser.add_argument("--compute-pointwise-error", action="store_true", default=False, help="Produces SquaredError.h5 if present")
    parser.add_argument("--case-label", type=str, help="BL or SBLI. Used to organize and title plots")
    parser.add_argument("--n-modes", type=int, dest="n_modes", help="Number of modes to plot")
    parser.add_argument("--block-rows", type=int, default=16, help="Row batch size for POD blocks")
    parser.add_argument("--time-batch", type=int, default=16, help="Time batch size for loading")

    args = parser.parse_args()

    # Access the input.json file
    input_dir = Path(args.results_dir, "input.json")
    with open(input_dir, "r") as f:
        cfg_json = json.load(f)
        config = Config.from_json(cfg_json)

    # Create the required paths
    if config.jet.jet_method_name == "OpenLoop":
        data_path = args.results_dir / "distribute_save"
    elif config.jet.jet_method_name == "Classical":
        data_path = args.results_dir / "distribute_save"
    else:
        data_path = args.results_dir / "LB_EvalData" / args.episode_tag
    sa_path = data_path / "span_averages.h5"
    traj_path = data_path / "trajectories.h5"

    output_dir = args.results_dir / "modeling_results"
    output_dir.mkdir(exist_ok=True)
    plot_dir = args.results_dir / "modeling_results" / "modeling_figs"
    plot_dir.mkdir(exist_ok=True) 

    # Model type selection (as of 2/1/26 only dmdc)
    method = args.model_type.lower()
    if method == "dmdc":
        velocity_components = _parse_components(args.velocity_components)
        # dmdc routine selection (as of 1/13/26 must be run sequentially)
        if args.workflow == "run_dmdc":
            run_dmdc(
                sa_path, 
                traj_path, 
                output_dir, 
                args.energy,
                velocity_components,
                args.block_rows,
                args.time_batch,
            )
        elif args.workflow == "validate_rom":
            validate_rom_on_case(
                args.validation_sa,
                args.validation_traj,
                output_dir,
                args.energy,
                args.case_label,
                velocity_components,
                args.time_batch,
                args.compute_pointwise_error,
            )
        elif args.workflow == "plot_modes":
            label = f"{int(args.energy)}pct"
            red_dir = output_dir / "ReducedModels"
            sigma_path = red_dir / f"Sigma_{label}.npy"
            basis_path = red_dir / f"DMDcBasis_{label}.npy"
            A_path = red_dir / f"A_red_matrix_{label}.npy"

            fig, _ = plot_pod_energy_spectrum(
                sigma_path=sigma_path,
                max_modes=args.n_modes,
                energy_pct=args.energy,
            )
            fig.savefig(plot_dir / "pod_energy_spectrum.png", dpi=300)
            plt.close(fig)

            fig, _ = plot_pod_modes(
                basis_path=basis_path,
                sigma_path=sigma_path,
                n_modes=args.n_modes,
                nx=config.grid.nx,
                ny=config.grid.ny,
                n_comp=args.n_comp,
                velocity_components=velocity_components,
                output_path=args.results_dir,
            )
            fig.savefig(plot_dir / "pod_modes.png", dpi=300)
            plt.close(fig)

            fig, _ = plot_dmd_modes(
                A_path=A_path,
                basis_path=basis_path,
                n_modes=args.n_modes,
                nx=config.grid.nx,
                ny=config.grid.ny,
                n_comp=args.n_comp,
                velocity_components=velocity_components,
                output_path=args.results_dir,
            )
            fig.savefig(plot_dir / "dmd_modes.png", dpi=300)
            plt.close(fig)
    else:
        print(f"Unknown model type: {args.model_type}")
        
if __name__ == "__main__":
    main()
