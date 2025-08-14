"""Entry point for modeling utilities."""
import argparse
import json
import sys
from pathlib import Path
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Script imports
from config import Config
from dmdc import run_dmdc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modeling on StreamsML output")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--episode-tag", default="", help="Episode tag used for learning based runs")
    parser.add_argument("--model-type", default="dmdc", help="Modeling method to apply")
    args = parser.parse_args()

    input_dir = Path(args.results_dir, "input.json")
    with open(input_dir, "r") as f:
        cfg_json = json.load(f)
        config = Config.from_json(cfg_json)

    if config.jet.jet_method_name == "OpenLoop":
        data_path = args.results_dir / "distribute_save"
    elif config.jet.jet_method_name == "Classical":
        print("Classical Output Not yet known")
        sys.exit(0)
    else:
        data_path = args.results_dir / "LB_EvalData" / args.episode_tag

    sa_path = data_path / "span_averages.h5"
    traj_path = data_path / "trajectories.h5"
    output_dir = args.results_dir / "modeling_results"
    output_dir.mkdir(exist_ok=True)

    method = args.model_type.lower()
    if method == "dmdc":
        run_dmdc(sa_path, traj_path, output_dir)
    else:
        print(f"Unknown model type: {args.model_type}")
        
if __name__ == "__main__":
    main()
