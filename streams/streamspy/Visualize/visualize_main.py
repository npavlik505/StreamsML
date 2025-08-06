"""Entry point for visualization utilities."""

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
from SNAPSHOT import run_snapshot
from ANIMATION import run_animation

def main() -> None:
    parser = argparse.ArgumentParser(description="Run visualization on StreamsML output")
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--episode-tag", help="Episode tag used for learning based runs")
    parser.add_argument("--variable", default="rho", help="Variable to visualize (rho,u,v,w,E)")
    parser.add_argument("--snapshot-number", type= int, default= 1, help="Choose the snapshot to visualize")
    parser.add_argument("--vis-type", type=str, help="Visualization type")
    args = parser.parse_args()
    
    input_dir = Path( args.results_dir , "input.json" )
    with open(input_dir, "r") as f:
        cfg_json = json.load(f)
        config = Config.from_json(cfg_json)
        
    if config.jet.jet_method_name == "OpenLoop":
        analysis_path = args.results_dir / "distribute_save"
        
    elif config.jet.jet_method_name == "Classical":
        print('Classical Output Not yet known')
        sys.exit(0)
        
    else:
        analysis_path = args.results_dir / "LB_EvalData" / args.episode_tag

    sa_path = analysis_path / "span_averages.h5"
    output_dir = args.results_dir / "visualization_results"
    output_dir.mkdir(exist_ok=True)

    method = args.vis_type.lower()
    print(f'{method} being generated')
    if method == "snapshot":
        run_snapshot(sa_path, output_dir, args.variable, args.snapshot_number)
    elif method == "animation":
        run_animation(sa_path, output_dir, args.variable)
    else:
        print(f"Unknown visualization type: {args.vis_type}")

if __name__ == "__main__":
    main()
