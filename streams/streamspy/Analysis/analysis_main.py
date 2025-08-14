"""Entry point for analysis utilities."""
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
from POD import run_pod
from DMD import run_dmd
from PYSENSORS import run_pysensors



if __name__ == "__main__":

    def main() -> None:
        parser = argparse.ArgumentParser(description="Run analysis on StreamsML output")
        parser.add_argument("--results_dir", type=Path, required=True, help="Path to output directory")
        parser.add_argument("--episode-tag", default="", help="Episode tag used for learning based runs")
        parser.add_argument("--analysis-method", default="POD", help="Analysis method to apply")
        parser.add_argument("--num-sensors", type=int, default=10, help="Number of sensors to select")
        args = parser.parse_args()
        
        input_dir = Path( args.results_dir , "input.json" )
        # with open("args.results_dir/input/input.json", "r") as f:
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

        dxdy_path = args.results_dir / "distribute_save"
        sa_path = analysis_path / "span_averages.h5"
        output_dir = args.results_dir / "analysis_results"

        method = args.analysis_method.lower()
        if method == "pod":
            run_pod(sa_path, dxdy_path, output_dir)
        elif method == "dmd":
            run_dmd(sa_path, output_dir)
        elif method == "pysensors":
            run_pysensors(sa_path, output_dir, args.num_sensors)
        else:
            print(f"Unknown analysis method: {args.analysis_method}")
        
if __name__ == "__main__":
    main()
