'''RL METRICS - Observation Magnitude, Actuation, Reward'''

from pathlib import Path
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Create a plot for Obs Mag, Actuation, and Reward for RL Training and Eval
def run_rlmetrics(rl_metrics_path: Path, output_dir: Path) -> None:

    rl_training_metrics = rl_metrics_path / 'training.h5'
    rl_eval_metrics = rl_metrics_path / 'evaluation.h5'

    # Extract and mean center the data from the training.h5 file
    with h5py.File(rl_training_metrics, "r") as tgd:
        episodes, steps = tgd["time"].shape
            
        for ep in range(episodes):  
            t_time = tgd["time"][ep,:]
            
            t_amp = tgd["amplitude"][ep,:].astype(float)
            t_amp_mc = t_amp - np.nanmean(t_amp)
            
            t_reward = tgd["reward"][ep,:].astype(float)
            t_reward_mc = t_reward - np.nanmean(t_reward)
            
            # plot
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
            axes[0].plot(t_time, t_amp_mc)
            axes[0].set_ylabel("Actuation (mean-centered)")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(t_time, t_reward_mc)
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Reward (mean-centered)")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f'Training - Episode {ep}')
            fig.tight_layout()
            fig.savefig(output_dir / f'training_metrics_ep{ep:04d}.png', dpi=150)
            plt.close(fig)
        
    # Extract and mean center the data from the evaluation.h5 file
    with h5py.File(rl_eval_metrics, "r") as egd:
        episodes, steps = egd["time"].shape
            
        for ep in range(episodes):  
            e_time = egd["time"][ep,:]
            
            e_amp = egd["amplitude"][ep,:].astype(float)
            e_amp_mc = e_amp - np.nanmean(e_amp)
            
            e_reward = egd["reward"][ep,:].astype(float)
            e_reward_mc = e_reward - np.nanmean(e_reward)
            
            # plot
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
            axes[0].plot(e_time, e_amp_mc)
            axes[0].set_ylabel("Actuation (mean-centered)")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(e_time, e_reward_mc)
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Reward (mean-centered)")
            axes[1].grid(True, alpha=0.3)

            fig.suptitle(f'Evaluation - Episode {ep}')
            fig.tight_layout()
            fig.savefig(output_dir / f"eval_metrics_ep{ep:04d}.png", dpi=150)
            plt.close(fig)
